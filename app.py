from flask import Flask, jsonify, send_from_directory, render_template, request
import requests
import time
from datetime import datetime, timedelta
import threading
import json
import joblib
import numpy as np
from sklearn.ensemble import IsolationForest
from river import anomaly, preprocessing, compose
from sklearn.preprocessing import LabelEncoder # Explicitly import LabelEncoder for type hinting/clarity
import os
import csv
import atexit # Import atexit for cleanup on app shutdown

app = Flask(__name__, static_folder='static', template_folder='templates')

# --- Configuration ---
# IP address of the ESP32 acting as the Active Receiver (the one running the AP and WebServer).
# IMPORTANT: Replace with the actual IP address of your ESP32's SoftAP.
# Default ESP32 SoftAP IP is usually 192.168.4.1
# ESP32_IP = '192.168.4.1' # This is no longer used directly for remote access

# The Cloudflare Tunnel URL that points to your ESP32's base IP (e.g., 192.168.4.1)
# Make sure this matches the hostname provided by cloudflared tunnel.
# This URL is temporary for a quick tunnel and will change if you restart cloudflared without a named tunnel.
CLOUDFARE_TUNNEL_BASE_URL = 'https://restricted-encouraging-seeking-quarterly.trycloudflare.com'

# The endpoint on the ESP32 WebServer that provides sensor data.
ESP32_SENSOR_ENDPOINT = f'{CLOUDFARE_TUNNEL_BASE_URL}/sensor_data'
# The endpoint on the ESP32 WebServer to change its operating mode.
ESP32_SET_MODE_ENDPOINT = f'{CLOUDFARE_TUNNEL_BASE_URL}/set_mode'
# How often the Flask backend will poll the ESP32 for new data (in seconds).
POLLING_INTERVAL_SECONDS = 1
# How many seconds of historical overall data to retain for the frontend's charts.
# This aligns with the MAX_DATA_POINTS in the frontend (60 points * 1 sec/point = 1 minute).
FRONTEND_DATA_RETENTION_SECONDS = 60
# Timeout for forward filling. If a node hasn't been reported by the ESP32
# for this many seconds (as seen by Flask), its data is considered stale.
DATA_STALE_TIMEOUT_SECONDS = 60 # 1 minute

# File paths for pre-trained models and encoders
ISOLATION_FOREST_MODEL_PATH = 'offline_isolation_forest.joblib'
SCALER_PATH = 'offline_scaler.joblib'
MAC_ENCODER_PATH = 'mac_encoder.joblib'

# --- CSV Logging Configuration ---
CSV_LOG_DIRECTORY = 'csv_logs'
current_csv_file_path = None
csv_file_handle = None
csv_writer = None
is_logging_active = False
csv_lock = threading.Lock() # Lock for thread-safe CSV writing

# --- Global Data Stores ---
# Stores the latest data for each individual node, including anomaly scores and location.
# Keyed by node ID (string).
# Example: {
#   '1': {'rain': 100, 'soil': 500, 'vibration': 0.1, 'tilt': 5.0, 'mac': 'AA:BB:CC:DD:EE:FF',
#         'iso_score': 0.05, 'river_score': 15.2, 'warning_level_numerical': 0, 'warning_level_text': 'Low',
#         'latitude': 28.7041, 'longitude': 77.1025, 'last_seen_by_flask': datetime_obj},
# }
latest_node_data = {}

# Stores historical overall (averaged) sensor data.
# This list is used to provide the 'overall_data' to the frontend, which then plots it.
# Each entry is a dictionary: {'timestamp': datetime_obj, 'rain': avg_rain, ...}
overall_historical_data = []

# A lock to ensure thread-safe access to the global data stores,
# as `fetch_data_from_esp32` runs in a separate thread.
data_lock = threading.Lock()

# --- User-defined Node Locations (Non-persistent for now) ---
# This dictionary will store manually set locations, overriding the dummy ones.
# Key: nodeId (string), Value: {'latitude': float, 'longitude': float}
user_defined_node_locations = {}

# --- ML Model Instances ---
if_model: IsolationForest = None
scaler: preprocessing.StandardScaler = None
mac_encoder: LabelEncoder = None
river_model = anomaly.HalfSpaceTrees(seed=42) # Initialize River HalfSpaceTrees model

def load_ml_models():
    """Loads the pre-trained Isolation Forest model, scaler, and MAC encoder."""
    global if_model, scaler, mac_encoder
    try:
        if_model = joblib.load(ISOLATION_FOREST_MODEL_PATH)
        print(f"Successfully loaded Isolation Forest model from {ISOLATION_FOREST_MODEL_PATH}")
    except FileNotFoundError:
        print(f"Error: Isolation Forest model not found at {ISOLATION_FOREST_MODEL_PATH}. Please ensure it exists.")
        if_model = None
    except Exception as e:
        print(f"Error loading Isolation Forest model: {e}")
        if_model = None
    
    try:
        scaler = joblib.load(SCALER_PATH)
        print(f"Successfully loaded Scaler from {SCALER_PATH}")
    except FileNotFoundError:
        print(f"Error: Scaler not found at {SCALER_PATH}. Please ensure it exists.")
        scaler = None
    except Exception as e:
        print(f"Error loading Scaler: {e}")
        scaler = None

    try:
        mac_encoder = joblib.load(MAC_ENCODER_PATH)
        print(f"Successfully loaded MAC Encoder from {MAC_ENCODER_PATH}")
    except FileNotFoundError:
        print(f"Error: MAC Encoder not found at {MAC_ENCODER_PATH}. Please ensure it exists.")
        mac_encoder = None
    except Exception as e:
        print(f"Error loading MAC Encoder: {e}")
        mac_encoder = None

    if not (if_model and scaler and mac_encoder):
        print("Warning: Not all models/preprocessors loaded. Anomaly detection might be limited.")


def get_warning_level(iso_score, river_score):
    """
    Applies ensemble logic to determine the warning level as a numerical score (0.000 to 2.000)
    and a text level.
    Returns: (numerical_level, text_level)
    """
    iso_anomaly_threshold = -0.05
    river_anomaly_threshold = 20

    # Define assumed min/max possible scores for normalization
    # These might need tuning based on actual model outputs
    MIN_ISO_SCORE_EXPECTED = -0.5 # Most anomalous for IF (lower score)
    MAX_ISO_SCORE_EXPECTED = 0.5 # Most normal for IF (higher score)

    MIN_RIVER_SCORE_EXPECTED = 0   # Most normal for River (lower score)
    MAX_RIVER_SCORE_EXPECTED = 100 # Most anomalous for River (higher score)

    iso_contribution = 0.0
    river_contribution = 0.0

    if iso_score is not None:
        if iso_score < iso_anomaly_threshold:
            # Scale iso_score from [MIN_ISO_SCORE_EXPECTED, iso_anomaly_threshold] to [1, 0]
            # A score of MIN_ISO_SCORE_EXPECTED yields 1.0, iso_anomaly_threshold yields 0.0
            if (iso_anomaly_threshold - MIN_ISO_SCORE_EXPECTED) > 0:
                iso_contribution = (iso_anomaly_threshold - iso_score) / (iso_anomaly_threshold - MIN_ISO_SCORE_EXPECTED)
                iso_contribution = min(max(iso_contribution, 0.0), 1.0) # Clamp between 0 and 1
            else: # Handle case where threshold is same as min_expected (unlikely but for robustness)
                iso_contribution = 1.0 if iso_score <= iso_anomaly_threshold else 0.0
        # If iso_score is >= iso_anomaly_threshold, contribution is 0, which is already set.

    if river_score is not None:
        if river_score > river_anomaly_threshold:
            # Scale river_score from [river_anomaly_threshold, MAX_RIVER_SCORE_EXPECTED] to [0, 1]
            # A score of river_anomaly_threshold yields 0.0, MAX_RIVER_SCORE_EXPECTED yields 1.0
            if (MAX_RIVER_SCORE_EXPECTED - river_anomaly_threshold) > 0:
                river_contribution = (river_score - river_anomaly_threshold) / (MAX_RIVER_SCORE_EXPECTED - river_anomaly_threshold)
                river_contribution = min(max(river_contribution, 0.0), 1.0) # Clamp between 0 and 1
            else: # Handle case where threshold is same as max_expected (unlikely)
                river_contribution = 1.0 if river_score >= river_anomaly_threshold else 0.0
        # If river_score is <= river_anomaly_threshold, contribution is 0, which is already set.

    # Sum the contributions to get a numerical score between 0.0 and 2.0
    numerical_level = iso_contribution + river_contribution
    numerical_level = round(numerical_level, 3) # Round to 3 decimal places as requested

    # Determine text level based on the new numerical score ranges
    text_level = "Low"
    if numerical_level >= 0.5 and numerical_level < 1.5:
        text_level = "Mid"
    elif numerical_level >= 1.5:
        text_level = "High"

    return (numerical_level, text_level)

def init_csv_logger():
    """Initializes the CSV logging file."""
    global current_csv_file_path, csv_file_handle, csv_writer, is_logging_active

    with csv_lock:
        if csv_file_handle:
            print("CSV logger already active. Closing existing file before starting new one.")
            close_csv_logger()

        if not os.path.exists(CSV_LOG_DIRECTORY):
            os.makedirs(CSV_LOG_DIRECTORY)
            print(f"Created CSV log directory: {CSV_LOG_DIRECTORY}")

        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_csv_file_path = os.path.join(CSV_LOG_DIRECTORY, f"anomaly_log_{timestamp_str}.csv")

        try:
            csv_file_handle = open(current_csv_file_path, 'w', newline='')
            csv_writer = csv.writer(csv_file_handle)
            # Write header row
            csv_writer.writerow([
                'Timestamp', 'NodeID', 'MAC_Address',
                'Rain', 'Soil', 'Vibration', 'Tilt',
                'ISO_Score', 'River_Score', 'Warning_Level_Numerical', 'Warning_Level_Text'
            ])
            is_logging_active = True
            print(f"CSV logging started. Data will be saved to: {current_csv_file_path}")
            return True
        except IOError as e:
            print(f"Error opening CSV file for logging: {e}")
            current_csv_file_path = None
            csv_file_handle = None
            csv_writer = None
            is_logging_active = False
            return False

def close_csv_logger():
    """Closes the CSV logging file."""
    global csv_file_handle, csv_writer, is_logging_active
    with csv_lock:
        if csv_file_handle:
            try:
                csv_file_handle.close()
                print(f"CSV logging stopped. File closed: {current_csv_file_path}")
            except IOError as e:
                print(f"Error closing CSV file: {e}")
            finally:
                csv_file_handle = None
                csv_writer = None
                is_logging_active = False
        else:
            print("CSV logger not active or already closed.")


# --- Data Fetching Thread Function ---
def fetch_data_from_esp32():
    """
    This function runs in a separate thread. It periodically polls the ESP32
    Active Receiver for the latest sensor data from all known nodes.
    It updates the `latest_node_data` and `overall_historical_data` global variables,
    and performs anomaly detection.
    It also implements the forward filling timeout by removing stale node data.
    """
    global latest_node_data, overall_historical_data, river_model, is_logging_active, user_defined_node_locations

    # Dummy coordinates for demonstration. In a real app, these would come from ESP32 or a config.
    # We'll assign them based on node ID to make them distinct.
    BASE_LAT = 20.5937
    BASE_LON = 78.9629
    LAT_OFFSET_PER_NODE = 0.1
    LON_OFFSET_PER_NODE = 0.15

    while True:
        try:
            print(f"\n--- Polling Cycle Started ({datetime.now().strftime('%H:%M:%S')}) ---")
            print(f"Attempting to fetch data from ESP32 at {ESP32_SENSOR_ENDPOINT}...")
            response = requests.get(ESP32_SENSOR_ENDPOINT, timeout=5) # Increased timeout for remote connection
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

            esp32_nodes_data = response.json()
            print(f"Received raw data from ESP32: {esp32_nodes_data}")
            current_flask_time = datetime.now()

            with data_lock:
                nodes_reported_in_this_poll = set()

                for node_data in esp32_nodes_data:
                    node_id = str(node_data.get('nodeId'))
                    
                    if node_id:
                        rain = node_data.get('rain')
                        soil = node_data.get('soil')
                        vibration = node_data.get('vibration')
                        tilt = node_data.get('tilt')
                        mac = node_data.get('mac')

                        iso_score = None
                        river_score = None
                        warning_level_numerical = 0.0 # Initialize as float
                        warning_level_text = "Low"

                        # Determine coordinates: prioritize user-defined, then dummy, then N/A
                        latitude = None
                        longitude = None

                        if node_id in user_defined_node_locations:
                            latitude = user_defined_node_locations[node_id]['latitude']
                            longitude = user_defined_node_locations[node_id]['longitude']
                            print(f"Using user-defined location for Node {node_id}: [{latitude}, {longitude}]")
                        else:
                            try:
                                node_idx = int(node_id) - 1 # Assuming node IDs start from 1
                                latitude = BASE_LAT + (node_idx * LAT_OFFSET_PER_NODE)
                                longitude = BASE_LON + (node_idx * LON_OFFSET_PER_NODE)
                            except ValueError:
                                # Fallback for non-numeric node IDs or errors, keep as None
                                pass # latitude and longitude remain None

                        # Perform anomaly detection if models are loaded and data is complete
                        if all(v is not None for v in [rain, soil, vibration, tilt]) and \
                           if_model and scaler and mac_encoder:
                            
                            features_list = [float(rain), float(soil), float(vibration), float(tilt)]
                            
                            encoded_mac = None
                            if mac is not None:
                                try:
                                    if mac not in mac_encoder.classes_:
                                        # Dynamically add new MAC to encoder classes
                                        mac_encoder.classes_ = np.append(mac_encoder.classes_, mac)
                                        encoded_mac = mac_encoder.transform([mac])[0]
                                        print(f"Info: Added new MAC '{mac}' to encoder classes for Node {node_id}. New encoded value: {encoded_mac}")
                                    else:
                                        encoded_mac = mac_encoder.transform([mac])[0]
                                except Exception as e:
                                    print(f"Error encoding MAC '{mac}': {e}. Defaulting encoded_mac to 0.")
                                    encoded_mac = 0
                            else:
                                encoded_mac = 0 # Default if MAC is missing

                            features_list.append(float(encoded_mac))
                            input_array = np.array(features_list).reshape(1, -1)

                            # Isolation Forest Score
                            if scaler:
                                scaled_input = scaler.transform(input_array)
                                iso_score = if_model.decision_function(scaled_input)[0]
                            
                            # River HalfSpaceTrees Score (predict and learn)
                            river_input = {
                                'rain': float(rain),
                                'soil': float(soil),
                                'vibration': float(vibration),
                                'tilt': float(tilt),
                                'encoded_mac': float(encoded_mac)
                            }
                            print(f"[DEBUG - River Model] Node {node_id} - River Input: {river_input}")
                            river_score = river_model.score_one(river_input)
                            print(f"[DEBUG - River Model] Node {node_id} - Raw River Score: {river_score}")
                            river_model.learn_one(river_input) # Learn incrementally

                            # Ensemble Logic
                            warning_level_numerical, warning_level_text = get_warning_level(iso_score, river_score)
                            print(f"[DEBUG - River Model] Node {node_id} - Warning Level Numerical: {warning_level_numerical}, Text: {warning_level_text}")

                        # Update the node's data and its 'last_seen_by_flask' timestamp
                        latest_node_data[node_id] = {
                            'rain': rain,
                            'soil': soil,
                            'vibration': vibration,
                            'tilt': tilt,
                            'mac': mac,
                            'iso_score': iso_score,
                            'river_score': river_score,
                            'warning_level_numerical': warning_level_numerical,
                            'warning_level_text': warning_level_text,
                            'latitude': latitude, # Added latitude
                            'longitude': longitude, # Added longitude
                            'last_seen_by_flask': current_flask_time
                        }
                        nodes_reported_in_this_poll.add(node_id)

                        # --- CSV Logging ---
                        if is_logging_active and csv_writer:
                            with csv_lock:
                                try:
                                    csv_writer.writerow([
                                        current_flask_time.isoformat(),
                                        node_id,
                                        mac,
                                        rain,
                                        soil,
                                        vibration,
                                        tilt,
                                        iso_score,
                                        river_score,
                                        warning_level_numerical,
                                        warning_level_text
                                    ])
                                    csv_file_handle.flush() # Ensure data is written to disk immediately
                                except Exception as e:
                                    print(f"Error writing to CSV for node {node_id}: {e}")
                
                # --- Implement Forward Filling Timeout (Remove Stale Data) ---
                nodes_to_remove = []
                for node_id, data in list(latest_node_data.items()):
                    time_since_last_seen = current_flask_time - data['last_seen_by_flask']
                    
                    if node_id not in nodes_reported_in_this_poll:
                        if time_since_last_seen.total_seconds() > DATA_STALE_TIMEOUT_SECONDS:
                            nodes_to_remove.append(node_id)
                
                for node_id in nodes_to_remove:
                    print(f"Removing stale data for Node {node_id}.")
                    del latest_node_data[node_id]
                    # Also remove from user_defined_node_locations if node becomes stale
                    if node_id in user_defined_node_locations:
                        del user_defined_node_locations[node_id]
                        print(f"Removed user-defined location for stale Node {node_id}.")


                # --- Calculate Overall Averages for Historical Data ---
                if latest_node_data:
                    total_rain = 0.0
                    total_soil = 0.0
                    total_vibration = 0.0
                    total_tilt = 0.0
                    total_iso_score = 0.0
                    total_river_score = 0.0
                    total_warning_level = 0.0 # Initialize as float
                    
                    active_node_count = 0
                    iso_score_count = 0
                    river_score_count = 0
                    warning_level_count = 0

                    for node_id, data in latest_node_data.items():
                        if data['rain'] is not None:
                            total_rain += float(data['rain'])
                        if data['soil'] is not None:
                            total_soil += float(data['soil'])
                        if data['vibration'] is not None:
                            total_vibration += float(data['vibration'])
                        if data['tilt'] is not None:
                            total_tilt += float(data['tilt'])
                        
                        if data['iso_score'] is not None:
                            total_iso_score += data['iso_score']
                            iso_score_count += 1
                        if data['river_score'] is not None:
                            total_river_score += data['river_score']
                            river_score_count += 1
                        if data['warning_level_numerical'] is not None:
                            total_warning_level += data['warning_level_numerical']
                            warning_level_count += 1

                        active_node_count += 1

                    if active_node_count > 0:
                        overall_avg = {
                            'timestamp': current_flask_time,
                            'rain': total_rain / active_node_count,
                            'soil': total_soil / active_node_count,
                            'vibration': total_vibration / active_node_count,
                            'tilt': total_tilt / active_node_count,
                            'iso_score': total_iso_score / iso_score_count if iso_score_count > 0 else None,
                            'river_score': total_river_score / river_score_count if river_score_count > 0 else None,
                            'warning_level_numerical': total_warning_level / warning_level_count if warning_level_count > 0 else None
                        }
                        overall_historical_data.append(overall_avg)

                        cutoff_time = current_flask_time - timedelta(seconds=FRONTEND_DATA_RETENTION_SECONDS);
                        overall_historical_data = [
                            entry for entry in overall_historical_data if entry['timestamp'] >= cutoff_time
                        ]
                    else:
                        print("No valid data from active nodes to calculate overall averages.")
                        overall_historical_data = []
                else:
                    print("No node data available in latest_node_data to calculate overall averages.")
                    overall_historical_data = []

            print(f"Data fetched and processed successfully. Active nodes in Flask: {len(latest_node_data)}")
            print(f"--- Polling Cycle Ended ---\n")

        except requests.exceptions.Timeout:
            print(f"Connection to ESP32 via Cloudflare Tunnel timed out. Retrying...")
        except requests.exceptions.ConnectionError as e:
            print(f"Failed to connect to ESP32 via Cloudflare Tunnel. Check tunnel status. Error: {e}")
        except requests.exceptions.HTTPError as e:
            print(f"HTTP error from ESP32 via Cloudflare Tunnel: {e.response.status_code} - {e.response.text}")
        except json.JSONDecodeError:
            print("Error decoding JSON from ESP32 response via Cloudflare Tunnel. Check ESP32's /sensor_data output format.")
        except Exception as e:
            print(f"An unexpected error occurred in data fetching thread: {e}")

        time.sleep(POLLING_INTERVAL_SECONDS)

# --- Flask Routes ---

@app.route('/api/live_data')
def live_data():
    """
    API endpoint for the web frontend to fetch live sensor data, including anomaly scores.
    """
    with data_lock:
        latest_overall = overall_historical_data[-1] if overall_historical_data else {
            'rain': None, 'soil': None, 'vibration': None, 'tilt': None,
            'iso_score': None, 'river_score': None, 'warning_level_numerical': None
        }

        nodes_for_frontend = {}
        for node_id, data in latest_node_data.items():
            # Prioritize user-defined locations if available
            lat_to_send = user_defined_node_locations.get(node_id, {}).get('latitude', data['latitude'])
            lon_to_send = user_defined_node_locations.get(node_id, {}).get('longitude', data['longitude'])

            nodes_for_frontend[node_id] = {
                'rain': data['rain'],
                'soil': data['soil'],
                'vibration': data['vibration'],
                'tilt': data['tilt'],
                'mac': data['mac'], # Include MAC address
                'iso_score': data['iso_score'],
                'river_score': data['river_score'],
                'warning_level_numerical': data['warning_level_numerical'],
                'warning_level_text': data['warning_level_text'],
                'latitude': lat_to_send, # Use potentially overridden latitude
                'longitude': lon_to_send, # Use potentially overridden longitude
                'timestamp': data['last_seen_by_flask'].isoformat()
            }

        response_data = {
            'timestamp': datetime.now().isoformat(),
            'overall_data': {
                'rain': latest_overall['rain'],
                'soil': latest_overall['soil'],
                'vibration': latest_overall['vibration'],
                'tilt': latest_overall['tilt'],
                'iso_score': latest_overall['iso_score'],
                'river_score': latest_overall['river_score'],
                'warning_level_numerical': latest_overall['warning_level_numerical']
            },
            'nodes': nodes_for_frontend,
            'is_logging_active': is_logging_active # Send logging status to frontend
        }
    return jsonify(response_data)

@app.route('/api/set_node_location', methods=['POST'])
def set_node_location():
    """
    API endpoint to manually set or update a node's location.
    This location will override any dummy or ESP32-reported location for the session.
    """
    data = request.get_json()
    node_id = data.get('nodeId')
    latitude = data.get('latitude')
    longitude = data.get('longitude')

    if not node_id or latitude is None or longitude is None:
        return jsonify({"status": "error", "message": "Node ID, latitude, and longitude are required."}), 400

    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        return jsonify({"status": "error", "message": "Latitude and longitude must be numbers."}), 400

    with data_lock:
        user_defined_node_locations[node_id] = {'latitude': latitude, 'longitude': longitude}
        # If the node is currently active, update its live data immediately
        if node_id in latest_node_data:
            latest_node_data[node_id]['latitude'] = latitude
            latest_node_data[node_id]['longitude'] = longitude
            print(f"Updated live data for Node {node_id} with user-defined location.")
        else:
            print(f"Stored user-defined location for Node {node_id}. Node not currently active in live data.")

    print(f"User set location for Node {node_id}: Lat={latitude}, Lon={longitude}")
    return jsonify({"status": "success", "message": f"Location for Node {node_id} set successfully."})


@app.route('/api/set_mode', methods=['POST'])
def set_mode():
    """
    API endpoint for the web frontend to set the ESP32's operating mode.
    """
    data = request.get_json()
    mode = data.get('mode')

    if mode not in ['data_collection', 'standby']:
        return jsonify({"status": "error", "message": "Invalid mode specified. Must be 'data_collection' or 'standby'."}), 400

    try:
        # Use the Cloudflare Tunnel URL for setting mode
        esp32_response = requests.get(f"{ESP32_SET_MODE_ENDPOINT}?mode={mode}", timeout=5)
        esp32_response.raise_for_status()
        
        esp32_data = esp32_response.json()
        if esp32_data.get("status") == "success":
            print(f"Successfully set ESP32 to {mode} mode.")
            with data_lock:
                overall_historical_data.clear()
                latest_node_data.clear()
                user_defined_node_locations.clear() # Clear user locations on mode change
                # Re-initialize river model when mode changes to ensure fresh learning
                global river_model
                river_model = anomaly.HalfSpaceTrees(seed=42)
            return jsonify({"status": "success", "message": f"ESP32 set to {mode} mode.", "esp32_response": esp32_data})
        else:
            print(f"ESP32 reported an error setting mode: {esp32_data.get('error', 'Unknown error')}")
            return jsonify({"status": "error", "message": f"ESP32 failed to set mode: {esp32_data.get('error', 'Unknown error')}", "esp32_response": esp32_data}), 500

    except requests.exceptions.Timeout:
        print(f"Timeout while trying to set mode on ESP32 via Cloudflare Tunnel.")
        return jsonify({"status": "error", "message": "Timeout connecting to ESP32 to set mode."}), 500
    except requests.exceptions.ConnectionError as e:
        print(f"Connection error to ESP32 via Cloudflare Tunnel while setting mode: {e}")
        return jsonify({"status": "error", "message": "Connection error to ESP32 to set mode."}), 500
    except requests.exceptions.HTTPError as e:
        print(f"HTTP error from ESP32 via Cloudflare Tunnel while setting mode: {e.response.status_code} - {e.response.text}")
        return jsonify({"status": "error", "message": f"HTTP error from ESP32: {e.response.status_code} - {e.response.text}"}), 500
    except json.JSONDecodeError:
        print("Error decoding JSON from ESP32 response when setting mode via Cloudflare Tunnel.")
        return jsonify({"status": "error", "message": "Invalid JSON response from ESP32 when setting mode."}), 500
    except Exception as e:
        print(f"An unexpected error occurred while setting mode: {e}")
        return jsonify({"status": "error", "message": f"An unexpected error occurred: {e}"}), 500

@app.route('/api/start_csv_logging', methods=['POST'])
def start_csv_logging():
    """API endpoint to start CSV logging."""
    global is_logging_active
    if is_logging_active:
        return jsonify({"status": "info", "message": "CSV logging is already active."})
    
    if init_csv_logger():
        return jsonify({"status": "success", "message": "CSV logging started."})
    else:
        return jsonify({"status": "error", "message": "Failed to start CSV logging."}), 500

@app.route('/api/stop_csv_logging', methods=['POST'])
def stop_csv_logging():
    """API endpoint to stop CSV logging."""
    global is_logging_active
    if not is_logging_active:
        return jsonify({"status": "info", "message": "CSV logging is not active."})

    close_csv_logger()
    return jsonify({"status": "success", "message": "CSV logging stopped."})

@app.route('/api/download_csv')
def download_csv():
    """API endpoint to download the current CSV log file."""
    global current_csv_file_path
    if current_csv_file_path and os.path.exists(current_csv_file_path):
        # Ensure the file is closed before sending for download
        close_csv_logger() # Close it if it's still open for logging

        directory = os.path.dirname(current_csv_file_path)
        filename = os.path.basename(current_csv_file_path)
        print(f"Attempting to send file: {filename} from directory: {directory}")
        return send_from_directory(directory, filename, as_attachment=True)
    else:
        return jsonify({"status": "error", "message": "No CSV file available for download or file not found."}), 404

# --- New Routes for Home and About Pages ---
@app.route('/')
def home_page():
    """
    Serves the home HTML page (`home.html`) from the templates folder.
    This will now be the default route.
    """
    print("Serving home.html from templates folder...")
    return render_template('home.html')

@app.route('/dashboard')
def index():
    """
    Serves the main dashboard HTML page (`index.html`) from the templates folder.
    Renamed from '/' to '/dashboard'.
    """
    print("Serving index.html (dashboard) from templates folder...")
    return render_template('index.html')

@app.route('/about')
def about_page():
    """
    Serves the about HTML page (`about.html`) from the templates folder.
    """
    print("Serving about.html from templates folder...")
    return render_template('about.html')

@app.route('/map')
def map_page():
    """
    Serves the map HTML page (`map.html`) from the templates folder.
    """
    print("Serving map.html from templates folder...")
    return render_template('map.html')

# --- Main Execution Block ---
if __name__ == '__main__':
    load_ml_models() # Load models once at startup
    data_fetch_thread = threading.Thread(target=fetch_data_from_esp32, daemon=True)
    data_fetch_thread.start()
    print("Data fetching and ML processing thread started.")

    # Register close_csv_logger to be called when the app shuts down
    atexit.register(close_csv_logger)

    print("Starting Flask app on http://0.0.0.0:5000")
    app.run(host='0.0.0.0', port=5000, debug=True)
