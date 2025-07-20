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

print("Flask application starting up...")

# Initialize Flask app.
# static_folder and template_folder are correctly set if your structure matches.
app = Flask(__name__, static_folder='static', template_folder='templates')
print(f"Flask app initialized with static_folder='{app.static_folder}' and template_folder='{app.template_folder}'")

# --- Configuration ---
# The Cloudflare Tunnel URL that points to your ESP32's base IP (e.g., 192.168.4.1)
# IMPORTANT: This URL is CRITICAL for Render deployment.
# The 'trycloudflare.com' URLs are temporary. If you restart your Cloudflare Tunnel
# without a persistent named tunnel, this URL will change.
# Render's server needs to be able to reach this URL to fetch data from your ESP32.
# You MUST ensure this URL is up-to-date and accessible from Render's network.
CLOUDFARE_TUNNEL_BASE_URL = os.environ.get('CLOUDFARE_TUNNEL_BASE_URL', 'https://performed-equation-stack-bracelet.trycloudflare.com')
print(f"Configured CLOUDFARE_TUNNEL_BASE_URL: {CLOUDFARE_TUNNEL_BASE_URL}")

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
print(f"Model paths configured: IF='{ISOLATION_FOREST_MODEL_PATH}', Scaler='{SCALER_PATH}', MAC='{MAC_ENCODER_PATH}'")

# --- CSV Logging Configuration ---
CSV_LOG_DIRECTORY = 'csv_logs'
current_csv_file_path = None
csv_file_handle = None
csv_writer = None
is_logging_active = False
csv_lock = threading.Lock() # Lock for thread-safe CSV writing
print(f"CSV log directory configured: {CSV_LOG_DIRECTORY}")

# --- Global Data Stores ---
# Stores the latest data for each individual node, including anomaly scores and location.
# Keyed by node ID (string).
# Example: {
#   '1': {'rain': 100, 'soil': 500, 'vibration': 0.1, 'tilt': 5.0, 'mac': 'AA:BB:CC:DD:EE:FF',
#         'iso_score': 0.05, 'river_score': 15.2, 'warning_level_numerical': 0, 'warning_level_text': 'Low',
#         'latitude': 28.7041, 'longitude': 77.1025, 'last_seen_by_flask': datetime_obj},
# }
latest_node_data = {}
print("Initialized latest_node_data.")

# Stores historical overall (averaged) sensor data.
# This list is used to provide the 'overall_data' to the frontend, which then plots it.
# Each entry is a dictionary: {'timestamp': datetime_obj, 'rain': avg_rain, ...}
overall_historical_data = []
print("Initialized overall_historical_data.")

# A lock to ensure thread-safe access to the global data stores,
# as `fetch_data_from_esp32` runs in a separate thread.
data_lock = threading.Lock()
print("Initialized data_lock.")

# --- User-defined Node Locations (Non-persistent for now) ---
# This dictionary will store manually set locations, overriding the dummy ones.
# Key: nodeId (string), Value: {'latitude': float, 'longitude': float}
user_defined_node_locations = {}
print("Initialized user_defined_node_locations.")

# --- ML Model Instances ---
if_model: IsolationForest = None
scaler: preprocessing.StandardScaler = None
mac_encoder: LabelEncoder = None
river_model = anomaly.HalfSpaceTrees(seed=42) # Initialize River HalfSpaceTrees model
print("Initialized ML model placeholders.")

def load_ml_models():
    """Loads the pre-trained Isolation Forest model, scaler, and MAC encoder."""
    global if_model, scaler, mac_encoder
    print("Attempting to load ML models...")
    try:
        if_model = joblib.load(ISOLATION_FOREST_MODEL_PATH)
        print(f"DEBUG: Successfully loaded Isolation Forest model from {ISOLATION_FOREST_MODEL_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Isolation Forest model not found at {ISOLATION_FOREST_MODEL_PATH}. Please ensure it exists in your repo.")
        if_model = None
    except Exception as e:
        print(f"ERROR: Error loading Isolation Forest model: {e}")
        if_model = None
    
    try:
        scaler = joblib.load(SCALER_PATH)
        print(f"DEBUG: Successfully loaded Scaler from {SCALER_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Scaler not found at {SCALER_PATH}. Please ensure it exists in your repo.")
        scaler = None
    except Exception as e:
        print(f"ERROR: Error loading Scaler: {e}")
        scaler = None

    try:
        mac_encoder = joblib.load(MAC_ENCODER_PATH)
        print(f"DEBUG: Successfully loaded MAC Encoder from {MAC_ENCODER_PATH}")
    except FileNotFoundError:
        print(f"ERROR: MAC Encoder not found at {MAC_ENCODER_PATH}. Please ensure it exists in your repo.")
        mac_encoder = None
    except Exception as e:
        print(f"ERROR: Error loading MAC Encoder: {e}")
        mac_encoder = None

    if not (if_model and scaler and mac_encoder):
        print("WARNING: Not all models/preprocessors loaded. Anomaly detection might be limited or fail.")
    else:
        print("DEBUG: All ML models and preprocessors loaded successfully.")


def get_warning_level(iso_score, river_score):
    """
    Applies ensemble logic to determine the warning level as a numerical score (0.000 to 2.000)
    and a text level.
    Returns: (numerical_level, text_level)
    """
    iso_anomaly_threshold = -0.05
    river_anomaly_threshold = 20

    MIN_ISO_SCORE_EXPECTED = -0.5
    MAX_ISO_SCORE_EXPECTED = 0.5

    MIN_RIVER_SCORE_EXPECTED = 0
    MAX_RIVER_SCORE_EXPECTED = 100

    iso_contribution = 0.0
    river_contribution = 0.0

    if iso_score is not None:
        if iso_score < iso_anomaly_threshold:
            if (iso_anomaly_threshold - MIN_ISO_SCORE_EXPECTED) > 0:
                iso_contribution = (iso_anomaly_threshold - iso_score) / (iso_anomaly_threshold - MIN_ISO_SCORE_EXPECTED)
                iso_contribution = min(max(iso_contribution, 0.0), 1.0)
            else:
                iso_contribution = 1.0 if iso_score <= iso_anomaly_threshold else 0.0

    if river_score is not None:
        if river_score > river_anomaly_threshold:
            if (MAX_RIVER_SCORE_EXPECTED - river_anomaly_threshold) > 0:
                river_contribution = (river_score - river_anomaly_threshold) / (MAX_RIVER_SCORE_EXPECTED - river_anomaly_threshold)
                river_contribution = min(max(river_contribution, 0.0), 1.0)
            else:
                river_contribution = 1.0 if river_score >= river_anomaly_threshold else 0.0

    numerical_level = iso_contribution + river_contribution
    numerical_level = round(numerical_level, 3)

    text_level = "Low"
    if numerical_level >= 0.5 and numerical_level < 1.5:
        text_level = "Mid"
    elif numerical_level >= 1.5:
        text_level = "High"
    
    print(f"DEBUG: Calculated warning level: Numerical={numerical_level}, Text={text_level}")
    return (numerical_level, text_level)

def init_csv_logger():
    """Initializes the CSV logging file."""
    global current_csv_file_path, csv_file_handle, csv_writer, is_logging_active
    print("DEBUG: Attempting to initialize CSV logger.")
    with csv_lock:
        if csv_file_handle:
            print("INFO: CSV logger already active. Closing existing file before starting new one.")
            close_csv_logger()

        if not os.path.exists(CSV_LOG_DIRECTORY):
            os.makedirs(CSV_LOG_DIRECTORY)
            print(f"DEBUG: Created CSV log directory: {CSV_LOG_DIRECTORY}")

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
            print(f"INFO: CSV logging started. Data will be saved to: {current_csv_file_path}")
            return True
        except IOError as e:
            print(f"ERROR: Error opening CSV file for logging: {e}")
            current_csv_file_path = None
            csv_file_handle = None
            csv_writer = None
            is_logging_active = False
            return False

def close_csv_logger():
    """Closes the CSV logging file."""
    global csv_file_handle, csv_writer, is_logging_active
    print("DEBUG: Attempting to close CSV logger.")
    with csv_lock:
        if csv_file_handle:
            try:
                csv_file_handle.close()
                print(f"INFO: CSV logging stopped. File closed: {current_csv_file_path}")
            except IOError as e:
                print(f"ERROR: Error closing CSV file: {e}")
            finally:
                csv_file_handle = None
                csv_writer = None
                is_logging_active = False
        else:
            print("INFO: CSV logger not active or already closed.")


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
        print(f"\n--- Polling Cycle Started ({datetime.now().strftime('%H:%M:%S')}) ---")
        print(f"DEBUG: Attempting to fetch data from ESP32 at {ESP32_SENSOR_ENDPOINT}...")
        try:
            # This is where your Render server tries to connect to your Cloudflare Tunnel.
            # If the tunnel URL is invalid or not reachable, this will fail.
            response = requests.get(ESP32_SENSOR_ENDPOINT, timeout=5) # Increased timeout for remote connection
            response.raise_for_status() # Raise an HTTPError for bad responses (4xx or 5xx)

            esp32_nodes_data = response.json()
            print(f"DEBUG: Received raw data from ESP32: {esp32_nodes_data}")
            current_flask_time = datetime.now()

            with data_lock:
                nodes_reported_in_this_poll = set()

                for node_data in esp32_nodes_data:
                    node_id = str(node_data.get('nodeId'))
                    print(f"DEBUG: Processing data for Node ID: {node_id}")
                    
                    if node_id:
                        rain = node_data.get('rain')
                        soil = node_data.get('soil')
                        vibration = node_data.get('vibration')
                        tilt = node_data.get('tilt')
                        mac = node_data.get('mac')
                        print(f"DEBUG: Node {node_id} - Raw Sensor Data: Rain={rain}, Soil={soil}, Vib={vibration}, Tilt={tilt}, MAC={mac}")

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
                            print(f"DEBUG: Using user-defined location for Node {node_id}: [{latitude}, {longitude}]")
                        else:
                            try:
                                node_idx = int(node_id) - 1 # Assuming node IDs start from 1
                                latitude = BASE_LAT + (node_idx * LAT_OFFSET_PER_NODE)
                                longitude = BASE_LON + (node_idx * LON_OFFSET_PER_NODE)
                                print(f"DEBUG: Using dummy location for Node {node_id}: [{latitude}, {longitude}]")
                            except ValueError:
                                print(f"WARNING: Node ID '{node_id}' is not numeric. Cannot assign dummy location.")
                                pass # latitude and longitude remain None

                        # Perform anomaly detection if models are loaded and data is complete
                        if all(v is not None for v in [rain, soil, vibration, tilt]) and \
                           if_model and scaler and mac_encoder:
                            print(f"DEBUG: Performing anomaly detection for Node {node_id}")
                            features_list = [float(rain), float(soil), float(vibration), float(tilt)]
                            
                            encoded_mac = None
                            if mac is not None:
                                try:
                                    if mac not in mac_encoder.classes_:
                                        # Dynamically add new MAC to encoder classes
                                        # This might not be ideal for production if MACs are truly dynamic
                                        # and you want a fixed set for the model. Consider re-training or
                                        # handling unknown MACs differently.
                                        mac_encoder.classes_ = np.append(mac_encoder.classes_, mac)
                                        encoded_mac = mac_encoder.transform([mac])[0]
                                        print(f"INFO: Added new MAC '{mac}' to encoder classes for Node {node_id}. New encoded value: {encoded_mac}")
                                    else:
                                        encoded_mac = mac_encoder.transform([mac])[0]
                                except Exception as e:
                                    print(f"ERROR: Error encoding MAC '{mac}': {e}. Defaulting encoded_mac to 0.")
                                    encoded_mac = 0
                            else:
                                encoded_mac = 0 # Default if MAC is missing
                            print(f"DEBUG: Node {node_id} - Encoded MAC: {encoded_mac}")

                            features_list.append(float(encoded_mac))
                            input_array = np.array(features_list).reshape(1, -1)
                            print(f"DEBUG: Node {node_id} - Input array for models: {input_array}")

                            # Isolation Forest Score
                            if scaler:
                                scaled_input = scaler.transform(input_array)
                                iso_score = if_model.decision_function(scaled_input)[0]
                                print(f"DEBUG: Node {node_id} - ISO Score: {iso_score}")
                            
                            # River HalfSpaceTrees Score (predict and learn)
                            river_input = {
                                'rain': float(rain),
                                'soil': float(soil),
                                'vibration': float(vibration),
                                'tilt': float(tilt),
                                'encoded_mac': float(encoded_mac)
                            }
                            print(f"DEBUG: Node {node_id} - River Input: {river_input}")
                            river_score = river_model.score_one(river_input)
                            print(f"DEBUG: Node {node_id} - Raw River Score: {river_score}")
                            river_model.learn_one(river_input) # Learn incrementally

                            # Ensemble Logic
                            warning_level_numerical, warning_level_text = get_warning_level(iso_score, river_score)
                            print(f"DEBUG: Node {node_id} - Final Warning Level: Numerical={warning_level_numerical}, Text={warning_level_text}")
                        else:
                            print(f"WARNING: Skipping anomaly detection for Node {node_id} due to missing data or unloaded models.")

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
                        print(f"DEBUG: Updated latest_node_data for Node {node_id}.")

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
                                    print(f"DEBUG: Logged data for Node {node_id} to CSV.")
                                except Exception as e:
                                    print(f"ERROR: Error writing to CSV for node {node_id}: {e}")
                
                # --- Implement Forward Filling Timeout (Remove Stale Data) ---
                nodes_to_remove = []
                for node_id, data in list(latest_node_data.items()):
                    time_since_last_seen = current_flask_time - data['last_seen_by_flask']
                    
                    if node_id not in nodes_reported_in_this_poll:
                        if time_since_last_seen.total_seconds() > DATA_STALE_TIMEOUT_SECONDS:
                            nodes_to_remove.append(node_id)
                
                for node_id in nodes_to_remove:
                    print(f"INFO: Removing stale data for Node {node_id}.")
                    del latest_node_data[node_id]
                    # Also remove from user_defined_node_locations if node becomes stale
                    if node_id in user_defined_node_locations:
                        del user_defined_node_locations[node_id]
                        print(f"INFO: Removed user-defined location for stale Node {node_id}.")


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
                        print(f"DEBUG: Appended new overall average data: {overall_avg}")

                        cutoff_time = current_flask_time - timedelta(seconds=FRONTEND_DATA_RETENTION_SECONDS);
                        overall_historical_data = [
                            entry for entry in overall_historical_data if entry['timestamp'] >= cutoff_time
                        ]
                        print(f"DEBUG: Trimmed historical data. Current length: {len(overall_historical_data)}")
                    else:
                        print("INFO: No valid data from active nodes to calculate overall averages. Clearing historical data.")
                        overall_historical_data = []
                else:
                    print("INFO: No node data available in latest_node_data to calculate overall averages. Clearing historical data.")
                    overall_historical_data = []

            print(f"INFO: Data fetched and processed successfully. Active nodes in Flask: {len(latest_node_data)}")
            print(f"--- Polling Cycle Ended ---\n")

        except requests.exceptions.Timeout:
            print(f"ERROR: Connection to ESP32 via Cloudflare Tunnel timed out. Check tunnel status and network connectivity. Retrying...")
            # This error in Render logs means Render couldn't reach your tunnel.
        except requests.exceptions.ConnectionError as e:
            print(f"ERROR: Failed to connect to ESP32 via Cloudflare Tunnel. Check tunnel status and URL. Error: {e}")
            # This error in Render logs means Render couldn't establish a connection.
        except requests.exceptions.HTTPError as e:
            print(f"ERROR: HTTP error from ESP32 via Cloudflare Tunnel: {e.response.status_code} - {e.response.text}")
            # This error means the tunnel was reached, but the ESP32 returned an HTTP error.
        except json.JSONDecodeError:
            print("ERROR: Error decoding JSON from ESP32 response via Cloudflare Tunnel. Check ESP32's /sensor_data output format.")
        except Exception as e:
            print(f"CRITICAL ERROR: An unexpected error occurred in data fetching thread: {e}")

        time.sleep(POLLING_INTERVAL_SECONDS)

# --- Flask Routes ---

@app.route('/api/live_data')
def live_data():
    """
    API endpoint for the web frontend to fetch live sensor data, including anomaly scores.
    This is the endpoint your frontend JavaScript should be calling.
    """
    print("DEBUG: /api/live_data endpoint hit.")
    with data_lock:
        latest_overall = overall_historical_data[-1] if overall_historical_data else {
            'rain': None, 'soil': None, 'vibration': None, 'tilt': None,
            'iso_score': None, 'river_score': None, 'warning_level_numerical': None
        }
        print(f"DEBUG: Latest overall data: {latest_overall}")

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
        print(f"DEBUG: Nodes prepared for frontend: {nodes_for_frontend}")

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
    print("DEBUG: Sending live_data JSON response.")
    return jsonify(response_data)

@app.route('/api/set_node_location', methods=['POST'])
def set_node_location():
    """
    API endpoint to manually set or update a node's location.
    This location will override any dummy or ESP32-reported location for the session.
    """
    print("DEBUG: /api/set_node_location endpoint hit.")
    data = request.get_json()
    node_id = data.get('nodeId')
    latitude = data.get('latitude')
    longitude = data.get('longitude')
    print(f"DEBUG: Received data for set_node_location: NodeID={node_id}, Lat={latitude}, Lon={longitude}")

    if not node_id or latitude is None or longitude is None:
        print("ERROR: Missing data for set_node_location.")
        return jsonify({"status": "error", "message": "Node ID, latitude, and longitude are required."}), 400

    if not isinstance(latitude, (int, float)) or not isinstance(longitude, (int, float)):
        print("ERROR: Invalid data types for latitude/longitude in set_node_location.")
        return jsonify({"status": "error", "message": "Latitude and longitude must be numbers."}), 400

    with data_lock:
        user_defined_node_locations[node_id] = {'latitude': latitude, 'longitude': longitude}
        # If the node is currently active, update its live data immediately
        if node_id in latest_node_data:
            latest_node_data[node_id]['latitude'] = latitude
            latest_node_data[node_id]['longitude'] = longitude
            print(f"INFO: Updated live data for Node {node_id} with user-defined location.")
        else:
            print(f"INFO: Stored user-defined location for Node {node_id}. Node not currently active in live data.")

    print(f"INFO: User set location for Node {node_id}: Lat={latitude}, Lon={longitude}. Sending success response.")
    return jsonify({"status": "success", "message": f"Location for Node {node_id} set successfully."})


@app.route('/api/set_mode', methods=['POST'])
def set_mode():
    """
    API endpoint for the web frontend to set the ESP32's operating mode.
    """
    print("DEBUG: /api/set_mode endpoint hit.")
    data = request.get_json()
    mode = data.get('mode')
    print(f"DEBUG: Received mode for set_mode: {mode}")

    if mode not in ['data_collection', 'standby']:
        print(f"ERROR: Invalid mode '{mode}' specified for set_mode.")
        return jsonify({"status": "error", "message": "Invalid mode specified. Must be 'data_collection' or 'standby'."}), 400

    try:
        # Use the Cloudflare Tunnel URL for setting mode
        print(f"DEBUG: Sending mode change request to ESP32 at {ESP32_SET_MODE_ENDPOINT}?mode={mode}")
        esp32_response = requests.get(f"{ESP32_SET_MODE_ENDPOINT}?mode={mode}", timeout=5)
        esp32_response.raise_for_status()
        
        esp32_data = esp32_response.json()
        print(f"DEBUG: ESP32 response for set_mode: {esp32_data}")
        if esp32_data.get("status") == "success":
            print(f"INFO: Successfully set ESP32 to {mode} mode. Clearing data stores.")
            with data_lock:
                overall_historical_data.clear()
                latest_node_data.clear()
                user_defined_node_locations.clear() # Clear user locations on mode change
                # Re-initialize river model when mode changes to ensure fresh learning
                global river_model
                river_model = anomaly.HalfSpaceTrees(seed=42)
            print("INFO: Data stores cleared and River model re-initialized.")
            return jsonify({"status": "success", "message": f"ESP32 set to {mode} mode.", "esp32_response": esp32_data})
        else:
            print(f"ERROR: ESP32 reported an error setting mode: {esp32_data.get('error', 'Unknown error')}")
            return jsonify({"status": "error", "message": f"ESP32 failed to set mode: {esp32_data.get('error', 'Unknown error')}", "esp32_response": esp32_data}), 500

    except requests.exceptions.Timeout:
        print(f"ERROR: Timeout while trying to set mode on ESP32 via Cloudflare Tunnel.")
        return jsonify({"status": "error", "message": "Timeout connecting to ESP32 to set mode."}), 500
    except requests.exceptions.ConnectionError as e:
        print(f"ERROR: Connection error to ESP32 via Cloudflare Tunnel while setting mode: {e}")
        return jsonify({"status": "error", "message": "Connection error to ESP32 to set mode."}), 500
    except requests.exceptions.HTTPError as e:
        print(f"ERROR: HTTP error from ESP32 via Cloudflare Tunnel while setting mode: {e.response.status_code} - {e.response.text}")
        return jsonify({"status": "error", "message": f"HTTP error from ESP32: {e.response.status_code} - {e.response.text}"}), 500
    except json.JSONDecodeError:
        print("ERROR: Error decoding JSON from ESP32 response when setting mode via Cloudflare Tunnel.")
        return jsonify({"status": "error", "message": "Invalid JSON response from ESP32 when setting mode."}), 500
    except Exception as e:
        print(f"CRITICAL ERROR: An unexpected error occurred while setting mode: {e}")
        return jsonify({"status": "error", "message": f"An unexpected error occurred: {e}"}), 500

@app.route('/api/start_csv_logging', methods=['POST'])
def start_csv_logging():
    """API endpoint to start CSV logging."""
    print("DEBUG: /api/start_csv_logging endpoint hit.")
    global is_logging_active
    if is_logging_active:
        print("INFO: CSV logging is already active. No action taken.")
        return jsonify({"status": "info", "message": "CSV logging is already active."})
    
    if init_csv_logger():
        print("INFO: CSV logging successfully started.")
        return jsonify({"status": "success", "message": "CSV logging started."})
    else:
        print("ERROR: Failed to start CSV logging.")
        return jsonify({"status": "error", "message": "Failed to start CSV logging."}), 500

@app.route('/api/stop_csv_logging', methods=['POST'])
def stop_csv_logging():
    """API endpoint to stop CSV logging."""
    print("DEBUG: /api/stop_csv_logging endpoint hit.")
    global is_logging_active
    if not is_logging_active:
        print("INFO: CSV logging is not active. No action taken.")
        return jsonify({"status": "info", "message": "CSV logging is not active."})

    close_csv_logger()
    print("INFO: CSV logging successfully stopped.")
    return jsonify({"status": "success", "message": "CSV logging stopped."})

@app.route('/api/download_csv')
def download_csv():
    """API endpoint to download the current CSV log file."""
    print("DEBUG: /api/download_csv endpoint hit.")
    global current_csv_file_path
    if current_csv_file_path and os.path.exists(current_csv_file_path):
        # Ensure the file is closed before sending for download
        close_csv_logger() # Close it if it's still open for logging
        print("INFO: CSV file closed before download.")

        directory = os.path.dirname(current_csv_file_path)
        filename = os.path.basename(current_csv_file_path)
        print(f"INFO: Attempting to send file for download: {filename} from directory: {directory}")
        return send_from_directory(directory, filename, as_attachment=True)
    else:
        print("WARNING: No CSV file available for download or file not found.")
        return jsonify({"status": "error", "message": "No CSV file available for download or file not found."}), 404

# --- Routes for Frontend Pages ---
@app.route('/')
def home_page():
    """
    Serves the home HTML page (`home.html`) from the templates folder.
    This will now be the default route.
    """
    print("DEBUG: Serving home.html from templates folder...")
    return render_template('home.html')

@app.route('/dashboard')
def index():
    """
    Serves the main dashboard HTML page (`index.html`) from the templates folder.
    Renamed from '/' to '/dashboard'.
    """
    print("DEBUG: Serving index.html (dashboard) from templates folder...")
    return render_template('index.html')

@app.route('/about')
def about_page():
    """
    Serves the about HTML page (`about.html`) from the templates folder.
    """
    print("DEBUG: Serving about.html from templates folder...")
    return render_template('about.html')

@app.route('/map')
def map_page():
    """
    Serves the map HTML page (`map.html`) from the templates folder.
    """
    print("DEBUG: Serving map.html from templates folder...")
    return render_template('map.html')

# --- Main Execution Block ---
if __name__ == '__main__':
    print("INFO: Running Flask app in local development mode (via __main__).")
    load_ml_models() # Load models once at startup
    data_fetch_thread = threading.Thread(target=fetch_data_from_esp32, daemon=True)
    data_fetch_thread.start()
    print("INFO: Data fetching and ML processing thread started.")

    # Register close_csv_logger to be called when the app shuts down
    atexit.register(close_csv_logger)
    print("INFO: CSV logger cleanup registered with atexit.")

    # In a Render deployment, this `app.run()` block is typically NOT used.
    # Render uses a WSGI server like Gunicorn (configured via a `Procfile`)
    # to run your Flask application in production.
    # The `Procfile` should look something like: `web: gunicorn app:app`
    # (assuming your Flask app instance is named `app` in `app.py`).
    # For local testing, ensure your ESP32 is accessible at the Cloudflare Tunnel URL.
    port = int(os.environ.get('PORT', 5000)) # Use Render's PORT env var if available, else 5000
    print(f"INFO: Starting Flask app on http://0.0.0.0:{port} (debug=True for local).")
    app.run(host='0.0.0.0', port=port, debug=True)

