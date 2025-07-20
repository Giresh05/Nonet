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
app = Flask(__name__, static_folder='static', template_folder='templates')
print(f"Flask app initialized with static_folder='{app.static_folder}' and template_folder='{app.template_folder}'")

# --- Configuration ---
# The API key to secure the new data submission endpoint.
# Set this in your Render environment variables.
API_SECRET_KEY = "SensorNet_Alpha_77!"
if not API_SECRET_KEY:
    print("WARNING: API_SECRET_KEY environment variable not set. The /api/submit_data endpoint is insecure.")

# How many seconds of historical overall data to retain for the frontend's charts.
FRONTEND_DATA_RETENTION_SECONDS = 60
# Timeout for forward filling. If a node hasn't been seen for this many seconds,
# its data is considered stale and will be removed.
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
latest_node_data = {}
print("Initialized latest_node_data.")
overall_historical_data = []
print("Initialized overall_historical_data.")
data_lock = threading.Lock()
print("Initialized data_lock.")
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
        print(f"ERROR: Isolation Forest model not found at {ISOLATION_FOREST_MODEL_PATH}. Please ensure it exists.")
        if_model = None
    except Exception as e:
        print(f"ERROR: Error loading Isolation Forest model: {e}")
        if_model = None
    
    try:
        scaler = joblib.load(SCALER_PATH)
        print(f"DEBUG: Successfully loaded Scaler from {SCALER_PATH}")
    except FileNotFoundError:
        print(f"ERROR: Scaler not found at {SCALER_PATH}. Please ensure it exists.")
        scaler = None
    except Exception as e:
        print(f"ERROR: Error loading Scaler: {e}")
        scaler = None

    try:
        mac_encoder = joblib.load(MAC_ENCODER_PATH)
        print(f"DEBUG: Successfully loaded MAC Encoder from {MAC_ENCODER_PATH}")
    except FileNotFoundError:
        print(f"ERROR: MAC Encoder not found at {MAC_ENCODER_PATH}. Please ensure it exists.")
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
    and a text level. Returns: (numerical_level, text_level)
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

def process_incoming_data(esp32_nodes_data):
    """
    Processes a list of node data, performs anomaly detection, updates global data stores,
    and handles logging.
    """
    global latest_node_data, overall_historical_data, river_model, is_logging_active, user_defined_node_locations
    BASE_LAT = 20.5937
    BASE_LON = 78.9629
    LAT_OFFSET_PER_NODE = 0.1
    LON_OFFSET_PER_NODE = 0.15

    print(f"DEBUG: Processing incoming data: {esp32_nodes_data}")
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
                warning_level_numerical = 0.0
                warning_level_text = "Low"

                latitude = None
                longitude = None

                if node_id in user_defined_node_locations:
                    latitude = user_defined_node_locations[node_id]['latitude']
                    longitude = user_defined_node_locations[node_id]['longitude']
                    print(f"DEBUG: Using user-defined location for Node {node_id}: [{latitude}, {longitude}]")
                else:
                    try:
                        node_idx = int(node_id) - 1
                        latitude = BASE_LAT + (node_idx * LAT_OFFSET_PER_NODE)
                        longitude = BASE_LON + (node_idx * LON_OFFSET_PER_NODE)
                        print(f"DEBUG: Using dummy location for Node {node_id}: [{latitude}, {longitude}]")
                    except ValueError:
                        print(f"WARNING: Node ID '{node_id}' is not numeric. Cannot assign dummy location.")
                        pass

                if all(v is not None for v in [rain, soil, vibration, tilt]) and \
                   if_model and scaler and mac_encoder:
                    print(f"DEBUG: Performing anomaly detection for Node {node_id}")
                    features_list = [float(rain), float(soil), float(vibration), float(tilt)]
                    
                    encoded_mac = None
                    if mac is not None:
                        try:
                            if mac not in mac_encoder.classes_:
                                mac_encoder.classes_ = np.append(mac_encoder.classes_, mac)
                                encoded_mac = mac_encoder.transform([mac])[0]
                                print(f"INFO: Added new MAC '{mac}' to encoder classes for Node {node_id}. New encoded value: {encoded_mac}")
                            else:
                                encoded_mac = mac_encoder.transform([mac])[0]
                        except Exception as e:
                            print(f"ERROR: Error encoding MAC '{mac}': {e}. Defaulting encoded_mac to 0.")
                            encoded_mac = 0
                    else:
                        encoded_mac = 0
                    print(f"DEBUG: Node {node_id} - Encoded MAC: {encoded_mac}")

                    features_list.append(float(encoded_mac))
                    input_array = np.array(features_list).reshape(1, -1)
                    print(f"DEBUG: Node {node_id} - Input array for models: {input_array}")

                    if scaler:
                        scaled_input = scaler.transform(input_array)
                        iso_score = if_model.decision_function(scaled_input)[0]
                        print(f"DEBUG: Node {node_id} - ISO Score: {iso_score}")
                    
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
                    river_model.learn_one(river_input)

                    warning_level_numerical, warning_level_text = get_warning_level(iso_score, river_score)
                    print(f"DEBUG: Node {node_id} - Final Warning Level: Numerical={warning_level_numerical}, Text={text_level}")
                else:
                    print(f"WARNING: Skipping anomaly detection for Node {node_id} due to missing data or unloaded models.")

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
                    'latitude': latitude,
                    'longitude': longitude,
                    'last_seen_by_flask': current_flask_time
                }
                nodes_reported_in_this_poll.add(node_id)
                print(f"DEBUG: Updated latest_node_data for Node {node_id}.")

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
                            csv_file_handle.flush()
                            print(f"DEBUG: Logged data for Node {node_id} to CSV.")
                        except Exception as e:
                            print(f"ERROR: Error writing to CSV for node {node_id}: {e}")
        
        nodes_to_remove = []
        for node_id, data in list(latest_node_data.items()):
            time_since_last_seen = current_flask_time - data['last_seen_by_flask']
            
            if node_id not in nodes_reported_in_this_poll:
                if time_since_last_seen.total_seconds() > DATA_STALE_TIMEOUT_SECONDS:
                    nodes_to_remove.append(node_id)
        
        for node_id in nodes_to_remove:
            print(f"INFO: Removing stale data for Node {node_id}.")
            del latest_node_data[node_id]

        if latest_node_data:
            total_rain, total_soil, total_vibration, total_tilt = 0.0, 0.0, 0.0, 0.0
            total_iso_score, total_river_score, total_warning_level = 0.0, 0.0, 0.0
            
            active_node_count, iso_score_count, river_score_count, warning_level_count = 0, 0, 0, 0

            for node_id, data in latest_node_data.items():
                if data['rain'] is not None: total_rain += float(data['rain'])
                if data['soil'] is not None: total_soil += float(data['soil'])
                if data['vibration'] is not None: total_vibration += float(data['vibration'])
                if data['tilt'] is not None: total_tilt += float(data['tilt'])
                
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

                cutoff_time = current_flask_time - timedelta(seconds=FRONTEND_DATA_RETENTION_SECONDS)
                overall_historical_data = [
                    entry for entry in overall_historical_data if entry['timestamp'] >= cutoff_time
                ]
                print(f"DEBUG: Trimmed historical data. Current length: {len(overall_historical_data)}")
            else:
                overall_historical_data = []
        else:
            overall_historical_data = []

    print(f"INFO: Data processed successfully. Active nodes in Flask: {len(latest_node_data)}")


# --- Flask Routes ---

@app.route('/api/live_data')
def live_data():
    """API endpoint for the web frontend to fetch processed live data."""
    print("DEBUG: /api/live_data endpoint hit.")
    with data_lock:
        latest_overall = overall_historical_data[-1] if overall_historical_data else {
            'rain': None, 'soil': None, 'vibration': None, 'tilt': None,
            'iso_score': None, 'river_score': None, 'warning_level_numerical': None
        }
        
        nodes_for_frontend = {}
        for node_id, data in latest_node_data.items():
            nodes_for_frontend[node_id] = {
                'rain': data['rain'],
                'soil': data['soil'],
                'vibration': data['vibration'],
                'tilt': data['tilt'],
                'mac': data['mac'],
                'iso_score': data['iso_score'],
                'river_score': data['river_score'],
                'warning_level_numerical': data['warning_level_numerical'],
                'warning_level_text': data['warning_level_text'],
                'latitude': data['latitude'],
                'longitude': data['longitude'],
                'timestamp': data['last_seen_by_flask'].isoformat()
            }
        
        response_data = {
            'timestamp': datetime.now().isoformat(),
            'overall_data': latest_overall,
            'nodes': nodes_for_frontend,
            'is_logging_active': is_logging_active
        }
    return jsonify(response_data)

@app.route('/api/submit_data', methods=['POST'])
def submit_data():
    """New API endpoint to receive data pushed from the local PC script."""
    submitted_key = request.headers.get('X-API-KEY')
    if not API_SECRET_KEY or submitted_key != API_SECRET_KEY:
        print(f"ERROR: Unauthorized access to /api/submit_data. Submitted key: '{submitted_key}'")
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    data = request.get_json()
    if not data or not isinstance(data, list):
        print(f"ERROR: Invalid JSON received: {data}")
        return jsonify({"status": "error", "message": "Invalid JSON data. A list of nodes is expected."}), 400

    process_incoming_data(data)
    return jsonify({"status": "success", "message": "Data received."})


@app.route('/api/set_node_location', methods=['POST'])
def set_node_location():
    """API endpoint to manually set or update a node's location."""
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
        if node_id in latest_node_data:
            latest_node_data[node_id]['latitude'] = latitude
            latest_node_data[node_id]['longitude'] = longitude
            print(f"INFO: Updated live data for Node {node_id} with user-defined location.")
        else:
            print(f"INFO: Stored user-defined location for Node {node_id}. Node not currently active.")

    print(f"INFO: User set location for Node {node_id}: Lat={latitude}, Lon={longitude}. Sending success response.")
    return jsonify({"status": "success", "message": f"Location for Node {node_id} set successfully."})


@app.route('/api/set_mode', methods=['POST'])
def set_mode():
    data = request.get_json()
    mode = data.get('mode')
    
    # This function now only affects the Flask app's state, as it can't reach the ESP32 directly.
    # It's kept for potential future use or to control server-side logic (like logging).
    
    if mode not in ['data_collection', 'standby']:
        return jsonify({"status": "error", "message": "Invalid mode specified"}), 400
        
    print(f"INFO: /api/set_mode called. Clearing data stores for mode change simulation.")
    with data_lock:
        overall_historical_data.clear()
        latest_node_data.clear()
        user_defined_node_locations.clear()
        global river_model
        river_model = anomaly.HalfSpaceTrees(seed=42)
    print("INFO: Data stores cleared and River model re-initialized.")
    return jsonify({"status": "success", "message": f"Flask server state reset for {mode} mode."})


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
        close_csv_logger() # Ensure the file is closed before sending
        directory = os.path.dirname(current_csv_file_path)
        filename = os.path.basename(current_csv_file_path)
        print(f"Attempting to send file: {filename} from directory: {directory}")
        return send_from_directory(directory, filename, as_attachment=True)
    else:
        return jsonify({"status": "error", "message": "No CSV file available for download or file not found."}), 404

# --- Routes for Frontend Pages ---
@app.route('/')
def home_page():
    print("DEBUG: Serving home.html from templates folder...")
    return render_template('home.html')

@app.route('/dashboard')
def index():
    print("DEBUG: Serving index.html (dashboard) from templates folder...")
    return render_template('index.html')

@app.route('/about')
def about_page():
    print("DEBUG: Serving about.html from templates folder...")
    return render_template('about.html')

@app.route('/map')
def map_page():
    print("DEBUG: Serving map.html from templates folder...")
    return render_template('map.html')

# --- Main Execution Block ---
if __name__ == '__main__':
    print("INFO: Running Flask app in local development mode (via __main__).")
    load_ml_models()
    atexit.register(close_csv_logger)
    print("INFO: CSV logger cleanup registered with atexit.")
    
    # The data fetching thread is removed as data is now pushed to the server.

    port = int(os.environ.get('PORT', 5000))
    print(f"INFO: Starting Flask app on http://0.0.0.0:{port} (debug=True for local).")
    app.run(host='0.0.0.0', port=port, debug=True)
