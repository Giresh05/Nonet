from flask import Flask, jsonify, send_from_directory, render_template, request
from datetime import datetime
import threading
import os
import csv
import atexit

print("Flask application starting up...")
app = Flask(__name__, static_folder='static', template_folder='templates')

# --- Configuration ---
API_SECRET_KEY = "SensorNet_Alpha_77!"
if not API_SECRET_KEY:
    print("WARNING: API_SECRET_KEY environment variable not set. The /api/submit_data endpoint is insecure.")

# --- CSV Logging Configuration (can remain on server) ---
CSV_LOG_DIRECTORY = 'csv_logs'
current_csv_file_path = None
csv_file_handle = None
csv_writer = None
is_logging_active = False
csv_lock = threading.Lock()

# --- Global Data Stores (server-side cache) ---
latest_node_data = {}
latest_overall_data = {}
user_defined_node_locations = {} # Kept for the /api/set_node_location endpoint
data_lock = threading.Lock()

# --- All ML and Calculation logic has been removed ---
# --- It now lives in local_forwarder.py ---

def init_csv_logger():
    """Initializes the CSV logging file."""
    global current_csv_file_path, csv_file_handle, csv_writer, is_logging_active
    with csv_lock:
        if csv_file_handle:
            close_csv_logger()
        if not os.path.exists(CSV_LOG_DIRECTORY):
            os.makedirs(CSV_LOG_DIRECTORY)
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        current_csv_file_path = os.path.join(CSV_LOG_DIRECTORY, f"anomaly_log_{timestamp_str}.csv")
        try:
            csv_file_handle = open(current_csv_file_path, 'w', newline='')
            csv_writer = csv.writer(csv_file_handle)
            csv_writer.writerow([
                'Timestamp', 'NodeID', 'MAC_Address', 'Rain', 'Soil', 'Vibration', 'Tilt',
                'ISO_Score', 'River_Score', 'Warning_Level_Numerical', 'Warning_Level_Text'
            ])
            is_logging_active = True
            return True
        except IOError as e:
            print(f"ERROR: Error opening CSV file for logging: {e}")
            is_logging_active = False
            return False

def close_csv_logger():
    """Closes the CSV logging file."""
    global csv_file_handle, csv_writer, is_logging_active
    with csv_lock:
        if csv_file_handle:
            csv_file_handle.close()
            csv_file_handle = None
            csv_writer = None
            is_logging_active = False

atexit.register(close_csv_logger)

# --- Flask Routes ---

@app.route('/api/submit_data', methods=['POST'])
def submit_data():
    """
    Receives a fully processed payload from the local forwarder and caches it.
    """
    global latest_node_data, latest_overall_data
    
    # 1. Authenticate the request
    submitted_key = request.headers.get('X-API-KEY')
    if not API_SECRET_KEY or submitted_key != API_SECRET_KEY:
        return jsonify({"status": "error", "message": "Unauthorized"}), 401

    # 2. Get the pre-processed data
    payload = request.get_json()
    if not payload:
        return jsonify({"status": "error", "message": "Invalid JSON"}), 400

    # 3. Update the server's global data stores
    with data_lock:
        latest_node_data = payload.get('nodes', {})
        latest_overall_data = payload.get('overall_data', {})
        
        if is_logging_active and csv_writer:
            with csv_lock:
                for node_id, data in latest_node_data.items():
                    try:
                        csv_writer.writerow([
                            data.get('last_seen', datetime.now().isoformat()),
                            node_id,
                            data.get('mac'),
                            data.get('rain'),
                            data.get('soil'),
                            data.get('vibration'),
                            data.get('tilt'),
                            data.get('iso_score'),
                            data.get('river_score'),
                            data.get('warning_level_numerical'),
                            data.get('warning_level_text')
                        ])
                    except Exception as e:
                        print(f"Error writing to CSV: {e}")


    return jsonify({"status": "success", "message": "Payload received."})


@app.route('/api/live_data')
def live_data():
    """API endpoint for the web frontend to fetch the cached live data."""
    with data_lock:
        # Apply any user-defined location overrides before sending to frontend
        for node_id, location in user_defined_node_locations.items():
            if node_id in latest_node_data:
                latest_node_data[node_id]['latitude'] = location['latitude']
                latest_node_data[node_id]['longitude'] = location['longitude']
                
        response_data = {
            'timestamp': datetime.now().isoformat(),
            'overall_data': latest_overall_data,
            'nodes': latest_node_data,
            'is_logging_active': is_logging_active
        }
    return jsonify(response_data)
    

@app.route('/api/set_node_location', methods=['POST'])
def set_node_location():
    """Manually sets a node's location in the server's cache."""
    data = request.get_json()
    node_id = data.get('nodeId')
    latitude = data.get('latitude')
    longitude = data.get('longitude')

    if not all([node_id, latitude, longitude]):
        return jsonify({"status": "error", "message": "Missing data"}), 400

    with data_lock:
        user_defined_node_locations[node_id] = {'latitude': latitude, 'longitude': longitude}
        
    return jsonify({"status": "success", "message": f"Location for Node {node_id} cached."})


@app.route('/api/set_mode', methods=['POST'])
def set_mode():
    """This endpoint now only clears server-side data, as mode is controlled locally."""
    with data_lock:
        latest_node_data.clear()
        latest_overall_data.clear()
        user_defined_node_locations.clear()
    return jsonify({"status": "success", "message": "Server-side data has been cleared."})


@app.route('/api/start_csv_logging', methods=['POST'])
def start_csv_logging():
    if init_csv_logger():
        return jsonify({"status": "success", "message": "CSV logging started."})
    else:
        return jsonify({"status": "error", "message": "Failed to start CSV logging."}), 500


@app.route('/api/stop_csv_logging', methods=['POST'])
def stop_csv_logging():
    close_csv_logger()
    return jsonify({"status": "success", "message": "CSV logging stopped."})


@app.route('/api/download_csv')
def download_csv():
    if current_csv_file_path and os.path.exists(current_csv_file_path):
        close_csv_logger()
        return send_from_directory(CSV_LOG_DIRECTORY, os.path.basename(current_csv_file_path), as_attachment=True)
    else:
        return jsonify({"status": "error", "message": "No CSV file to download."}), 404

# --- Routes for Frontend Pages ---
@app.route('/')
def home_page():
    return render_template('home.html')

@app.route('/dashboard')
def index():
    return render_template('index.html')

@app.route('/about')
def about_page():
    return render_template('about.html')

@app.route('/map')
def map_page():
    return render_template('map.html')

# --- Main Execution Block ---
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
