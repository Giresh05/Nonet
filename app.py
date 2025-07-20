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
data_lock = threading.Lock()

# --- All ML and Calculation logic has been removed ---
# --- It now lives in local_forwarder.py ---

def init_csv_logger():
    # This function remains unchanged
    global current_csv_file_path, csv_file_handle, csv_writer, is_logging_active
    # ... (code for init_csv_logger is the same as before)
    pass # Placeholder for brevity

def close_csv_logger():
    # This function remains unchanged
    global csv_file_handle, csv_writer, is_logging_active
    # ... (code for close_csv_logger is the same as before)
    pass # Placeholder for brevity

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
        
        # (Optional) CSV Logging of received data
        if is_logging_active and csv_writer:
             with csv_lock:
                for node_id, data in latest_node_data.items():
                    # This part can be adapted if you want to continue logging on the server
                    pass

    return jsonify({"status": "success", "message": "Payload received."})


@app.route('/api/live_data')
def live_data():
    """API endpoint for the web frontend to fetch the cached live data."""
    with data_lock:
        response_data = {
            'timestamp': datetime.now().isoformat(),
            'overall_data': latest_overall_data,
            'nodes': latest_node_data,
            'is_logging_active': is_logging_active
        }
    return jsonify(response_data)
    
# --- All other routes remain the same ---
# (/api/set_node_location, /api/start_csv_logging, /, /dashboard, etc.)
# For brevity, they are not repeated here.
@app.route('/api/set_node_location', methods=['POST'])
def set_node_location():
    # This function can remain on the server, but it will only affect the server's cache
    # until the next payload from the forwarder overwrites it.
    # For a persistent change, the location logic would also need to move to the forwarder.
    return jsonify({"status": "info", "message": "Location setting is disabled in this architecture."})

# ... (other routes like start/stop logging, and page routes)

# --- Main Execution Block ---
if __name__ == '__main__':
    # No ML models to load at startup
    atexit.register(close_csv_logger)
    port = int(os.environ.get('PORT', 5000))
    print(f"INFO: Starting lightweight Flask app on http://0.0.0.0:{port}.")
    app.run(host='0.0.0.0', port=port, debug=True)
