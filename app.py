from flask import Flask, jsonify, send_from_directory, render_template, request, abort
import os
from datetime import datetime, timedelta
import threading
from functools import wraps

# ==============================================================================
# Flask App Initialization
# ==============================================================================
# We initialize the Flask app, specifying the folders for our static assets (CSS, JS, images)
# and our HTML templates. This is standard setup for a Flask web application.
app = Flask(__name__, static_folder='static', template_folder='templates')

# ==============================================================================
# Configuration
# ==============================================================================
# This section defines how the application behaves.
# RENDER_API_SECRET_KEY: A secret password. The local `app.py` must send this
#                        key to prove it's allowed to submit data. For security,
#                        we fetch this from an environment variable on the server.
# FRONTEND_DATA_RETENTION_SECONDS: How long (in seconds) to keep data for the charts.
#                                  60 seconds means the charts will show a 1-minute history.
RENDER_API_SECRET_KEY = 'SensorNet_Alpha_77!'
FRONTEND_DATA_RETENTION_SECONDS = 60

# ==============================================================================
# Global Data Stores & Threading Lock
# ==============================================================================
# These variables will hold our sensor data in memory. They are "global" so that
# different parts of the app (like receiving data and displaying it) can access them.
#
# latest_node_data: A dictionary to store the most recent data for each sensor node.
#                   Example: {'1': {'rain': 100, 'soil': 500, ...}, '2': {...}}
# overall_historical_data: A list to store the average data of all nodes over time.
#                          This is used to draw the main trend charts on the dashboard.
# data_lock: A lock to prevent race conditions. Since the server can receive new data
#            while a user is trying to view it, this lock ensures that the data isn't
#            corrupted by simultaneous access.
latest_node_data = {}
overall_historical_data = []
data_lock = threading.Lock()

# ==============================================================================
# Security Decorator
# ==============================================================================
# This is a Python "decorator" for security. We'll wrap our data submission
# endpoint with it. It checks for a special header ('X-API-KEY') in the incoming
# request and makes sure the key matches our secret key. If it doesn't match,
# it rejects the request with a 403 Forbidden error.
def require_api_key(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if request.headers.get('X-API-KEY') and request.headers.get('X-API-KEY') == RENDER_API_SECRET_KEY:
            return f(*args, **kwargs)
        else:
            print("Aborting request due to missing or invalid API key.")
            abort(403)  # Forbidden
    return decorated_function

# ==============================================================================
# API Endpoints (The "Backend" for our Webpage)
# ==============================================================================

@app.route('/api/submit_data', methods=['POST'])
@require_api_key
def submit_data():
    """
    Receives processed data from the local `app.py` instance.
    This is the primary data ingestion point for the Render server.
    """
    # Get the JSON data sent by the local controller
    data = request.get_json()
    if not data:
        return jsonify({"status": "error", "message": "No data provided."}), 400

    # Extract the node data and overall average data from the payload
    nodes_payload = data.get('nodes', {})
    overall_payload = data.get('overall_data', {})
    
    # Use the lock to safely update our global data stores
    with data_lock:
        global latest_node_data, overall_historical_data
        
        # Update the latest data for each node
        latest_node_data = nodes_payload
        
        # Add the new overall average data to our historical list
        if overall_payload:
            # Convert timestamp string from payload back to a datetime object
            overall_payload['timestamp'] = datetime.fromisoformat(overall_payload['timestamp'])
            overall_historical_data.append(overall_payload)
        
        # Trim the historical data list to keep it from growing indefinitely.
        # We only keep the last `FRONTEND_DATA_RETENTION_SECONDS` worth of data.
        cutoff_time = datetime.now() - timedelta(seconds=FRONTEND_DATA_RETENTION_SECONDS)
        overall_historical_data = [
            entry for entry in overall_historical_data if entry['timestamp'] >= cutoff_time
        ]

    print(f"Successfully received and processed data for {len(nodes_payload)} nodes.")
    return jsonify({"status": "success", "message": "Data received successfully."})


@app.route('/api/live_data')
def live_data():
    """
    Provides the live sensor data to the frontend dashboard.
    The JavaScript on the webpage calls this endpoint every second to get new data.
    """
    with data_lock:
        # Get the most recent overall data point, or provide a default empty structure
        latest_overall = overall_historical_data[-1] if overall_historical_data else {
            'rain': None, 'soil': None, 'vibration': None, 'tilt': None,
            'iso_score': None, 'river_score': None, 'warning_level_numerical': None
        }

        # Prepare the data to be sent as JSON
        response_data = {
            'timestamp': datetime.now().isoformat(),
            'overall_data': latest_overall,
            'nodes': latest_node_data,
            # These features are disabled on the remote server
            'is_logging_active': False 
        }
    return jsonify(response_data)

# ==============================================================================
# Page-Serving Routes (The "Frontend")
# ==============================================================================
# These routes simply return the HTML pages. Flask uses the `render_template`
# function to find and return the correct HTML file from the `templates` folder.

@app.route('/')
def home():
    """Serves the home page."""
    return render_template('home.html')

@app.route('/dashboard')
def dashboard():
    """Serves the main dashboard page."""
    return render_template('index.html')

@app.route('/map')
def map_page():
    """Serves the map HTML page."""
    return render_template('map.html')

@app.route('/about')
def about():
    """Serves the about page."""
    return render_template('about.html')

# ==============================================================================
# Main Execution Block
# ==============================================================================
# This block runs when the script is executed directly.
if __name__ == '__main__':
    # The host '0.0.0.0' makes the server accessible from other devices on the network.
    # `debug=True` is useful for development but should be turned off for production.
    # Render will handle running this in a production-ready way.
    app.run(host='0.0.0.0', port=5000, debug=True)
