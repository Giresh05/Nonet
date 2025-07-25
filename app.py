import os
from flask import Flask, jsonify, request, render_template
from functools import wraps
import threading
from datetime import datetime

# Initialize Flask App, specifying the folders for templates and static files
app = Flask(__name__, template_folder='templates', static_folder='static')

# --- Configuration ---
# It's best practice to get the secret key from an environment variable on Render.
# This keeps your key secure and out of the source code.
# The local server must send a request with a header 'X-API-KEY' matching this value.
API_SECRET_KEY = os.environ.get('RENDER_API_SECRET_KEY', 'SensorNet_Alpha_77!')

# --- In-Memory Data Store ---
# This dictionary will hold the most recent complete data payload received from the local server.
# We use a lock to ensure thread-safe updates and reads, as web servers handle multiple requests concurrently.
latest_data_store = {
    'nodes': {},
    'overall_data': {},
    'last_updated': None
}
data_lock = threading.Lock()

# --- Security Decorator ---
def api_key_required(f):
    """
    A decorator to protect routes with an API key. It checks for the 'X-API-KEY' header.
    """
    @wraps(f)
    def decorated_function(*args, **kwargs):
        # Get the API key from the request headers
        provided_key = request.headers.get('X-API-KEY')
        
        # Check if the key is missing or incorrect
        if not provided_key or provided_key != API_SECRET_KEY:
            # Log the failed attempt for security monitoring
            print(f"SECURITY ALERT: Unauthorized access attempt from IP {request.remote_addr}.")
            # Return a 403 Forbidden error
            return jsonify({"status": "error", "message": "Unauthorized: Invalid or missing API key."}), 403
        
        # If the key is valid, proceed with the original function
        return f(*args, **kwargs)
    return decorated_function

# --- API Routes ---

@app.route('/api/submit_data', methods=['POST'])
@api_key_required
def submit_data():
    """
    This is the main endpoint for receiving data from the local Flask server.
    It's protected by the API key decorator.
    """
    # Get the JSON payload from the incoming request
    new_data = request.get_json()

    # Basic validation to ensure the payload has the expected structure
    if not new_data or 'nodes' not in new_data or 'overall_data' not in new_data:
        return jsonify({"status": "error", "message": "Invalid data format."}), 400

    # Use a lock to safely update the shared data store
    with data_lock:
        global latest_data_store
        latest_data_store['nodes'] = new_data['nodes']
        latest_data_store['overall_data'] = new_data['overall_data']
        latest_data_store['last_updated'] = datetime.now().isoformat()
    
    # Log the successful data reception for monitoring purposes
    print(f"Successfully received and stored data at {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}.")
    
    # Send a success response back to the local server
    return jsonify({"status": "success", "message": "Data received successfully."}), 200

@app.route('/api/live_data', methods=['GET'])
def get_live_data():
    """
    This endpoint provides the latest stored data to any client (e.g., a cloud dashboard).
    """
    with data_lock:
        # Return a copy of the latest data
        return jsonify(latest_data_store)

# --- Page Serving Routes ---

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

# --- Main Execution Block ---
# This block is standard for deploying Flask apps on services like Render.
if __name__ == '__main__':
    # Render's web service will use a production-ready server like Gunicorn,
    # but this is useful for local testing.
    # The host '0.0.0.0' makes the server accessible from outside its container.
    app.run(host='0.0.0.0', port=5001, debug=True)
