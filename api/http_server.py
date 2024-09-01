from .dependency_installer import *
from flask import Flask, send_from_directory, render_template
from flask_socketio import SocketIO, emit
from flask_restx import Api
import logging
import time
from .controllers import register_blueprints
import psutil
import GPUtil
import threading

# Initialize Flask app
title = f"Resource Monitor"
app = Flask(title, static_folder='web/assets', template_folder='web/templates')
app.config['CORS_HEADERS'] = 'Content-Type'

# Initialize Flask-RESTx API
api = Api(app, version='1.0', title=title, description='API for system resource monitoring')

# Register blueprints (API endpoints)
register_blueprints(app, api)

# Initialize SocketIO with the Flask app
socketio = SocketIO(app, cors_allowed_origins="*")

# Cache for system usage data
cache = {
    'timestamp': 0,
    'data': {
        'cpu': 0,
        'ram': 0,
        'gpu': 0,
        'vram': 0,
        'hdd': 0,
        'temp': 0
    }
}
CACHE_DURATION = 1  # Cache duration in seconds

# Suppress the Flask development server warning
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)  # Set level to ERROR to suppress warnings

def get_cache(current_time):
    # Get CPU utilization
    cpu_percent = psutil.cpu_percent(interval=0)

    # Get Memory utilization
    mem = psutil.virtual_memory()
    mem_percent = mem.percent

    # Get GPU utilization (considering only the first GPU)
    gpus = GPUtil.getGPUs()
    gpu_percent = gpus[0].load * 100 if gpus else 0

    # Get VRAM usage (considering only the first GPU)
    vram_usage = 0
    if gpus:
        used = gpus[0].memoryUsed
        total = gpus[0].memoryTotal
        vram_usage = (used / total) * 100

    # Get HDD usage (assuming usage of the primary disk)
    hdd = psutil.disk_usage('/')
    hdd_percent = hdd.percent

    # Get temperature (if available)
    temperature = gpus[0].temperature if gpus else 0

    # Update the cache
    cache['data'] = {
        'cpu': cpu_percent,
        'ram': mem_percent,
        'gpu': gpu_percent,
        'vram': vram_usage,  # Convert bytes to MB
        'hdd': hdd_percent,
        'temp': temperature  # Add temperature
    }
    cache['timestamp'] = current_time

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('web', filename)

@socketio.on('connect')
def handle_connect():
    # Emit initial data
    current_time = time.time()
    get_cache(current_time)
    emit('data_update', cache['data'])

@socketio.on('disconnect')
def handle_disconnect():
    pass

def background_thread():
    while True:
        current_time = time.time()
        get_cache(current_time)
        socketio.emit('data_update', cache['data'])
        time.sleep(.5) 

def run_app():
    time.sleep(1)  # Sleep for a short while to let the server start
    # Start the background thread for emitting data
    socketio.start_background_task(target=background_thread)
    # Run the Flask app with SocketIO
    socketio.run(app, port=5000)

# Start Flask app in a separate thread
thread = threading.Thread(target=run_app)
thread.start()