from flask import Flask, send_from_directory, jsonify
from flask_restx import Api
import threading
import logging
from flask_cors import CORS

# Adjusted import for fooocus_version and shared
from api.controllers import register_blueprints
import fooocus_version
import shared
import args_manager
import os
import gradio as gr
import dependency_installer


dependency_installer.check_flask_installed()
dependency_installer.check_GPUtil_installed()
dependency_installer.check_tkinter_installed()

def load_page(filename):
    """Load an HTML file as a string and return it"""
    file_path = os.path.join("web", filename)
    with open(file_path, 'r') as file:
        content = file.read()
    return content

def addResourceMonitor():
    ceq = None
    with gr.Row():
        ceq = gr.HTML(load_page('templates/perf-monitor/index.html'))

    return ceq


# Cache for system usage data
cache = {
    'timestamp': 0,
    'data': {
        'cpu': 0,
        'memory': 0,
        'gpu': 0,
        'vram': 0,
        'hdd': 0
    }
}
CACHE_DURATION = 1  # Cache duration in seconds



# Suppress the Flask development server warning
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)  # Set level to ERROR to suppress warnings

title = f"Fooocus version: {fooocus_version.version}"
app = Flask(title, static_folder='web', template_folder='web')
app.config['CORS_HEADERS'] = 'Content-Type'
api = Api(app, version='1.0', title=title, description='Fooocus REST API')

# Register blueprints (API endpoints)
register_blueprints(app, api)

# Enable CORS for all origins
CORS(app, resources={r"/*": {"origins": "*"}})

gradio_app = shared.gradio_root
# Serve static files from the 'web' folder

@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('web', filename)

@app.route('/config')
def config():
    return jsonify({
        'base_url': f"http://{str(args_manager.args.listen)}:5000"
    })


def run_app():
    app.run(port=5000)


# Start Flask app in a separate thread
thread = threading.Thread(target=run_app)
thread.start()

print(
    f" * REST API Server Running at http://{str(args_manager.args.listen)}:5000 or {str(args_manager.args.listen)}:5000")
print(
    f" * Open http://{str(args_manager.args.listen)}:5000 or {str(args_manager.args.listen)}:5000 in a browser to view REST endpoints")