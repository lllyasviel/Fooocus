
from .dependency_installer import *
from flask import Flask, send_from_directory, jsonify, render_template
from flask_restx import Api
import threading
import logging
from flask_cors import CORS
from .controllers import register_blueprints
import os


def load_page(filename):
    """Load an HTML file as a string and return it"""
    file_path = os.path.join("web", filename)
    with open(file_path, 'r') as file:
        content = file.read()
    return content


# Suppress the Flask development server warning
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)  # Set level to ERROR to suppress warnings

title = f"Elegant Resource Monitor"
app = Flask(title, static_folder='web/assets', template_folder='web/templates')
app.config['CORS_HEADERS'] = 'Content-Type'
api = Api(app, version='1.0', title=title,
          description='Elegant Resource Monitor REST API')

# Register blueprints (API endpoints)
register_blueprints(app, api)

# Enable CORS for all origins
CORS(app, resources={r"/*": {"origins": "*"}})


@app.route('/<path:filename>')
def serve_static(filename):
    return send_from_directory('web', filename)


def run_app():
    app.run(port=5000)


# Start Flask app in a separate thread
thread = threading.Thread(target=run_app)
thread.start()
