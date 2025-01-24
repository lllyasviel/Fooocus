import json
import os
from flask import Blueprint, jsonify, request
from flask_restx import Api, Resource, fields, Namespace

# Create a Blueprint for the settings controller
settings_bp = Blueprint('settings', __name__)
settings_api = Api(settings_bp, version='1.0', title='Settings API',
                   description='API for managing settings')

# Define a namespace for settings
settings_ns = Namespace('settings', description='Settings operations')

# Define the model for settings
settings_model = settings_ns.model('Setting', {
    'key': fields.String(required=True, description='The key of the setting'),
    'value': fields.String(required=True, description='The value of the setting')
})

# File to persist settings data
SETTINGS_FILE = 'settings.json'


def load_settings():
    if os.path.exists(SETTINGS_FILE):
        with open(SETTINGS_FILE, 'r') as file:
            return json.load(file)
    return {}


def save_settings(settings):
    with open(SETTINGS_FILE, 'w') as file:
        json.dump(settings, file, indent=4)


# Load initial data
settings_store = load_settings()


@settings_ns.route('/')
class SettingsList(Resource):
    def get(self):
        """List all settings"""
        return jsonify({'settings': list(settings_store.values())})

    @settings_ns.expect(settings_model)
    def post(self):
        """Create or update a setting"""
        setting = request.json
        key = setting['key']
        settings_store[key] = setting
        save_settings(settings_store)  # Save to file
        return jsonify(setting)


@settings_ns.route('/<string:key>')
class SettingItem(Resource):
    def get(self, key):
        """Get a setting by key"""
        setting = settings_store.get(key)
        if setting is None:
            return {'message': 'Setting not found'}, 404
        return jsonify(setting)

    def delete(self, key):
        """Delete a setting by key"""
        if key not in settings_store:
            return {'message': 'Setting not found'}, 404
        del settings_store[key]
        save_settings(settings_store)  # Save to file
        return {'message': 'Setting deleted'}
