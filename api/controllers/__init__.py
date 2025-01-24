import os
import importlib
from flask import Blueprint
from flask_restx import Namespace


def register_blueprints(app, api):
    """Register all Blueprints to the Flask app automatically."""
    controllers_dir = os.path.dirname(__file__)
    for filename in os.listdir(controllers_dir):
        if filename.endswith('_controller.py') and filename != '__init__.py':
            module_name = filename[:-3]  # Remove ".py"
            module = importlib.import_module(
                f'.{module_name}', package=__package__)
            for attribute_name in dir(module):
                attribute = getattr(module, attribute_name)
                if isinstance(attribute, Namespace):
                    api.add_namespace(attribute)

                if isinstance(attribute, Blueprint):
                    app.register_blueprint(
                        attribute)
                    print(f"Registered blueprint: {attribute_name}")