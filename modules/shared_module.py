# shared_module.py
import json

def read_model_config_path(json_file_path):
    """
    This function reads a JSON file and returns its content.

    'json_file_path' is the path to the JSON file.

    The function opens the JSON file in read mode, loads its content into the 'model_paths' variable, and then returns 'model_paths'.
    """
    with open(json_file_path, 'r') as f:
        model_paths = json.load(f)
    return model_paths