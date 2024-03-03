# download_models.py
import json
import os

from modules.model_loader import load_file_from_url


def download_models(url, selected, file_name=None):
    with open('./model_config_path.json', 'r') as f:
        model_paths = json.load(f)

    path = model_paths.get(selected)
    path = os.path.abspath(path)

    try:
        load_file_from_url(url, model_dir=path, progress=True, file_name=file_name)
        message = f"Download successful! Model saved to {path}."
    except Exception as e:
        message = f"Download failed! Please check the URL and try again."

    return message
