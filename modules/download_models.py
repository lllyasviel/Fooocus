# download_models.py
import os

from modules import config
from modules.model_loader import load_file_from_url


def download_models(url, selected, file_name=None):
    model_paths = config.config_paths
    paths = model_paths[selected]
    if isinstance(paths, list):
        path = os.path.join(*paths)
    else:
        path = paths
    try:
        load_file_from_url(url, model_dir=path, progress=True, file_name=file_name)
        message = f"Download successful! Model saved to {path}."
    except Exception as e:
        message = f"Download failed! Please check the URL and try again."

    return message
