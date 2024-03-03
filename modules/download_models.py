# download_models.py
from modules.model_loader import load_file_from_url
from modules.shared_module import read_model_config_path

def download_models(url, selected, file_name=None):
    """
    This function downloads models from a given URL and saves them to a specified path.

    'url' is the URL from which the model will be downloaded.

    'selected' is the key to get the path from the 'model_paths' dictionary where the downloaded model will be saved.

    'file_name' is an optional parameter. If provided, the downloaded file will be saved with this name. If not provided, the original file name from the URL will be used.

    The function first reads the 'model_config_path.json' file to get the 'model_paths' dictionary.

    The function then gets the path where the model will be saved from the 'model_paths' dictionary using the 'selected' key.

    The function then tries to download the file from the URL and save it to the path. If the download is successful, a success message is returned. If the download fails, an error message is returned.
    """
    model_paths = read_model_config_path("./model_config_path.json")
    path = model_paths.get(selected)

    try:

        load_file_from_url(url, model_dir=path, progress=True, file_name=file_name)
        success_message = f"Download successful! Model saved to {path}."
    except Exception as e:
        success_message = f"Download failed! please check url if it is correct."

    return success_message