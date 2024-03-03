import os
from urllib.parse import urlparse
from typing import Optional


def load_file_from_url(
        url: str,
        *,
        model_dir: str,
        progress: bool = True,
        file_name: Optional[str] = None,
) -> str:
    """
    This function downloads a file from a given URL and saves it to a specified directory.

    'url' is the URL from which the file will be downloaded.

    'model_dir' is the directory where the downloaded file will be saved.

    'progress' is a boolean that indicates whether to display a progress bar during the download. The default value is True.

    'file_name' is an optional parameter. If provided, the downloaded file will be saved with this name. If not provided, the original file name from the URL will be used.

    The function first creates the 'model_dir' directory if it does not exist.

    If 'file_name' is not provided, the function parses the 'url' to get the file name.

    The function then checks if the file already exists in the 'model_dir' directory. If the file does not exist, the function tries to download the file from the 'url' and save it to the 'model_dir' directory. If the download fails, an error message is printed.

    The function returns the path to the downloaded file.
    """
    os.makedirs(model_dir, exist_ok=True)
    if not file_name:
        # if file_name is not provided, the file name is extracted from the url.
        parts = urlparse(url)
        file_name = os.path.basename(parts.path)
    cached_file = os.path.abspath(os.path.join(model_dir, file_name))
    if not os.path.exists(cached_file):
        try:
            # if the file does not exist, it is downloaded from the url and saved to the model_dir directory.
            print(f'Downloading: "{url}" to {cached_file}\n')
            from torch.hub import download_url_to_file
            download_url_to_file(url, cached_file, progress=progress)
        except Exception as e:
           # if the download fails, an error message is printed.
            print(f"Failed to download {url} to {cached_file}: {e}")

    return cached_file