import sys
import os
import shutil
import zipfile
import importlib
import urllib.request
from modules.launch_util import  run_pip
import torch

def detect_python_version():
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}"
    is_embedded = hasattr(sys, '_base_executable') or (sys.base_prefix != sys.prefix and not hasattr(sys, 'real_prefix'))
    return version_str, is_embedded


def check_tkinter_installed():
    version_str, is_embedded = detect_python_version()
    print(f"Detected Python version: {version_str}")
    print(f"Is Embedded Python: {is_embedded}")
    try:
        import tkinter
        tkinter_installed = True
    except ImportError:
        tkinter_installed = False
    if not tkinter_installed or (is_embedded and not tkinter_installed):
        install_tkinter(version_str)

    
def check_GPUtil_installed():
    if not torch.cuda.is_available():
        return False

    try:
        import GPUtil
        return True
    except ImportError:
        import_GPUtil()
        return False
    

        
def check_flask_installed():
    if not torch.cuda.is_available():
        return False

    try:
        import flask
        return True
    except ImportError:
        import_flask()
        return False


def download_and_unzip_tkinter():
    url = "https://github.com/ChrisColeTech/tkinter-standalone/releases/download/1.0.0/tkinter-standalone.zip"
    zip_path = "tkinter-standalone.zip"
    print(f"Downloading {url}...")
    urllib.request.urlretrieve(url, zip_path)

    print("Unzipping tkinter-standalone.zip...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("tkinter-standalone")

    os.remove(zip_path)
    print("Download and extraction complete.")


def copy_tkinter_files(version_str):
    src_folder = os.path.join("tkinter-standalone", version_str, "python_embedded")
    number_only = version_str.replace(".","")
    python_zip = f"python{number_only}"
    python_zip_path = os.path.join(src_folder, f"{python_zip}.zip")
    with zipfile.ZipFile(python_zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(src_folder, python_zip))


    if not os.path.exists(src_folder):
        print(f"Error: No tkinter files for Python {version_str}")
        return

    python_dir = os.path.dirname(sys.executable)
    print(f"Copying tkinter files from {src_folder} to {python_dir}...")
    shutil.copytree(src_folder, python_dir, dirs_exist_ok=True)

    print("Tkinter files copied successfully.")
    shutil.rmtree("tkinter-standalone", ignore_errors=True)


def install_tkinter(version_str):
    download_and_unzip_tkinter()
    copy_tkinter_files(version_str)
    import_tkinter()
    
def import_tkinter():
    try:
        tkinter = importlib.import_module("tkinter")
        return tkinter
    except ImportError:
        print("Failed to import Tkinter after installation.")
        return None
    
    
def import_GPUtil():
    run_pip(f"install GPUtil")

    try:
        GPUtil = importlib.import_module("GPUtil", desc="GPU Performance Monitor" )
        return GPUtil
    except ImportError:
        print("Failed to import GPUtil after installation.")
        return None
    
def import_flask():
    run_pip(f"install flask flask-restx flask-cors", desc="Flask Rest API")

    try:
        flask = importlib.import_module("flask")
        restx = importlib.import_module("flask-restx")
        return restx
    except ImportError:
        print("Failed to import flask after installation.")
        return None

check_tkinter_installed()
check_GPUtil_installed()
check_flask_installed()