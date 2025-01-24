import subprocess
import sys
import os
import shutil
import zipfile
import importlib
import urllib.request
import re


re_requirement = re.compile(r"\s*([-\w]+)\s*(?:==\s*([-+.\w]+))?\s*")

python = sys.executable
default_command_live = (os.environ.get('LAUNCH_LIVE_OUTPUT') == "1")
index_url = os.environ.get('INDEX_URL', "")

modules_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.dirname(modules_path)


def detect_python_version():
    version = sys.version_info
    version_str = f"{version.major}.{version.minor}"
    is_embedded = hasattr(sys, '_base_executable') or (
        sys.base_prefix != sys.prefix and not hasattr(sys, 'real_prefix'))
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

    try:
        import GPUtil
        import psutil
        return True
    except ImportError:
        import_GPUtil()
        return False


def check_flask_installed():

    try:
        import flask
        import flask_restx
        import flask_cors

        import flask_socketio
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
    src_folder = os.path.join("tkinter-standalone",
                              version_str, "python_embedded")
    number_only = version_str.replace(".", "")
    python_zip = f"python{number_only}"
    python_zip_path = os.path.join(src_folder, f"{python_zip}.zip")
    with zipfile.ZipFile(python_zip_path, 'r') as zip_ref:
        zip_ref.extractall(os.path.join(src_folder, python_zip))

    if not os.path.exists(src_folder):
        print(f"Error: No tkinter files for Python {version_str}")
        return

    # Define paths
    python_dir = os.path.dirname(sys.executable)
    pth_filename = f"{python_zip}._pth"
    pth_path_src = os.path.join(src_folder, pth_filename)
    pth_path_dest = os.path.join(python_dir, pth_filename)

    # Copy the .pth file from python_dir to src_folder
    if os.path.exists(pth_path_dest):
        shutil.copy(pth_path_dest, pth_path_src)
    else:
        print(f"Error: {pth_filename} not found in {python_dir}")
        return

    # Modify the .pth file
    with open(pth_path_src, 'a') as pth_file:
        pth_file.write(f'\n./{python_zip}\n')
        pth_file.write('./Scripts\n')
        pth_file.write('./DLLs\n')

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
        print("tkinter module loaded successfully.")
        return tkinter
    except ModuleNotFoundError as e:
        print(f"Module not found: {e}")
    except ImportError as e:
        print("Failed to import Tkinter after installation.")
        print(f"An error occurred: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return None


def import_GPUtil():
    run_pip(f"install GPUtil psutil", desc="GPU Utility for NVIDIA GPUs")

    try:
        GPUtil = importlib.import_module(
            "GPUtil")
        psutil = importlib.import_module(
            "psutil")
        return GPUtil
    except ImportError:
        print("Failed to import GPUtil after installation.")
        return None


def import_flask():
    run_pip(f"install flask flask-restx flask-cors flask_socketio", desc="Flask Rest API")

    try:
        flask = importlib.import_module("flask")
        restx = importlib.import_module("flask-restx")
        flask_socketio = importlib.import_module("flask_socketio")
        return restx
    except ImportError:
        print("Failed to import flask after installation.")
        return None


def run(command, desc=None, errdesc=None, custom_env=None, live: bool = default_command_live) -> str:
    if desc is not None:
        print(desc)

    run_kwargs = {
        "args": command,
        "shell": True,
        "env": os.environ if custom_env is None else custom_env,
        "encoding": 'utf8',
        "errors": 'ignore',
    }

    if not live:
        run_kwargs["stdout"] = run_kwargs["stderr"] = subprocess.PIPE

    result = subprocess.run(**run_kwargs)

    if result.returncode != 0:
        error_bits = [
            f"{errdesc or 'Error running command'}.",
            f"Command: {command}",
            f"Error code: {result.returncode}",
        ]
        if result.stdout:
            error_bits.append(f"stdout: {result.stdout}")
        if result.stderr:
            error_bits.append(f"stderr: {result.stderr}")
        raise RuntimeError("\n".join(error_bits))

    return (result.stdout or "")


def run_pip(command, desc=None, live=default_command_live):
    try:
        index_url_line = f' --index-url {index_url}' if index_url != '' else ''
        return run(f'"{python}" -m pip {command} --prefer-binary{index_url_line}', desc=f"Installing {desc}",
                   errdesc=f"Couldn't install {desc}", live=live)
    except Exception as e:
        print(e)
        print(f'CMD Failed {desc}: {command}')
        return None


check_tkinter_installed()
check_GPUtil_installed()
check_flask_installed()
