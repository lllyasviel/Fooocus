import os
import importlib
import importlib.util
import subprocess
import sys
import re
import logging
import importlib.metadata
import packaging.version


logging.getLogger("torch.distributed.nn").setLevel(logging.ERROR)  # sshh...
logging.getLogger("xformers").addFilter(lambda record: 'A matching Triton is not available' not in record.getMessage())

re_requirement = re.compile(r"\s*([-_a-zA-Z0-9]+)\s*(?:==\s*([-+_.a-zA-Z0-9]+))?\s*")

python = sys.executable
default_command_live = (os.environ.get('LAUNCH_LIVE_OUTPUT') == "1")
index_url = os.environ.get('INDEX_URL', "")

modules_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.dirname(modules_path)


def is_installed(package):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        return False

    return spec is not None


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


def requirements_met(requirements_file):
    """
    Does a simple parse of a requirements.txt file to determine if all requirements in it
    are already installed. Returns True if so, False if not installed or parsing fails.
    """
    # Regex pattern for parsing requirements.txt
    re_requirement = re.compile(r"([^=<>!]+)([=<>!]*)([^=<>!]+)?")

    with open(requirements_file, "r", encoding="utf8") as file:
        for line in file:
            line = line.strip()
            if line == "" or line.startswith('#'):  # Skip empty lines and comments
                continue

            m = re.match(re_requirement, line)
            if m is None:
                print(f"Warning: Unable to parse line: '{line}'")
                return False

            package = m.group(1).strip()
            version_required = (m.group(3) or "").strip()

            if version_required == "":
                # If no specific version is required, skip version check
                continue

            try:
                if package not in importlib.metadata.distributions():
                    print(f"Package not installed: {package}")
                    return False
                version_installed = importlib.metadata.version(package)
            except Exception as e:
                print(f"Error checking version for {package}: {e}")
                return False

            if packaging.version.parse(version_required) != packaging.version.parse(version_installed):
                print(f"Version mismatch for {package}: Required {version_required}, Installed {version_installed}")
                return False

    return True
