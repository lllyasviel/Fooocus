import os
import re
import shutil
import subprocess
import sys

import importlib.metadata
import packaging.version
from packaging.requirements import Requirement


def is_installed(package, version=None, strict=True):
    has_package = None
    try:
        has_package = importlib.metadata.version(package)
        if has_package is not None:
            if version is not None:
                installed_version = packaging.version.parse(has_package)
                target_version = packaging.version.parse(version)
                return installed_version == target_version if strict else installed_version >= target_version
            else:
                return True
        else:
            return False
    except Exception:
        return False


def run(command, desc=None, errdesc=None, custom_env=None, live=False):
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
            f"{errdesc or 'Error running command'}. Command: {command}",
            f"Error code: {result.returncode}",
        ]
        if result.stdout:
            error_bits.append(f"stdout: {result.stdout}")
        if result.stderr:
            error_bits.append(f"stderr: {result.stderr}")
        raise RuntimeError("\n".join(error_bits))

    return result.stdout or ""


def python(command, desc=None, errdesc=None, custom_env=None, live=False):
    return run(f'"{sys.executable}" {command}', desc, errdesc, custom_env, live)


def run_pip(command, desc=None, live=False):
    return python(f'-m pip {command}', desc=f"Installing {desc}", errdesc=f"Couldn't install {desc}", live=live)


def requirements_met(requirements_file):
    with open(requirements_file, "r", encoding="utf8") as file:
        for line in file:
            line = line.strip()
            if line == "" or line.startswith('#'):
                continue

            requirement = Requirement(line)
            package = requirement.name

            try:
                version_installed = importlib.metadata.version(package)
                installed_version = packaging.version.parse(version_installed)

                # Check if the installed version satisfies the requirement
                if installed_version not in requirement.specifier:
                    print(f"Version mismatch for {package}: Installed version {version_installed} does not meet requirement {requirement}")
                    return False
            except Exception as e:
                print(f"Error checking version for {package}: {e}")
                return False

    return True


def delete_folder_content(folder, prefix=None):
    result = True
    prefix = prefix or ''

    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
        print(f'{prefix}Temp dir did not exist; created {folder}')
        return True

    for filename in os.listdir(folder):
        file_path = os.path.join(folder, filename)
        try:
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f'{prefix}Failed to delete {file_path}. Reason: {e}')
            result = False

    return result
