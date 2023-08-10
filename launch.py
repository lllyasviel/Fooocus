import os
import sys
import platform

from modules.launch_util import commit_hash, fooocus_tag, is_installed, run, python, run_pip, repo_dir, git_clone


REINSTALL_ALL = True


def prepare_environment():
    torch_index_url = os.environ.get('TORCH_INDEX_URL', "https://download.pytorch.org/whl/cu118")
    torch_command = os.environ.get('TORCH_COMMAND', f"pip install torch==2.0.1 torchvision==0.15.2 --extra-index-url {torch_index_url}")
    requirements_file = os.environ.get('REQS_FILE', "requirements_versions.txt")

    xformers_package = os.environ.get('XFORMERS_PACKAGE', 'xformers==0.0.20')

    comfy_repo = os.environ.get('COMFY_REPO', "https://github.com/comfyanonymous/ComfyUI.git")
    comfy_commit_hash = os.environ.get('COMFY_COMMIT_HASH', "5ac96897e9782805cd5e8fe85bd98ad03eae2b6f")

    commit = commit_hash()
    tag = fooocus_tag

    print(f"Python {sys.version}")
    print(f"Version: {tag}")
    print(f"Commit hash: {commit}")

    git_clone(comfy_repo, repo_dir('StabilityAI-official-comfyui'), "Inference Engine", comfy_commit_hash)

    if REINSTALL_ALL or not is_installed("torch") or not is_installed("torchvision"):
        run(f'"{python}" -m {torch_command}', "Installing torch and torchvision", "Couldn't install torch", live=True)

    if REINSTALL_ALL or not is_installed("xformers"):
        if platform.system() == "Windows":
            if platform.python_version().startswith("3.10"):
                run_pip(f"install -U -I --no-deps {xformers_package}", "xformers", live=True)
            else:
                print("Installation of xformers is not supported in this version of Python.")
                print("You can also check this and build manually: https://github.com/AUTOMATIC1111/stable-diffusion-webui/wiki/Xformers#building-xformers-on-windows-by-duckness")
                if not is_installed("xformers"):
                    exit(0)
        elif platform.system() == "Linux":
            run_pip(f"install -U -I --no-deps {xformers_package}", "xformers")



prepare_environment()

