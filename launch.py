import os
import sys

from modules.launch_util import commit_hash, fooocus_tag


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


prepare_environment()

