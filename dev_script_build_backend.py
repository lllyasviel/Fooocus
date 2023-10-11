import os
import shutil


from modules.launch_util import git_clone


def onerror(func, path, exc_info):
    import stat
    if not os.access(path, os.W_OK):
        os.chmod(path, stat.S_IWUSR)
        func(path)
    else:
        raise 'Failed to invoke "shutil.rmtree", git management failed.'


def get_empty_folder(path):
    if os.path.isdir(path) or os.path.exists(path):
        shutil.rmtree(path, onerror=onerror)
    os.makedirs(path, exist_ok=True)
    return path


comfy_repo = "https://github.com/comfyanonymous/ComfyUI"
comfy_commit_hash = "be903eb2e2921f03a3a03dc9d6b0c6437ae201f5"
comfy_temp_path = get_empty_folder(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend', 'comfy_temp'))
comfy_core_path = get_empty_folder(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend', 'headless'))

git_clone(comfy_repo, comfy_temp_path, "ComfyUI", comfy_commit_hash)


def get_item(name):
    shutil.move(os.path.join(comfy_temp_path, name), os.path.join(comfy_core_path, name))


get_item('comfy')
get_item('comfy_extras')
get_item('custom_nodes')
get_item('latent_preview.py')
get_item('folder_paths.py')
get_item('nodes.py')

shutil.rmtree(comfy_temp_path, onerror=onerror)

print('Backend is built.')
