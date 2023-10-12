import os
import shutil
import stat
import fnmatch

from modules.launch_util import run


def onerror(func, path, execinfo):
    os.chmod(path, stat.S_IWUSR)
    func(path)


def get_empty_folder(path):
    if os.path.isdir(path) or os.path.exists(path):
        shutil.rmtree(path, onerror=onerror)
    os.makedirs(path, exist_ok=True)
    return path


def git_clone(url, dir, hash=None):
    run(f'git clone {url} {dir}')


def findReplace(directory, find, replace, filePattern):
    for path, dirs, files in os.walk(os.path.abspath(directory)):
        for filename in fnmatch.filter(files, filePattern):
            filepath = os.path.join(path, filename)
            with open(filepath, encoding='utf-8') as f:
                s = f.read()
            s = s.replace(find, replace)
            with open(filepath, "w", encoding='utf-8') as f:
                f.write(s)


repo = "https://github.com/comfyanonymous/ComfyUI"
commit_hash = None

temp_path = get_empty_folder(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend', 'temp'))
core_path = get_empty_folder(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'backend', 'headless'))

git_clone(repo, temp_path, commit_hash)


def get_item(name, rename=None):
    if rename is None:
        rename = name
    shutil.move(os.path.join(temp_path, name), os.path.join(core_path, rename))


get_item('comfy', 'fcbh')
get_item('comfy_extras', 'fcbh_extras')
get_item('latent_preview.py')
get_item('folder_paths.py')
get_item('nodes.py')
get_item('LICENSE')

shutil.rmtree(temp_path, onerror=onerror)

findReplace("./backend", "comfy", "fcbh", "*.py")
findReplace("./backend", "Comfy", "FCBH", "*.py")
findReplace("./backend", "FCBHUI", "fcbh_backend", "*.py")
findReplace("./backend", "os.path.dirname(os.path.realpath(__file__))",
            "os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))", "folder_paths.py")

print('Backend is built.')
