
import os

def get_files_from_folder(folder_path, exensions=None, name_filter=None):
    if not os.path.isdir(folder_path):
        raise ValueError("Folder path is not a valid directory.")

    filenames = []

    for root, dirs, files in os.walk(folder_path, topdown=False):
        relative_path = os.path.relpath(root, folder_path)
        if relative_path == ".":
            relative_path = ""
        for filename in sorted(files, key=lambda s: s.casefold()):
            _, file_extension = os.path.splitext(filename)
            if (exensions is None or file_extension.lower() in exensions) and (name_filter is None or name_filter in _):
                path = os.path.join(relative_path, filename)
                filenames.append(path)

    return filenames
