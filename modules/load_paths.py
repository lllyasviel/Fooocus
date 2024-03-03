"""
This script contains functions for handling folders and paths.

Importing the 'os' module which provides a way of using operating system dependent functionality.
The 'os' module provides a portable way of using operating system dependent functionality such as reading or writing to the file system, starting or killing processes, etc.

Importing the 'json' module which provides a way of working with JSON data.
The 'json' module provides a way of encoding and decoding JSON data.
"""
import os
import json

def get_folders_and_paths(root_folder):
    """
    This function takes a root folder as input and returns a dictionary containing all the folders and their paths in the root folder and its subdirectories.

    'root_folder' is the path to the root folder.

    'folder_data' is a dictionary that will contain the folders and their paths.

    The function iterates over all the items in the root folder. If an item is a directory, its name and path are added to the 'folder_data' dictionary.

    The function is called recursively to handle subdirectories.
    """
    folder_data = {}

    for folder_name in os.listdir(root_folder):
        folder_path = os.path.join(root_folder, folder_name)
        if os.path.isdir(folder_path):
            folder_data[folder_name] = folder_path

            subfolder_data = get_folders_and_paths(folder_path)
            folder_data.update(subfolder_data)

    return folder_data

def save_to_json(data, json_file):
    """
    This function takes data and a json file as input and writes the data to the json file.

    'data' is the data to be written to the json file.

    'json_file' is the path to the json file.

    The data is written to the json file with an indentation of 4 spaces.
    """
    with open(json_file, 'w') as f:
        json.dump(data, f, indent=4)

def get_model_paths():
    """
    This function gets the paths of all the models in the 'models' directory and its subdirectories and saves them to a json file.

    The function first gets the absolute path of the script's directory.

    The root folder is set to the 'models' directory in the script's directory.

    If the root folder does not exist, an error message is printed and the function returns.

    The function then gets all the folders and their paths in the root folder and its subdirectories.

    The function then iterates over all the folders and their paths. If a folder name contains a path separator, the folder is a subdirectory. The function then updates the 'folder_data' dictionary to contain the subdirectory and its path and adds the parent directory to the 'items_to_delete' list.

    The function then deletes all the items in the 'items_to_delete' list from the 'folder_data' dictionary.

    The function then saves the 'folder_data' dictionary to a json file.

    The function then prints a message indicating that the folder data has been saved to the json file.
    """
    script_directory = os.path.dirname(os.path.abspath(__file__))

    root_folder = os.path.join(script_directory, "../models/")

    if not os.path.exists(root_folder):
        print("Error: The specified folder does not exist.")
        return

    folder_data = get_folders_and_paths(root_folder)

    items_to_delete = []
    for folder_name, folder_path in folder_data.items():
        if os.path.sep in folder_name:
            parent_folder_name, subfolder_name = folder_name.split(os.path.sep, 1)
            parent_folder_path = folder_data[parent_folder_name]
            folder_data[subfolder_name] = os.path.join(parent_folder_path, folder_name)
            items_to_delete.append(folder_name)

    for item in items_to_delete:
        del folder_data[item]

    json_file_name = "model_config_path.json"
    json_file_path = os.path.join('./', json_file_name)

    save_to_json(folder_data, json_file_path)

    print(f"Folder data has been saved to {json_file_path}.")