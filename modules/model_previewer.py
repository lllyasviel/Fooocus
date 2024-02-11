import os
import json

# Constants
CHECKPOINTS_DIR = 'models/checkpoints'
LORAS_DIR = 'models/loras'
OUTPUT_FOLDER = 'outputs'
PREVIEW_LOG_FILE = 'preview_log.json'

def read_json_file(file_path):
    """ Reads a JSON file and returns its contents, or creates a new file if it doesn't exist. """
    try:
        if not file_exists(file_path):
            with open(file_path, 'w') as file:
                json.dump({}, file)
            return {}

        with open(file_path, 'r') as file:
            return json.load(file)
    except IOError as e:
        print(f"Error reading file {file_path}: {e}")
        return {}

def update_json_file(file_path, data):
    """ Writes updated data to a JSON file. """
    try:
        with open(file_path, 'w') as file:
            json.dump(data, file, indent=4)
    except IOError as e:
        print(f"Error writing to file {file_path}: {e}")

def file_exists(file_path):
    """ Checks if a file exists at the given path. """
    return os.path.exists(file_path)

def verify_and_cleanup_data(json_data, base_folder):
    """ Verifies the existence of files and cleans up JSON data. """
    cleaned_data = {}
    for safetensor, images in json_data.items():
        safetensor_path = os.path.join(base_folder, safetensor)
        if file_exists(safetensor_path):
            existing_images = [img for img in images if file_exists(os.path.join(OUTPUT_FOLDER, img))]
            if existing_images:
                cleaned_data[safetensor] = existing_images
    return cleaned_data

def get_cleaned_data(json_path, base_folder):
    data = read_json_file(json_path)
    cleaned_data = verify_and_cleanup_data(data, base_folder)
    return cleaned_data

def process_directory(directory):
    """ Process a single directory (checkpoints or loras). """
    json_path = os.path.join(directory, PREVIEW_LOG_FILE)
    cleaned_data = get_cleaned_data(json_path, directory)
    try:
        with open(json_path, 'w') as f:
            json.dump(cleaned_data, f, indent=4)
    except IOError as e:
        print(f"Error writing to file {json_path}: {e}")

def cleanup():
    """ Cleans up the JSON files in both checkpoints and loras directories. """
    process_directory(CHECKPOINTS_DIR)
    process_directory(LORAS_DIR)
 
def add_preview(model_name, image_location, directory):
    """ Adds a new image location to the preview list of a given model file. """
    print(f"Adding new preview '{image_location}' for '{directory}/{model_name}'")
    json_path = os.path.join(directory, PREVIEW_LOG_FILE)
    data = read_json_file(json_path)

    if model_name not in data:
        data[model_name] = []
    if image_location not in data[model_name]:
        data[model_name].append(image_location)
    update_json_file(json_path, data)

def add_preview_for_checkpoint(model_name, image_location):
    """ Adds a new image location for the given model file in checkpoints. """
    add_preview(model_name, image_location, CHECKPOINTS_DIR)

def add_preview_image_for_lora(model_name, image_location):
    """ Adds a new image location for the given model file in loras. """
    add_preview(model_name, image_location, LORAS_DIR)
    
def add_preview_by_attempt(base_model_name, refiner_model_name, loras, image_location):
    print(f"Attempting to add new preview for base model '{base_model_name}', refiner model '{refiner_model_name}' or for lora model '{loras}' to image location '{image_location}'")
    
    # Add preview based on the only one lora name
    active_loras = [lora for lora in loras if lora[0] != 'None']
    if len(active_loras) == 1: 
        active_lora_name = active_loras[0][0]
        add_preview_image_for_lora(active_lora_name, image_location)
    
    # Add preview based on only one model name if possible
    if len(active_loras) == 0: 
        if refiner_model_name == "None":
            add_preview_for_checkpoint(base_model_name, image_location)
        elif "_SD_" in refiner_model_name:
            add_preview_for_checkpoint(refiner_model_name, image_location)
    
def get_preview(model_name, directory):
    json_path = os.path.join(directory, PREVIEW_LOG_FILE)
    cleaned_data = get_cleaned_data(json_path, directory)
    return get_preview_from_data(model_name, cleaned_data)

def get_preview_from_data(model_name, data):
    """ Retrieves the latest available image for the given model file. """
    images = data.get(model_name, [])
    if images:
        latest_image = sorted(images, reverse=True)[0]
        latest_image_path = OUTPUT_FOLDER + "/" + latest_image
        if file_exists(latest_image_path):
            return latest_image_path
            print(f"Verbose Debug: File exists for model '{model_name}' at path '{latest_image_path}'.")
        else:
            print(f"Verbose Debug: File does not exist for model '{model_name}' at path '{latest_image_path}'.")
    else:
        print(f"Verbose Debug: No images found for model '{model_name}' in data.")
    return None

def get_all_previews(directory):
    """ Retrieves the latest available image for all. """
    json_path = os.path.join(directory, PREVIEW_LOG_FILE)
    print(f"Verbose Debug: Get previews from '{json_path}'.")
    data = read_json_file(json_path)
    
    valid_previews = {}

    # Find all files in the specified directory (only first level)
    for filename in os.listdir(directory):
        image_path = get_preview_from_data(filename, data)
        if image_path is not None:
            print(f"Verbose Debug: Valid preview found for '{filename}'.")
            valid_previews[filename] = image_path
        else:
            print(f"Verbose Debug: No valid preview found for '{filename}'.")

    return valid_previews

def get_all_previews_for_checkpoints():
    """ Retrieves the available images for a list of all model names in checkpoints. """
    return get_all_previews(CHECKPOINTS_DIR)

def get_all_previews_for_loras():
    """ Retrieves the available images for a list of all model names in loras. """
    return get_all_previews(LORAS_DIR)