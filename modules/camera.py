import json
import os

from modules.util import normalize_key


def load_presets(file_path):
    """
    This function loads presets from a given file path. It reads the file, parses the JSON content, and stores the presets in a dictionary.
    Each preset is a tuple containing a prompt and a weight.

    Args:
        file_path (str): The path to the file containing the presets.

    Returns:
        dict: A dictionary where the keys are the normalized names of the presets and the values are tuples containing the prompt and the weight of each preset.
    """
    presets = {}
    try:
        with open(file_path, encoding='utf-8') as f:
            for preset in json.load(f):
                key = normalize_key(preset['name'])
                prompt = preset.get('prompt', '')
                weight = preset.get('default', 1)
                presets[key] = (prompt, weight)
    except Exception as e:
        print(str(e))
        print(f'Failed to load camera presets file {file_path}')
    return presets


# Paths to the files containing the camera angles and distances presets
angles_path = os.path.join(os.path.dirname(__file__), '../camera_presets/angles.json')
distances_path = os.path.join(os.path.dirname(__file__), '../camera_presets/distances.json')

# Load the camera angles and distances presets
camera_angles = load_presets(angles_path)
camera_distances = load_presets(distances_path)

# Get the names of the camera angles and distances presets for use in the UI
camera_angle_names = list(camera_angles.keys())
camera_distance_names = list(camera_distances.keys())


def get_preset_weight(preset, name):
    """
    This function retrieves the weight of a given preset.

    Args:
        preset (str): The type of the preset ('angle' or 'distance').
        name (str): The name of the preset.

    Returns:
        str: The weight of the preset.
    """
    _, w = _get_preset(preset, name)
    return w


def apply_camera_preset(preset, name, weight, positive):
    """
    This function applies a camera preset. It replaces the placeholders in the presets' prompt with the given weight and positive value.

    Args:
        preset (str): The type of the preset ('angle' or 'distance').
        name (str): The name of the preset.
        weight (str): The weight to replace the '{weight}' placeholder in the presets' prompt.
        positive (str): The value to replace the '{prompt}' placeholder in the presets' prompt.

    Returns:
        str: The presets' prompt with the placeholders replaced by the given weight and positive value.
    """
    p, _ = _get_preset(preset, name)
    return p.replace('{weight}', str(weight)).replace('{prompt}', positive)


def _get_preset(preset, name):
    """
    This function retrieves a preset from the appropriate dictionary based on the preset type.

    Args:
        preset (str): The type of the preset ('angle' or 'distance').
        name (str): The name of the preset.

    Returns:
        tuple: The preset (a tuple containing the prompt and the weight).

    Raises:
        ValueError: If the preset type is not 'angle' or 'distance'.
    """
    if preset == 'angle':
        camera_presets = camera_angles
    elif preset == 'distance':
        camera_presets = camera_distances
    else:
        raise ValueError(f'Unknown preset type : {preset}')

    return camera_presets[name]
