# https://huggingface.co/spaces/SmilingWolf/wd-v1-4-tags
# https://github.com/pythongosssss/ComfyUI-WD14-Tagger/blob/main/wd14tagger.py

# {
#     "wd-v1-4-moat-tagger-v2": "https://huggingface.co/SmilingWolf/wd-v1-4-moat-tagger-v2",
#     "wd-v1-4-convnextv2-tagger-v2": "https://huggingface.co/SmilingWolf/wd-v1-4-convnextv2-tagger-v2",
#     "wd-v1-4-convnext-tagger-v2": "https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger-v2",
#     "wd-v1-4-convnext-tagger": "https://huggingface.co/SmilingWolf/wd-v1-4-convnext-tagger",
#     "wd-v1-4-vit-tagger-v2": "https://huggingface.co/SmilingWolf/wd-v1-4-vit-tagger-v2"
# }


import numpy as np
import csv
import onnxruntime as ort

from PIL import Image
from onnxruntime import InferenceSession
from modules.config import path_clip_vision
from modules.model_loader import load_file_from_url


global_model = None
global_csv = None


def default_interrogator(image_rgb, threshold=0.35, character_threshold=0.85, exclude_tags=""):
    global global_model, global_csv

    model_name = "wd-v1-4-moat-tagger-v2"

    model_onnx_filename = load_file_from_url(
        url=f'https://huggingface.co/lllyasviel/misc/resolve/main/{model_name}.onnx',
        model_dir=path_clip_vision,
        file_name=f'{model_name}.onnx',
    )

    model_csv_filename = load_file_from_url(
        url=f'https://huggingface.co/lllyasviel/misc/resolve/main/{model_name}.csv',
        model_dir=path_clip_vision,
        file_name=f'{model_name}.csv',
    )

    if global_model is not None:
        model = global_model
    else:
        model = InferenceSession(model_onnx_filename, providers=ort.get_available_providers())
        global_model = model

    input = model.get_inputs()[0]
    height = input.shape[1]

    image = Image.fromarray(image_rgb)  # RGB
    ratio = float(height)/max(image.size)
    new_size = tuple([int(x*ratio) for x in image.size])
    image = image.resize(new_size, Image.LANCZOS)
    square = Image.new("RGB", (height, height), (255, 255, 255))
    square.paste(image, ((height-new_size[0])//2, (height-new_size[1])//2))

    image = np.array(square).astype(np.float32)
    image = image[:, :, ::-1]  # RGB -> BGR
    image = np.expand_dims(image, 0)

    if global_csv is not None:
        csv_lines = global_csv
    else:
        csv_lines = []
        with open(model_csv_filename) as f:
            reader = csv.reader(f)
            next(reader)
            for row in reader:
                csv_lines.append(row)
        global_csv = csv_lines

    tags = []
    general_index = None
    character_index = None
    for line_num, row in enumerate(csv_lines):
        if general_index is None and row[2] == "0":
            general_index = line_num
        elif character_index is None and row[2] == "4":
            character_index = line_num
        tags.append(row[1])

    label_name = model.get_outputs()[0].name
    probs = model.run([label_name], {input.name: image})[0]

    result = list(zip(tags, probs[0]))

    general = [item for item in result[general_index:character_index] if item[1] > threshold]
    character = [item for item in result[character_index:] if item[1] > character_threshold]

    all = character + general
    remove = [s.strip() for s in exclude_tags.lower().split(",")]
    all = [tag for tag in all if tag[0] not in remove]

    res = ", ".join((item[0].replace("(", "\\(").replace(")", "\\)") for item in all)).replace('_', ' ')
    return res
