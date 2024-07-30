# -*- coding: utf-8 -*-

""" File utils

Use for managing generated files

@file: file_utils.py
@author: Konie
@update: 2024-03-22
"""
import base64
import datetime
import hashlib
from io import BytesIO
import os
import json
from pathlib import Path
from typing import List

import numpy as np
from PIL import Image
from PIL.PngImagePlugin import PngInfo

from apis.utils.img_utils import narray_to_base64img

from modules.config import path_outputs

SCRIPT_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), '../..'))

output_dir = os.path.join(SCRIPT_PATH, 'outputs')

STATIC_SERVER_BASE = 'http://127.0.0.1:7866'


def delete_tasks(tasks: list) -> None:
    """
    Delete tasks from database
    Args:
        tasks: list of tasks
    """
    files_to_delete = []
    for task in tasks:
        if task['result'] is None:
            continue
        files_to_delete.extend(task['result'])
    if len(files_to_delete) == 0:
        return
    for file in url_path(files_to_delete):
        delete_output_file(file)


def save_output_file(
        img: np.ndarray,
        image_meta: dict = None,
        image_name: str = '',
        extension: str = 'png') -> str:
    """
    Save np image to file
    Args:
        img: np.ndarray image to save
        image_meta: dict of image metadata
        image_name: str of image name
        extension: str of image extension
    Returns:
        str of file name
    """
    current_time = datetime.datetime.now()
    date_string = current_time.strftime("%Y-%m-%d")

    filename = os.path.join(date_string, image_name + '.' + extension)
    file_path = os.path.join(path_outputs, filename)

    if extension not in ['png', 'jpg', 'webp']:
        extension = 'png'
    image_format = Image.registered_extensions()['.' + extension]

    if image_meta is None:
        image_meta = {}

    meta = None
    if extension == 'png' and image_meta != {}:
        meta = PngInfo()
        meta.add_text("parameters", json.dumps(image_meta))
        meta.add_text("fooocus_scheme", image_meta['metadata_scheme'])

    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    Image.fromarray(img).save(
        file_path,
        format=image_format,
        pnginfo=meta,
        optimize=True)
    return Path(filename).as_posix()


def delete_output_file(file_path: str):
    """
    Delete files specified in the output directory
    Args:
        file_path: str of file name
    """
    file_name = os.path.basename(file_path)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        print(f'[Fooocus API] {file_name} not exists or is not a file')
    try:
        os.remove(file_path)
        print(f'[Fooocus API] Delete output file: {file_name}')
    except OSError:
        print(f'[Fooocus API] Delete output file failed: {file_name}')


def output_file_to_base64img(filename: str | None) -> str | None:
    """
    Convert an image file to a base64 string.
    Args:
        filename: str of file name
    return: str of base64 string
    """
    if filename is None:
        return None
    file_path = os.path.join(path_outputs, filename)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return None

    img = Image.open(file_path)
    output_buffer = BytesIO()
    img.save(output_buffer, format='PNG')
    byte_data = output_buffer.getvalue()
    base64_str = base64.b64encode(byte_data).decode('utf-8')
    return base64_str


def output_file_to_bytesimg(filename: str | None) -> bytes | None:
    """
    Convert an image file to a bytes string.
    Args:
        filename: str of file name
    return: bytes of image data
    """
    if filename is None:
        return None
    file_path = os.path.join(path_outputs, filename)
    if not os.path.exists(file_path) or not os.path.isfile(file_path):
        return None

    img = Image.open(file_path)
    output_buffer = BytesIO()
    img.save(output_buffer, format='PNG')
    byte_data = output_buffer.getvalue()
    return byte_data


def get_file_serve_url(filename: str | None) -> str | None:
    """
    Get the static serve url of an image file.
    Args:
        filename: str of file name
    return: str of static serve url
    """
    if filename is None:
        return None
    return STATIC_SERVER_BASE + filename.replace('\\', '/')


def save_base64(base64_str: str | np.ndarray, file_dir: str) -> str:
    """
    Save a base64 string to a file.
    Args:
        base64_str: str of base64 string
        file_dir: str of file path
    """
    if not isinstance(base64_str, str):
        base64_str = narray_to_base64img(base64_str)
    if base64_str is None or base64_str == '' or base64_str.lower() == 'none':
        return ''
    sha256 = hashlib.sha256(base64_str.encode('utf-8')).hexdigest()
    file_path = os.path.join(file_dir, f'{sha256}.png')
    if os.path.exists(file_path):
        return file_path
    try:
        img_data = base64.b64decode(base64_str)
    except Exception as e:
        print(f'[Fooocus API] Decode base64 string failed: {e}')
        return ''
    with open(file_path, 'wb') as f:
        f.write(img_data)
    return file_path


def to_http(path: str, dir_name: str) -> str:
    """
    Convert a file path to an HTTP URL.
    Args:
        path: str of file path
        dir_name: str of directory name
    """
    if path == '':
        return ''
    path = Path(path).as_posix()
    uri = '/'.join(path.split('/')[-2:])
    file_name = path.rsplit('/', maxsplit=1)[-1]
    if dir_name == 'outputs':
        return f"{STATIC_SERVER_BASE}/outputs/{uri}"
    return f"{STATIC_SERVER_BASE}/inputs/{file_name}"


def url_path(result: list) -> list:
    """
    Converts the result to a list of URL paths.
    :param result: The result to convert.
    :return: The list of URL paths.
    """
    url_or_path = []
    if len(result) == 0:
        return url_or_path
    if str.startswith(result[0], 'http'):
        for res in result:
            uri = '/'.join(res.split('/')[-2:])
            url_or_path.append(os.path.join(path_outputs, uri))
        return url_or_path
    for res in result:
        url_or_path.append(to_http(res, "outputs"))
    return url_or_path


def change_filename(source: List[str], target: str, ext: str) -> list:
    """
    Change file name
    :param source:
    :param target:
    :param ext:
    return
    """
    if target in (None, '', 'none', 'None'):
        return source
    results = []
    images = len(source)
    for index in range(images):
        target_name = f"{target}-{str(index)}.{ext}"
        source_path = Path(source[index]).as_posix()
        source_dir = os.path.dirname(source_path)
        target_path = os.path.normpath(
            Path(os.path.join(source_dir, target_name)).as_posix()
        )
        try:
            os.makedirs(os.path.dirname(target_path), exist_ok=True)
            os.rename(source_path, target_path)
            results.append(target_path)
        except:
            results.append(source_path)
    return results
