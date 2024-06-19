"""
Calls the worker with the given params.
"""
import asyncio
import json

from fastapi import Response
from apis.utils.api_utils import params_to_params
from apis.utils.img_utils import (
    narray_to_base64img,
    base64_from_path
)
from apis.models.requests import CommonRequest
from modules.async_worker import AsyncTask, async_tasks


def convert_yield_to_json(item: list):
    """
    Converts the yield to a JSON string.
    :param yield: The yield to convert.
    :return: The JSON string.
    """
    if item[0] == "preview":
        try:
            data = {
                "progress": item[1][0],
                "preview": narray_to_base64img(item[1][2]),
                "message": item[1][1],
                "images": []
            }
            return f"{json.dumps(data)}\n"
        except Exception as e:
            print(e)

    data = {
        "progress": 100,
        "preview": None,
        "message": "",
        "images": [base64_from_path(image) for image in item[1]]
    }
    return f"{json.dumps(data)}\n"


async def stream_output(request: CommonRequest):
    """
    Calls the worker with the given params.
    :param request: The request object containing the params.
    """
    params = params_to_params(request)
    task = AsyncTask(args=params)
    async_tasks.append(task)
    while True:
        await asyncio.sleep(1)
        try:
            text = convert_yield_to_json(task.yields[-1])
        except IndexError:
            continue
        yield text
        if task.yields[-1][0] == "finish":
            break


async def binary_output(request: CommonRequest):
    """
    Calls the worker with the given params.
    :param request: The request object containing the params.
    """
    request.image_number = 1
    params = params_to_params(request)
    task = AsyncTask(args=params)
    async_tasks.append(task)
    while True:
        await asyncio.sleep(1)
        progress = task.yields[-1][0]
        image = task.yields[-1][1][-1]
        if progress == "finish":
            print(image)
            with open(image, "rb") as f:
                image = f.read()
            break
    return Response(image, media_type="image/png")
