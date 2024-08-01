"""
Query routes.
"""
import os
import re
import json

from datetime import datetime
from typing import List

from fastapi import APIRouter, Depends, Header, Response
from fastapi.responses import JSONResponse
from starlette.responses import FileResponse

from apis.models.response import AllModelNamesResponse
from apis.utils.api_utils import api_key_auth
from apis.utils.call_worker import current_task, session
from apis.utils.file_utils import delete_tasks
from apis.utils.img_utils import convert_image
from apis.utils.sql_client import GenerateRecord
from modules.async_worker import async_tasks

from modules.config import path_outputs


def date_to_timestamp(date: str) -> int | None:
    """
    Converts the date to a timestamp.
    :param date: The ISO 8601 date to convert.
    :return: The timestamp in millisecond.
    """
    pattern = r'\A\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}'
    if date is None:
        return None
    try:
        date = re.match(pattern, date).group()
    except AttributeError:
        return None
    return int(datetime.fromisoformat(date).timestamp()) * 1000


async def tasks_info(task_id: str = None):
    """
    Returns the tasks.
    :param task_id: The task ID to filter by.
    :return: The tasks.
    """
    ct = await current_task()
    try:
        ct_task_id = ct[0]['task_id']
    except IndexError:
        ct_task_id = None

    if ct_task_id is not None and ct_task_id == task_id:
        return ct[0]
    if task_id:
        query = session.query(GenerateRecord).filter_by(task_id=task_id).first()
        if query is None:
            return []
        result = json.loads(str(query))
        try:
            result["req_params"] = json.loads(result["req_params"])
        except TypeError:
            pass
        return result
    return async_tasks


secure_router = APIRouter(
    dependencies=[Depends(api_key_auth)]
)

router = APIRouter()


@secure_router.get("/tasks", tags=["Query"])
async def get_tasks(
        query: str = "all",
        page: int = 0,
        page_size: int = 10,
        start_at: str = None,
        end_at: str = datetime.now().isoformat(),
        action: str = None):
    """
    Get all tasks.
    :param query: The type of tasks to filter by. One of all, history, current, pending
    :param page: The page number to return. used for history and pending
    :param page_size: The number of tasks to return per page.
    :param start_at: The start time to filter by.
    :param end_at: The end time to filter by.
    :param action: Delete only.
    :return: The tasks.
    """
    start_at = date_to_timestamp(start_at)
    end_at = date_to_timestamp(end_at)
    action = action.lower() if action is not None else None
    if start_at is None or start_at >= end_at:
        start_at, end_at = None, None

    if action == 'delete':
        try:
            query_result = session.query(GenerateRecord).filter(GenerateRecord.in_queue_mills >= start_at).filter(GenerateRecord.in_queue_mills <= end_at).all()
            tasks = [json.loads(str(task)) for task in query_result]
            delete_tasks(tasks)
            session.query(GenerateRecord).filter(GenerateRecord.in_queue_mills >= start_at).filter(GenerateRecord.in_queue_mills <= end_at).delete()
            session.commit()
        except Exception as e:
            print(e)
        return

    historys, current, pending = [], [], []
    pending_ids = [task.task_id for task in async_tasks]

    if query in ('all', 'history') and action != "delete":
        if start_at is not None:
            query_history = session.query(GenerateRecord).filter(GenerateRecord.task_id.not_in(pending_ids)).filter(GenerateRecord.in_queue_mills >= start_at).filter(GenerateRecord.in_queue_mills <= end_at).all()
        else:
            query_history = session.query(GenerateRecord).filter(GenerateRecord.task_id.not_in(pending_ids)).order_by(GenerateRecord.id.desc()).limit(page_size).offset(page * page_size).all()
        for q in query_history:
            result = json.loads(str(q))
            historys.append(result)
    if query in ('all', 'current'):
        current = await current_task()
    if query in ('all', 'pending'):

        query_pending = session.query(GenerateRecord).filter(GenerateRecord.task_id.in_(pending_ids)).all()
        for q in query_pending:
            result = json.loads(str(q))
            pending.append(result)
        start_index = page * page_size
        end_index = (page + 1) * page_size
        max_page = len(pending) / page_size if len(pending) / page_size == len(pending) // page_size else len(pending) // page_size + 1
        if page > max_page:
            pending = []
        else:
            pending = pending[start_index:end_index]

    return JSONResponse({
        "history": historys,
        "current": current,
        "pending": pending
    })


@secure_router.get("/tasks/{task_id}", tags=["Query"])
async def get_task(task_id: str):
    """
    Get a specific task by its ID.
    """
    return JSONResponse(await tasks_info(task_id))


@router.get("/outputs/{dir_name}/{file_name}", tags=["Query"])
async def get_output(dir_name: str, file_name: str, accept: str = Header(None)):
    """
    Get a specific output by its ID.
    """
    if not os.path.exists(f"{path_outputs}/{dir_name}/{file_name}"):
        return Response(status_code=404)

    accept_formats = ('png', 'jpg', 'jpeg', 'webp')
    try:
        _, ext = accept.lower().split("/")
        if ext not in accept_formats:
            ext = None
    except ValueError:
        ext = None

    if not file_name.endswith(accept_formats):
        return Response(status_code=404)

    if ext is None:
        try:
            return FileResponse(f"{path_outputs}/{dir_name}/{file_name}")
        except FileNotFoundError:
            return Response(status_code=404)
    img = await convert_image(f"{path_outputs}/{dir_name}/{file_name}", ext)
    return Response(content=img, media_type=f"image/{ext}")


@router.get("/inputs/{file_name}", tags=["Query"])
async def get_input(file_name: str, accept: str = Header(None)):
    """
    Get a specific input by its ID.
    """
    if not os.path.exists(f"inputs/{file_name}"):
        return Response(status_code=404)

    accept_formats = ('png', 'jpg', 'jpeg', 'webp')
    try:
        _, ext = accept.lower().split("/")
        if ext not in accept_formats:
            ext = None
    except ValueError:
        ext = None
    if ext is None:
        try:
            return FileResponse(f"inputs/{file_name}")
        except FileNotFoundError:
            return Response(status_code=404)

    img = await convert_image(f"inputs/{file_name}", ext)
    return Response(content=img, media_type=f"image/{ext}")


@secure_router.get(
        path="/v1/engines/all-models",
        response_model=AllModelNamesResponse,
        description="Get all filenames of base model and lora",
        tags=["Query"])
def all_models():
    """Refresh and return all models"""
    from modules import config
    config.update_files()
    models = AllModelNamesResponse(
        model_filenames=config.model_filenames,
        lora_filenames=config.lora_filenames)
    return models


@secure_router.get(
        path="/v1/engines/styles",
        response_model=List[str],
        description="Get all legal Fooocus styles",
        tags=['Query'])
def all_styles():
    """Return all available styles"""
    from modules.sdxl_styles import legal_style_names
    return legal_style_names
