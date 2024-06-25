"""
Query routes.
"""
import json
from fastapi import APIRouter
from fastapi.responses import JSONResponse
from modules.async_worker import async_tasks
from apis.utils.call_worker import current_task, session
from apis.utils.sql_client import GenerateRecord
from apis.models.response import RecordResponse


def tasks_info(task_id: str = None):
    """
    Returns the tasks.
    :param task_id: The task ID to filter by.
    :return: The tasks.
    """
    if task_id:
        query = session.query(GenerateRecord).filter_by(task_id=task_id).first()
        if query is None:
            return []
        result = json.loads(str(query))
        result["req_params"] = json.loads(result["req_params"])
        return result
    return async_tasks


router = APIRouter()

@router.get("/tasks")
async def get_tasks(
    query: str = "all",
    page: int = 0,
    page_size: int = 10):
    """
    Get all tasks.
    :param type: The type of tasks to filter by. One of all, history, current, pending
    :param page: The page number to return. used for history and pending
    :param page_size: The number of tasks to return per page.
    :return: The tasks.
    """
    historys, current, pending = [], [], []
    if query in ('all', 'history'):
        query_history = session.query(GenerateRecord).order_by(GenerateRecord.id.desc()).limit(page_size).offset(page * page_size).all()
        for q in query_history:
            result = json.loads(str(q))
            result["req_params"] = json.loads(result["req_params"])
            historys.append(result)
    if query in ('all', 'current'):
        current = await current_task()
    if query in ('all', 'pending'):
        pending = [task.task_id for task in async_tasks]
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


@router.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """
    Get a specific task by its ID.
    """
    return JSONResponse(tasks_info(task_id))
