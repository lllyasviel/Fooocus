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
async def get_tasks():
    """
    Get all tasks.
    """
    historys = []
    query = session.query(GenerateRecord).order_by(GenerateRecord.id.desc()).limit(20).all()
    for q in query:
        result = json.loads(str(q))
        result["req_params"] = json.loads(result["req_params"])
        historys.append(result)
    return JSONResponse({
        "history": historys,
        "current": await current_task(),
        "pending": [task.task_id for task in async_tasks]
    })


@router.get("/tasks/{task_id}")
async def get_task(task_id: str):
    """
    Get a specific task by its ID.
    """
    return JSONResponse(tasks_info(task_id))
