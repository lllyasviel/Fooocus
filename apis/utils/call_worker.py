"""
Calls the worker with the given params.
"""
import asyncio
import copy
import os
import json
import uuid
import datetime
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from fastapi import Response
from apis.models.response import RecordResponse
from apis.utils.api_utils import params_to_params
from apis.utils.file_utils import save_base64
from apis.utils.sql_client import GenerateRecord

from apis.utils.img_utils import (
    narray_to_base64img,
    base64_from_path
)
from apis.models.requests import CommonRequest
from modules.async_worker import AsyncTask, async_tasks


class CurrentTask:
    """
    Current task class.
    """
    ct = None

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INPUT_PATH = os.path.join(ROOT_DIR, '..', 'inputs')
OUT_PATH = os.path.join(ROOT_DIR, '..', 'outputs')

engine = create_engine(
    f"sqlite:///{OUT_PATH}/db.sqlite3",
    connect_args={"check_same_thread": False},
    future=True
)
Session = sessionmaker(bind=engine, autoflush=True)
session = Session()


# todo: use argument to specify hosts
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
            url_or_path.append(os.path.join(OUT_PATH, uri))
        return url_or_path
    for res in result:
        path = Path(res).as_posix()
        uri = '/'.join(path.split('/')[-2:])
        url_or_path.append(f"http://127.0.0.1:7866/outputs/{uri}")
    return url_or_path


def pre_worker(request: CommonRequest):
    """
    Pre-processes the request.
    :param request: The request to pre-process.
    :return: The pre-processed request.
    """
    os.makedirs(INPUT_PATH, exist_ok=True)
    req_copy = copy.deepcopy(request)
    req_copy.inpaint_input_image = save_base64(req_copy.inpaint_input_image, INPUT_PATH)
    cn_imgs = []
    for cn in req_copy.controlnet_image:
        cn.cn_img = save_base64(cn.cn_img, INPUT_PATH)
        cn_imgs.append(cn)
    req_copy.controlnet_image = cn_imgs
    return req_copy


def post_worker(task: AsyncTask, started_at: int):
    """
    Posts the task to the worker.
    :param task: The task to post.
    :param started_at: The time the task started.
    :return: The task.
    """
    try:
        query = session.query(GenerateRecord).filter(GenerateRecord.task_id == task.task_id).first()
        query.start_mills = started_at
        query.finish_mills = int(datetime.datetime.now().timestamp() * 1000)
        query.task_status = "finished"
        query.progress = 100
        query.result = url_path(task.results)
        session.commit()
    except Exception as e:
        print(e)
    CurrentTask.ct = None
    return task


async def execute_in_background(task: AsyncTask, raw_req: CommonRequest, in_queue_mills):
    """
    Executes the request in the background.
    :param request: The request to execute.
    :param raw_req: The raw request.
    :param in_queue_mills: The time the request was enqueued.
    :return: The response.
    """
    finished = False
    started = False
    while not finished:
        await asyncio.sleep(0.01)
        if len(task.yields) > 0:
            if not started:
                started = True
                started_at = int(datetime.datetime.now().timestamp() * 1000)
                CurrentTask.ct = RecordResponse(
                    task_id=task.task_id,
                    req_params=json.loads(raw_req.model_dump_json()),
                    in_queue_mills=in_queue_mills,
                    start_mills=started_at,
                    task_status="running",
                    progress=0
                )
            flag, product = task.yields.pop(0)
            if flag == 'preview':
                if len(task.yields) > 0:
                    if task.yields[0][0] == 'preview':
                        continue
                percentage, _, image = product
                CurrentTask.ct.progress = percentage
                CurrentTask.ct.preview = narray_to_base64img(image)
            if flag == 'finish':
                finished = True
                post_worker(task=task, started_at=started_at)


async def stream_output(request: CommonRequest):
    """
    Calls the worker with the given params.
    :param request: The request object containing the params.
    """
    raw_req = pre_worker(request)
    params = params_to_params(request)
    task = AsyncTask(args=params)
    async_tasks.append(task)
    in_queue_mills=int(datetime.datetime.now().timestamp() * 1000)
    session.add(GenerateRecord(
        task_id=task.task_id,
        req_params=raw_req.model_dump_json(),
        webhook_url=raw_req.webhook_url,
        in_queue_mills=in_queue_mills
    ))
    session.commit()

    started = False
    finished = False
    while not finished:
        await asyncio.sleep(0.01)
        if len(task.yields) > 0:
            if not started:
                started = True
                started_at = int(datetime.datetime.now().timestamp() * 1000)
                CurrentTask.ct = RecordResponse(
                    task_id=task.task_id,
                    req_params=json.loads(raw_req.model_dump_json()),
                    in_queue_mills=in_queue_mills,
                    start_mills=started_at,
                    task_status="running",
                    progress=0,
                    result=[]
                )
            flag, product = task.yields.pop(0)
            if flag == 'preview':
                if len(task.yields) > 0:
                    if task.yields[0][0] == 'preview':
                        continue
                percentage, title, image = product
                text = json.dumps({
                    "progress": percentage,
                    "preview": narray_to_base64img(image),
                    "message": title,
                    "images": []
                })
                CurrentTask.ct.progress = percentage
                CurrentTask.ct.preview = narray_to_base64img(image)
                yield f"{text}\n"
            if flag == 'results':
                print(task.results)
                print(task.processing)
            if flag == 'finish':
                text = json.dumps({
                    "progress": 100,
                    "preview": None,
                    "message": "",
                    "images": url_path(task.results)
                })
                post_worker(task=task, started_at=started_at)
                yield f"{text}\n"
                finished = True


async def binary_output(request: CommonRequest):
    """
    Calls the worker with the given params.
    :param request: The request object containing the params.
    """
    request.image_number = 1
    raw_req = pre_worker(request)
    params = params_to_params(request)
    task = AsyncTask(args=params, task_id=uuid.uuid4().hex)
    async_tasks.append(task)
    in_queue_mills=int(datetime.datetime.now().timestamp() * 1000)
    session.add(GenerateRecord(
        task_id=task.task_id,
        req_params=raw_req.model_dump_json(),
        webhook_url=raw_req.webhook_url,
        in_queue_mills=in_queue_mills
    ))
    session.commit()

    started = False
    finished = False
    while not finished:
        await asyncio.sleep(0.01)
        if len(task.yields) > 0:
            if not started:
                started = True
                started_at = int(datetime.datetime.now().timestamp() * 1000)
                CurrentTask.ct = RecordResponse(
                    task_id=task.task_id,
                    req_params=json.loads(raw_req.model_dump_json()),
                    in_queue_mills=in_queue_mills,
                    start_mills=started_at,
                    task_status="running",
                    progress=0,
                    result=[]
                )
            flag, product = task.yields.pop(0)
            if flag == 'preview':
                if len(task.yields) > 0:
                    if task.yields[0][0] == 'preview':
                        continue
                percentage, _, image = product
                CurrentTask.ct.progress = percentage
                CurrentTask.ct.preview = narray_to_base64img(image)
            if flag == 'finish':
                finished = True
                post_worker(task=task, started_at=started_at)

    with open (task.results[0], "rb") as f:
        image = f.read()
    return Response(image, media_type="image/png")


async def async_worker(request: CommonRequest) -> RecordResponse:
    """
    Calls the worker with the given params.
    :param request: The request object containing the params.
    """
    raw_req = pre_worker(request)
    task_id = uuid.uuid4().hex
    task = AsyncTask(
        task_id=task_id,
        args=params_to_params(request)
    )
    async_tasks.append(task)
    in_queue_mills = int(datetime.datetime.now().timestamp() * 1000)
    session.add(GenerateRecord(
        task_id=task.task_id,
        req_params=raw_req.model_dump_json(),
        webhook_url=raw_req.webhook_url,
        in_queue_mills=in_queue_mills
    ))
    session.commit()

    asyncio.create_task(execute_in_background(task, raw_req, in_queue_mills))

    return RecordResponse(task_id=task_id, task_status="pending").model_dump()


async def current_task():
    """
    Returns the current task.
    """
    if CurrentTask.ct is None:
        return []
    return [CurrentTask.ct.model_dump()]
