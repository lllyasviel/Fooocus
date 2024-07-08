"""
Do something after generate
"""
import datetime
import os

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apis.models.base import CurrentTask
from apis.utils.file_utils import url_path
from apis.utils.sql_client import GenerateRecord
from apis.utils.web_hook import send_result_to_web_hook
from modules.async_worker import AsyncTask

from apis.utils import file_utils


ROOT_DIR = file_utils.SCRIPT_PATH
INPUT_PATH = os.path.join(ROOT_DIR, 'inputs')
OUT_PATH = os.path.join(ROOT_DIR, 'outputs')

engine = create_engine(
    f"sqlite:///{OUT_PATH}/db.sqlite3",
    connect_args={"check_same_thread": False},
    future=True
)
Session = sessionmaker(bind=engine, autoflush=True)
session = Session()


async def post_worker(task: AsyncTask, started_at: int):
    """
    Posts the task to the worker.
    :param task: The task to post.
    :param started_at: The time the task started.
    :return: The task.
    """
    task_status = "finished"
    if task.last_stop in ['stop', 'skip']:
        task_status = task.last_stop
    try:
        query = session.query(GenerateRecord).filter(GenerateRecord.task_id == task.task_id).first()
        query.start_mills = started_at
        query.finish_mills = int(datetime.datetime.now().timestamp() * 1000)
        query.task_status = task_status
        query.progress = 100
        query.result = url_path(task.results)
        finally_result = str(query)
        session.commit()
        await send_result_to_web_hook(query.webhook_url, finally_result)
        return finally_result
    except Exception as e:
        print(e)
    CurrentTask.ct = None
