"""
Do something after generate
"""
import datetime
import os
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from apis.models.base import CurrentTask
from apis.utils.sql_client import GenerateRecord
from modules.async_worker import AsyncTask


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
