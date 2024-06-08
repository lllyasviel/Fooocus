import time
import uuid
from typing import List
from apis.models.task import TaskStatus


class TaskObj:
    """Task object"""
    def __init__(self, req_params: object):
        self.accept: str = "application/json"
        self.task_id: str = str(uuid.uuid4())
        self.req_param: object = req_params
        self.in_queue_mills: int = int(time.time()*1000)
        self.start_mills: int = 0
        self.finish_mills: int = 0
        self.task_status: TaskStatus | None = None
        self.progress: int = 0
        self.task_step_preview: str | None = None
        self.webhook_url: str | None = None
        self.task_result: List = []

    def update(self, attribute: str, value):
        """
        Update task obj
        Args:
            attribute: attribute name
            value: value
        """
        setattr(self, attribute, value)
        return getattr(self, attribute)
