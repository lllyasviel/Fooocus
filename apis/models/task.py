"""
Task and job related models
"""
from enum import Enum


class TaskStatus(str, Enum):
    """
    Task status
    """
    waiting = 'WAITING'
    running = 'RUNNING'
    success = 'SUCCESS'
    error = 'ERROR'
    cancel = 'CANCEL'


class AsyncJobStage(str, Enum):
    """
    Async job stage
    """
    waiting = 'WAITING'
    running = 'RUNNING'
    success = 'SUCCESS'
    error = 'ERROR'
