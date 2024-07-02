"""
The response model is used to define the structure of the response object that will be returned by the API.
"""
from typing import List
from pydantic import (
    BaseModel,
    ConfigDict, Field
)


class RecordResponse(BaseModel):
    """
    The RecordResponse class defines the structure of the response object for a single record.
    """
    id: int = Field(default=-1, title="ID", description="The unique identifier of the record.")
    task_id: str = Field(default="", title="Task ID", description="The ID of the task associated with the record.")
    req_params: dict = Field(default={}, title="Request Parameters", description="The request parameters associated with the record.")
    in_queue_mills: int = Field(default=-1, title="In Queue Milliseconds", description="The time in milliseconds when the record was added to the queue.")
    start_mills: int = Field(default=-1, title="Start Milliseconds", description="The time in milliseconds when the record started processing.")
    finish_mills: int = Field(default=-1, title="Finish Milliseconds", description="The time in milliseconds when the record finished processing.")
    task_status: str = Field(default="", title="Task Status", description="The status of the task associated with the record.")
    progress: float = Field(default=-1, title="Progress", description="The progress of the task associated with the record.")
    preview: str = Field(default="", title="Preview", description="The preview of the task associated with the record.")
    webhook_url: str = Field(default="", title="Webhook URL", description="The webhook URL associated with the record.")
    result: List = Field(default=[], title="Result", description="The result of the task associated with the record.")


class AllModelNamesResponse(BaseModel):
    """
    all model list response
    """
    model_filenames: List[str] = Field(description="All available model filenames")
    lora_filenames: List[str] = Field(description="All available lora filenames")

    model_config = ConfigDict(
        protected_namespaces=('protect_me_', 'also_protect_')
    )
