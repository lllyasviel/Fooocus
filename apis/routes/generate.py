"""
Generate API routes
"""
from typing import Annotated
from fastapi import (
    APIRouter,
    Header
)
from sse_starlette.sse import EventSourceResponse
from apis.utils.call_worker import (
    stream_output,
    binary_output
)
from apis.models.requests import CommonRequest

router = APIRouter()


@router.post("/generate/", summary="Generate API V2 routes")
async def generate_routes(
        common_request: CommonRequest,
        accept: Annotated[str | None, Header()] = None):
    """
    Generate API routes
    """
    if accept == "application/json":
        return EventSourceResponse(
            stream_output(request=common_request)
        )

    return await binary_output(request=common_request)
