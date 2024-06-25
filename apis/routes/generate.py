"""
Generate API routes
"""
from fastapi import (
    APIRouter
)
from sse_starlette.sse import EventSourceResponse
from apis.utils.call_worker import (
    async_worker,
    stream_output,
    binary_output
)
from apis.models.requests import CommonRequest

router = APIRouter()


@router.post("/generate/", summary="Generate API V2 routes")
async def generate_routes(common_request: CommonRequest):
    """
    Generate API routes
    """
    if common_request.stream_output:
        return EventSourceResponse(
            stream_output(request=common_request)
        )

    if common_request.async_process:
        return await async_worker(request=common_request)

    return await binary_output(request=common_request)
