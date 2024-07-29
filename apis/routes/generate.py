"""
Generate API routes
"""
from fastapi import (
    APIRouter, Depends, Header, Query, UploadFile
)
from sse_starlette.sse import EventSourceResponse

from apis.models.base import (
    DescribeImageResponse,
    DescribeImageType,
    CurrentTask, GenerateMaskRequest
)
from apis.utils.api_utils import api_key_auth
from apis.utils.call_worker import (
    async_worker,
    stream_output,
    binary_output,
    generate_mask as gm
)
from apis.models.requests import CommonRequest, DescribeImageRequest
from apis.utils.img_utils import read_input_image
from modules.util import HWC3

secure_router = APIRouter(
    dependencies=[Depends(api_key_auth)]
)


@secure_router.post(
    path="/v1/engine/generate/",
    summary="Generate endpoint all in one",
    tags=["GenerateV1"])
async def generate_routes(
        common_request: CommonRequest,
        accept: str = Header(None)):
    """
    Generate API routes
    """
    try:
        accept, ext = accept.lower().split("/")
        if ext not in ["png", "jpg", "jpeg", "webp"]:
            ext = 'png'
    except ValueError:
        ext = 'png'
        pass

    if accept == "image":
        return await binary_output(
            request=common_request,
            ext=ext)

    if common_request.stream_output:
        return EventSourceResponse(
            stream_output(request=common_request)
        )

    if common_request.async_process:
        return await async_worker(request=common_request)

    return await async_worker(request=common_request, wait_for_result=True)


@secure_router.post(
    path="/v1/tools/generate_mask",
    summary="Generate mask endpoint",
    tags=["GenerateV1"])
async def generate_mask(mask_options: GenerateMaskRequest) -> str:
    """
    Generate mask endpoint
    """
    return await gm(request=mask_options)


@secure_router.post(
    path="/v1/tools/describe-image",
    response_model=DescribeImageResponse,
    tags=["GenerateV1"])
async def describe_image(
        request: DescribeImageRequest):
    """\nDescribe image\n
    Describe image, Get tags from an image
    Arguments:
        request {DescribeImageRequest} -- Describe image request
    Returns:
        DescribeImageResponse -- Describe image response, a string
    """
    image = request.image
    image_type = request.image_type
    if image_type == DescribeImageType.photo:
        from extras.interrogate import default_interrogator as default_interrogator_photo
        interrogator = default_interrogator_photo
    else:
        from extras.wd14tagger import default_interrogator as default_interrogator_anime
        interrogator = default_interrogator_anime
    img = HWC3(await read_input_image(image))
    result = interrogator(img)
    return DescribeImageResponse(describe=result)


@secure_router.post("/v1/engine/control", tags=["GenerateV1"])
async def stop_engine(action: str):
    """Stop or skip engine"""
    if action not in ["stop", "skip"]:
        return {"message": "Invalid control action"}
    if CurrentTask.task is None:
        return {"message": "No task running"}
    from ldm_patched.modules import model_management
    ct = CurrentTask.task
    ct.last_stop = action
    if ct.processing:
        model_management.interrupt_current_processing()
    return {"message": f"task {action}ed"}
