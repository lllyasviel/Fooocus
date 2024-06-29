"""
Generate API routes
"""
from fastapi import (
    APIRouter, Depends, Query, UploadFile
)
from sse_starlette.sse import EventSourceResponse

from apis.models.base import DescribeImageResponse, DescribeImageType
from apis.utils.api_utils import api_key_auth
from apis.utils.call_worker import (
    async_worker,
    stream_output,
    binary_output
)
from apis.models.requests import CommonRequest
from apis.utils.img_utils import read_input_image
from modules.util import HWC3


secure_router = APIRouter(
    dependencies=[Depends(api_key_auth)]
)


@secure_router.post("/v1/engine/generate/", summary="Generate endpoint all in one")
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


@secure_router.post(
        path="/v1/tools/describe-image",
        response_model=DescribeImageResponse,
        tags=["GenerateV1"])
def describe_image(
    image: UploadFile,
    image_type: DescribeImageType = Query(
        DescribeImageType.photo,
        description="Image type, 'Photo' or 'Anime'")):
    """\nDescribe image\n
    Describe image, Get tags from an image
    Arguments:
        image {UploadFile} -- Image to get tags
        image_type {DescribeImageType} -- Image type, 'Photo' or 'Anime'
    Returns:
        DescribeImageResponse -- Describe image response, a string
    """
    if image_type == DescribeImageType.photo:
        from extras.interrogate import default_interrogator as default_interrogator_photo
        interrogator = default_interrogator_photo
    else:
        from extras.wd14tagger import default_interrogator as default_interrogator_anime
        interrogator = default_interrogator_anime
    img = HWC3(read_input_image(image))
    result = interrogator(img)
    return DescribeImageResponse(describe=result)
