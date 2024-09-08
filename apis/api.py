"""
Entry for startup fastapi server
"""
import os
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
from urllib.parse import urlparse

import uvicorn

from apis.routes.generate import secure_router as generate
from apis.routes.query import secure_router as query
from apis.routes.query import router
from apis.utils import file_utils
from apis.utils import api_utils

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow access from all sources
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all request headers
)


app.include_router(query)
app.include_router(generate)
app.include_router(router)


@app.get("/", tags=["Query"])
async def root():
    """
    Root endpoint
    :return: root endpoint
    """
    return RedirectResponse("/docs")


def run_server(arguments):
    """
    Run the FastAPI server
    :param arguments: command line arguments
    """
    if arguments.apikey != "":
        api_utils.APIKEY_AUTH = arguments.apikey

    os.environ["WEBHOOK_URL"] = arguments.webhook_url
    try:
        api_port = int(os.environ['API_PORT'])
    except KeyError:
        try:
            api_port = int(arguments.port) + 1
        except TypeError:
            api_port = int(os.environ["GRADIO_SERVER_PORT"]) + 1

    # Parse the base_url to handle cases where it includes a scheme
    parsed_url = urlparse(arguments.base_url)

    if parsed_url.scheme:
        # If a scheme is provided, use the full URL
        file_utils.STATIC_SERVER_BASE = f"{arguments.base_url}"
    else:
        file_utils.STATIC_SERVER_BASE = f"http://{arguments.base_url}:{api_port}"
    uvicorn.run(app, host=arguments.listen, port=api_port)
