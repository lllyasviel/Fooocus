"""
Entry for startup fastapi server
"""
import os
from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from apis.routes.generate import secure_router as generate
from apis.routes.query import secure_router as query
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
        api_port = int(arguments.port) + 1
    except TypeError:
        api_port = int(os.environ["GRADIO_SERVER_PORT"]) + 1
    file_utils.STATIC_SERVER_BASE = f"http://{arguments.base_url}:{api_port}"
    uvicorn.run(app, host=arguments.listen, port=api_port)
