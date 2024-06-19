"""
Entry for startup fastapi server
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from apis.routes.generate import router as generate


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow access from all sources
    allow_credentials=True,
    allow_methods=["*"],  # Allow all HTTP methods
    allow_headers=["*"],  # Allow all request headers
)


# app.include_router(query)
app.include_router(generate)


def run_server(arguments):
    """
    Run the FastAPI server
    :param arguments: command line arguments
    """
    try:
        API_PORT = int(arguments.port) + 1
    except TypeError:
        API_PORT = int(os.environ["GRADIO_SERVER_PORT"]) + 1
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
