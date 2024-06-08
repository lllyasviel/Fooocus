"""
Entry for startup fastapi server
"""
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from apis.routes.generate import router as generate


API_PORT = int(os.environ["GRADIO_SERVER_PORT"]) + 1
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


def run_server():
    """
    Run the FastAPI server
    """
    uvicorn.run(app, host="0.0.0.0", port=API_PORT)
