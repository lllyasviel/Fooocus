"""
Entry for startup fastapi server
"""
import os
from fastapi import FastAPI, Response
from fastapi.responses import FileResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from apis.routes.generate import router as generate
from apis.routes.query import router as query


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


@app.get("/")
async def root():
    """
    Root endpoint
    :return: root endpoint
    """
    return RedirectResponse("/docs")


@app.get("/outputs/{file_name}")
async def serve_outputs(file_name: str):
    """
    Serve outputs directory
    :param file_name: file name
    :return: file content
    """
    print(file_name)
    if file_name.split('.')[-1] in ['sqlite3', 'html']:
        return Response(status_code=404)
    return FileResponse(f"outputs/{file_name}")

app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


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
