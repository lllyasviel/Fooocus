
import gradio as gr
import os
from .http_server import *

def addResourceMonitor():
    ceq = None
    with gr.Row():
        ceq = gr.HTML(load_page('templates/perf-monitor/index.html'))

    return ceq

def load_page(filename):
    """Load an HTML file as a string and return it"""
    file_path = os.path.join("web", filename)
    with open(file_path, 'r') as file:
        content = file.read()
    return content
