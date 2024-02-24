from rembg import remove
import gradio as gr
from PIL import Image

def rembg_run(path, progress=gr.Progress(track_tqdm=True)):
    input = Image.open(path)
    output = remove(input)
    return output