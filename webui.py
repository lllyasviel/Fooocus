import gradio as gr

from modules.sdxl_styles import apply_style
from modules.default_pipeline import process


def generate_clicked(positive_prompt):

    p, n = apply_style('cinematic-default', positive_prompt, '')

    print(p)
    print(n)

    return process(positive_prompt=p,
                   negative_prompt=n)


block = gr.Blocks()
with block:
    with gr.Row():
        with gr.Column(scale=0.7):
            gallery = gr.Gallery(label='Gallery', show_label=False, object_fit='contain', height=768)
            with gr.Row():
                with gr.Column(scale=0.85):
                    prompt = gr.Textbox(show_label=False, placeholder="Type prompt here.", container=False)
                with gr.Column(scale=0.15, min_width=0):
                    run_button = gr.Button(label="Generate", value="Generate")
            with gr.Row():
                advanced_checkbox = gr.Checkbox(label='Advanced', value=False, container=False)
        with gr.Column(scale=0.3):
            with gr.Group():
                gr.Textbox()
        run_button.click(fn=generate_clicked, inputs=[prompt], outputs=[gallery])

block.launch()
