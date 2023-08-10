import gradio as gr
from modules.default_pipeline import process


def generate_clicked(positive_prompt):
    return process(positive_prompt=positive_prompt, negative_prompt='bad, ugly')


block = gr.Blocks().queue()
with block:
    with gr.Row():
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", value='a handsome man in forest')
            run_button = gr.Button(label="Run")
        with gr.Column():
            result_gallery = gr.Gallery(label='Output', show_label=False, elem_id="gallery").style(grid=2, height='auto')
    run_button.click(fn=generate_clicked, inputs=[prompt], outputs=[result_gallery])


block.launch()
