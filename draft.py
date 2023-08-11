import gradio as gr

from modules.sdxl_styles import apply_style
from modules.default_pipeline import process


def generate_clicked(positive_prompt):

    p, n = apply_style('cinematic-default', positive_prompt, '')

    print(p)
    print(n)

    return process(positive_prompt=p,
                   negative_prompt=n)


block = gr.Blocks().queue()
with block:
    with gr.Column():
        prompt = gr.Textbox(label="Prompt", value='a handsome man in forest')
        run_button = gr.Button(label="Run")
        result_gallery = gr.Gallery(label='Fooocus', show_label=True, elem_id="gallery",
                                    object_fit='contain', height=768)
    run_button.click(fn=generate_clicked, inputs=[prompt], outputs=[result_gallery])


block.launch()
