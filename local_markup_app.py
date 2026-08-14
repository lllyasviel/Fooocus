from __future__ import annotations

import gradio as gr

from local_markup.gradio_helpers import build_markup_ui_outputs, preset_to_ui_outputs
from local_markup.presets import PRESETS


def understand_edit(instruction: str):
    prompt, plan = build_markup_ui_outputs(instruction)
    return prompt, plan


def apply_preset(preset_name: str | None):
    instruction, prompt, plan = preset_to_ui_outputs(preset_name)
    return instruction, prompt, plan


def build_app() -> gr.Blocks:
    with gr.Blocks(title="Fooocus Local Markup Assistant") as demo:
        gr.Markdown("# Fooocus Local Markup Assistant\nDescribe an image edit in plain English, then copy the generated prompt into Fooocus Inpaint.")
        with gr.Row():
            with gr.Column():
                instruction = gr.Textbox(
                    label="Plain-English Edit Instruction",
                    placeholder="Example: make the shirt black and remove the trash from the table",
                    lines=3,
                )
                preset = gr.Dropdown(
                    label="Quick Edit Preset",
                    choices=list(PRESETS.keys()),
                    value=None,
                )
                with gr.Row():
                    understand_button = gr.Button("Understand Edit", variant="primary")
                    preset_button = gr.Button("Use Preset")
            with gr.Column():
                prompt = gr.Textbox(label="Fooocus Inpaint Prompt", lines=5)
                plan = gr.Markdown(label="Edit Plan")

        understand_button.click(understand_edit, inputs=instruction, outputs=[prompt, plan], queue=False)
        preset_button.click(apply_preset, inputs=preset, outputs=[instruction, prompt, plan], queue=False)
    return demo


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=7871, inbrowser=True)
