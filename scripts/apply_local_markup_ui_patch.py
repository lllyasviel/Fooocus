from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEBUI = ROOT / "webui.py"

HELPER = """

def local_markup_understand_edit(user_instruction):
    from local_markup.ui_adapter import build_markup_ui_outputs

    if user_instruction is None or str(user_instruction).strip() == '':
        return '', '', 'Enter an edit instruction first.', ''
    return build_markup_ui_outputs(str(user_instruction))
"""

OLD_WIDGET_BLOCK = """                                inpaint_mode = gr.Dropdown(choices=modules.flags.inpaint_options, value=modules.config.default_inpaint_method, label='Method')\n                                inpaint_additional_prompt = gr.Textbox(placeholder=\"Describe what you want to inpaint.\", elem_id='inpaint_additional_prompt', label='Inpaint Additional Prompt', visible=False)\n                                outpaint_selections = gr.CheckboxGroup(choices=['Left', 'Right', 'Top', 'Bottom'], value=[], label='Outpaint Direction')\n"""

NEW_WIDGET_BLOCK = """                                inpaint_mode = gr.Dropdown(choices=modules.flags.inpaint_options, value=modules.config.default_inpaint_method, label='Method')\n                                local_markup_instruction = gr.Textbox(placeholder=\"Plain English edit: make the shirt black, remove table clutter, clean background...\", label='Understand Edit')\n                                local_markup_button = gr.Button(value='Understand Edit')\n                                local_markup_summary = gr.Textbox(label='Edit Plan', interactive=False, visible=True)\n                                local_markup_negative_prompt = gr.Textbox(label='Suggested Negative Prompt', interactive=False, visible=False)\n                                inpaint_additional_prompt = gr.Textbox(placeholder=\"Describe what you want to inpaint.\", elem_id='inpaint_additional_prompt', label='Inpaint Additional Prompt', visible=False)\n                                outpaint_selections = gr.CheckboxGroup(choices=['Left', 'Right', 'Top', 'Bottom'], value=[], label='Outpaint Direction')\n"""

OLD_CLICK_BLOCK = """                                example_inpaint_prompts.click(lambda x: x[0], inputs=example_inpaint_prompts, outputs=inpaint_additional_prompt, show_progress=False, queue=False)\n\n                            with gr.Column(visible=modules.config.default_inpaint_advanced_masking_checkbox) as inpaint_mask_generation_col:\n"""

NEW_CLICK_BLOCK = """                                example_inpaint_prompts.click(lambda x: x[0], inputs=example_inpaint_prompts, outputs=inpaint_additional_prompt, show_progress=False, queue=False)\n                                local_markup_button.click(local_markup_understand_edit, inputs=local_markup_instruction, outputs=[inpaint_additional_prompt, inpaint_mask_dino_prompt_text, local_markup_summary, local_markup_negative_prompt], queue=False, show_progress=False)\n\n                            with gr.Column(visible=modules.config.default_inpaint_advanced_masking_checkbox) as inpaint_mask_generation_col:\n"""


def apply_patch() -> None:
    text = WEBUI.read_text(encoding="utf-8")
    if "def local_markup_understand_edit" not in text:
        marker = "\n\nreload_javascript()"
        if marker not in text:
            raise RuntimeError("Could not find reload_javascript marker")
        text = text.replace(marker, HELPER + marker, 1)

    if "local_markup_instruction" not in text:
        if OLD_WIDGET_BLOCK not in text:
            raise RuntimeError("Could not find inpaint widget block")
        text = text.replace(OLD_WIDGET_BLOCK, NEW_WIDGET_BLOCK, 1)

    if "local_markup_button.click" not in text:
        if OLD_CLICK_BLOCK not in text:
            raise RuntimeError("Could not find inpaint click block")
        text = text.replace(OLD_CLICK_BLOCK, NEW_CLICK_BLOCK, 1)

    WEBUI.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    apply_patch()
    print("Applied local markup UI patch to webui.py")
