import gradio as gr

from local_markup.easy_sdxl import build_easy_sdxl_outputs
from local_markup.gradio_helpers import build_markup_ui_outputs
from local_markup.photo_bundle import BUNDLE_PRESETS, SAFE_OUTFIT_MAP, build_bundle_outputs
from local_markup.presets import PRESETS, get_preset
from local_markup.style_explainer import describe_style_markdown, list_style_names, style_recommendations_for_goal


WORKFLOWS = ["Generate New Image", "Edit This Image", "Use Image as Reference", "Improve / Enhance"]
AREAS = ["Face", "Glasses", "Hair", "Shirt / Clothes", "Background", "Object", "Whole Image", "I will draw the mask"]
STYLE_NAMES = list_style_names()
BUNDLE_NAMES = list(BUNDLE_PRESETS.keys())
OUTFIT_NAMES = list(SAFE_OUTFIT_MAP.keys())


def plan_easy_sdxl(workflow, area, instruction):
    return build_easy_sdxl_outputs(workflow, area, instruction)


def headshot_polo_preset():
    return "Edit This Image", "Shirt / Clothes", "Remove glasses and change suit and tie to a clean navy blue polo shirt"


def remove_jacket_preset():
    return "Edit This Image", "Shirt / Clothes", "Remove only the jacket. Keep the same person, same face, same pose, same background, and replace the jacket area with a clean professional shirt."


def build_one_click_recipe(instruction):
    goal = instruction or "Remove only the jacket"
    positive, inpaint, detection, negative, settings, checklist = build_easy_sdxl_outputs("Edit This Image", "Shirt / Clothes", goal)
    recipe = (
        "ONE-CLICK EXACT EDIT RECIPE\n\n"
        "1. Fooocus tab: Inpaint or Outpaint\n"
        "2. Enable Advanced Masking Features.\n"
        "3. Mask generation model: sam\n"
        "4. Detection prompt: jacket, suit jacket, blazer, coat\n"
        "5. Click Generate mask from image.\n"
        "6. Review mask. It should cover only the jacket/clothes area, not the face.\n"
        "7. Paste the inpaint prompt below.\n"
        "8. Generate.\n\n"
        "Expected result: same headshot, same face, same lighting, jacket removed/replaced only in the masked clothing area."
    )
    return positive, inpaint, "jacket, suit jacket, blazer, coat", negative, settings, recipe


def explain_style(style_name, sample_prompt):
    return describe_style_markdown(style_name, sample_prompt)


def recommend_styles(goal):
    return style_recommendations_for_goal(goal)


def plan_photo_bundle(bundle, outfit, goal):
    return build_bundle_outputs(bundle, outfit, goal)


def understand_edit(instruction):
    prompt, plan = build_markup_ui_outputs(instruction)
    return prompt, plan


def apply_preset(preset_name):
    if not preset_name:
        return "", "", "Choose a preset first."
    preset = get_preset(preset_name)
    instruction = preset["instruction"]
    prompt, plan = build_markup_ui_outputs(instruction)
    return instruction, prompt, plan


def build_app():
    with gr.Blocks(title="Easy SDXL Control Center") as demo:
        gr.Markdown(
            "# Easy SDXL Control Center\n"
            "Use simple choices on top while keeping Fooocus SDXL concepts underneath: Inpaint, Image Prompt, Enhance, steps, CFG, denoise, seed, styles, and LoRAs."
        )
        with gr.Tab("Easy SDXL Planner"):
            gr.Markdown(
                "### Use this as your SDXL copilot\n"
                "For exact edits, use **Edit This Image** and Fooocus **Inpaint or Outpaint**. "
                "For a new picture inspired by the upload, use **Use Image as Reference**."
            )
            with gr.Row():
                with gr.Column():
                    workflow = gr.Radio(label="What are you trying to do?", choices=WORKFLOWS, value="Edit This Image")
                    area = gr.Radio(label="Area / SDXL target", choices=AREAS, value="I will draw the mask")
                    instruction = gr.Textbox(label="Plain-English instruction", placeholder="Example: remove glasses and change the suit to a navy polo shirt", lines=3)
                    with gr.Row():
                        plan_button = gr.Button("Plan SDXL Settings", variant="primary")
                        headshot_button = gr.Button("Headshot: no glasses + polo")
                        jacket_button = gr.Button("One-click recipe: remove jacket")
                with gr.Column():
                    settings = gr.Textbox(label="Recommended SDXL settings", lines=4)
                    checklist = gr.Textbox(label="What to do next in Fooocus", lines=12)
            with gr.Row():
                positive_prompt = gr.Textbox(label="Main prompt", lines=4)
                inpaint_prompt = gr.Textbox(label="Inpaint prompt", lines=4)
            with gr.Row():
                detection_prompt = gr.Textbox(label="Detection / mask prompt", lines=2)
                negative_prompt = gr.Textbox(label="Negative prompt", lines=3)

            plan_button.click(plan_easy_sdxl, inputs=[workflow, area, instruction], outputs=[positive_prompt, inpaint_prompt, detection_prompt, negative_prompt, settings, checklist], queue=False)
            headshot_button.click(headshot_polo_preset, outputs=[workflow, area, instruction], queue=False).then(plan_easy_sdxl, inputs=[workflow, area, instruction], outputs=[positive_prompt, inpaint_prompt, detection_prompt, negative_prompt, settings, checklist], queue=False)
            jacket_button.click(remove_jacket_preset, outputs=[workflow, area, instruction], queue=False).then(build_one_click_recipe, inputs=[instruction], outputs=[positive_prompt, inpaint_prompt, detection_prompt, negative_prompt, settings, checklist], queue=False)

        with gr.Tab("Photo Bundle Builder"):
            gr.Markdown(
                "### Build a safe personal image bundle from a few reference photos\n"
                "Use this for standing portraits, swimming/resort images, business/casual sets, and lifestyle variations. "
                "Swimwear is supported. Nudity or undressing generation is intentionally not supported."
            )
            with gr.Row():
                with gr.Column():
                    bundle = gr.Dropdown(label="Bundle type", choices=BUNDLE_NAMES, value="Swimming / Resort")
                    outfit = gr.Dropdown(label="Safe outfit direction", choices=OUTFIT_NAMES, value="swimwear")
                    bundle_goal = gr.Textbox(label="Goal", value="make me standing near a pool ready for swimming", lines=3)
                    bundle_button = gr.Button("Build Photo Bundle Prompts", variant="primary")
                with gr.Column():
                    bundle_identity_prompt = gr.Textbox(label="Identity / main prompt", lines=5)
                    bundle_negative_prompt = gr.Textbox(label="Negative prompt / safety guardrails", lines=5)
            bundle_plan = gr.Textbox(label="Bundle shot list and expectations", lines=16)
            bundle_button.click(plan_photo_bundle, inputs=[bundle, outfit, bundle_goal], outputs=[bundle_identity_prompt, bundle_negative_prompt, bundle_plan], queue=False)

        with gr.Tab("Style Expectations"):
            gr.Markdown(
                "### Know what each style is doing before you generate\n"
                "Pick a style and preview the positive/negative style text that will affect your image."
            )
            with gr.Row():
                with gr.Column():
                    style_name = gr.Dropdown(label="Fooocus style", choices=STYLE_NAMES, value="Fooocus Photograph")
                    style_goal = gr.Textbox(label="Your goal", value="professional realistic headshot", lines=2)
                    with gr.Row():
                        style_button = gr.Button("Explain Style", variant="primary")
                        recommend_button = gr.Button("Recommend Styles")
                with gr.Column():
                    style_output = gr.Markdown()
                    style_recs = gr.Textbox(label="Style recommendations", lines=8)
            style_button.click(explain_style, inputs=[style_name, style_goal], outputs=style_output, queue=False)
            recommend_button.click(recommend_styles, inputs=[style_goal], outputs=style_recs, queue=False)

        with gr.Tab("Classic Markup Assistant"):
            gr.Markdown("Use this when you only want a quick Fooocus inpaint prompt and a short edit plan.")
            with gr.Row():
                with gr.Column():
                    classic_instruction = gr.Textbox(label="Plain-English edit instruction", lines=3)
                    preset = gr.Dropdown(label="Quick edit preset", choices=list(PRESETS.keys()), value=None)
                    with gr.Row():
                        understand_button = gr.Button("Understand Edit", variant="primary")
                        preset_button = gr.Button("Use Preset")
                with gr.Column():
                    prompt = gr.Textbox(label="Fooocus inpaint prompt", lines=5)
                    plan = gr.Markdown(label="Edit plan")
            understand_button.click(understand_edit, inputs=classic_instruction, outputs=[prompt, plan], queue=False)
            preset_button.click(apply_preset, inputs=preset, outputs=[classic_instruction, prompt, plan], queue=False)
    return demo


if __name__ == "__main__":
    build_app().launch(server_name="127.0.0.1", server_port=7871, inbrowser=True)
