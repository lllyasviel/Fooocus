from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEBUI = ROOT / "webui.py"

HELPER_START = "# EASY_SDXL_WEBUI_HELPER_START"
HELPER_END = "# EASY_SDXL_WEBUI_HELPER_END"
PANEL_START = "# EASY_SDXL_WEBUI_PANEL_START"
PANEL_END = "# EASY_SDXL_WEBUI_PANEL_END"
BINDING_START = "# EASY_SDXL_WEBUI_BINDING_START"
BINDING_END = "# EASY_SDXL_WEBUI_BINDING_END"

HELPER = r'''
# EASY_SDXL_WEBUI_HELPER_START
def easy_sdxl_webui_plan(instruction, goal):
    from local_markup.easy_sdxl import build_easy_sdxl_outputs
    from local_markup.photo_bundle import build_bundle_plan
    from local_markup.style_explainer import style_recommendations_for_goal
    import modules.flags as _flags

    text = (instruction or "").strip()
    if not text:
        text = "Remove only the jacket. Keep the same person, same face, same pose, and same background. Replace the jacket area with a clean professional shirt."

    # Default exact-edit values.
    current_tab_value = "inpaint"
    uov_value = _flags.disabled
    inpaint_mode_value = _flags.inpaint_option_modify
    advanced_mask_value = True
    mask_model_value = "sam"
    first_ip_type = _flags.default_ip
    first_ip_stop = _flags.default_parameters[_flags.default_ip][0]
    first_ip_weight = _flags.default_parameters[_flags.default_ip][1]

    if goal == "Swimming / resort bundle":
        plan = build_bundle_plan("Swimming / Resort", "swimwear", text or "make me standing near a pool ready for swimming")
        guide = plan.as_text() + "\n\nUse Image Prompt or FaceSwap with your reference photos for identity consistency."
        return plan.identity_prompt, "", "", plan.negative_prompt, guide, True, current_tab_value, uov_value, inpaint_mode_value, advanced_mask_value, mask_model_value, first_ip_type, first_ip_stop, first_ip_weight

    if goal == "Stand up full-body":
        plan = build_bundle_plan("Lifestyle Starter", "casual", text or "standing full-body realistic portrait")
        guide = plan.as_text() + "\n\nBest workflow: Image Prompt with 2-5 references. A headshot alone can inspire a standing image, but full-body reference improves accuracy."
        current_tab_value = "ip"
        first_ip_type = _flags.cn_ip_face
        first_ip_stop = _flags.default_parameters[_flags.cn_ip_face][0]
        first_ip_weight = _flags.default_parameters[_flags.cn_ip_face][1]
        return plan.identity_prompt, "", "", plan.negative_prompt, guide, True, current_tab_value, uov_value, inpaint_mode_value, advanced_mask_value, mask_model_value, first_ip_type, first_ip_stop, first_ip_weight

    guided_map = {
        "Image Prompt reference": {
            "tab": "ip",
            "ip_type": _flags.cn_ip,
            "prompt": text or "create a new high-quality image inspired by the uploaded reference image",
            "guide": "Upload the reference image in Image Prompt. This creates a new image inspired by the reference; it is not an exact edit.",
        },
        "FaceSwap identity reference": {
            "tab": "ip",
            "ip_type": _flags.cn_ip_face,
            "prompt": text or "same adult person identity from the reference photo, realistic portrait, natural expression",
            "guide": "Upload a clear face reference in Image Prompt. Type is set to FaceSwap for stronger identity preservation.",
        },
        "PyraCanny structure control": {
            "tab": "ip",
            "ip_type": _flags.cn_canny,
            "prompt": text or "use the uploaded image structure and edges to guide a clean realistic result",
            "guide": "Upload a structure reference. Type is set to PyraCanny, which follows edges/contours strongly.",
        },
        "CPDS composition control": {
            "tab": "ip",
            "ip_type": _flags.cn_cpds,
            "prompt": text or "use the uploaded image composition and depth to guide a realistic result",
            "guide": "Upload a composition reference. Type is set to CPDS, which follows composition/depth without copying every edge.",
        },
        "Upscale image": {
            "tab": "uov",
            "ip_type": _flags.default_ip,
            "prompt": text or "improve the uploaded image quality, sharpness, and professional finish",
            "guide": "Upload image in Upscale or Variation. Method is set to Upscale (2x). Use this when the image is good and you want it larger/cleaner.",
            "uov": _flags.upscale_2,
        },
        "Subtle variation": {
            "tab": "uov",
            "ip_type": _flags.default_ip,
            "prompt": text or "make a subtle realistic variation while preserving the original look",
            "guide": "Upload image in Upscale or Variation. Method is set to Vary (Subtle). Use this for small changes.",
            "uov": _flags.subtle_variation,
        },
        "Strong variation": {
            "tab": "uov",
            "ip_type": _flags.default_ip,
            "prompt": text or "make a stronger creative variation inspired by the uploaded image",
            "guide": "Upload image in Upscale or Variation. Method is set to Vary (Strong). Use this when you want a noticeably different version.",
            "uov": _flags.strong_variation,
        },
        "Auto mask clothing": {
            "tab": "inpaint",
            "ip_type": _flags.default_ip,
            "prompt": text or "same person, preserve face and background, edit only the masked clothing area, seamless realistic clothing edit",
            "guide": "Use Inpaint. Upload image, enable Advanced Masking, model sam, detection prompt clothing/jacket/shirt, click Generate mask from image, review mask, then Generate.",
            "detect": "jacket, shirt, clothing, outfit",
        },
        "Auto mask background": {
            "tab": "inpaint",
            "ip_type": _flags.default_ip,
            "prompt": text or "same person, preserve face and body, replace only the background with a clean realistic background",
            "guide": "Use Inpaint. Upload image, enable Advanced Masking, model sam, detection prompt background, generate mask, review, then Generate.",
            "detect": "background",
        },
        "U2Net person/background mask": {
            "tab": "inpaint",
            "ip_type": _flags.default_ip,
            "prompt": text or "clean realistic edit using automatic segmentation mask",
            "guide": "Use Inpaint advanced masking with a U2Net-family model when you want broad person/background segmentation instead of text-directed SAM detection.",
            "mask_model": "u2net_human_seg",
            "detect": "",
        },
    }

    if goal in guided_map:
        item = guided_map[goal]
        current_tab_value = item.get("tab", "ip")
        uov_value = item.get("uov", _flags.disabled)
        first_ip_type = item.get("ip_type", _flags.default_ip)
        first_ip_stop = _flags.default_parameters.get(first_ip_type, _flags.default_parameters[_flags.default_ip])[0]
        first_ip_weight = _flags.default_parameters.get(first_ip_type, _flags.default_parameters[_flags.default_ip])[1]
        mask_model_value = item.get("mask_model", "sam")
        detection = item.get("detect", "")
        positive = item["prompt"]
        negative = "changed identity, distorted face, bad hands, low quality, blurry, unrealistic anatomy, artifacts"
        guide = item["guide"] + "\n\nTool selected: " + goal
        return positive, positive if current_tab_value == "inpaint" else "", detection, negative, guide, True, current_tab_value, uov_value, inpaint_mode_value, advanced_mask_value, mask_model_value, first_ip_type, first_ip_stop, first_ip_weight

    area = "Shirt / Clothes"
    if goal == "Remove glasses":
        area = "Glasses"
    elif goal == "Change background":
        area = "Background"
    elif goal == "Improve face/headshot":
        area = "Face"
    elif goal == "Swimwear outfit":
        area = "Shirt / Clothes"
        text = text or "replace clothing with tasteful swim trunks suitable for swimming, non-explicit, natural fit"

    positive, inpaint, detection, negative, settings, checklist = build_easy_sdxl_outputs("Edit This Image", area, text)

    if goal == "Remove jacket":
        detection = "jacket, suit jacket, blazer, coat"
        inpaint = "same person, preserve facial identity, age, skin tone, expression, camera angle, and lighting, remove only the jacket, replace the jacket area with a clean professional shirt, professional realistic headshot, seamless edit, natural fabric texture, no visible retouching artifacts"
        positive = inpaint
    elif goal == "Swimwear outfit":
        detection = "shirt, jacket, clothing, outfit"
        inpaint = "same adult person, preserve facial identity, age, skin tone, face shape, expression, camera angle, and lighting, wearing tasteful swim trunks suitable for swimming, beach or pool ready, non-explicit, natural fit, realistic body proportions, professional realistic lifestyle photo, seamless clothing edit"
        positive = inpaint
        negative = negative + ", unsafe adult content, see-through clothing, sexualized pose"

    guide = (
        "Easy SDXL plan ready. Use Inpaint or Outpaint for exact edits. "
        "Upload the image, enable Advanced Masking Features, use SAM, click Generate mask from image, review the mask, then Generate.\n\n"
        + checklist
        + "\n\nStyle advice:\n"
        + style_recommendations_for_goal(text)
    )
    return positive, inpaint, detection, negative, guide, True, current_tab_value, uov_value, inpaint_mode_value, advanced_mask_value, mask_model_value, first_ip_type, first_ip_stop, first_ip_weight
# EASY_SDXL_WEBUI_HELPER_END
'''

PANEL = r'''
                    # EASY_SDXL_WEBUI_PANEL_START
                    with gr.Accordion("Easy SDXL Exact Edit + Guided Image Tools", open=True):
                        gr.Markdown("One place for exact edits, reference images, FaceSwap, structure controls, upscale/variation, auto masks, and safe personal photo bundles. This writes directly into Fooocus controls.")
                        with gr.Row():
                            easy_sdxl_goal = gr.Radio(
                                label="One-click goal / tool",
                                choices=[
                                    "Remove jacket", "Remove glasses", "Change clothing", "Swimwear outfit", "Stand up full-body", "Swimming / resort bundle",
                                    "Image Prompt reference", "FaceSwap identity reference", "PyraCanny structure control", "CPDS composition control",
                                    "Upscale image", "Subtle variation", "Strong variation",
                                    "Auto mask clothing", "Auto mask background", "U2Net person/background mask",
                                    "Change background", "Improve face/headshot"
                                ],
                                value="Remove jacket",
                            )
                            easy_sdxl_instruction = gr.Textbox(
                                label="Plain-English edit, reference goal, or bundle goal",
                                value="Remove only the jacket. Keep the same person, same face, same pose, same background, and replace the jacket area with a clean professional shirt.",
                                lines=3,
                            )
                        easy_sdxl_apply = gr.Button(value="Set Up Tool", variant="primary")
                        easy_sdxl_guide = gr.Textbox(label="What this tool set up and what to do next", lines=12, interactive=False)
                    # EASY_SDXL_WEBUI_PANEL_END
'''

BINDING = r'''
        # EASY_SDXL_WEBUI_BINDING_START
        easy_sdxl_apply.click(
            easy_sdxl_webui_plan,
            inputs=[easy_sdxl_instruction, easy_sdxl_goal],
            outputs=[
                prompt,
                inpaint_additional_prompt,
                inpaint_mask_dino_prompt_text,
                negative_prompt,
                easy_sdxl_guide,
                input_image_checkbox,
                current_tab,
                uov_method,
                inpaint_mode,
                inpaint_advanced_masking_checkbox,
                inpaint_mask_model,
                ip_types[0],
                ip_stops[0],
                ip_weights[0],
            ],
            queue=False,
            show_progress=False,
        )
        # EASY_SDXL_WEBUI_BINDING_END
'''


def replace_marked_block(text: str, start: str, end: str, replacement: str) -> tuple[str, bool]:
    start_index = text.find(start)
    if start_index == -1:
        return text, False
    block_start = text.rfind("\n", 0, start_index)
    if block_start == -1:
        block_start = start_index
    block_end = text.find(end, start_index)
    if block_end == -1:
        return text, False
    block_end = text.find("\n", block_end)
    if block_end == -1:
        block_end = len(text)
    new_text = text[:block_start] + "\n" + replacement + text[block_end:]
    return new_text, new_text != text


def main():
    text = WEBUI.read_text(encoding="utf-8")
    changed = False

    text, did_replace = replace_marked_block(text, HELPER_START, HELPER_END, HELPER)
    changed = changed or did_replace
    text, did_replace = replace_marked_block(text, PANEL_START, PANEL_END, PANEL)
    changed = changed or did_replace
    text, did_replace = replace_marked_block(text, BINDING_START, BINDING_END, BINDING)
    changed = changed or did_replace

    if HELPER_START not in text:
        text = text.replace("reload_javascript()", HELPER + "\nreload_javascript()", 1)
        changed = True

    if PANEL_START not in text:
        anchor = "                    if isinstance(default_prompt, str) and default_prompt != '':\n                        shared.gradio_root.load(lambda: default_prompt, outputs=prompt)\n"
        text = text.replace(anchor, anchor + PANEL, 1)
        changed = True

    if BINDING_START not in text:
        anchor = "        inpaint_mode.change(inpaint_mode_change, inputs=[inpaint_mode, inpaint_engine_state], outputs=[\n            inpaint_additional_prompt, outpaint_selections, example_inpaint_prompts,\n            inpaint_disable_initial_latent, inpaint_engine,\n            inpaint_strength, inpaint_respective_field\n        ], show_progress=False, queue=False)\n"
        text = text.replace(anchor, anchor + BINDING, 1)
        changed = True

    if changed:
        WEBUI.write_text(text, encoding="utf-8")
        print("Easy SDXL WebUI integration applied/repaired.")
    else:
        print("Easy SDXL WebUI integration already healthy.")


if __name__ == "__main__":
    main()
