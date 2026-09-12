# Easy Image Tools UX

This document explains the guided tools added on top of Fooocus so users do not have to understand every SDXL/Fooocus concept before getting useful results.

## New panel

The WebUI patcher adds:

```txt
Easy SDXL Exact Edit + Guided Image Tools
```

Button:

```txt
Set Up Tool
```

The panel writes directly into existing Fooocus controls. It does not replace the engine.

## Guided tools

### Image Prompt reference

Use when the uploaded image should inspire a new image.

Sets:

- Input Image enabled
- Current workflow target: Image Prompt
- First image prompt type: ImagePrompt
- Recommended prompt/negative prompt

### FaceSwap identity reference

Use when identity preservation matters.

Sets:

- Input Image enabled
- Current workflow target: Image Prompt
- First image prompt type: FaceSwap
- Face reference guidance

### PyraCanny structure control

Use when the output should follow edges/contours from the uploaded reference.

Sets:

- Input Image enabled
- First image prompt type: PyraCanny
- Edge/structure guidance

### CPDS composition control

Use when the output should follow broad composition/depth without copying every edge.

Sets:

- Input Image enabled
- First image prompt type: CPDS
- Composition guidance

### Upscale image

Use when the image is already good and needs to be larger/cleaner.

Sets:

- Input Image enabled
- Current workflow target: Upscale or Variation
- Method: Upscale (2x)

### Subtle variation

Use when the output should stay close to the original image.

Sets:

- Input Image enabled
- Method: Vary (Subtle)

### Strong variation

Use when the output should be noticeably different but inspired by the original.

Sets:

- Input Image enabled
- Method: Vary (Strong)

### Auto mask clothing

Use for clothing/object-area edits.

Sets:

- Input Image enabled
- Inpaint mode: Modify Content
- Advanced Masking enabled
- Mask generation model: SAM
- Detection prompt: jacket, shirt, clothing, outfit

### Auto mask background

Use for background replacement.

Sets:

- Input Image enabled
- Inpaint mode: Modify Content
- Advanced Masking enabled
- Mask generation model: SAM
- Detection prompt: background

### U2Net person/background mask

Use for broad segmentation rather than text-directed SAM detection.

Sets:

- Input Image enabled
- Inpaint mode: Modify Content
- Advanced Masking enabled
- Mask generation model: u2net_human_seg

## What still requires user action

The panel sets the right controls, but the user still needs to upload the image into the correct visible input area and click Generate or Generate mask from image where appropriate.

The next UX pass should add JavaScript tab-switching so the visible tab changes automatically after Set Up Tool.

## Why this is safe

The guided tools reuse Fooocus controls already present in `webui.py`:

- `input_image_checkbox`
- `current_tab`
- `uov_method`
- `inpaint_mode`
- `inpaint_advanced_masking_checkbox`
- `inpaint_mask_model`
- `inpaint_mask_dino_prompt_text`
- first Image Prompt type/stop/weight controls

This avoids a second generation path and reduces break risk.
