# Fooocus Capability and Guardrails Audit

## Purpose

This audit maps what the app can do today and where guardrails should live. The goal is to keep Fooocus powerful for SDXL image generation, inpainting, outfit changes, image bundles, and personal creative workflows while making safety behavior explicit, testable, and configurable where appropriate.

## Current application shape

Fooocus is a local Gradio application. The main UI is built in `webui.py` with `gr.Blocks(...).queue()`. The app creates a main prompt area, gallery, generate controls, image-input tabs, and an advanced settings column.

Primary entry points:

- `launch.py`: environment preparation and app startup.
- `webui.py`: main Gradio UI and event wiring.
- `modules/async_worker.py`: parses UI controls into an `AsyncTask`, prepares generation, runs diffusion, and saves/logs results.
- `modules/config.py`: loads presets, config file, env values, paths, defaults, model folders, output folders, and safety checker path.
- `modules/flags.py`: enumerates feature choices such as inpaint modes, image-prompt control modes, mask models, samplers, schedulers, output formats, and performance levels.
- `extras/inpaint_mask.py`: automatic mask generation using rembg-style models and GroundingDINO + SAM.
- `extras/censor.py`: image-output safety checker/censor path.
- `modules/private_logger.py`: image save, metadata save, HTML log, and copy-to-clipboard metadata.
- `local_markup/*`: new Easy SDXL, style explanation, and photo bundle planning layer.

## User-facing capabilities

### 1. Text-to-image generation

The main prompt box feeds the generation pipeline through the `prompt` control. Styles, negative prompt, performance, model, LoRA, seed, sampler, scheduler, CFG, sharpness, and output format affect generation.

Guardrail implication:

- Prompt-level guardrails should run before `AsyncTask` is created or inside `AsyncTask.__init__` so they apply to all generation paths.
- UI-only guardrails are not enough because users can paste directly into the prompt.

### 2. Image Prompt / reference image generation

The `Image Prompt` tab supports multiple image inputs and control types:

- ImagePrompt
- FaceSwap
- PyraCanny
- CPDS

These are defined in `modules/flags.py` and parsed by `AsyncTask` into `cn_tasks`.

Guardrail implication:

- Reference-image workflows can preserve identity or composition. Guardrails should cover identity-sensitive output and user consent expectations.
- FaceSwap is identity-sensitive and deserves separate policy controls.

### 3. Upscale / variation

The `Upscale or Variation` tab allows subtle/strong variation and upscale modes.

Guardrail implication:

- Variation can drift identity or clothing. Safer UI should clearly label this as "new image inspired by reference," not exact editing.

### 4. Inpaint / outpaint

The `Inpaint or Outpaint` tab supports upload/sketch mask, inpaint prompt, outpaint directions, and several inpaint methods:

- Inpaint or Outpaint default
- Improve Detail
- Modify Content

The app parses `inpaint_input_image`, `inpaint_additional_prompt`, and `inpaint_mask_image` into `AsyncTask`.

Guardrail implication:

- Inpainting is the highest-risk edit area because users can target clothing/body/face regions.
- Guardrails should run on both main prompt and inpaint prompt.
- Mask-aware guardrails should consider target type: face, clothing, body, background, object.

### 5. Automatic mask generation

Fooocus already supports automatic mask generation using mask models:

- u2net
- u2netp
- u2net_human_seg
- u2net_cloth_seg
- silueta
- isnet-general-use
- isnet-anime
- sam

SAM uses GroundingDINO prompt, thresholds, max detections, and model type.

Guardrail implication:

- This is the right path for one-click scan/mask UX.
- Do not create a second masking stack unless needed.
- Guardrails can restrict or warn on detection prompts for sensitive body/clothing targets.

### 6. Enhance

Enhance can process images with upscale/variation and per-tab masks/prompts.

Guardrail implication:

- Enhance uses a separate set of prompts, negatives, mask models, and inpaint controls. Guardrails must cover `enhance_ctrls`, not just the main inpaint tab.

### 7. Describe / interrogate

Describe can generate prompts from uploaded images and optionally apply styles.

Guardrail implication:

- Image-to-prompt can leak sensitive descriptions into prompt fields. Add a review step or sanitize step before auto-applying.

### 8. Styles

Styles are loaded from JSON files in `sdxl_styles/` through `modules/sdxl_styles.py`. `legal_style_names` includes `Fooocus V2`, `Random Style`, and all loaded style keys.

Guardrail implication:

- Styles should be treated as prompt modifiers.
- Random Style should be discouraged for exact identity edits.
- Style explainability is useful because some styles can push outputs toward unexpected aesthetics.

### 9. Models and LoRAs

The advanced Models tab lets users select base model, refiner, LoRAs, and weights.

Guardrail implication:

- Model/LoRA selection can change safety behavior significantly.
- Guardrails should log model/LoRA selection and optionally warn on unknown or unreviewed assets.

### 10. Metadata and logging

Generated images are saved to output folders and a private `log.html`. Metadata can be embedded depending on settings. Logs include prompt, negative prompt, styles, performance, resolution, CFG, model, refiner, seed, LoRAs, sampler, scheduler, VAE, metadata scheme, and version.

Guardrail implication:

- Logs are valuable for auditability.
- They may contain sensitive prompts and reference details.
- Provide a privacy mode that disables image log and metadata for personal images.

### 11. Safety checker / censor

`extras/censor.py` uses a Stable Diffusion safety checker. `modules/async_worker.py` calls censoring when `modules.config.default_black_out_nsfw` or `black_out_nsfw` is true.

Guardrail implication:

- Output censoring is already present but should be treated as a final fallback, not the only guardrail.
- Prompt/input/request-time guardrails should come before generation.

### 12. Easy SDXL and photo bundle layer

The new layer adds:

- Easy SDXL planner
- Exact-edit prompt planning
- Style expectations
- Safe photo bundle builder
- Environment controls for swimwear bundles and strict photo guardrails

Guardrail implication:

- This layer improves UX but does not stop users from bypassing it via direct prompt input.
- Hard guardrails must be placed at the main generation boundary.

## Recommended guardrail architecture

### Layer 1: UI guidance

Purpose: prevent accidental misuse and reduce confusion.

Where:

- `webui.py`
- `local_markup_app.py`
- Easy SDXL webui patcher

Features:

- Clear mode labels: Generate, Reference, Exact Edit, Enhance, Bundle.
- Style expectations and warnings.
- One-click safe recipes for common tasks.
- Explicit notes when using reference images and identity-preserving workflows.

### Layer 2: Prompt sanitizer and classifier

Purpose: catch direct prompt edits and pasted prompts.

Where:

- Best: before `worker.AsyncTask(args=args)` in `get_task` or inside `AsyncTask.__init__` after parsing.
- Also cover Enhance controls and inpaint prompts.

Recommended module:

```txt
modules/guardrails.py
```

Recommended API:

```python
class GuardrailDecision:
    allowed: bool
    severity: str
    reason: str
    sanitized_prompt: str
    warnings: list[str]

validate_generation_request(task_or_fields) -> GuardrailDecision
sanitize_prompt(prompt: str, context: str) -> GuardrailDecision
```

Controls:

```env
KJB_GUARDRAILS_MODE=strict|balanced|off
KJB_BLOCK_EXPLICIT_ADULT=true
KJB_ALLOW_SWIMWEAR_BUNDLES=true
KJB_ALLOW_FACE_SWAP=true
KJB_PRIVACY_MODE=false
KJB_LOG_GUARDRAIL_DECISIONS=true
```

Important: Some categories should not have an `allow=true` escape hatch if the app is being distributed or used with untrusted inputs.

### Layer 3: Workflow-specific rules

Purpose: treat each mode according to risk.

Rules:

- Text-to-image: validate main prompt and negative prompt.
- Image Prompt: validate main prompt plus reference-image mode. Warn for FaceSwap.
- Inpaint: validate main prompt, inpaint prompt, detection prompt, and mask target category.
- Enhance: validate enhancement prompts and mask prompts.
- Describe: validate generated description before applying to main prompt.
- Bundle: validate bundle prompt and outfit category.

### Layer 4: Output safety fallback

Purpose: catch failures after generation.

Where:

- Existing censor path in `extras/censor.py` and `async_worker.py`.

Recommendations:

- Keep output censoring on by default in balanced/strict mode.
- Add log entry when censoring occurs.
- Do not rely only on output censoring.

### Layer 5: Audit logging and privacy

Purpose: make behavior traceable without leaking more than necessary.

Where:

- `modules/private_logger.py`
- new `modules/guardrail_logger.py`

Recommendations:

- Log guardrail decision reason, mode, and sanitized category, not full sensitive prompts if privacy mode is on.
- Privacy mode should set:
  - disable image log
  - disable metadata
  - avoid prompt retention in logs

## Capability risk matrix

| Capability | Power | Risk | Guardrail priority |
|---|---:|---:|---:|
| Text-to-image | High | Medium | High |
| Image Prompt | High | Medium/High | High |
| FaceSwap | High | High | Very high |
| Inpaint clothing/body | High | High | Very high |
| Inpaint face | High | High | Very high |
| Auto mask generation | High | Medium/High | High |
| Enhance | Medium | Medium | Medium |
| Describe | Medium | Medium | Medium |
| Style selection | Medium | Low/Medium | Medium |
| LoRA/model selection | High | Medium/High | High |
| Metadata/logging | Medium | Privacy risk | High |

## Immediate implementation plan

### Phase 1: Non-breaking audit and docs

- Add this capability audit.
- Keep safe photo bundle env docs.
- Keep Gradio pinned.
- Keep CUDA Torch pinned in local instructions.

### Phase 2: Central guardrail module

Add `modules/guardrails.py` with:

- env mode parser
- prompt validator
- context labels: main, inpaint, enhance, describe, bundle, style
- decision object
- safe prompt rewrite helpers

### Phase 3: Generation-boundary enforcement

Modify `webui.get_task` or `AsyncTask.__init__` to validate:

- prompt
- negative prompt
- inpaint additional prompt
- SAM detection prompt if available
- enhance prompts
- photo bundle prompts

### Phase 4: UI transparency

Add a small guardrail status panel:

- Current guardrail mode
- Swimwear allowed true/false
- FaceSwap allowed true/false
- Privacy mode true/false
- Output censor true/false

### Phase 5: Tests

Add tests for:

- safe headshot edit allowed
- swimwear bundle allowed when env true
- swimwear fallback when env false
- direct prompt safety decisions
- privacy mode logging behavior
- FaceSwap disabled/enabled behavior

## Recommended default env

```env
KJB_GUARDRAILS_MODE=balanced
KJB_BLOCK_EXPLICIT_ADULT=true
KJB_ALLOW_SWIMWEAR_BUNDLES=true
KJB_ALLOW_FACE_SWAP=true
KJB_PRIVACY_MODE=false
KJB_LOG_GUARDRAIL_DECISIONS=true
```

## Final recommendation

Keep Fooocus as the powerful SDXL engine. Put Easy SDXL on top for guided workflows, then add central guardrails at the task boundary so every path is covered, including direct prompt typing, inpaint prompts, image prompt, face swap, enhance, and generated bundle prompts.
