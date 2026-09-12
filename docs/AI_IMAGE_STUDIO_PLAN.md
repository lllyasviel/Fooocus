# AI Image Studio Plan

## Why this exists

Patching Fooocus `webui.py` directly is brittle. The correct architecture is to keep Fooocus as the SDXL engine and build a clean AI-assisted control interface beside it.

## Files

```txt
ai_studio_app.py
local_markup/ai_studio_agent.py
RUN_AI_STUDIO.bat
scripts/remove_easy_sdxl_webui.py
docs/AI_IMAGE_STUDIO_PLAN.md
```

## Current version

The current AI Studio is a standalone Gradio app at:

```txt
http://127.0.0.1:7872
```

It does not modify Fooocus core UI. It helps the user decide which workflow to use and prepares the prompt, negative prompt, reference strategy, mask strategy, shot prompts, hand-off recipe, and next steps.

## Clean Fooocus rule

`RUN_FOOOCUS_LOCAL.bat` now removes old experimental Easy SDXL WebUI patch blocks before launching Fooocus. This keeps Fooocus clean and prevents the old accordion panel from interfering with the real engine UI.

Manual cleanup command:

```powershell
python scripts\remove_easy_sdxl_webui.py
python -m py_compile webui.py
```

## User flow

1. Start Fooocus normally with `RUN_FOOOCUS_LOCAL.bat`.
2. Start the new AI Studio with `RUN_AI_STUDIO.bat`.
3. Upload 1-3 reference/source images into AI Studio.
4. Tell the agent the goal.
5. The agent selects the workflow:
   - Text to Image
   - Inpaint / Outpaint
   - Image Prompt / Face Reference
   - PyraCanny / CPDS
   - Upscale / Variation
   - Bundle Builder
6. The user reviews the plan.
7. The user copies the **Best first shot prompt** into Fooocus first.
8. The user generates candidates one shot at a time.

## What the agent does

The agent infers:

- whether the task is an exact edit
- whether identity preservation matters
- whether a photo bundle is being requested
- whether structure/pose/composition control is needed
- whether upscale/variation is the right path
- whether masks are required

It outputs:

- selected tool
- Fooocus area
- best first shot prompt
- shot-by-shot prompt bundle
- negative prompt
- reference strategy
- mask strategy
- recommended settings
- Fooocus hand-off recipe
- next steps
- warnings / guardrails

## Top-shelf method for personal photo bundles

For a personal photo bundle, the agent should not send one vague prompt like "create a varied personal photo bundle." It now generates specific shots:

1. Resort / pool full-body shot
2. Beach / resort walking shot
3. Upper-body resort portrait
4. Professional social profile portrait

Each shot is generated one at a time. This is more reliable than asking SDXL to produce a whole bundle from one prompt.

## Recommended Fooocus method for identity bundles

Use:

```txt
Fooocus area: Image Prompt
Reference image 1: clear face/source image
Reference image 2: optional upper-body/style reference
Reference image 3: optional full-body/pose reference
Styles: Fooocus Photograph + Fooocus Enhance + Fooocus Sharp
Performance: Speed for drafts, Quality for final
Image number: 2 per shot
```

For standing/full-body outputs, a single headshot can inspire an image, but body accuracy improves with a full-body or pose reference.

## Why this is safer

The new UI does not monkey-patch Fooocus UI. It can be iterated quickly and tested separately.

The repo audit confirms that Fooocus has multiple powerful paths: text-to-image, Image Prompt, FaceSwap, PyraCanny, CPDS, Upscale/Variation, Inpaint/Outpaint, automatic mask generation, Enhance, Describe, Styles, LoRAs, metadata/logging, and output safety checks. Guardrails should live at the future job boundary, not only in the UI.

## Next engineering phase

Add a backend adapter that can submit jobs directly to:

1. Local Fooocus through a stable API/queue adapter, or
2. RunPod / remote worker through a provider adapter.

That would remove hand-off/copy-paste entirely.

## Future architecture

```txt
AI Image Studio UI
        ↓
Workflow Agent
        ↓
Job API / Queue
        ↓
Provider Adapter
        ↓
Fooocus Local OR RunPod Worker
        ↓
Gallery / History / Metadata
```

## Guardrails

Guardrails should live in the job API / task boundary, not only in the UI. The Studio Agent should advise, but backend validation must enforce.
