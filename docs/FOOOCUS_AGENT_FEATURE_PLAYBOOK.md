# Fooocus Agent Feature Playbook

## Goal

AI Image Studio should know how to use every major Fooocus feature and choose the right stack for any scenario.

The agent should not just say "use Image Prompt" or "use Inpaint." It should explain:

- what feature stack to use
- why that stack fits the scenario
- what Fooocus area to use
- what settings matter
- what user action is required
- what risks or quality traps to watch for

## Feature families

### Create

- Text to Image

Use for new images from words. Avoid for identity-preserving or exact-edit work.

### Reference / identity

- Image Prompt
- FaceSwap / face reference

Use Image Prompt when an uploaded image should inspire the output. Use FaceSwap/face reference when the same adult person should stay recognizable.

### Structure / composition

- PyraCanny
- CPDS

Use PyraCanny for hard edge/pose/outline following. Use CPDS for softer composition/depth guidance.

### Exact editing

- Inpaint
- Outpaint

Use Inpaint when changing a specific area while preserving the rest. Use Outpaint when expanding the image outside its borders.

### Masking

- SAM / GroundingDINO
- U2Net-family segmentation

Use SAM for text-directed object masks like jacket, glasses, shirt, background, car, object. Use U2Net for broad person/background/clothing segmentation.

### Explore / polish

- Upscale
- Subtle Variation
- Strong Variation
- Enhance

Use Variation for alternatives, not exact edits. Use Upscale/Enhance after choosing winners.

### Understanding / style / repeatability

- Describe / Interrogate
- Styles
- LoRAs
- Seed control
- Metadata/privacy

Use Describe to draft prompts from images. Use styles carefully because styles can change identity. Use seeds to refine a good result. Use metadata/privacy awareness for personal work.

## Scenario examples

### Remove jacket from same photo

Feature stack:

```txt
Inpaint -> SAM mask -> conservative realistic styles -> seed lock -> upscale winner
```

Why:

The user wants an exact clothing-area edit, not a new image.

### Make me stand up full body

Feature stack:

```txt
FaceSwap / face reference -> CPDS or PyraCanny pose reference -> Image Prompt -> realistic styles -> seed lock
```

Why:

A headshot alone can preserve face but not body. The agent should ask for or recommend a full-body/pose reference.

### Generate photo bundle from a few photos

Feature stack:

```txt
FaceSwap / identity reference -> Image Prompt -> optional CPDS/PyraCanny pose reference -> shot-by-shot prompts -> upscale/enhance winners
```

Why:

SDXL works better when generating one specific shot at a time instead of one vague bundle prompt.

### Change background

Feature stack:

```txt
Inpaint -> SAM or U2Net mask -> background prompt -> review mask -> generate candidates
```

Why:

Background changes should preserve the subject.

### Make image cleaner/larger

Feature stack:

```txt
Upscale -> Enhance -> locked seed if refining
```

Why:

Do not use variation for a final clean-up unless creative drift is acceptable.

## Implementation

The feature knowledge lives in:

```txt
local_markup/fooocus_feature_playbook.py
```

The agent imports this playbook and adds:

- scenario summary
- feature reasoning
- feature stack recommendations
- risks and user actions

The UI remains in:

```txt
ai_studio_app.py
```

## Next step

The next major engineering step is a job adapter. The agent currently plans and guides. The adapter should eventually execute jobs directly through a local Fooocus API/queue or a RunPod backend.
