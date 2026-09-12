# Feature Replication Checklist

The experimental system must preserve current behavior before replacing the legacy UI/runtime path.

## Core Generation

- [ ] Text prompt generation
- [ ] Negative prompt support
- [ ] Seeded replay
- [ ] Random seed behavior
- [ ] Batch image count
- [ ] Aspect ratio presets
- [ ] Width/height overrides
- [ ] Performance modes
- [ ] Sampler selection
- [ ] Scheduler selection
- [ ] Guidance settings
- [ ] Sharpness settings
- [ ] Advanced ADM settings
- [ ] Clip skip
- [ ] FreeU

## Models

- [ ] Base checkpoint selection
- [ ] Refiner checkpoint selection
- [ ] Refiner switch behavior
- [ ] Refiner swap methods
- [ ] VAE selection
- [ ] LoRA selection
- [ ] Multiple LoRAs with weights
- [ ] Model downloads
- [ ] Model file discovery
- [ ] Model hash/cache behavior

## Prompt System

- [ ] Style selections
- [ ] Style ordering
- [ ] Prompt expansion
- [ ] Wildcard expansion
- [ ] Prompt metadata logging
- [ ] Parameter import/export

## Image Workflows

- [ ] Upscale
- [ ] Variation
- [ ] Image prompt
- [ ] Face image prompt
- [ ] Canny control workflow
- [ ] CPDS control workflow
- [ ] Inpaint
- [ ] Outpaint
- [ ] Assisted mask generation
- [ ] Enhance tabs
- [ ] Enhanced output ordering

## Output and Metadata

- [ ] Save final images
- [ ] Save intermediate images when enabled
- [ ] Disable image log when configured
- [ ] Embed or store generation metadata
- [ ] Preserve output format selection
- [ ] Build image grid/wall
- [ ] Store replayable job payload
- [ ] Store output timing

## Runtime Controls

- [ ] Start job
- [ ] Stop job
- [ ] Skip current image
- [ ] Report progress text
- [ ] Report preview image
- [ ] Report final result
- [ ] Handle failure with useful error
- [ ] Clean temporary files

## New Platform Additions

- [ ] HTTP job creation
- [ ] Database-backed job status
- [ ] Queue-backed worker execution
- [ ] Realtime progress events
- [ ] Asset storage abstraction
- [ ] Worker health reporting
- [ ] Model registry
- [ ] Project folders
- [ ] User-owned history
- [ ] API keys
- [ ] Admin model console
- [ ] Benchmarks

## Acceptance Rule

A checkbox is not complete until there is either:

1. A passing automated test,
2. A repeatable manual validation script, or
3. A written reason the feature is deferred.
