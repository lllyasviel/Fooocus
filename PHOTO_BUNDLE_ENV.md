# Photo Bundle Environment Controls

The photo bundle workflow supports safe personal image sets such as headshots, standing portraits, resort photos, fitness looks, business photos, casual photos, and tasteful swimwear.

It does not include an environment switch that enables explicit nude or undressing generation.

## Supported flags

```env
KJB_ALLOW_SWIMWEAR_BUNDLES=true
KJB_STRICT_PHOTO_GUARDRAILS=true
```

## KJB_ALLOW_SWIMWEAR_BUNDLES

When true, the bundle builder can create tasteful swimming/resort/swimwear prompts.

```env
KJB_ALLOW_SWIMWEAR_BUNDLES=true
```

When false, swimwear requests fall back to resort outfit prompts.

```env
KJB_ALLOW_SWIMWEAR_BUNDLES=false
```

## KJB_STRICT_PHOTO_GUARDRAILS

When true, the bundle builder stays limited to the built-in safe outfit categories:

- swimwear
- resort
- fitness
- business
- casual

```env
KJB_STRICT_PHOTO_GUARDRAILS=true
```

When false, the planner allows more wording flexibility in user goals, but the workflow still uses safe non-explicit outfit directions and still adds a negative prompt against unsafe adult content.

```env
KJB_STRICT_PHOTO_GUARDRAILS=false
```

## Recommended reference-photo set

Use 2-5 clear photos:

1. Face close-up
2. Upper-body photo
3. Full-body photo if available
4. Natural expression photo
5. Optional style/outfit reference

A single headshot can inspire a standing or full-body image, but full-body accuracy is better when a full-body reference is available.
