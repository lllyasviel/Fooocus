"""
Fooocus CLI Worker

This module handles the programmatic generation of images using the Fooocus pipeline.
It is invoked by launch.py when CLI arguments are detected.
"""
import os
import random
import torch

import modules.default_pipeline as pipeline
import modules.async_worker as worker
import modules.flags as flags
import modules.config
from modules.util import get_enabled_loras
from modules.sdxl_styles import apply_style
from modules.private_logger import log

def run_cli(args):
    prompt = args.prompt
    negative_prompt = args.negative_prompt
    seed = args.seed
    
    performance = args.performance
    aspect_ratio = args.aspect_ratio
    
    # 1. Configuration & Models
    # Styles
    if args.style:
        styles = args.style
    else:
        styles = ["Fooocus V2", "Fooocus Photograph", "Fooocus Enhance"]
        
    # Models
    base_model = args.base_model if args.base_model else modules.config.default_base_model_name
    refiner_model = args.refiner_model if args.refiner_model else modules.config.default_refiner_model_name
    
    # LoRAs (convert 'name:weight' to (enabled, name, weight))
    if args.lora:
        loras = []
        for l in args.lora:
            try:
                if ':' in l:
                    parts = l.rsplit(':', 1)
                    name = parts[0]
                    weight = float(parts[1])
                else:
                    name = l
                    weight = 1.0
                loras.append((True, name, weight))
            except Exception as e:
                print(f"[CLI] Warning: Could not parse LoRA '{l}': {e}")
    else:
        loras = modules.config.default_loras

    print(f"\n[CLI] Settings:")
    print(f"  Prompt: {prompt}")
    print(f"  Negative: {negative_prompt}")
    print(f"  Styles: {styles}")
    print(f"  Models: Base='{base_model}', Refiner='{refiner_model}'")
    print(f"  LoRAs: {len(loras)} enabled")
    print(f"  Performance: {performance}")
    
    # Refresh Pipeline
    try:
        pipeline.refresh_everything(
            refiner_model_name=refiner_model,
            base_model_name=base_model,
            loras=get_enabled_loras(loras),
            vae_name=modules.config.default_vae 
        )
    except Exception as e:
        print(f"[CLI] Error loading models: {e}")
        return

    # 2. Process Prompts
    positive_basic_workloads = [prompt]
    negative_basic_workloads = [negative_prompt]
    
    for s in styles:
        if s == "Fooocus V2":
            continue
        try:
            p, n, _ = apply_style(s, positive=prompt)
            positive_basic_workloads += p
            negative_basic_workloads += n
        except Exception as e:
            print(f"[CLI] Warning: Failed to apply style '{s}': {e}")
    
    # 3. Expansion
    if "Fooocus V2" in styles:
        seed_for_expansion = seed if seed != -1 else random.randint(0, 2**32 - 1)
        expansion = pipeline.final_expansion(prompt, seed_for_expansion)
        print(f"[CLI] Expansion: {expansion}")
        positive_prompts = positive_basic_workloads + [expansion]
    else:
        expansion = ""
        positive_prompts = positive_basic_workloads
    
    # 4. Encoding
    print("[CLI] Encoding prompts...")
    c = pipeline.clip_encode(texts=positive_prompts, pool_top_k=len(positive_basic_workloads))
    uc = pipeline.clip_encode(texts=negative_basic_workloads, pool_top_k=len(negative_basic_workloads))
    
    # 5. Parameters
    perf_obj = flags.Performance(performance)
    steps = perf_obj.steps()
    if args.steps > 0:
        steps = args.steps
    
    switch = int(steps * args.refiner_switch)
    
    try:
        if '\u00d7' in aspect_ratio:
            w_h = aspect_ratio.split('\u00d7')
        elif 'x' in aspect_ratio:
            w_h = aspect_ratio.split('x')
        else:
            w_h = aspect_ratio.split(',') # fallback? no, probably just error out to default
            
        width = int(w_h[0])
        height = int(w_h[1])
    except:
        width, height = 1152, 896
        
    print(f"[CLI] Resolution: {width}x{height}, Steps: {steps}, Sampler: {args.sampler}")

    # Initialize PatchSettings
    from modules.patch import PatchSettings, patch_settings
    pid = os.getpid()
    patch_settings[pid] = PatchSettings(
        sharpness=args.sharpness,
        adm_scaler_end=0.3,
        positive_adm_scale=1.5,
        negative_adm_scale=0.8,
        controlnet_softness=0.25,
        adaptive_cfg=7.0
    )

    # 6. Generation Loop
    def callback(step, x0, x, total_steps, y):
        print(f"  Step {step}/{total_steps}", end="\r")

    total_images = args.image_number
    print(f"[CLI] Generating {total_images} image(s)...")

    for i in range(total_images):
        current_seed = seed
        if current_seed == -1:
            current_seed = random.randint(0, 2**32 - 1)
        elif i > 0:
            current_seed += 1 # Auto-increment seed
            
        print(f"\n[CLI] Image {i+1}/{total_images} (Seed: {current_seed})")

        with torch.no_grad():
            imgs = pipeline.process_diffusion(
                positive_cond=c,
                negative_cond=uc,
                steps=steps,
                switch=switch,
                width=width,
                height=height,
                image_seed=current_seed,
                callback=callback,
                sampler_name=args.sampler,
                scheduler_name=args.scheduler,
                latent=None,
                denoise=1.0,
                cfg_scale=args.cfg_scale,
                refiner_swap_method="joint"
            )
        
        # 7. Save
        log_task = {
            'log_positive_prompt': prompt,
            'log_negative_prompt': negative_prompt,
            'expansion': expansion,
            'styles': styles,
            'task_seed': current_seed,
            'base_model_name': base_model
        }
        
        for img_idx, img in enumerate(imgs):
            d = [('Prompt', 'prompt', prompt)]
            path = log(img, d, None, "png", log_task)
            print(f"[CLI] Saved to: {path}")
            
            if args.output:
                if total_images > 1:
                    # Append index to filename if multiple images
                    root, ext = os.path.splitext(args.output)
                    final_output = f"{root}_{i+1}{ext}"
                else:
                    final_output = args.output
                
                import shutil
                shutil.copy(path, final_output)
                print(f"[CLI] Copied to: {final_output}")
    
    print("\n[CLI] Generation complete!")
