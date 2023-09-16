import threading
import torch

buffer = []
outputs = []


def worker():
    global buffer, outputs

    import time
    import shared
    import random
    import copy
    import modules.default_pipeline as pipeline
    import modules.core as core
    import modules.flags as flags
    import modules.path
    import modules.patch
    import modules.virtual_memory as virtual_memory

    from modules.sdxl_styles import apply_style, aspect_ratios, fooocus_expansion
    from modules.private_logger import log
    from modules.expansion import safe_str
    from modules.util import join_prompts, remove_empty_str, HWC3, resize_image

    try:
        async_gradio_app = shared.gradio_root
        flag = f'''App started successful. Use the app with {str(async_gradio_app.local_url)} or {str(async_gradio_app.server_name)}:{str(async_gradio_app.server_port)}'''
        if async_gradio_app.share:
            flag += f''' or {async_gradio_app.share_url}'''
        print(flag)
    except Exception as e:
        print(e)

    def progressbar(number, text):
        print(f'[Fooocus] {text}')
        outputs.append(['preview', (number, text, None)])

    @torch.no_grad()
    @torch.inference_mode()
    def handler(task):
        prompt, negative_prompt, style_selections, performance_selction, \
            aspect_ratios_selction, image_number, image_seed, sharpness, \
            base_model_name, refiner_model_name, \
            l1, w1, l2, w2, l3, w3, l4, w4, l5, w5, \
            input_image_checkbox, \
            uov_method, uov_input_image = task

        loras = [(l1, w1), (l2, w2), (l3, w3), (l4, w4), (l5, w5)]

        raw_style_selections = copy.deepcopy(style_selections)

        uov_method = uov_method.lower()

        if fooocus_expansion in style_selections:
            use_expansion = True
            style_selections.remove(fooocus_expansion)
        else:
            use_expansion = False

        use_style = len(style_selections) > 0
        modules.patch.sharpness = sharpness
        initial_latent = None
        denoising_strength = 1.0
        tiled = False

        if performance_selction == 'Speed':
            steps = 30
            switch = 20
        else:
            steps = 60
            switch = 40

        pipeline.clear_all_caches()  # save memory

        width, height = aspect_ratios[aspect_ratios_selction]

        if input_image_checkbox:
            progressbar(0, 'Image processing ...')
            if uov_method != flags.disabled and uov_input_image is not None:
                uov_input_image = HWC3(uov_input_image)
                if 'vary' in uov_method:
                    uov_input_image = resize_image(uov_input_image, width=width, height=height)
                    if 'subtle' in uov_method:
                        denoising_strength = 0.5
                    if 'strong' in uov_method:
                        denoising_strength = 0.85
                    initial_pixels = core.numpy_to_pytorch(uov_input_image)
                    progressbar(0, 'VAE encoding ...')
                    initial_latent = core.encode_vae(vae=pipeline.xl_base_patched.vae, pixels=initial_pixels)
                elif 'upscale' in uov_method:
                    H, W, C = uov_input_image.shape
                    if '1.5x' in uov_method:
                        f = 1.5
                    elif '2x' in uov_method:
                        f = 2.0
                    else:
                        f = 1.0
                    width = int(W * f)
                    height = int(H * f)
                    print(f'Upscaling image from {str((H, W))} to {str((height, width))}.')
                    uov_input_image = resize_image(uov_input_image, width=width, height=height)
                    tiled = True
                    denoising_strength = 0.57732154
                    # steps = int(steps * 0.618)
                    # switch = int(switch * 0.618)
                    initial_pixels = core.numpy_to_pytorch(uov_input_image)
                    progressbar(0, 'VAE encoding ...')
                    initial_latent = core.encode_vae(vae=pipeline.xl_base_patched.vae, pixels=initial_pixels, tiled=True)
                    B, C, H, W = initial_latent['samples'].shape
                    width = W * 8
                    height = H * 8
                    print(f'Final resolution is {str((height, width))}.')

        progressbar(1, 'Initializing ...')

        raw_prompt = prompt
        raw_negative_prompt = negative_prompt

        prompts = remove_empty_str([safe_str(p) for p in prompt.split('\n')], default='')
        negative_prompts = remove_empty_str([safe_str(p) for p in negative_prompt.split('\n')], default='')

        prompt = prompts[0]
        negative_prompt = negative_prompts[0]

        extra_positive_prompts = prompts[1:] if len(prompts) > 1 else []
        extra_negative_prompts = negative_prompts[1:] if len(negative_prompts) > 1 else []

        seed = image_seed
        max_seed = int(1024 * 1024 * 1024)
        if not isinstance(seed, int):
            seed = random.randint(1, max_seed)
        if seed < 0:
            seed = - seed
        seed = seed % max_seed

        progressbar(3, 'Loading models ...')

        pipeline.refresh_everything(
            refiner_model_name=refiner_model_name,
            base_model_name=base_model_name,
            loras=loras)

        progressbar(3, 'Processing prompts ...')

        positive_basic_workloads = []
        negative_basic_workloads = []

        if use_style:
            for s in style_selections:
                p, n = apply_style(s, positive=prompt)
                positive_basic_workloads.append(p)
                negative_basic_workloads.append(n)
        else:
            positive_basic_workloads.append(prompt)

        negative_basic_workloads.append(negative_prompt)  # Always use independent workload for negative.

        positive_basic_workloads = positive_basic_workloads + extra_positive_prompts
        negative_basic_workloads = negative_basic_workloads + extra_negative_prompts

        positive_basic_workloads = remove_empty_str(positive_basic_workloads, default=prompt)
        negative_basic_workloads = remove_empty_str(negative_basic_workloads, default=negative_prompt)

        positive_top_k = len(positive_basic_workloads)
        negative_top_k = len(negative_basic_workloads)

        tasks = [dict(
            task_seed=seed + i,
            positive=positive_basic_workloads,
            negative=negative_basic_workloads,
            expansion='',
            c=[None, None],
            uc=[None, None],
        ) for i in range(image_number)]

        if use_expansion:
            for i, t in enumerate(tasks):
                progressbar(5, f'Preparing Fooocus text #{i + 1} ...')
                expansion = pipeline.expansion(prompt, t['task_seed'])
                print(f'[Prompt Expansion] New suffix: {expansion}')
                t['expansion'] = expansion
                t['positive'] = copy.deepcopy(t['positive']) + [join_prompts(prompt, expansion)]  # Deep copy.

        for i, t in enumerate(tasks):
            progressbar(7, f'Encoding base positive #{i + 1} ...')
            t['c'][0] = pipeline.clip_encode(sd=pipeline.xl_base_patched, texts=t['positive'],
                                             pool_top_k=positive_top_k)

        for i, t in enumerate(tasks):
            progressbar(9, f'Encoding base negative #{i + 1} ...')
            t['uc'][0] = pipeline.clip_encode(sd=pipeline.xl_base_patched, texts=t['negative'],
                                              pool_top_k=negative_top_k)

        if pipeline.xl_refiner is not None:
            virtual_memory.load_from_virtual_memory(pipeline.xl_refiner.clip.cond_stage_model)

            for i, t in enumerate(tasks):
                progressbar(11, f'Encoding refiner positive #{i + 1} ...')
                t['c'][1] = pipeline.clip_encode(sd=pipeline.xl_refiner, texts=t['positive'],
                                                 pool_top_k=positive_top_k)

            for i, t in enumerate(tasks):
                progressbar(13, f'Encoding refiner negative #{i + 1} ...')
                t['uc'][1] = pipeline.clip_encode(sd=pipeline.xl_refiner, texts=t['negative'],
                                                  pool_top_k=negative_top_k)

            virtual_memory.try_move_to_virtual_memory(pipeline.xl_refiner.clip.cond_stage_model)

        results = []
        all_steps = steps * image_number

        def callback(step, x0, x, total_steps, y):
            done_steps = current_task_id * steps + step
            outputs.append(['preview', (
                int(15.0 + 85.0 * float(done_steps) / float(all_steps)),
                f'Step {step}/{total_steps} in the {current_task_id + 1}-th Sampling',
                y)])

        outputs.append(['preview', (13, 'Starting tasks ...', None)])
        for current_task_id, task in enumerate(tasks):
            imgs = pipeline.process_diffusion(
                positive_cond=task['c'],
                negative_cond=task['uc'],
                steps=steps,
                switch=switch,
                width=width,
                height=height,
                image_seed=task['task_seed'],
                callback=callback,
                latent=initial_latent,
                denoise=denoising_strength,
                tiled=tiled
            )

            for x in imgs:
                d = [
                    ('Prompt', raw_prompt),
                    ('Negative Prompt', raw_negative_prompt),
                    ('Fooocus V2 Expansion', task['expansion']),
                    ('Styles', str(raw_style_selections)),
                    ('Performance', performance_selction),
                    ('Resolution', str((width, height))),
                    ('Sharpness', sharpness),
                    ('Base Model', base_model_name),
                    ('Refiner Model', refiner_model_name),
                    ('Seed', task['task_seed'])
                ]
                for n, w in loras:
                    if n != 'None':
                        d.append((f'LoRA [{n}] weight', w))
                log(x, d, single_line_number=3)

            results += imgs

        outputs.append(['results', results])
        return

    while True:
        time.sleep(0.01)
        if len(buffer) > 0:
            task = buffer.pop(0)
            handler(task)
    pass


threading.Thread(target=worker, daemon=True).start()
