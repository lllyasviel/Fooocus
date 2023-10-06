import threading


buffer = []
outputs = []


def worker():
    global buffer, outputs

    import traceback
    import numpy as np
    import torch
    import time
    import shared
    import random
    import copy
    import modules.default_pipeline as pipeline
    import modules.core as core
    import modules.flags as flags
    import modules.path
    import modules.patch
    import comfy.model_management
    import modules.inpaint_worker as inpaint_worker

    from modules.sdxl_styles import apply_style, aspect_ratios, fooocus_expansion
    from modules.private_logger import log
    from modules.expansion import safe_str
    from modules.util import join_prompts, remove_empty_str, HWC3, resize_image, image_is_generated_in_current_ui
    from modules.upscaler import perform_upscale

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
        execution_start_time = time.perf_counter()

        prompt, negative_prompt, style_selections, performance_selection, \
            aspect_ratios_selection, image_number, image_seed, sharpness, adm_scaler_positive, adm_scaler_negative, adm_scaler_end, guidance_scale, adaptive_cfg, sampler_name, scheduler_name, \
            overwrite_step, overwrite_switch, overwrite_width, overwrite_height, overwrite_vary_strength, overwrite_upscale_strength, \
            base_model_name, refiner_model_name, \
            l1, w1, l2, w2, l3, w3, l4, w4, l5, w5, \
            input_image_checkbox, current_tab, \
            uov_method, uov_input_image, outpaint_selections, inpaint_input_image, \
            ip1_img, ip2_img, ip3_img, ip4_img, ip1_type, ip2_type, ip3_type, ip4_type = task

        outpaint_selections = [o.lower() for o in outpaint_selections]

        loras = [(l1, w1), (l2, w2), (l3, w3), (l4, w4), (l5, w5)]

        image_prompts = {k: [] for k in flags.ip_list}
        for v, k in [(ip1_img, ip1_type), (ip2_img, ip2_type), (ip3_img, ip3_type), (ip4_img, ip4_type)]:
            if v is not None:
                image_prompts[k].append(v)

        raw_style_selections = copy.deepcopy(style_selections)

        uov_method = uov_method.lower()

        if fooocus_expansion in style_selections:
            use_expansion = True
            style_selections.remove(fooocus_expansion)
        else:
            use_expansion = False

        use_style = len(style_selections) > 0

        modules.patch.adaptive_cfg = adaptive_cfg
        print(f'[Parameters] Adaptive CFG = {modules.patch.adaptive_cfg}')

        modules.patch.sharpness = sharpness
        print(f'[Parameters] Sharpness = {modules.patch.sharpness}')

        modules.patch.positive_adm_scale = adm_scaler_positive
        modules.patch.negative_adm_scale = adm_scaler_negative
        modules.patch.adm_scaler_end = adm_scaler_end
        print(f'[Parameters] ADM Scale = {modules.patch.positive_adm_scale} : {modules.patch.negative_adm_scale} : {modules.patch.adm_scaler_end}')

        cfg_scale = float(guidance_scale)
        print(f'[Parameters] CFG = {cfg_scale}')

        initial_latent = None
        denoising_strength = 1.0
        tiled = False
        inpaint_worker.current_task = None

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
        pipeline.refresh_everything(refiner_model_name=refiner_model_name, base_model_name=base_model_name, loras=loras)

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
                expansion = pipeline.final_expansion(prompt, t['task_seed'])
                print(f'[Prompt Expansion] New suffix: {expansion}')
                t['expansion'] = expansion
                t['positive'] = copy.deepcopy(t['positive']) + [join_prompts(prompt, expansion)]  # Deep copy.

        for i, t in enumerate(tasks):
            progressbar(7, f'Encoding base positive #{i + 1} ...')
            t['c'][0] = pipeline.clip_encode(texts=t['positive'], pool_top_k=positive_top_k)

        for i, t in enumerate(tasks):
            progressbar(9, f'Encoding base negative #{i + 1} ...')
            t['uc'][0] = pipeline.clip_encode(texts=t['negative'], pool_top_k=negative_top_k)

        if pipeline.final_refiner is not None:
            for i, t in enumerate(tasks):
                progressbar(11, f'Encoding refiner positive #{i + 1} ...')
                t['c'][1] = pipeline.clip_separate(t['c'][0])

            for i, t in enumerate(tasks):
                progressbar(13, f'Encoding refiner negative #{i + 1} ...')
                t['uc'][1] = pipeline.clip_separate(t['uc'][0])

        # Spatial Parameters
        if performance_selection == 'Speed':
            steps = 30
            switch = 20
        else:
            steps = 60
            switch = 40

        if overwrite_step > 0:
            steps = overwrite_step

        if overwrite_switch > 0:
            switch = overwrite_switch

        width, height = aspect_ratios[aspect_ratios_selection]

        if overwrite_width > 0:
            width = overwrite_width

        if overwrite_height > 0:
            height = overwrite_height

        if input_image_checkbox:
            progressbar(13, 'Image processing ...')
            if current_tab == 'uov' and uov_method != flags.disabled and uov_input_image is not None:
                uov_input_image = HWC3(uov_input_image)
                if 'vary' in uov_method:
                    if not image_is_generated_in_current_ui(uov_input_image, ui_width=width, ui_height=height):
                        uov_input_image = resize_image(uov_input_image, width=width, height=height)
                        print(f'Resolution corrected - users are uploading their own images.')
                    else:
                        print(f'Processing images generated by Fooocus.')
                    if 'subtle' in uov_method:
                        denoising_strength = 0.5
                    if 'strong' in uov_method:
                        denoising_strength = 0.85
                    if overwrite_vary_strength > 0:
                        denoising_strength = overwrite_vary_strength
                    initial_pixels = core.numpy_to_pytorch(uov_input_image)
                    progressbar(13, 'VAE encoding ...')
                    initial_latent = core.encode_vae(vae=pipeline.final_vae, pixels=initial_pixels)
                    B, C, H, W = initial_latent['samples'].shape
                    width = W * 8
                    height = H * 8
                    print(f'Final resolution is {str((height, width))}.')
                elif 'upscale' in uov_method:
                    H, W, C = uov_input_image.shape
                    progressbar(13, f'Upscaling image from {str((H, W))} ...')

                    uov_input_image = core.numpy_to_pytorch(uov_input_image)
                    uov_input_image = perform_upscale(uov_input_image)
                    uov_input_image = core.pytorch_to_numpy(uov_input_image)[0]
                    print(f'Image upscaled.')

                    if '1.5x' in uov_method:
                        f = 1.5
                    elif '2x' in uov_method:
                        f = 2.0
                    else:
                        f = 1.0

                    width_f = int(width * f)
                    height_f = int(height * f)

                    if image_is_generated_in_current_ui(uov_input_image, ui_width=width_f, ui_height=height_f):
                        uov_input_image = resize_image(uov_input_image, width=int(W * f), height=int(H * f))
                        print(f'Processing images generated by Fooocus.')
                    else:
                        uov_input_image = resize_image(uov_input_image, width=width_f, height=height_f)
                        print(f'Resolution corrected - users are uploading their own images.')

                    H, W, C = uov_input_image.shape
                    image_is_super_large = H * W > 2800 * 2800

                    if 'fast' in uov_method:
                        direct_return = True
                    elif image_is_super_large:
                        print('Image is too large. Directly returned the SR image. '
                              'Usually directly return SR image at 4K resolution '
                              'yields better results than SDXL diffusion.')
                        direct_return = True
                    else:
                        direct_return = False

                    if direct_return:
                        d = [('Upscale (Fast)', '2x')]
                        log(uov_input_image, d, single_line_number=1)
                        outputs.append(['results', [uov_input_image]])
                        return

                    tiled = True
                    denoising_strength = 0.382

                    if performance_selection == 'Speed':
                        steps = 18
                        switch = 12
                    else:
                        steps = 36
                        switch = 24

                    if overwrite_upscale_strength > 0:
                        denoising_strength = overwrite_upscale_strength
                    if overwrite_step > 0:
                        steps = overwrite_step
                    if overwrite_switch > 0:
                        switch = overwrite_switch

                    initial_pixels = core.numpy_to_pytorch(uov_input_image)
                    progressbar(13, 'VAE encoding ...')

                    initial_latent = core.encode_vae(vae=pipeline.final_vae, pixels=initial_pixels, tiled=True)
                    B, C, H, W = initial_latent['samples'].shape
                    width = W * 8
                    height = H * 8
                    print(f'Final resolution is {str((height, width))}.')
            if current_tab == 'inpaint' and isinstance(inpaint_input_image, dict):
                inpaint_image = inpaint_input_image['image']
                inpaint_mask = inpaint_input_image['mask'][:, :, 0]
                if isinstance(inpaint_image, np.ndarray) and isinstance(inpaint_mask, np.ndarray) \
                        and (np.any(inpaint_mask > 127) or len(outpaint_selections) > 0):
                    if len(outpaint_selections) > 0:
                        H, W, C = inpaint_image.shape
                        if 'top' in outpaint_selections:
                            inpaint_image = np.pad(inpaint_image, [[int(H * 0.3), 0], [0, 0], [0, 0]], mode='edge')
                            inpaint_mask = np.pad(inpaint_mask, [[int(H * 0.3), 0], [0, 0]], mode='constant', constant_values=255)
                        if 'bottom' in outpaint_selections:
                            inpaint_image = np.pad(inpaint_image, [[0, int(H * 0.3)], [0, 0], [0, 0]], mode='edge')
                            inpaint_mask = np.pad(inpaint_mask, [[0, int(H * 0.3)], [0, 0]], mode='constant', constant_values=255)

                        H, W, C = inpaint_image.shape
                        if 'left' in outpaint_selections:
                            inpaint_image = np.pad(inpaint_image, [[0, 0], [int(H * 0.3), 0], [0, 0]], mode='edge')
                            inpaint_mask = np.pad(inpaint_mask, [[0, 0], [int(H * 0.3), 0]], mode='constant', constant_values=255)
                        if 'right' in outpaint_selections:
                            inpaint_image = np.pad(inpaint_image, [[0, 0], [0, int(H * 0.3)], [0, 0]], mode='edge')
                            inpaint_mask = np.pad(inpaint_mask, [[0, 0], [0, int(H * 0.3)]], mode='constant', constant_values=255)

                        inpaint_image = np.ascontiguousarray(inpaint_image.copy())
                        inpaint_mask = np.ascontiguousarray(inpaint_mask.copy())

                    inpaint_worker.current_task = inpaint_worker.InpaintWorker(image=inpaint_image, mask=inpaint_mask,
                                                                               is_outpaint=len(outpaint_selections) > 0)

                    # print(f'Inpaint task: {str((height, width))}')
                    # outputs.append(['results', inpaint_worker.current_task.visualize_mask_processing()])
                    # return

                    progressbar(13, 'Downloading inpainter ...')
                    inpaint_head_model_path, inpaint_patch_model_path = modules.path.downloading_inpaint_models()

                    progressbar(13, 'Loading inpainter ...')
                    pipeline.refresh_everything(refiner_model_name=refiner_model_name, base_model_name=base_model_name,
                                                loras=loras + [(inpaint_patch_model_path, 1.0)])

                    progressbar(13, 'VAE encoding ...')
                    inpaint_pixels = core.numpy_to_pytorch(inpaint_worker.current_task.image_ready)
                    initial_latent = core.encode_vae(vae=pipeline.final_vae, pixels=inpaint_pixels)
                    inpaint_latent = initial_latent['samples']
                    B, C, H, W = inpaint_latent.shape
                    inpaint_mask = core.numpy_to_pytorch(inpaint_worker.current_task.mask_ready[None])
                    inpaint_mask = torch.nn.functional.avg_pool2d(inpaint_mask, (8, 8))
                    inpaint_mask = torch.nn.functional.interpolate(inpaint_mask, (H, W), mode='bilinear')
                    inpaint_worker.current_task.load_latent(latent=inpaint_latent, mask=inpaint_mask)

                    progressbar(13, 'VAE inpaint encoding ...')

                    inpaint_mask = (inpaint_worker.current_task.mask_ready > 0).astype(np.float32)
                    inpaint_mask = torch.tensor(inpaint_mask).float()

                    vae_dict = core.encode_vae_inpaint(
                        mask=inpaint_mask, vae=pipeline.final_vae, pixels=inpaint_pixels)

                    inpaint_latent = vae_dict['samples']
                    inpaint_mask = vae_dict['noise_mask']
                    inpaint_worker.current_task.load_inpaint_guidance(latent=inpaint_latent, mask=inpaint_mask, model_path=inpaint_head_model_path)

                    B, C, H, W = inpaint_latent.shape
                    height, width = inpaint_worker.current_task.image_raw.shape[:2]
                    print(f'Final resolution is {str((height, width))}, latent is {str((H * 8, W * 8))}.')

                    sampler_name = 'dpmpp_fooocus_2m_sde_inpaint_seamless'
            if current_tab == 'ip':
                image_prompts_ip_tasks = image_prompts[flags.ip_ip]
                image_prompts_structure_tasks = image_prompts[flags.ip_structure]
                if len(image_prompts_ip_tasks) > 0:
                    print(f'NotImplementedError: image_prompts_ip_tasks = {len(image_prompts_ip_tasks)}')
                if len(image_prompts_structure_tasks) > 0:
                    print(f'NotImplementedError: image_prompts_structure_tasks = {len(image_prompts_structure_tasks)}')

        print(f'[Parameters] Sampler = {sampler_name} - {scheduler_name}')
        print(f'[Parameters] Steps = {steps} - {switch}')

        results = []
        all_steps = steps * image_number

        def callback(step, x0, x, total_steps, y):
            done_steps = current_task_id * steps + step
            outputs.append(['preview', (
                int(15.0 + 85.0 * float(done_steps) / float(all_steps)),
                f'Step {step}/{total_steps} in the {current_task_id + 1}-th Sampling',
                y)])

        preparation_time = time.perf_counter() - execution_start_time
        print(f'Preparation time: {preparation_time:.2f} seconds')

        outputs.append(['preview', (13, 'Moving model to GPU ...', None)])
        execution_start_time = time.perf_counter()
        comfy.model_management.load_models_gpu([pipeline.final_unet])
        moving_time = time.perf_counter() - execution_start_time
        print(f'Moving model to GPU: {moving_time:.2f} seconds')

        outputs.append(['preview', (13, 'Starting tasks ...', None)])
        for current_task_id, task in enumerate(tasks):
            execution_start_time = time.perf_counter()

            try:
                imgs = pipeline.process_diffusion(
                    positive_cond=task['c'],
                    negative_cond=task['uc'],
                    steps=steps,
                    switch=switch,
                    width=width,
                    height=height,
                    image_seed=task['task_seed'],
                    callback=callback,
                    sampler_name=sampler_name,
                    scheduler_name=scheduler_name,
                    latent=initial_latent,
                    denoise=denoising_strength,
                    tiled=tiled,
                    cfg_scale=cfg_scale
                )

                if inpaint_worker.current_task is not None:
                    imgs = [inpaint_worker.current_task.post_process(x) for x in imgs]

                for x in imgs:
                    d = [
                        ('Prompt', raw_prompt),
                        ('Negative Prompt', raw_negative_prompt),
                        ('Fooocus V2 Expansion', task['expansion']),
                        ('Styles', str(raw_style_selections)),
                        ('Performance', performance_selection),
                        ('Resolution', str((width, height))),
                        ('Sharpness', sharpness),
                        ('Guidance Scale', guidance_scale),
                        ('ADM Guidance', str((adm_scaler_positive, adm_scaler_negative))),
                        ('Base Model', base_model_name),
                        ('Refiner Model', refiner_model_name),
                        ('Sampler', sampler_name),
                        ('Scheduler', scheduler_name),
                        ('Seed', task['task_seed'])
                    ]
                    for n, w in loras:
                        if n != 'None':
                            d.append((f'LoRA [{n}] weight', w))
                    log(x, d, single_line_number=3)

                results += imgs
            except comfy.model_management.InterruptProcessingException as e:
                print('User stopped')
                break

            execution_time = time.perf_counter() - execution_start_time
            print(f'Generating and saving time: {execution_time:.2f} seconds')

        outputs.append(['results', results])

        pipeline.prepare_text_encoder(async_call=True)
        return

    while True:
        time.sleep(0.01)
        if len(buffer) > 0:
            task = buffer.pop(0)
            try:
                handler(task)
            except:
                traceback.print_exc()
                outputs.append(['results', []])
    pass


threading.Thread(target=worker, daemon=True).start()
