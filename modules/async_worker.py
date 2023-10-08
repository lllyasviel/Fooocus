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
    import fooocus_extras.preprocessors as preprocessors
    import modules.inpaint_worker as inpaint_worker
    import modules.advanced_parameters as advanced_parameters
    import fooocus_extras.ip_adapter as ip_adapter

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
    def handler(args):
        execution_start_time = time.perf_counter()

        args.reverse()

        prompt = args.pop()
        negative_prompt = args.pop()
        style_selections = args.pop()
        performance_selection = args.pop()
        aspect_ratios_selection = args.pop()
        image_number = args.pop()
        image_seed = args.pop()
        sharpness = args.pop()
        guidance_scale = args.pop()
        base_model_name = args.pop()
        refiner_model_name = args.pop()
        loras = [(args.pop(), args.pop()) for _ in range(5)]
        input_image_checkbox = args.pop()
        current_tab = args.pop()
        uov_method = args.pop()
        uov_input_image = args.pop()
        outpaint_selections = args.pop()
        inpaint_input_image = args.pop()

        cn_tasks = {flags.cn_ip: [], flags.cn_canny: [], flags.cn_cpds: []}
        for _ in range(4):
            cn_img = args.pop()
            cn_stop = args.pop()
            cn_weight = args.pop()
            cn_type = args.pop()
            if cn_img is not None:
                cn_tasks[cn_type].append([cn_img, cn_stop, cn_weight])

        outpaint_selections = [o.lower() for o in outpaint_selections]
        loras_raw = copy.deepcopy(loras)
        raw_style_selections = copy.deepcopy(style_selections)
        uov_method = uov_method.lower()

        if fooocus_expansion in style_selections:
            use_expansion = True
            style_selections.remove(fooocus_expansion)
        else:
            use_expansion = False

        use_style = len(style_selections) > 0

        modules.patch.adaptive_cfg = advanced_parameters.adaptive_cfg
        print(f'[Parameters] Adaptive CFG = {modules.patch.adaptive_cfg}')

        modules.patch.sharpness = sharpness
        print(f'[Parameters] Sharpness = {modules.patch.sharpness}')

        modules.patch.positive_adm_scale = advanced_parameters.adm_scaler_positive
        modules.patch.negative_adm_scale = advanced_parameters.adm_scaler_negative
        modules.patch.adm_scaler_end = advanced_parameters.adm_scaler_end
        print(f'[Parameters] ADM Scale = {modules.patch.positive_adm_scale} : {modules.patch.negative_adm_scale} : {modules.patch.adm_scaler_end}')

        cfg_scale = float(guidance_scale)
        print(f'[Parameters] CFG = {cfg_scale}')

        initial_latent = None
        denoising_strength = 1.0
        tiled = False
        inpaint_worker.current_task = None
        width, height = aspect_ratios[aspect_ratios_selection]
        skip_prompt_processing = False

        raw_prompt = prompt
        raw_negative_prompt = negative_prompt

        inpaint_image = None
        inpaint_mask = None
        inpaint_head_model_path = None
        controlnet_canny_path = None
        controlnet_cpds_path = None
        clip_vision_path, ip_negative_path, ip_adapter_path = None, None, None

        seed = image_seed
        max_seed = int(1024 * 1024 * 1024)
        if not isinstance(seed, int):
            seed = random.randint(1, max_seed)
        if seed < 0:
            seed = - seed
        seed = seed % max_seed

        if performance_selection == 'Speed':
            steps = 30
            switch = 20
        else:
            steps = 60
            switch = 40

        sampler_name = advanced_parameters.sampler_name
        scheduler_name = advanced_parameters.scheduler_name

        goals = []
        tasks = []

        if input_image_checkbox:
            progressbar(13, 'Image processing ...')
            if (current_tab == 'uov' or (current_tab == 'ip' and advanced_parameters.mixing_image_prompt_and_vary_upscale)) \
                    and uov_method != flags.disabled and uov_input_image is not None:
                uov_input_image = HWC3(uov_input_image)
                if 'vary' in uov_method:
                    goals.append('vary')
                elif 'upscale' in uov_method:
                    goals.append('upscale')
                    if 'fast' in uov_method:
                        skip_prompt_processing = True
                    else:
                        if performance_selection == 'Speed':
                            steps = 18
                            switch = 12
                        else:
                            steps = 36
                            switch = 24
                    progressbar(13, 'Downloading upscale models ...')
                    modules.path.downloading_upscale_model()
            if (current_tab == 'inpaint' or (current_tab == 'ip' and advanced_parameters.mixing_image_prompt_and_inpaint))\
                    and isinstance(inpaint_input_image, dict):
                inpaint_image = inpaint_input_image['image']
                inpaint_mask = inpaint_input_image['mask'][:, :, 0]
                inpaint_image = HWC3(inpaint_image)
                if isinstance(inpaint_image, np.ndarray) and isinstance(inpaint_mask, np.ndarray) \
                        and (np.any(inpaint_mask > 127) or len(outpaint_selections) > 0):
                    progressbar(13, 'Downloading inpainter ...')
                    inpaint_head_model_path, inpaint_patch_model_path = modules.path.downloading_inpaint_models()
                    loras += [(inpaint_patch_model_path, 1.0)]
                    goals.append('inpaint')
                    sampler_name = 'dpmpp_fooocus_2m_sde_inpaint_seamless'
            if current_tab == 'ip' or \
                    advanced_parameters.mixing_image_prompt_and_inpaint or \
                    advanced_parameters.mixing_image_prompt_and_vary_upscale:
                goals.append('cn')
                progressbar(13, 'Downloading control models ...')
                if len(cn_tasks[flags.cn_canny]) > 0:
                    controlnet_canny_path = modules.path.downloading_controlnet_canny()
                if len(cn_tasks[flags.cn_cpds]) > 0:
                    controlnet_cpds_path = modules.path.downloading_controlnet_cpds()
                if len(cn_tasks[flags.cn_ip]) > 0:
                    clip_vision_path, ip_negative_path, ip_adapter_path = modules.path.downloading_ip_adapters()
                progressbar(13, 'Loading control models ...')

        # Load or unload CNs
        pipeline.refresh_controlnets([controlnet_canny_path, controlnet_cpds_path])
        ip_adapter.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_path)

        if advanced_parameters.overwrite_step > 0:
            steps = advanced_parameters.overwrite_step

        if advanced_parameters.overwrite_switch > 0:
            switch = advanced_parameters.overwrite_switch

        if advanced_parameters.overwrite_width > 0:
            width = advanced_parameters.overwrite_width

        if advanced_parameters.overwrite_height > 0:
            height = advanced_parameters.overwrite_height

        print(f'[Parameters] Sampler = {sampler_name} - {scheduler_name}')
        print(f'[Parameters] Steps = {steps} - {switch}')

        progressbar(1, 'Initializing ...')

        if not skip_prompt_processing:

            prompts = remove_empty_str([safe_str(p) for p in prompt.split('\n')], default='')
            negative_prompts = remove_empty_str([safe_str(p) for p in negative_prompt.split('\n')], default='')

            prompt = prompts[0]
            negative_prompt = negative_prompts[0]

            extra_positive_prompts = prompts[1:] if len(prompts) > 1 else []
            extra_negative_prompts = negative_prompts[1:] if len(negative_prompts) > 1 else []

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
                c=None,
                uc=None,
            ) for i in range(image_number)]

            if use_expansion:
                for i, t in enumerate(tasks):
                    progressbar(5, f'Preparing Fooocus text #{i + 1} ...')
                    expansion = pipeline.final_expansion(prompt, t['task_seed'])
                    print(f'[Prompt Expansion] New suffix: {expansion}')
                    t['expansion'] = expansion
                    t['positive'] = copy.deepcopy(t['positive']) + [join_prompts(prompt, expansion)]  # Deep copy.

            for i, t in enumerate(tasks):
                progressbar(7, f'Encoding positive #{i + 1} ...')
                t['c'] = pipeline.clip_encode(texts=t['positive'], pool_top_k=positive_top_k)

            for i, t in enumerate(tasks):
                progressbar(10, f'Encoding negative #{i + 1} ...')
                t['uc'] = pipeline.clip_encode(texts=t['negative'], pool_top_k=negative_top_k)

        if len(goals) > 0:
            progressbar(13, 'Image processing ...')

        if 'vary' in goals:
            if not image_is_generated_in_current_ui(uov_input_image, ui_width=width, ui_height=height):
                uov_input_image = resize_image(uov_input_image, width=width, height=height)
                print(f'Resolution corrected - users are uploading their own images.')
            else:
                print(f'Processing images generated by Fooocus.')
            if 'subtle' in uov_method:
                denoising_strength = 0.5
            if 'strong' in uov_method:
                denoising_strength = 0.85
            if advanced_parameters.overwrite_vary_strength > 0:
                denoising_strength = advanced_parameters.overwrite_vary_strength
            initial_pixels = core.numpy_to_pytorch(uov_input_image)
            progressbar(13, 'VAE encoding ...')
            initial_latent = core.encode_vae(vae=pipeline.final_vae, pixels=initial_pixels)
            B, C, H, W = initial_latent['samples'].shape
            width = W * 8
            height = H * 8
            print(f'Final resolution is {str((height, width))}.')

        if 'upscale' in goals:
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

            if advanced_parameters.overwrite_upscale_strength > 0:
                denoising_strength = advanced_parameters.overwrite_upscale_strength

            initial_pixels = core.numpy_to_pytorch(uov_input_image)
            progressbar(13, 'VAE encoding ...')

            initial_latent = core.encode_vae(vae=pipeline.final_vae, pixels=initial_pixels, tiled=True)
            B, C, H, W = initial_latent['samples'].shape
            width = W * 8
            height = H * 8
            print(f'Final resolution is {str((height, width))}.')

        if 'inpaint' in goals:
            if len(outpaint_selections) > 0:
                H, W, C = inpaint_image.shape
                if 'top' in outpaint_selections:
                    inpaint_image = np.pad(inpaint_image, [[int(H * 0.3), 0], [0, 0], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[int(H * 0.3), 0], [0, 0]], mode='constant',
                                          constant_values=255)
                if 'bottom' in outpaint_selections:
                    inpaint_image = np.pad(inpaint_image, [[0, int(H * 0.3)], [0, 0], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[0, int(H * 0.3)], [0, 0]], mode='constant',
                                          constant_values=255)

                H, W, C = inpaint_image.shape
                if 'left' in outpaint_selections:
                    inpaint_image = np.pad(inpaint_image, [[0, 0], [int(H * 0.3), 0], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[0, 0], [int(H * 0.3), 0]], mode='constant',
                                          constant_values=255)
                if 'right' in outpaint_selections:
                    inpaint_image = np.pad(inpaint_image, [[0, 0], [0, int(H * 0.3)], [0, 0]], mode='edge')
                    inpaint_mask = np.pad(inpaint_mask, [[0, 0], [0, int(H * 0.3)]], mode='constant',
                                          constant_values=255)

                inpaint_image = np.ascontiguousarray(inpaint_image.copy())
                inpaint_mask = np.ascontiguousarray(inpaint_mask.copy())

            inpaint_worker.current_task = inpaint_worker.InpaintWorker(image=inpaint_image, mask=inpaint_mask,
                                                                       is_outpaint=len(outpaint_selections) > 0)

            # print(f'Inpaint task: {str((height, width))}')
            # outputs.append(['results', inpaint_worker.current_task.visualize_mask_processing()])
            # return

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
            inpaint_worker.current_task.load_inpaint_guidance(latent=inpaint_latent, mask=inpaint_mask,
                                                              model_path=inpaint_head_model_path)

            B, C, H, W = inpaint_latent.shape
            final_height, final_width = inpaint_worker.current_task.image_raw.shape[:2]
            height, width = H * 8, W * 8
            print(f'Final resolution is {str((final_height, final_width))}, latent is {str((height, width))}.')

        if 'cn' in goals:
            for task in cn_tasks[flags.cn_canny]:
                cn_img, cn_stop, cn_weight = task
                cn_img = resize_image(HWC3(cn_img), width=width, height=height)
                cn_img = preprocessors.canny_pyramid(cn_img)
                cn_img = HWC3(cn_img)
                task[0] = core.numpy_to_pytorch(cn_img)
                if advanced_parameters.debugging_cn_preprocessor:
                    outputs.append(['results', [cn_img]])
                    return
            for task in cn_tasks[flags.cn_cpds]:
                cn_img, cn_stop, cn_weight = task
                cn_img = resize_image(HWC3(cn_img), width=width, height=height)
                cn_img = preprocessors.cpds(cn_img)
                cn_img = HWC3(cn_img)
                task[0] = core.numpy_to_pytorch(cn_img)
                if advanced_parameters.debugging_cn_preprocessor:
                    outputs.append(['results', [cn_img]])
                    return
            for task in cn_tasks[flags.cn_ip]:
                cn_img, cn_stop, cn_weight = task
                cn_img = HWC3(cn_img)

                # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
                cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)

                task[0] = ip_adapter.preprocess(cn_img)
                if advanced_parameters.debugging_cn_preprocessor:
                    outputs.append(['results', [cn_img]])
                    return

            if len(cn_tasks[flags.cn_ip]) > 0:
                pipeline.final_unet = ip_adapter.patch_model(pipeline.final_unet, cn_tasks[flags.cn_ip])

        results = []
        all_steps = steps * image_number

        preparation_time = time.perf_counter() - execution_start_time
        print(f'Preparation time: {preparation_time:.2f} seconds')

        outputs.append(['preview', (13, 'Moving model to GPU ...', None)])
        execution_start_time = time.perf_counter()
        comfy.model_management.load_models_gpu([pipeline.final_unet])
        moving_time = time.perf_counter() - execution_start_time
        print(f'Moving model to GPU: {moving_time:.2f} seconds')

        outputs.append(['preview', (13, 'Starting tasks ...', None)])

        def callback(step, x0, x, total_steps, y):
            done_steps = current_task_id * steps + step
            outputs.append(['preview', (
                int(15.0 + 85.0 * float(done_steps) / float(all_steps)),
                f'Step {step}/{total_steps} in the {current_task_id + 1}-th Sampling',
                y)])

        for current_task_id, task in enumerate(tasks):
            execution_start_time = time.perf_counter()

            try:
                positive_cond, negative_cond = task['c'], task['uc']

                if 'cn' in goals:
                    for cn_flag, cn_path in [
                        (flags.cn_canny, controlnet_canny_path),
                        (flags.cn_cpds, controlnet_cpds_path)
                    ]:
                        for cn_img, cn_stop, cn_weight in cn_tasks[cn_flag]:
                            positive_cond, negative_cond = core.apply_controlnet(
                                positive_cond, negative_cond,
                                pipeline.loaded_ControlNets[cn_path], cn_img, cn_weight, 0, cn_stop)

                imgs = pipeline.process_diffusion(
                    positive_cond=positive_cond,
                    negative_cond=negative_cond,
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

                del task['c'], task['uc'], positive_cond, negative_cond  # Save memory

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
                        ('ADM Guidance', str((modules.patch.positive_adm_scale, modules.patch.negative_adm_scale))),
                        ('Base Model', base_model_name),
                        ('Refiner Model', refiner_model_name),
                        ('Sampler', sampler_name),
                        ('Scheduler', scheduler_name),
                        ('Seed', task['task_seed'])
                    ]
                    for n, w in loras_raw:
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
