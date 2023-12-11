import typing as tp
import copy
import dataclasses
import threading
import uuid


@dataclasses.dataclass
class TaskArgs:
    prompt: str
    negative_prompt: str
    style_selections: list[str]
    performance_selection: str
    aspect_ratios_selection: str
    image_number: int
    image_seed: int
    sharpness: float
    guidance_scale: float
    custom_steps: int
    base_model_name: str
    refiner_model_name: str
    refiner_switch: float
    loras: list[tuple[str, float]]
    input_image_checkbox: bool
    current_tab: str
    uov_method: str
    uov_input_image: tp.Any
    outpaint_selections: str
    inpaint_input_image: tp.Any
    inpaint_additional_prompt: str
    ip_ctrls: list


class AsyncTask:
    def __init__(self, execution_start_time, **kwargs):
        self.uuid = str(uuid.uuid4())
        self.name = f"[{self.uuid}] {kwargs.get('prompt')}"
        self.start_time = execution_start_time
        self.args = TaskArgs(**copy.deepcopy(kwargs))

        #
        self.yields = []
        self.results = []
        self.finished = False


async_tasks: list[AsyncTask] = []
running_tasks: list[AsyncTask] = []
results = []
events: list[tuple[str, tp.Any]] = []
states = {
    'progress_bar': (0, '...'),
    'preview': None,
}


def async_tasks_list():
    global async_tasks
    return [task.name for task in async_tasks]


def worker():
    import traceback
    import math
    import numpy as np
    import torch
    import time
    import shared
    import random
    import copy
    import modules.default_pipeline as pipeline
    import modules.core as core
    import modules.flags as flags
    import modules.config
    import modules.patch
    import fcbh.model_management
    import fooocus_extras.preprocessors as preprocessors
    import modules.inpaint_worker as inpaint_worker
    import modules.constants as constants
    import modules.advanced_parameters as advanced_parameters
    import fooocus_extras.ip_adapter as ip_adapter
    import fooocus_extras.face_crop

    from modules.sdxl_styles import apply_style, apply_wildcards, fooocus_expansion
    from modules.private_logger import log
    from modules.expansion import safe_str
    from modules.util import remove_empty_str, HWC3, resize_image, \
        get_image_shape_ceil, set_image_shape_ceil, get_shape_ceil, resample_image
    from modules.upscaler import perform_upscale

    try:
        async_gradio_app = shared.gradio_root
        flag = f'''App started successful. Use the app with {str(async_gradio_app.local_url)} or {str(async_gradio_app.server_name)}:{str(async_gradio_app.server_port)}'''
        if async_gradio_app.share:
            flag += f''' or {async_gradio_app.share_url}'''
        print(flag)
    except Exception as e:
        print(e)

    def progressbar(async_task, number, text):
        print(f'[Fooocus] {text}')
        states['progress_bar'] = (number, text)
        events.append(('Preview', (number, text, None)))

    def yield_preview(async_task, number, text, img):
        states['progress_bar'] = (number, text)
        states['preview'] = img
        events.append(('Preview', (number, text, img)))

    def yield_result(async_task, imgs, do_not_show_finished_images=False):
        if not isinstance(imgs, list):
            imgs = [imgs]

        async_task.results = async_task.results + imgs

        if do_not_show_finished_images:
            return

        events.append(('Results', async_task.results))

    def yield_finish(async_task):
        global results

        states['preview'] = None
        results.extend(async_task.results)
        events.append(('Finish', results))

    def build_image_wall(async_task):
        if not advanced_parameters.generate_image_grid:
            return

        results = async_task.results

        if len(results) < 2:
            return

        for img in results:
            if not isinstance(img, np.ndarray):
                return
            if img.ndim != 3:
                return

        H, W, C = results[0].shape

        for img in results:
            Hn, Wn, Cn = img.shape
            if H != Hn:
                return
            if W != Wn:
                return
            if C != Cn:
                return

        cols = float(len(results)) ** 0.5
        cols = int(math.ceil(cols))
        rows = float(len(results)) / float(cols)
        rows = int(math.ceil(rows))

        wall = np.zeros(shape=(H * rows, W * cols, C), dtype=np.uint8)

        for y in range(rows):
            for x in range(cols):
                if y * cols + x < len(results):
                    img = results[y * cols + x]
                    wall[y * H:y * H + H, x * W:x * W + W, :] = img

        # must use deep copy otherwise gradio is super laggy. Do not use list.append() .
        async_task.results = async_task.results + [wall]
        return

    @torch.no_grad()
    @torch.inference_mode()
    def handler(async_task: AsyncTask):
        execution_start_time = time.perf_counter()
        args = async_task.args

        cn_tasks = {x: [] for x in flags.ip_list}
        for _ in range(4):
            cn_img = args.ip_ctrls.pop()
            cn_stop = args.ip_ctrls.pop()
            cn_weight = args.ip_ctrls.pop()
            cn_type = args.ip_ctrls.pop()
            if cn_img is not None:
                cn_tasks[cn_type].append([cn_img, cn_stop, cn_weight])

        outpaint_selections = [o.lower() for o in args.outpaint_selections]
        base_model_additional_loras = []
        raw_style_selections = copy.deepcopy(args.style_selections)
        uov_method = args.uov_method.lower()

        if fooocus_expansion in args.style_selections:
            use_expansion = True
            args.style_selections.remove(fooocus_expansion)
        else:
            use_expansion = False

        use_style = len(args.style_selections) > 0

        if args.base_model_name == args.refiner_model_name:
            print(f'Refiner disabled because base model and refiner are same.')
            args.refiner_model_name = 'None'

        assert args.performance_selection in ['Speed', 'Quality', 'Extreme Speed', 'Custom']

        steps_map = {
            'Speed': 30,
            'Quality': 60,
            'Extreme Speed': 8,
        }
        steps = steps_map.get(args.performance_selection, args.custom_steps)

        if args.performance_selection == 'Extreme Speed':
            print('Enter LCM mode.')
            progressbar(async_task, 1, 'Downloading LCM components ...')
            args.loras += [(modules.config.downloading_sdxl_lcm_lora(), 1.0)]

            if args.refiner_model_name != 'None':
                print(f'Refiner disabled in LCM mode.')

            args.refiner_model_name = 'None'
            sampler_name = advanced_parameters.sampler_name = 'lcm'
            scheduler_name = advanced_parameters.scheduler_name = 'lcm'
            modules.patch.sharpness = args.sharpness = 0.0
            cfg_scale = args.guidance_scale = 1.0
            modules.patch.adaptive_cfg = advanced_parameters.adaptive_cfg = 1.0
            args.refiner_switch = 1.0
            modules.patch.positive_adm_scale = advanced_parameters.adm_scaler_positive = 1.0
            modules.patch.negative_adm_scale = advanced_parameters.adm_scaler_negative = 1.0
            modules.patch.adm_scaler_end = advanced_parameters.adm_scaler_end = 0.0

        modules.patch.adaptive_cfg = advanced_parameters.adaptive_cfg
        print(f'[Parameters] Adaptive CFG = {modules.patch.adaptive_cfg}')

        modules.patch.sharpness = args.sharpness
        print(f'[Parameters] Sharpness = {modules.patch.sharpness}')

        modules.patch.positive_adm_scale = advanced_parameters.adm_scaler_positive
        modules.patch.negative_adm_scale = advanced_parameters.adm_scaler_negative
        modules.patch.adm_scaler_end = advanced_parameters.adm_scaler_end
        print(f'[Parameters] ADM Scale = '
              f'{modules.patch.positive_adm_scale} : '
              f'{modules.patch.negative_adm_scale} : '
              f'{modules.patch.adm_scaler_end}')

        cfg_scale = float(args.guidance_scale)
        print(f'[Parameters] CFG = {cfg_scale}')

        initial_latent = None
        denoising_strength = 1.0
        tiled = False

        width, height = args.aspect_ratios_selection.replace('×', ' ').split(' ')[:2]
        width, height = int(width), int(height)

        skip_prompt_processing = False
        refiner_swap_method = advanced_parameters.refiner_swap_method

        inpaint_worker.current_task = None
        inpaint_parameterized = advanced_parameters.inpaint_engine != 'None'
        inpaint_image = None
        inpaint_mask = None
        inpaint_head_model_path = None

        use_synthetic_refiner = False

        controlnet_canny_path = None
        controlnet_cpds_path = None
        clip_vision_path, ip_negative_path, ip_adapter_path, ip_adapter_face_path = None, None, None, None

        seed = int(args.image_seed)
        print(f'[Parameters] Seed = {seed}')

        sampler_name = advanced_parameters.sampler_name
        scheduler_name = advanced_parameters.scheduler_name

        goals = []
        tasks = []

        if args.input_image_checkbox:
            if (args.current_tab == 'uov' or (
                    args.current_tab == 'ip' and advanced_parameters.mixing_image_prompt_and_vary_upscale)) \
                    and uov_method != flags.disabled and args.uov_input_image is not None:
                args.uov_input_image = HWC3(args.uov_input_image)
                if 'vary' in uov_method:
                    goals.append('vary')
                elif 'upscale' in uov_method:
                    goals.append('upscale')
                    if 'fast' in uov_method:
                        skip_prompt_processing = True
                    else:
                        steps = 18

                        if args.performance_selection == 'Speed':
                            steps = 18

                        if args.performance_selection == 'Quality':
                            steps = 36

                        if args.performance_selection == 'Extreme Speed':
                            steps = 8

                    progressbar(async_task, 1, 'Downloading upscale models ...')
                    modules.config.downloading_upscale_model()
            if (args.current_tab == 'inpaint' or (
                    args.current_tab == 'ip' and advanced_parameters.mixing_image_prompt_and_inpaint)) \
                    and isinstance(args.inpaint_input_image, dict):
                inpaint_image = args.inpaint_input_image['image']
                inpaint_mask = args.inpaint_input_image['mask'][:, :, 0]
                inpaint_image = HWC3(inpaint_image)
                if isinstance(inpaint_image, np.ndarray) and isinstance(inpaint_mask, np.ndarray) \
                        and (np.any(inpaint_mask > 127) or len(outpaint_selections) > 0):
                    if inpaint_parameterized:
                        progressbar(async_task, 1, 'Downloading inpainter ...')
                        modules.config.downloading_upscale_model()
                        inpaint_head_model_path, inpaint_patch_model_path = modules.config.downloading_inpaint_models(
                            advanced_parameters.inpaint_engine)
                        base_model_additional_loras += [(inpaint_patch_model_path, 1.0)]
                        print(f'[Inpaint] Current inpaint model is {inpaint_patch_model_path}')
                        if args.refiner_model_name == 'None':
                            use_synthetic_refiner = True
                            refiner_switch = 0.5
                    else:
                        inpaint_head_model_path, inpaint_patch_model_path = None, None
                        print(f'[Inpaint] Parameterized inpaint is disabled.')
                    if args.inpaint_additional_prompt != '':
                        if args.prompt == '':
                            prompt = args.inpaint_additional_prompt
                        else:
                            prompt = args.inpaint_additional_prompt + '\n' + args.prompt
                    goals.append('inpaint')
            if args.current_tab == 'ip' or \
                    advanced_parameters.mixing_image_prompt_and_inpaint or \
                    advanced_parameters.mixing_image_prompt_and_vary_upscale:
                goals.append('cn')
                progressbar(async_task, 1, 'Downloading control models ...')
                if len(cn_tasks[flags.cn_canny]) > 0:
                    controlnet_canny_path = modules.config.downloading_controlnet_canny()
                if len(cn_tasks[flags.cn_cpds]) > 0:
                    controlnet_cpds_path = modules.config.downloading_controlnet_cpds()
                if len(cn_tasks[flags.cn_ip]) > 0:
                    clip_vision_path, ip_negative_path, ip_adapter_path = modules.config.downloading_ip_adapters('ip')
                if len(cn_tasks[flags.cn_ip_face]) > 0:
                    clip_vision_path, ip_negative_path, ip_adapter_face_path = modules.config.downloading_ip_adapters(
                        'face')
                progressbar(async_task, 1, 'Loading control models ...')

        # Load or unload CNs
        pipeline.refresh_controlnets([controlnet_canny_path, controlnet_cpds_path])
        ip_adapter.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_path)
        ip_adapter.load_ip_adapter(clip_vision_path, ip_negative_path, ip_adapter_face_path)

        switch = int(round(steps * args.refiner_switch))

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

        progressbar(async_task, 1, 'Initializing ...')

        if not skip_prompt_processing:

            prompts = remove_empty_str([safe_str(p) for p in args.prompt.splitlines()], default='')
            negative_prompts = remove_empty_str([safe_str(p) for p in args.negative_prompt.splitlines()], default='')

            prompt = prompts[0]
            negative_prompt = negative_prompts[0]

            if prompt == '':
                # disable expansion when empty since it is not meaningful and influences image prompt
                use_expansion = False

            extra_positive_prompts = prompts[1:] if len(prompts) > 1 else []
            extra_negative_prompts = negative_prompts[1:] if len(negative_prompts) > 1 else []

            progressbar(async_task, 3, 'Loading models ...')
            pipeline.refresh_everything(refiner_model_name=args.refiner_model_name, base_model_name=args.base_model_name,
                                        loras=args.loras, base_model_additional_loras=base_model_additional_loras,
                                        use_synthetic_refiner=use_synthetic_refiner)

            progressbar(async_task, 3, 'Processing prompts ...')
            tasks = []
            for i in range(args.image_number):
                task_seed = (seed + i) % (constants.MAX_SEED + 1)  # randint is inclusive, % is not
                task_rng = random.Random(task_seed)  # may bind to inpaint noise in the future

                task_prompt = apply_wildcards(prompt, task_rng)
                task_negative_prompt = apply_wildcards(negative_prompt, task_rng)
                task_extra_positive_prompts = [apply_wildcards(pmt, task_rng) for pmt in extra_positive_prompts]
                task_extra_negative_prompts = [apply_wildcards(pmt, task_rng) for pmt in extra_negative_prompts]

                positive_basic_workloads = []
                negative_basic_workloads = []

                if use_style:
                    for s in args.style_selections:
                        p, n = apply_style(s, positive=task_prompt)
                        positive_basic_workloads = positive_basic_workloads + p
                        negative_basic_workloads = negative_basic_workloads + n
                else:
                    positive_basic_workloads.append(task_prompt)

                negative_basic_workloads.append(task_negative_prompt)  # Always use independent workload for negative.

                positive_basic_workloads = positive_basic_workloads + task_extra_positive_prompts
                negative_basic_workloads = negative_basic_workloads + task_extra_negative_prompts

                positive_basic_workloads = remove_empty_str(positive_basic_workloads, default=task_prompt)
                negative_basic_workloads = remove_empty_str(negative_basic_workloads, default=task_negative_prompt)

                tasks.append(dict(
                    task_seed=task_seed,
                    task_prompt=task_prompt,
                    task_negative_prompt=task_negative_prompt,
                    positive=positive_basic_workloads,
                    negative=negative_basic_workloads,
                    expansion='',
                    c=None,
                    uc=None,
                    positive_top_k=len(positive_basic_workloads),
                    negative_top_k=len(negative_basic_workloads),
                    log_positive_prompt='; '.join([task_prompt] + task_extra_positive_prompts),
                    log_negative_prompt='; '.join([task_negative_prompt] + task_extra_negative_prompts),
                ))

            if use_expansion:
                for i, t in enumerate(tasks):
                    progressbar(async_task, 5, f'Preparing Fooocus text #{i + 1} ...')
                    expansion = pipeline.final_expansion(t['task_prompt'], t['task_seed'])
                    print(f'[Prompt Expansion] {expansion}')
                    t['expansion'] = expansion
                    t['positive'] = copy.deepcopy(t['positive']) + [expansion]  # Deep copy.

            last_task = tasks[-1]

            async_task.positive_prompt = last_task['positive'][-1] or ''
            async_task.negative_prompt = last_task['negative'][-1] or ''
            events.append(('Prompts', (async_task.positive_prompt, async_task.negative_prompt)))

            for i, t in enumerate(tasks):
                progressbar(async_task, 7, f'Encoding positive #{i + 1} ...')
                t['c'] = pipeline.clip_encode(texts=t['positive'], pool_top_k=t['positive_top_k'])

            for i, t in enumerate(tasks):
                if abs(float(cfg_scale) - 1.0) < 1e-4:
                    t['uc'] = pipeline.clone_cond(t['c'])
                else:
                    progressbar(async_task, 10, f'Encoding negative #{i + 1} ...')
                    t['uc'] = pipeline.clip_encode(texts=t['negative'], pool_top_k=t['negative_top_k'])

        if len(goals) > 0:
            progressbar(async_task, 13, 'Image processing ...')

        if 'vary' in goals:
            if 'subtle' in uov_method:
                denoising_strength = 0.5
            if 'strong' in uov_method:
                denoising_strength = 0.85
            if advanced_parameters.overwrite_vary_strength > 0:
                denoising_strength = advanced_parameters.overwrite_vary_strength

            shape_ceil = get_image_shape_ceil(args.uov_input_image)
            if shape_ceil < 1024:
                print(f'[Vary] Image is resized because it is too small.')
                shape_ceil = 1024
            elif shape_ceil > 2048:
                print(f'[Vary] Image is resized because it is too big.')
                shape_ceil = 2048

            args.uov_input_image = set_image_shape_ceil(args.uov_input_image, shape_ceil)

            initial_pixels = core.numpy_to_pytorch(args.uov_input_image)
            progressbar(async_task, 13, 'VAE encoding ...')

            candidate_vae, _ = pipeline.get_candidate_vae(
                steps=steps,
                switch=switch,
                denoise=denoising_strength,
                refiner_swap_method=refiner_swap_method
            )

            initial_latent = core.encode_vae(vae=candidate_vae, pixels=initial_pixels)
            B, C, H, W = initial_latent['samples'].shape
            width = W * 8
            height = H * 8
            print(f'Final resolution is {str((height, width))}.')

        if 'upscale' in goals:
            H, W, C = args.uov_input_image.shape
            progressbar(async_task, 13, f'Upscaling image from {str((H, W))} ...')
            args.uov_input_image = perform_upscale(args.uov_input_image)
            print(f'Image upscaled.')

            if '1.5x' in uov_method:
                f = 1.5
            elif '2x' in uov_method:
                f = 2.0
            else:
                f = 1.0

            shape_ceil = get_shape_ceil(H * f, W * f)

            if shape_ceil < 1024:
                print(f'[Upscale] Image is resized because it is too small.')
                args.uov_input_image = set_image_shape_ceil(args.uov_input_image, 1024)
                shape_ceil = 1024
            else:
                args.uov_input_image = resample_image(args.uov_input_image, width=W * f, height=H * f)

            image_is_super_large = shape_ceil > 2800

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
                log(args.uov_input_image, d, single_line_number=1)
                yield_result(async_task, args.uov_input_image, do_not_show_finished_images=True)
                return

            tiled = True
            denoising_strength = 0.382

            if advanced_parameters.overwrite_upscale_strength > 0:
                denoising_strength = advanced_parameters.overwrite_upscale_strength

            initial_pixels = core.numpy_to_pytorch(args.uov_input_image)
            progressbar(async_task, 13, 'VAE encoding ...')

            candidate_vae, _ = pipeline.get_candidate_vae(
                steps=steps,
                switch=switch,
                denoise=denoising_strength,
                refiner_swap_method=refiner_swap_method
            )

            initial_latent = core.encode_vae(
                vae=candidate_vae,
                pixels=initial_pixels, tiled=True)
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
                advanced_parameters.inpaint_strength = 1.0
                advanced_parameters.inpaint_respective_field = 1.0

            denoising_strength = advanced_parameters.inpaint_strength

            inpaint_worker.current_task = inpaint_worker.InpaintWorker(
                image=inpaint_image,
                mask=inpaint_mask,
                use_fill=denoising_strength > 0.99,
                k=advanced_parameters.inpaint_respective_field
            )

            if advanced_parameters.debugging_inpaint_preprocessor:
                yield_result(async_task, inpaint_worker.current_task.visualize_mask_processing(),
                             do_not_show_finished_images=True)
                return

            progressbar(async_task, 13, 'VAE Inpaint encoding ...')

            inpaint_pixel_fill = core.numpy_to_pytorch(inpaint_worker.current_task.interested_fill)
            inpaint_pixel_image = core.numpy_to_pytorch(inpaint_worker.current_task.interested_image)
            inpaint_pixel_mask = core.numpy_to_pytorch(inpaint_worker.current_task.interested_mask)

            candidate_vae, candidate_vae_swap = pipeline.get_candidate_vae(
                steps=steps,
                switch=switch,
                denoise=denoising_strength,
                refiner_swap_method=refiner_swap_method
            )

            latent_inpaint, latent_mask = core.encode_vae_inpaint(
                mask=inpaint_pixel_mask,
                vae=candidate_vae,
                pixels=inpaint_pixel_image)

            latent_swap = None
            if candidate_vae_swap is not None:
                progressbar(async_task, 13, 'VAE SD15 encoding ...')
                latent_swap = core.encode_vae(
                    vae=candidate_vae_swap,
                    pixels=inpaint_pixel_fill)['samples']

            progressbar(async_task, 13, 'VAE encoding ...')
            latent_fill = core.encode_vae(
                vae=candidate_vae,
                pixels=inpaint_pixel_fill)['samples']

            inpaint_worker.current_task.load_latent(
                latent_fill=latent_fill, latent_mask=latent_mask, latent_swap=latent_swap)

            if inpaint_parameterized:
                pipeline.final_unet = inpaint_worker.current_task.patch(
                    inpaint_head_model_path=inpaint_head_model_path,
                    inpaint_latent=latent_inpaint,
                    inpaint_latent_mask=latent_mask,
                    model=pipeline.final_unet
                )

            if not advanced_parameters.inpaint_disable_initial_latent:
                initial_latent = {'samples': latent_fill}

            B, C, H, W = latent_fill.shape
            height, width = H * 8, W * 8
            final_height, final_width = inpaint_worker.current_task.image.shape[:2]
            print(f'Final resolution is {str((final_height, final_width))}, latent is {str((height, width))}.')

        if 'cn' in goals:
            for task in cn_tasks[flags.cn_canny]:
                cn_img, cn_stop, cn_weight = task
                cn_img = resize_image(HWC3(cn_img), width=width, height=height)

                if not advanced_parameters.skipping_cn_preprocessor:
                    cn_img = preprocessors.canny_pyramid(cn_img)

                cn_img = HWC3(cn_img)
                task[0] = core.numpy_to_pytorch(cn_img)
                if advanced_parameters.debugging_cn_preprocessor:
                    yield_result(async_task, cn_img, do_not_show_finished_images=True)
                    return
            for task in cn_tasks[flags.cn_cpds]:
                cn_img, cn_stop, cn_weight = task
                cn_img = resize_image(HWC3(cn_img), width=width, height=height)

                if not advanced_parameters.skipping_cn_preprocessor:
                    cn_img = preprocessors.cpds(cn_img)

                cn_img = HWC3(cn_img)
                task[0] = core.numpy_to_pytorch(cn_img)
                if advanced_parameters.debugging_cn_preprocessor:
                    yield_result(async_task, cn_img, do_not_show_finished_images=True)
                    return
            for task in cn_tasks[flags.cn_ip]:
                cn_img, cn_stop, cn_weight = task
                cn_img = HWC3(cn_img)

                # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
                cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)

                task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=ip_adapter_path)
                if advanced_parameters.debugging_cn_preprocessor:
                    yield_result(async_task, cn_img, do_not_show_finished_images=True)
                    return
            for task in cn_tasks[flags.cn_ip_face]:
                cn_img, cn_stop, cn_weight = task
                cn_img = HWC3(cn_img)

                if not advanced_parameters.skipping_cn_preprocessor:
                    cn_img = fooocus_extras.face_crop.crop_image(cn_img)

                # https://github.com/tencent-ailab/IP-Adapter/blob/d580c50a291566bbf9fc7ac0f760506607297e6d/README.md?plain=1#L75
                cn_img = resize_image(cn_img, width=224, height=224, resize_mode=0)

                task[0] = ip_adapter.preprocess(cn_img, ip_adapter_path=ip_adapter_face_path)
                if advanced_parameters.debugging_cn_preprocessor:
                    yield_result(async_task, cn_img, do_not_show_finished_images=True)
                    return

            all_ip_tasks = cn_tasks[flags.cn_ip] + cn_tasks[flags.cn_ip_face]

            if len(all_ip_tasks) > 0:
                pipeline.final_unet = ip_adapter.patch_model(pipeline.final_unet, all_ip_tasks)

        if advanced_parameters.freeu_enabled:
            print(f'FreeU is enabled!')
            pipeline.final_unet = core.apply_freeu(
                pipeline.final_unet,
                advanced_parameters.freeu_b1,
                advanced_parameters.freeu_b2,
                advanced_parameters.freeu_s1,
                advanced_parameters.freeu_s2
            )

        all_steps = steps * args.image_number

        print(f'[Parameters] Denoising Strength = {denoising_strength}')

        if isinstance(initial_latent, dict) and 'samples' in initial_latent:
            log_shape = initial_latent['samples'].shape
        else:
            log_shape = f'Image Space {(height, width)}'

        print(f'[Parameters] Initial Latent shape: {log_shape}')

        preparation_time = time.perf_counter() - execution_start_time
        print(f'Preparation time: {preparation_time:.2f} seconds')

        final_sampler_name = sampler_name
        final_scheduler_name = scheduler_name

        if scheduler_name == 'lcm':
            final_scheduler_name = 'sgm_uniform'
            if pipeline.final_unet is not None:
                pipeline.final_unet = core.opModelSamplingDiscrete.patch(
                    pipeline.final_unet,
                    sampling='lcm',
                    zsnr=False)[0]
            if pipeline.final_refiner_unet is not None:
                pipeline.final_refiner_unet = core.opModelSamplingDiscrete.patch(
                    pipeline.final_refiner_unet,
                    sampling='lcm',
                    zsnr=False)[0]
            print('Using lcm scheduler.')

        progressbar(async_task, 13, 'Moving model to GPU ...')

        def callback(step, x0, x, total_steps, y):
            done_steps = current_task_id * steps + step
            yield_preview(
                async_task,
                int(15.0 + 85.0 * float(done_steps) / float(all_steps)),
                f'Step {step}/{total_steps} in the {current_task_id + 1}-th Sampling',
                y,
            )

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
                    sampler_name=final_sampler_name,
                    scheduler_name=final_scheduler_name,
                    latent=initial_latent,
                    denoise=denoising_strength,
                    tiled=tiled,
                    cfg_scale=cfg_scale,
                    refiner_swap_method=refiner_swap_method
                )

                del task['c'], task['uc'], positive_cond, negative_cond  # Save memory

                if inpaint_worker.current_task is not None:
                    imgs = [inpaint_worker.current_task.post_process(x) for x in imgs]

                for x in imgs:
                    d = [
                        ('Prompt', task['log_positive_prompt']),
                        ('Negative Prompt', task['log_negative_prompt']),
                        ('Fooocus V2 Expansion', task['expansion']),
                        ('Styles', str(raw_style_selections)),
                        ('Performance', args.performance_selection),
                        ('Resolution', str((width, height))),
                        ('Sharpness', args.sharpness),
                        ('Guidance Scale', args.guidance_scale),
                        ('ADM Guidance', str((
                            modules.patch.positive_adm_scale,
                            modules.patch.negative_adm_scale,
                            modules.patch.adm_scaler_end))),
                        ('Base Model', args.base_model_name),
                        ('Refiner Model', args.refiner_model_name),
                        ('Refiner Switch', args.refiner_switch),
                        ('Sampler', sampler_name),
                        ('Scheduler', scheduler_name),
                        ('Seed', task['task_seed'])
                    ]
                    for n, w in args.loras:
                        if n != 'None':
                            d.append((f'LoRA [{n}] weight', w))
                    log(x, d, single_line_number=3)

                yield_result(async_task, imgs, do_not_show_finished_images=len(tasks) == 1)
            except fcbh.model_management.InterruptProcessingException as e:
                if shared.last_stop == 'skip':
                    print('User skipped')
                    continue
                else:
                    print('User stopped')
                    break

            execution_time = time.perf_counter() - execution_start_time
            print(f'Generating and saving time: {execution_time:.2f} seconds')

        return

    while True:
        time.sleep(0.01)
        if len(async_tasks) > 0:
            task = async_tasks.pop(0)
            running_tasks.append(task)
            try:
                handler(task)
            except:
                traceback.print_exc()
            finally:
                build_image_wall(task)
                yield_finish(task)
                running_tasks.remove(task)
                pipeline.prepare_text_encoder(async_call=True)
    pass


threading.Thread(target=worker, daemon=True).start()
