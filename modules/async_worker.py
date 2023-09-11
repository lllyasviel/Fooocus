import threading


buffer = []
outputs = []


def worker():
    global buffer, outputs

    import time
    import shared
    import random
    import modules.default_pipeline as pipeline
    import modules.path
    import modules.patch

    from modules.sdxl_styles import apply_style_negative, apply_style_positive, aspect_ratios
    from modules.private_logger import log
    from modules.expansion import safe_str

    try:
        async_gradio_app = shared.gradio_root
        flag = f'''App started successful. Use the app with {str(async_gradio_app.local_url)} or {str(async_gradio_app.server_name)}:{str(async_gradio_app.server_port)}'''
        if async_gradio_app.share:
            flag += f''' or {async_gradio_app.share_url}'''
        print(flag)
    except Exception as e:
        print(e)

    def handler(task):
        prompt, negative_prompt, style_selction, performance_selction, \
        aspect_ratios_selction, image_number, image_seed, sharpness, raw_mode, \
        base_model_name, refiner_model_name, \
        l1, w1, l2, w2, l3, w3, l4, w4, l5, w5 = task

        loras = [(l1, w1), (l2, w2), (l3, w3), (l4, w4), (l5, w5)]

        modules.patch.sharpness = sharpness

        outputs.append(['preview', (1, 'Initializing ...', None)])

        prompt = safe_str(prompt)
        negative_prompt = safe_str(negative_prompt)

        seed = image_seed
        max_seed = int(1024 * 1024 * 1024)
        if not isinstance(seed, int):
            seed = random.randint(1, max_seed)
        if seed < 0:
            seed = - seed
        seed = seed % max_seed

        outputs.append(['preview', (3, 'Load models ...', None)])

        pipeline.refresh_base_model(base_model_name)
        pipeline.refresh_refiner_model(refiner_model_name)
        pipeline.refresh_loras(loras)

        tasks = []
        if raw_mode:
            outputs.append(['preview', (5, 'Encoding negative text ...', None)])
            n_txt = apply_style_negative(style_selction, negative_prompt)
            n_cond = pipeline.process_prompt(n_txt)
            outputs.append(['preview', (9, 'Encoding positive text ...', None)])
            p_txt = apply_style_positive(style_selction, prompt)
            p_cond = pipeline.process_prompt(p_txt)

            for i in range(image_number):
                tasks.append(dict(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=seed + i,
                    n_cond=n_cond,
                    p_cond=p_cond,
                    real_positive_prompt=p_txt,
                    real_negative_prompt=n_txt
                ))
        else:
            for i in range(image_number):
                outputs.append(['preview', (5, f'Preparing positive text #{i + 1} ...', None)])
                current_seed = seed + i

                suffix = pipeline.expansion(prompt, current_seed)
                print(f'[Prompt Expansion] New suffix: {suffix}')

                p_txt = apply_style_positive(style_selction, prompt)
                p_txt = safe_str(p_txt) + suffix

                tasks.append(dict(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    seed=current_seed,
                    real_positive_prompt=p_txt,
                ))

            outputs.append(['preview', (9, 'Encoding negative text ...', None)])
            n_txt = apply_style_negative(style_selction, negative_prompt)
            n_cond = pipeline.process_prompt(n_txt)

            for i, t in enumerate(tasks):
                outputs.append(['preview', (12, f'Encoding positive text #{i + 1} ...', None)])
                t['p_cond'] = pipeline.process_prompt(t['real_positive_prompt'])
                t['real_negative_prompt'] = n_txt
                t['n_cond'] = n_cond

        if performance_selction == 'Speed':
            steps = 30
            switch = 20
        else:
            steps = 60
            switch = 40

        width, height = aspect_ratios[aspect_ratios_selction]

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
                positive_cond=task['p_cond'],
                negative_cond=task['n_cond'],
                steps=steps,
                switch=switch,
                width=width,
                height=height,
                image_seed=task['seed'],
                callback=callback)

            for x in imgs:
                d = [
                    ('Prompt', task['prompt']),
                    ('Negative Prompt', task['negative_prompt']),
                    ('Real Positive Prompt', task['real_positive_prompt']),
                    ('Real Negative Prompt', task['real_negative_prompt']),
                    ('Raw Mode', str(raw_mode)),
                    ('Style', style_selction),
                    ('Performance', performance_selction),
                    ('Resolution', str((width, height))),
                    ('Sharpness', sharpness),
                    ('Base Model', base_model_name),
                    ('Refiner Model', refiner_model_name),
                    ('Seed', task['seed'])
                ]
                for n, w in loras:
                    if n != 'None':
                        d.append((f'LoRA [{n}] weight', w))
                log(x, d)

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
