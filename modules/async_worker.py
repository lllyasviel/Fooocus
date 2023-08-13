import threading


buffer = []
outputs = []


def worker():
    global buffer, outputs

    import time
    import random
    import modules.default_pipeline as pipeline
    import modules.path

    from PIL import Image
    from modules.sdxl_styles import apply_style, aspect_ratios
    from modules.util import generate_temp_filename

    def handler(task):
        prompt, negative_prompt, style_selction, performance_selction, \
        aspect_ratios_selction, image_number, image_seed, base_model_name, refiner_model_name, \
        l1, w1, l2, w2, l3, w3, l4, w4, l5, w5 = task

        loras = [(l1, w1), (l2, w2), (l3, w3), (l4, w4), (l5, w5)]

        pipeline.refresh_base_model(base_model_name)
        pipeline.refresh_refiner_model(refiner_model_name)
        pipeline.refresh_loras(loras)
        pipeline.clean_prompt_cond_caches()

        p_txt, n_txt = apply_style(style_selction, prompt, negative_prompt)

        if performance_selction == 'Speed':
            steps = 30
            switch = 20
        else:
            steps = 60
            switch = 40

        width, height = aspect_ratios[aspect_ratios_selction]

        results = []
        seed = image_seed
        if not isinstance(seed, int) or seed < 0 or seed > 65535:
            seed = random.randint(1, 65535)

        all_steps = steps * image_number

        def callback(step, x0, x, total_steps, y):
            done_steps = i * steps + step
            outputs.append(['preview', (
                int(100.0 * float(done_steps) / float(all_steps)),
                f'Step {step}/{total_steps} in the {i}-th Sampling',
                y)])

        for i in range(image_number):
            imgs = pipeline.process(p_txt, n_txt, steps, switch, width, height, seed, callback=callback)

            for x in imgs:
                local_temp_filename = generate_temp_filename(folder=modules.path.temp_outputs_path, extension='png')
                Image.fromarray(x).save(local_temp_filename)

            seed += 1
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
