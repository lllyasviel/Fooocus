import gradio as gr
import modules.path
import random
import fooocus_version
import modules.default_pipeline as pipeline

from modules.sdxl_styles import apply_style, style_keys, aspect_ratios
from modules.cv2win32 import close_all_preview, save_image
from modules.util import generate_temp_filename


def generate_clicked(prompt, negative_prompt, style_selction, performance_selction,
                     aspect_ratios_selction, image_number, image_seed, base_model_name, refiner_model_name,
                     l1, w1, l2, w2, l3, w3, l4, w4, l5, w5, progress=gr.Progress()):

    loras = [(l1, w1), (l2, w2), (l3, w3), (l4, w4), (l5, w5)]

    pipeline.refresh_base_model(base_model_name)
    pipeline.refresh_refiner_model(refiner_model_name)
    pipeline.refresh_loras(loras)

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

    def callback(step, x0, x, total_steps):
        done_steps = i * steps + step
        progress(float(done_steps) / float(all_steps), f'Step {step}/{total_steps} in the {i}-th Sampling')

    for i in range(image_number):
        imgs = pipeline.process(p_txt, n_txt, steps, switch, width, height, seed, callback=callback)

        for x in imgs:
            local_temp_filename = generate_temp_filename(folder=modules.path.temp_outputs_path, extension='png')
            save_image(local_temp_filename, x)

        seed += 1
        results += imgs

    close_all_preview()
    return results


block = gr.Blocks(title='Fooocus ' + fooocus_version.version).queue()
with block:
    with gr.Row():
        with gr.Column():
            gallery = gr.Gallery(label='Gallery', show_label=False, object_fit='contain', height=720)
            with gr.Row():
                with gr.Column(scale=0.85):
                    prompt = gr.Textbox(show_label=False, placeholder="Type prompt here.", container=False, autofocus=True)
                with gr.Column(scale=0.15, min_width=0):
                    run_button = gr.Button(label="Generate", value="Generate")
            with gr.Row():
                advanced_checkbox = gr.Checkbox(label='Advanced', value=False, container=False)
        with gr.Column(scale=0.5, visible=False) as right_col:
            with gr.Tab(label='Setting'):
                performance_selction = gr.Radio(label='Performance', choices=['Speed', 'Quality'], value='Speed')
                aspect_ratios_selction = gr.Radio(label='Aspect Ratios (width × height)', choices=list(aspect_ratios.keys()),
                                                  value='1152×896')
                image_number = gr.Slider(label='Image Number', minimum=1, maximum=32, step=1, value=2)
                image_seed = gr.Number(label='Random Seed', value=-1, precision=0)
                negative_prompt = gr.Textbox(label='Negative Prompt', show_label=True, placeholder="Type prompt here.")
            with gr.Tab(label='Style'):
                style_selction = gr.Radio(show_label=False, container=True,
                                          choices=style_keys, value='cinematic-default')
            with gr.Tab(label='Advanced'):
                with gr.Row():
                    base_model = gr.Dropdown(label='SDXL Base Model', choices=modules.path.model_filenames, value=modules.path.default_base_model_name, show_label=True)
                    refiner_model = gr.Dropdown(label='SDXL Refiner', choices=['None'] + modules.path.model_filenames, value=modules.path.default_refiner_model_name, show_label=True)
                with gr.Accordion(label='LoRAs', open=True):
                    lora_ctrls = []
                    for i in range(5):
                        with gr.Row():
                            lora_model = gr.Dropdown(label=f'LoRA {i+1}', choices=['None'] + modules.path.lora_filenames, value=modules.path.default_lora_name if i == 0 else 'None')
                            lora_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.01, value=0.5)
                            lora_ctrls += [lora_model, lora_weight]
                model_refresh = gr.Button(label='Refresh', value='Refresh All Files', variant='secondary')

                def model_refresh_clicked():
                    modules.path.update_all_model_names()
                    results = []
                    results += [gr.update(choices=modules.path.model_filenames), gr.update(choices=['None'] + modules.path.model_filenames)]
                    for i in range(5):
                        results += [gr.update(choices=['None'] + modules.path.lora_filenames), gr.update()]
                    return results

                model_refresh.click(model_refresh_clicked, [], [base_model, refiner_model] + lora_ctrls)

        advanced_checkbox.change(lambda x: gr.update(visible=x), advanced_checkbox, right_col)
        ctrls = [
            prompt, negative_prompt, style_selction,
            performance_selction, aspect_ratios_selction, image_number, image_seed
        ]
        ctrls += [base_model, refiner_model] + lora_ctrls
        run_button.click(fn=generate_clicked, inputs=ctrls, outputs=[gallery])

block.launch(inbrowser=True)
