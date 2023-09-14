import gradio as gr
import random
import time
import shared
import argparse
import modules.path
import fooocus_version
import modules.html
import modules.async_worker as worker

from modules.sdxl_styles import style_keys, aspect_ratios, fooocus_expansion, default_styles

USE_WANDB_INTEGRATION = False
try:
    import os
    import wandb
    
    USE_WANDB_INTEGRATION = os.environ.get("WANDB_PROJECT") is not None and os.environ.get("WANDB_ENTITY") is not None
except:
    USE_WANDB_INTEGRATION = False


def fetch_wandb_history():
    project = os.environ["WANDB_PROJECT"]
    entity = os.environ["WANDB_ENTITY"]
    try:
        api = wandb.Api()
        runs = api.runs(f"{entity}/{project}")
        runs = [run for run in runs if run.state == "finished" and run.job_type == "text-to-image"]
        return project, entity, runs
    except:
        return project, entity, None


def generate_clicked(*args):
    yield gr.update(interactive=False), \
        gr.update(visible=True, value=modules.html.make_progress_html(1, 'Initializing ...')), \
        gr.update(visible=True, value=None), \
        gr.update(visible=False)

    worker.buffer.append(list(args))
    finished = False

    while not finished:
        time.sleep(0.01)
        if len(worker.outputs) > 0:
            flag, product = worker.outputs.pop(0)
            if flag == 'preview':
                percentage, title, image = product
                yield gr.update(interactive=False), \
                    gr.update(visible=True, value=modules.html.make_progress_html(percentage, title)), \
                    gr.update(visible=True, value=image) if image is not None else gr.update(), \
                    gr.update(visible=False)
            if flag == 'results':
                yield gr.update(interactive=True), \
                    gr.update(visible=False), \
                    gr.update(visible=False), \
                    gr.update(visible=True, value=product)
                finished = True
    return


shared.gradio_root = gr.Blocks(title='Fooocus ' + fooocus_version.version, css=modules.html.css).queue()
with shared.gradio_root:
    with gr.Row():
        with gr.Column():
            progress_window = gr.Image(label='Preview', show_label=True, height=640, visible=False)
            progress_html = gr.HTML(value=modules.html.make_progress_html(32, 'Progress 32%'), visible=False, elem_id='progress-bar', elem_classes='progress-bar')
            gallery = gr.Gallery(label='Gallery', show_label=False, object_fit='contain', height=720, visible=True)
            with gr.Row(elem_classes='type_row'):
                with gr.Column(scale=0.85):
                    prompt = gr.Textbox(show_label=False, placeholder="Type prompt here.", container=False, autofocus=True, elem_classes='type_row', lines=1024)
                with gr.Column(scale=0.15, min_width=0):
                    run_button = gr.Button(label="Generate", value="Generate", elem_classes='type_row')
            with gr.Row():
                advanced_checkbox = gr.Checkbox(label='Advanced', value=False, container=False)
        with gr.Column(scale=0.5, visible=False) as right_col:
            with gr.Tab(label='Setting'):
                performance_selction = gr.Radio(label='Performance', choices=['Speed', 'Quality'], value='Speed')
                aspect_ratios_selction = gr.Radio(label='Aspect Ratios', choices=list(aspect_ratios.keys()),
                                                  value='1152×896', info='width × height')
                image_number = gr.Slider(label='Image Number', minimum=1, maximum=32, step=1, value=2)
                negative_prompt = gr.Textbox(label='Negative Prompt', show_label=True, placeholder="Type prompt here.",
                                             info='Describing objects that you do not want to see.')
                seed_random = gr.Checkbox(label='Random', value=True)
                image_seed = gr.Number(label='Seed', value=0, precision=0, visible=False)

                def random_checked(r):
                    return gr.update(visible=not r)

                def refresh_seed(r, s):
                    if r:
                        return random.randint(1, 1024*1024*1024)
                    else:
                        return s

                seed_random.change(random_checked, inputs=[seed_random], outputs=[image_seed])

            with gr.Tab(label='Style'):
                style_selections = gr.CheckboxGroup(show_label=False, container=False,
                                                    choices=[fooocus_expansion] + style_keys,
                                                    value=[fooocus_expansion] + default_styles,
                                                    label='Image Style')
            with gr.Tab(label='Advanced'):
                with gr.Row():
                    base_model = gr.Dropdown(label='SDXL Base Model', choices=modules.path.model_filenames, value=modules.path.default_base_model_name, show_label=True)
                    refiner_model = gr.Dropdown(label='SDXL Refiner', choices=['None'] + modules.path.model_filenames, value=modules.path.default_refiner_model_name, show_label=True)
                with gr.Accordion(label='LoRAs', open=True):
                    lora_ctrls = []
                    for i in range(5):
                        with gr.Row():
                            lora_model = gr.Dropdown(label=f'SDXL LoRA {i+1}', choices=['None'] + modules.path.lora_filenames, value=modules.path.default_lora_name if i == 0 else 'None')
                            lora_weight = gr.Slider(label='Weight', minimum=-2, maximum=2, step=0.01, value=modules.path.default_lora_weight)
                            lora_ctrls += [lora_model, lora_weight]
                with gr.Row():
                    model_refresh = gr.Button(label='Refresh', value='\U0001f504 Refresh All Files', variant='secondary', elem_classes='refresh_button')
                with gr.Accordion(label='Advanced', open=False):
                    sharpness = gr.Slider(label='Sampling Sharpness', minimum=0.0, maximum=30.0, step=0.01, value=2.0)
                    gr.HTML('<a href="https://github.com/lllyasviel/Fooocus/discussions/117">\U0001F4D4 Document</a>')

                def model_refresh_clicked():
                    modules.path.update_all_model_names()
                    results = []
                    results += [gr.update(choices=modules.path.model_filenames), gr.update(choices=['None'] + modules.path.model_filenames)]
                    for i in range(5):
                        results += [gr.update(choices=['None'] + modules.path.lora_filenames), gr.update()]
                    return results

                model_refresh.click(model_refresh_clicked, [], [base_model, refiner_model] + lora_ctrls)
            
            if USE_WANDB_INTEGRATION:
                with gr.Tab(label="History") as history_tab:
                    project, entity, runs = fetch_wandb_history()
                    run_name_to_id = {run.name: run.id for run in runs} if runs is not None else {}
                    project_url = f"https://wandb.ai/{entity}/{project}"
                    gr.Markdown(f"## Project: [{entity}/{project}]({project_url})")
                    wandb_run_dropdown = gr.Dropdown(
                        choices=[run.name for run in runs] if runs is not None else [],
                        label="Past Experiments"
                    )
                    wandb_refresh_button = gr.Button(label='Refresh History', value='\U0001f504 Refresh History')
                    
                    history_details = gr.Markdown()
                    
                    def update_history_details(run_name):
                        project, entity, runs = fetch_wandb_history()
                        run_name_to_id = {run.name: run.id for run in runs} if runs is not None else {}
                        selected_run = run_name_to_id[run_name]
                        for run in runs:
                            if run.id == selected_run:
                                selected_run = run
                                break
                        content = f"### Experiment: [{run_name}](https://wandb.ai/{entity}/{project}/runs/{run_name_to_id[run_name]}) ({str(selected_run.created_at)})"
                        content += "\n|Configuration|Value|"
                        content += "\n|---|---|"
                        content += f"\n|Prompt|{selected_run.config['Prompt']}|"
                        content += f"\n|Negative Prompt|{selected_run.config['Negative Prompt']}|"
                        content += f"\n|Performance|{selected_run.config['Performance']}|"
                        content += f"\n|Resolution|{selected_run.config['Resolution']['width']}x{selected_run.config['Resolution']['height']}|"
                        content += f"\n|Image Seed|{selected_run.config['Image Seed']}|"
                        content += f"\n|Styles|{selected_run.config['Style']}|"
                        content += f"\n|Sharpness|{selected_run.config['Sharpness']}|"
                        content += f"\n|Number of Images|{selected_run.config['Number of Images']}|"
                        content += f"\n|Number of Steps|{selected_run.config['Number of Steps']}|"
                        content += f"\n|Base Model|{selected_run.config['Base Model']}|"
                        content += f"\n|Refiner Model|{selected_run.config['Refiner Model']}|"
                        for k, v in selected_run.config['LoRA Weights'].items():
                            content += f"\n|{k}|{v}|"
                        return content
                    
                    
                    def update_wandb_dropdown():
                        _, _, runs = fetch_wandb_history()
                        return gr.Dropdown.update(
                            choices=[run.name for run in runs] if runs is not None else [],
                            label="Past Experiments"
                        )
                        
                    
                    wandb_run_dropdown.change(fn=update_history_details, inputs=wandb_run_dropdown, outputs=history_details)
                    wandb_refresh_button.click(fn=update_wandb_dropdown, outputs=wandb_run_dropdown)

        advanced_checkbox.change(lambda x: gr.update(visible=x), advanced_checkbox, right_col)
        ctrls = [
            prompt, negative_prompt, style_selections,
            performance_selction, aspect_ratios_selction, image_number, image_seed, sharpness
        ]
        ctrls += [base_model, refiner_model] + lora_ctrls
        run_button.click(fn=refresh_seed, inputs=[seed_random, image_seed], outputs=image_seed)\
            .then(fn=generate_clicked, inputs=ctrls, outputs=[run_button, progress_html, progress_window, gallery])


parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=None, help="Set the listen port.")
parser.add_argument("--share", action='store_true', help="Set whether to share on Gradio.")
parser.add_argument("--listen", type=str, default=None, metavar="IP", nargs="?", const="0.0.0.0", help="Set the listen interface.")
args = parser.parse_args()
shared.gradio_root.launch(inbrowser=True, server_name=args.listen, server_port=args.port, share=args.share)
