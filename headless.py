import argparse
import time
import random
import modules.path
import fooocus_version
import modules.flags as flags
import modules.async_worker as worker
import comfy.model_management as model_management

def generate_images(*args):
    execution_start_time = time.perf_counter()
    last_progress_time = execution_start_time

    worker.buffer.append(list(args))
    finished = False

    while not finished:
        time.sleep(1)

        if len(worker.outputs) > 0:
            flag, product = worker.outputs.pop(0)

            if flag == 'preview':
                percentage, title, image = product
                print(f'Progress: {percentage}% - {title}')
                last_progress_time = time.perf_counter()

            elif flag == 'results':
                finished = True
                execution_time = time.perf_counter() - execution_start_time
                print(f'Total time: {execution_time:.2f} seconds')
                return product

    print("Image generation failed.")
    return None

# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument("--prompt", type=str, required=True, help="Input text prompt.")
parser.add_argument("--negative_prompt", type=str, required=False, default="", help="Negative text prompt.")
parser.add_argument("--styles", type=str, nargs='+', required=False, default=["Fooocus V2", "Default (Slightly Cinematic)"], help="List of style selections.")
parser.add_argument("--performance", type=str, required=False, default="Speed", help="Performance selection ('Speed' or 'Quality').")
parser.add_argument("--aspect_ratio", type=str, required=False, default="1024×1024", help="Aspect ratio selection.")
parser.add_argument("--image_number", type=int, required=False, default="2", help="Number of images to generate.")
parser.add_argument("--image_seed", type=int, required=False, default=random.randint(1, 1073741824), help="Random seed for image generation.")
parser.add_argument("--sharpness", type=float, required=False, default=2, help="Sampling sharpness.")
parser.add_argument("--base_model_name", type=str, required=False, default="sd_xl_base_1.0_0.9vae.safetensors", help="SDXL Base Model name.")
parser.add_argument("--refiner_model_name", type=str, required=False, default="sd_xl_refiner_1.0_0.9vae.safetensors", help="SDXL Refiner Model name.")
parser.add_argument("--l1", type=str, required=False, default="sd_xl_offset_example-lora_1.0.safetensors", help="LoRA Model 1 name.")
parser.add_argument("--w1", type=float, required=False, default=0.5, help="LoRA Model 1 weight.")
parser.add_argument("--l2", type=str, required=False, default="None", help="LoRA Model 2 name.")
parser.add_argument("--w2", type=float, required=False, default=0.5, help="LoRA Model 2 weight.")
parser.add_argument("--l3", type=str, required=False, default="None", help="LoRA Model 3 name.")
parser.add_argument("--w3", type=float, required=False, default=0.5, help="LoRA Model 3 weight.")
parser.add_argument("--l4", type=str, required=False, default="None", help="LoRA Model 4 name.")
parser.add_argument("--w4", type=float, required=False, default=0.5, help="LoRA Model 4 weight.")
parser.add_argument("--l5", type=str, required=False, default="None", help="LoRA Model 5 name.")
parser.add_argument("--w5", type=float, required=False, default=0.5, help="LoRA Model 5 weight.")
parser.add_argument("--current_tab", type=str, required=False, default="", help="Current tab.")
parser.add_argument("--use_input_image", type=bool, required=False, default=False, help="Input Image Checkbox value.")
parser.add_argument("--uov_method", type=str, required=False, default="disabled", help="Upscale or Variation Method.")
parser.add_argument("--uov_input_image", type=str, required=False, default=None, help="Upscale or Variation Input Image.")
parser.add_argument("--outpaint_selections", type=list, required=False, default=[], help="List of Outpaint selections.")
parser.add_argument("--inpaint_input_image", type=str, required=False, default=None, help="Inpaint Input Image.")
args = parser.parse_args()

# Call the function to generate images
result = generate_images(args.prompt, args.negative_prompt, args.style_selections, args.performance_selction,
                        args.aspect_ratios_selction, args.image_number, args.image_seed, args.sharpness,
                        args.base_model_name, args.refiner_model_name,
                        args.l1, args.w1, args.l2, args.w2, args.l3, args.w3, args.l4, args.w4, args.l5, args.w5,
                        args.input_image_checkbox, args.current_tab,
                        args.uov_method, args.uov_input_image, args.outpaint_selections, args.inpaint_input_image)

if result:
    print("Image generated successfully.")
else:
    print("Image generation failed.")
