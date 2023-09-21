import argparse
import time
import modules.path
import fooocus_version
import modules.flags as flags
import modules.async_worker as worker
import comfy.model_management as model_management

def generate_images(prompt, negative_prompt, style_selections, performance_selction, aspect_ratios_selction,
                    image_number, image_seed, sharpness, base_model_name, refiner_model_name,
                    l1, w1, l2, w2, l3, w3, l4, w4, l5, w5,
                    input_image_checkbox, current_tab,
                    uov_method, uov_input_image, outpaint_selections, inpaint_input_image):
    execution_start_time = time.perf_counter()
    last_progress_time = execution_start_time

    worker.buffer.append([prompt, negative_prompt, style_selections, performance_selction, aspect_ratios_selction,
                          image_number, image_seed, sharpness, base_model_name, refiner_model_name,
                          l1, w1, l2, w2, l3, w3, l4, w4, l5, w5,
                          input_image_checkbox, current_tab,
                          uov_method, uov_input_image, outpaint_selections, inpaint_input_image])
    finished = False

    while not finished:
        time.sleep(1)  # Sleep for 1 second

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
parser.add_argument("--negative_prompt", type=str, required=True, help="Negative text prompt.")
parser.add_argument("--style_selections", type=list, required=True, help="List of style selections.")
parser.add_argument("--performance_selction", type=str, required=True, help="Performance selection ('Speed' or 'Quality').")
parser.add_argument("--aspect_ratios_selction", type=str, required=True, help="Aspect ratio selection.")
parser.add_argument("--image_number", type=int, required=True, help="Number of images to generate.")
parser.add_argument("--image_seed", type=int, required=True, help="Random seed for image generation.")
parser.add_argument("--sharpness", type=float, required=True, help="Sampling sharpness.")
parser.add_argument("--base_model_name", type=str, required=True, help="SDXL Base Model name.")
parser.add_argument("--refiner_model_name", type=str, required=True, help="SDXL Refiner Model name.")
parser.add_argument("--l1", type=str, required=True, help="LoRA Model 1 name.")
parser.add_argument("--w1", type=float, required=True, help="LoRA Model 1 weight.")
parser.add_argument("--l2", type=str, required=True, help="LoRA Model 2 name.")
parser.add_argument("--w2", type=float, required=True, help="LoRA Model 2 weight.")
parser.add_argument("--l3", type=str, required=True, help="LoRA Model 3 name.")
parser.add_argument("--w3", type=float, required=True, help="LoRA Model 3 weight.")
parser.add_argument("--l4", type=str, required=True, help="LoRA Model 4 name.")
parser.add_argument("--w4", type=float, required=True, help="LoRA Model 4 weight.")
parser.add_argument("--l5", type=str, required=True, help="LoRA Model 5 name.")
parser.add_argument("--w5", type=float, required=True, help="LoRA Model 5 weight.")
parser.add_argument("--input_image_checkbox", type=bool, required=True, help="Input Image Checkbox value.")
parser.add_argument("--current_tab", type=str, required=True, help="Current tab.")
parser.add_argument("--uov_method", type=str, required=True, help="Upscale or Variation Method.")
parser.add_argument("--uov_input_image", type=str, required=True, help="Upscale or Variation Input Image.")
parser.add_argument("--outpaint_selections", type=list, required=True, help="List of Outpaint selections.")
parser.add_argument("--inpaint_input_image", type=str, required=True, help="Inpaint Input Image.")
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

This script takes all the required parameters for image generation and calls the generate_images function accordingly. It can be invoked from your launch.py script with the appropriate arguments to perform headless image generation.
