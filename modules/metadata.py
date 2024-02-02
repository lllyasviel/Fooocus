import json
import re
from pathlib import Path
from abc import ABC, abstractmethod
from PIL import Image

import modules.config
import fooocus_version
# import advanced_parameters
from modules.flags import MetadataScheme, Performance, Steps, lora_count_with_lcm
from modules.util import quote, unquote, extract_styles_from_prompt, is_json, calculate_sha256

re_param_code = r'\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)'
re_param = re.compile(re_param_code)
re_imagesize = re.compile(r"^(\d+)x(\d+)$")

hash_cache = {}


def get_sha256(filepath):
    global hash_cache

    if filepath not in hash_cache:
        hash_cache[filepath] = calculate_sha256(filepath)

    return hash_cache[filepath]

class MetadataParser(ABC):
    @abstractmethod
    def parse_json(self, metadata: dict) -> dict:
        raise NotImplementedError

    # TODO add data to parse
    @abstractmethod
    def parse_string(self, metadata: dict) -> str:
        raise NotImplementedError


class A1111MetadataParser(MetadataParser):

    fooocus_to_a1111 = {
        'negative_prompt': 'Negative prompt',
        'styles': 'Styles',
        'performance': 'Performance',
        'steps': 'Steps',
        'sampler': 'Sampler',
        'guidance_scale': 'CFG scale',
        'seed': 'Seed',
        'resolution': 'Size',
        'sharpness': 'Sharpness',
        'adm_guidance': 'ADM Guidance',
        'refiner_swap_method': 'Refiner Swap Method',
        'adaptive_cfg': 'Adaptive CFG',
        'overwrite_switch': 'Overwrite Switch',
        'freeu': 'FreeU',
        'base_model': 'Model',
        'base_model_hash': 'Model hash',
        'refiner_model': 'Refiner',
        'refiner_model_hash': 'Refiner hash',
        'lora_hashes': 'Lora hashes',
        'lora_weights': 'Lora weights',
        'version': 'Version'
    }

    def parse_json(self, metadata: str) -> dict:
        prompt = ''
        negative_prompt = ''

        done_with_prompt = False

        *lines, lastline = metadata.strip().split("\n")
        if len(re_param.findall(lastline)) < 3:
            lines.append(lastline)
            lastline = ''

        for line in lines:
            line = line.strip()
            if line.startswith(f"{self.fooocus_to_a1111['negative_prompt']}:"):
                done_with_prompt = True
                line = line[len(f"{self.fooocus_to_a1111['negative_prompt']}:"):].strip()
            if done_with_prompt:
                negative_prompt += ('' if negative_prompt == '' else "\n") + line
            else:
                prompt += ('' if prompt == '' else "\n") + line

        found_styles, prompt, negative_prompt = extract_styles_from_prompt(prompt, negative_prompt)

        data = {
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'styles': str(found_styles)
        }

        for k, v in re_param.findall(lastline):
            try:
                if v[0] == '"' and v[-1] == '"':
                    v = unquote(v)

                m = re_imagesize.match(v)
                if m is not None:
                    data[f'resolution'] = str((m.group(1), m.group(2)))
                else:
                    data[list(self.fooocus_to_a1111.keys())[list(self.fooocus_to_a1111.values()).index(k)]] = v
            except Exception:
                print(f"Error parsing \"{k}: {v}\"")

        # try to load performance based on steps, fallback for direct A1111 imports
        if 'steps' in data and 'performance' not in data:
            try:
                data['performance'] = Performance[Steps(int(data['steps'])).name].value
            except Exception:
                pass

        if 'base_model' in data:
            for filename in modules.config.model_filenames:
                path = Path(filename)
                if data['base_model'] == path.stem:
                    data['base_model'] = filename
                    break

        if 'lora_hashes' in data:
            # TODO optimize by using hash for matching. Problem is speed of creating the hash per model, even on startup
            lora_filenames = modules.config.lora_filenames.copy()
            lora_filenames.remove(modules.config.downloading_sdxl_lcm_lora())
            for li, lora in enumerate(data['lora_hashes'].split(', ')):
                name, _, weight = lora.split(': ')
                for filename in lora_filenames:
                    path = Path(filename)
                    if name == path.stem:
                        data[f'lora_combined_{li + 1}'] = f'{filename} : {weight}'
                        break

        return data

    def parse_string(self, metadata: dict) -> str:
        data = {k: v for _, k, v, _, _ in metadata}

        width, heigth = eval(data['resolution'])

        lora_hashes = []
        for index in range(lora_count_with_lcm):
            key = f'lora_name_{index + 1}'
            if key in data:
                lora_name = Path(data[f'lora_name_{index + 1}']).stem
                lora_weight = data[f'lora_weight_{index + 1}']
                lora_hash = data[f'lora_hash_{index + 1}']
                # workaround for Fooocus not knowing LoRA name in LoRA metadata
                lora_hashes.append(f'{lora_name}: {lora_hash}: {lora_weight}')
        lora_hashes_string = ', '.join(lora_hashes)

        generation_params = {
            self.fooocus_to_a1111['performance']: data['performance'],
            self.fooocus_to_a1111['steps']: data['steps'],
            self.fooocus_to_a1111['sampler']: data['sampler'],
            self.fooocus_to_a1111['seed']: data['seed'],
            self.fooocus_to_a1111['resolution']: f'{width}x{heigth}',
            self.fooocus_to_a1111['guidance_scale']: data['guidance_scale'],
            self.fooocus_to_a1111['sharpness']: data['sharpness'],
            self.fooocus_to_a1111['adm_guidance']: data['adm_guidance'],
            # TODO load model by name / hash
            self.fooocus_to_a1111['base_model']: Path(data['base_model']).stem,
            self.fooocus_to_a1111['base_model_hash']: data['base_model_hash']
        }

        if 'refiner_model' in data and data['refiner_model'] != 'None' and 'refiner_model_hash' in data:
            generation_params |= {
                self.fooocus_to_a1111['refiner_model']: Path(data['refiner_model']).stem,
                self.fooocus_to_a1111['refiner_model_hash']: data['refiner_model_hash']
            }

        for key in ['adaptive_cfg', 'overwrite_switch', 'refiner_swap_method', 'freeu']:
            if key in data:
                generation_params[self.fooocus_to_a1111[key]] = data[key]

        generation_params |= {
            self.fooocus_to_a1111['lora_hashes']: lora_hashes_string,
            self.fooocus_to_a1111['version']: data['version']
        }

        generation_params_text = ", ".join(
            [k if k == v else f'{k}: {quote(v)}' for k, v in generation_params.items() if v is not None])
        # TODO check if multiline positive prompt is correctly processed
        positive_prompt_resolved = ', '.join(data['full_prompt']) #TODO add loras to positive prompt if even possible
        negative_prompt_resolved = ', '.join(data['full_negative_prompt']) #TODO add loras to negative prompt if even possible
        negative_prompt_text = f"\nNegative prompt: {negative_prompt_resolved}" if negative_prompt_resolved else ""
        return f"{positive_prompt_resolved}{negative_prompt_text}\n{generation_params_text}".strip()


class FooocusMetadataParser(MetadataParser):

    def parse_json(self, metadata: dict) -> dict:
        model_filenames = modules.config.model_filenames.copy()
        lora_filenames = modules.config.lora_filenames.copy()

        for key, value in metadata.items():
            if value == '' or value == 'None':
                continue
            if key in ['base_model', 'refiner_model']:
                metadata[key] = self.replace_value_with_filename(key, value, model_filenames)
            elif key.startswith(('lora_combined_', 'lora_name_')):
                metadata[key] = self.replace_value_with_filename(key, value, lora_filenames)
            else:
                continue

        return metadata

    def parse_string(self, metadata: list) -> str:
        # remove model folder paths from metadata
        for li, (label, key, value, show_in_log, copy_in_log) in enumerate(metadata):
            if value == '' or value == 'None':
                continue
            if key in ['base_model', 'refiner_model'] or key.startswith(('lora_combined_', 'lora_name_')):
                if key.startswith('lora_combined_'):
                    name, weight = value.split(' : ')
                    name = Path(name).stem
                    value = f'{name} : {weight}'
                else:
                    value = Path(value).stem
                metadata[li] = (label, key, value, show_in_log, copy_in_log)

        return json.dumps({k: v for _, k, v, _, _ in metadata})
        # metadata = {
        #     # prompt with wildcards
        #     'prompt': raw_prompt, 'negative_prompt': raw_negative_prompt,
        #     # prompt with resolved wildcards
        #     'real_prompt': task['log_positive_prompt'], 'real_negative_prompt': task['log_negative_prompt'],
        #     # prompt with resolved wildcards, styles and prompt expansion
        #     'complete_prompt_positive': task['positive'], 'complete_prompt_negative': task['negative'],
        #     'styles': str(raw_style_selections),
        #     'seed': task['task_seed'], 'width': width, 'height': height,
        #     'sampler': sampler_name, 'scheduler': scheduler_name, 'performance': performance_selection,
        #     'steps': steps, 'refiner_switch': refiner_switch, 'sharpness': sharpness, 'cfg': cfg_scale,
        #     'base_model': base_model_name, 'base_model_hash': base_model_hash, 'refiner_model': refiner_model_name,
        #     'denoising_strength': denoising_strength,
        #     'freeu': advanced_parameters.freeu_enabled,
        #     'img2img': input_image_checkbox,
        #     'prompt_expansion': task['expansion']
        # }
        #
        # if advanced_parameters.freeu_enabled:
        #     metadata |= {
        #         'freeu_b1': advanced_parameters.freeu_b1, 'freeu_b2': advanced_parameters.freeu_b2,
        #         'freeu_s1': advanced_parameters.freeu_s1, 'freeu_s2': advanced_parameters.freeu_s2
        #     }
        #
        # if 'vary' in goals:
        #     metadata |= {
        #         'uov_method': uov_method
        #     }
        #
        # if 'upscale' in goals:
        #     metadata |= {
        #         'uov_method': uov_method, 'scale': f
        #     }
        #
        # if 'inpaint' in goals:
        #     if len(outpaint_selections) > 0:
        #         metadata |= {
        #             'outpaint_selections': outpaint_selections
        #         }
        #
        #     metadata |= {
        #         'inpaint_additional_prompt': inpaint_additional_prompt,
        #         'inpaint_mask_upload': advanced_parameters.inpaint_mask_upload_checkbox,
        #         'invert_mask': advanced_parameters.invert_mask_checkbox,
        #         'inpaint_disable_initial_latent': advanced_parameters.inpaint_disable_initial_latent,
        #         'inpaint_engine': advanced_parameters.inpaint_engine,
        #         'inpaint_strength': advanced_parameters.inpaint_strength,
        #         'inpaint_respective_field': advanced_parameters.inpaint_respective_field,
        #     }
        #
        # if 'cn' in goals:
        #     metadata |= {
        #         'canny_low_threshold': advanced_parameters.canny_low_threshold,
        #         'canny_high_threshold': advanced_parameters.canny_high_threshold,
        #     }
        #
        #     ip_list = {x: [] for x in flags.ip_list}
        #     cn_task_index = 1
        #     for cn_type in ip_list:
        #         for cn_task in cn_tasks[cn_type]:
        #             cn_img, cn_stop, cn_weight = cn_task
        #             metadata |= {
        #                 f'image_prompt_{cn_task_index}': {
        #                     'cn_type': cn_type, 'cn_stop': cn_stop, 'cn_weight': cn_weight,
        #                 }
        #             }
        #             cn_task_index += 1
        #
        # metadata |= {
        #     'software': f'Fooocus v{fooocus_version.version}',
        # }
        # TODO add metadata_created_by
        # if modules.config.metadata_created_by != '':
        #     metadata |= {
        #         'created_by': modules.config.metadata_created_by
        #     }
        # # return json.dumps(metadata, ensure_ascii=True) TODO check if possible
        # return json.dumps(metadata, ensure_ascii=False)

    @staticmethod
    def replace_value_with_filename(key, value, filenames):
        for filename in filenames:
            path = Path(filename)
            if key.startswith('lora_combined_'):
                name, weight = value.split(' : ')
                if name == path.stem:
                    return f'{filename} : {weight}'
            elif value == path.stem:
                return filename


def get_metadata_parser(metadata_scheme: MetadataScheme) -> MetadataParser:
    match metadata_scheme:
        case MetadataScheme.FOOOCUS:
            return FooocusMetadataParser()
        case MetadataScheme.A1111:
            return A1111MetadataParser()
        case _:
            raise NotImplementedError

# IGNORED_INFO_KEYS = {
#     'jfif', 'jfif_version', 'jfif_unit', 'jfif_density', 'dpi', 'exif',
#     'loop', 'background', 'timestamp', 'duration', 'progressive', 'progression',
#     'icc_profile', 'chromaticity', 'photoshop',
# }


def read_info_from_image(filepath) -> tuple[str | None, dict, MetadataScheme | None]:
    with Image.open(filepath) as image:
        items = (image.info or {}).copy()

    parameters = items.pop('parameters', None)
    if parameters is not None and is_json(parameters):
        parameters = json.loads(parameters)

    try:
        metadata_scheme = MetadataScheme(items.pop('fooocus_scheme', None))
    except Exception:
        metadata_scheme = None

    # broad fallback
    if metadata_scheme is None and isinstance(parameters, dict):
        metadata_scheme = modules.metadata.MetadataScheme.FOOOCUS

    if metadata_scheme is None and isinstance(parameters, str):
        metadata_scheme = modules.metadata.MetadataScheme.A1111

    # TODO code cleanup
    # if "exif" in items:
    #     exif_data = items["exif"]
    #     try:
    #         exif = piexif.load(exif_data)
    #     except OSError:
    #         # memory / exif was not valid so piexif tried to read from a file
    #         exif = None
    #     exif_comment = (exif or {}).get("Exif", {}).get(piexif.ExifIFD.UserComment, b'')
    #     try:
    #         exif_comment = piexif.helper.UserComment.load(exif_comment)
    #     except ValueError:
    #         exif_comment = exif_comment.decode('utf8', errors="ignore")
    #
    #     if exif_comment:
    #         items['exif comment'] = exif_comment
    #         parameters = exif_comment

    # for field in IGNORED_INFO_KEYS:
    #     items.pop(field, None)

    #     if items.get("Software", None) == "NovelAI":
    #         try:
    #             json_info = json.loads(items["Comment"])
    #             sampler = sd_samplers.samplers_map.get(json_info["sampler"], "Euler a")
    #
    #             geninfo = f"""{items["Description"]}
    # Negative prompt: {json_info["uc"]}
    # Steps: {json_info["steps"]}, Sampler: {sampler}, CFG scale: {json_info["scale"]}, Seed: {json_info["seed"]}, Size: {image.width}x{image.height}, Clip skip: 2, ENSD: 31337"""
    #         except Exception:
    #             errors.report("Error parsing NovelAI image generation parameters",
    #                           exc_info=True)

    return parameters, items, metadata_scheme
