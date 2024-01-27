import json
from abc import ABC, abstractmethod
from enum import Enum
from PIL import Image

import modules.config
import fooocus_version
# import advanced_parameters
from modules.util import quote, is_json


class MetadataScheme(Enum):
    FOOOCUS = 'fooocus'
    A1111 = 'a1111'


class MetadataParser(ABC):
    @abstractmethod
    def parse_json(self, metadata: dict):
        raise NotImplementedError

    # TODO add data to parse
    @abstractmethod
    def parse_string(self, metadata: dict) -> str:
        raise NotImplementedError


class A1111MetadataParser(MetadataParser):

    def parse_json(self, metadata: dict):
        # TODO add correct mapping
        pass

    def parse_string(self, metadata: dict) -> str:
        # TODO add correct mapping

        data = {k: v for _, k, v, _, _ in metadata}

        # TODO check if correct
        width, heigth = data['resolution'].split(', ')

        generation_params = {
            "Steps": data['steps'],
            "Sampler": data['sampler'],
            "CFG scale": data['guidance_scale'],
            "Seed": data['seed'],
            "Size": f"{width}x{heigth}",
            # "Model hash": base_model_hash,
            "Model": data['base_model'].split('.')[0],
            # "Lora hashes": lora_hashes_string,
            # "Denoising strength": data['denoising_strength'],
            "Version": f"Fooocus {data['version']}"
        }

        generation_params_text = ", ".join(
            [k if k == v else f'{k}: {quote(v)}' for k, v in generation_params.items() if v is not None])
        positive_prompt_resolved = ', '.join(data['full_prompt'])
        negative_prompt_resolved = ', '.join(data['full_negative_prompt'])
        negative_prompt_text = f"\nNegative prompt: {negative_prompt_resolved}" if negative_prompt_resolved else ""
        return f"{positive_prompt_resolved}{negative_prompt_text}\n{generation_params_text}".strip()


class FooocusMetadataParser(MetadataParser):

    def parse_json(self, metadata: dict):
        # TODO add mapping if necessary
        return metadata

    def parse_string(self, metadata: dict) -> str:

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
        # if modules.config.metadata_created_by != '':
        #     metadata |= {
        #         'created_by': modules.config.metadata_created_by
        #     }
        # # return json.dumps(metadata, ensure_ascii=True) TODO check if possible
        # return json.dumps(metadata, ensure_ascii=False)


def get_metadata_parser(metadata_scheme: str) -> MetadataParser:
    match metadata_scheme:
        case MetadataScheme.FOOOCUS.value:
            return FooocusMetadataParser()
        case MetadataScheme.A1111.value:
            return A1111MetadataParser()
        case _:
            raise NotImplementedError

# IGNORED_INFO_KEYS = {
#     'jfif', 'jfif_version', 'jfif_unit', 'jfif_density', 'dpi', 'exif',
#     'loop', 'background', 'timestamp', 'duration', 'progressive', 'progression',
#     'icc_profile', 'chromaticity', 'photoshop',
# }


def read_info_from_image(filepath) -> tuple[str | None, dict, str | None]:
    with Image.open(filepath) as image:
        items = (image.info or {}).copy()

    parameters = items.pop('parameters', None)
    if parameters is not None and is_json(parameters):
        parameters = json.loads(parameters)

    metadata_scheme = items.pop('fooocus_scheme', None)

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
