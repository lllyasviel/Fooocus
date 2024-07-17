import json
import re
from abc import ABC, abstractmethod
from pathlib import Path

import gradio as gr
from PIL import Image

import fooocus_version
import modules.config
import modules.sdxl_styles
from modules.flags import MetadataScheme, Performance, Steps
from modules.flags import SAMPLERS, CIVITAI_NO_KARRAS
from modules.hash_cache import sha256_from_cache
from modules.util import quote, unquote, extract_styles_from_prompt, is_json, get_file_from_folder_list

re_param_code = r'\s*(\w[\w \-/]+):\s*("(?:\\.|[^\\"])+"|[^,]*)(?:,|$)'
re_param = re.compile(re_param_code)
re_imagesize = re.compile(r"^(\d+)x(\d+)$")


def load_parameter_button_click(raw_metadata: dict | str, is_generating: bool, inpaint_mode: str):
    loaded_parameter_dict = raw_metadata
    if isinstance(raw_metadata, str):
        loaded_parameter_dict = json.loads(raw_metadata)
    assert isinstance(loaded_parameter_dict, dict)

    results = [len(loaded_parameter_dict) > 0]

    get_image_number('image_number', 'Image Number', loaded_parameter_dict, results)
    get_str('prompt', 'Prompt', loaded_parameter_dict, results)
    get_str('negative_prompt', 'Negative Prompt', loaded_parameter_dict, results)
    get_list('styles', 'Styles', loaded_parameter_dict, results)
    performance = get_str('performance', 'Performance', loaded_parameter_dict, results)
    get_steps('steps', 'Steps', loaded_parameter_dict, results)
    get_number('overwrite_switch', 'Overwrite Switch', loaded_parameter_dict, results)
    get_resolution('resolution', 'Resolution', loaded_parameter_dict, results)
    get_number('guidance_scale', 'Guidance Scale', loaded_parameter_dict, results)
    get_number('sharpness', 'Sharpness', loaded_parameter_dict, results)
    get_adm_guidance('adm_guidance', 'ADM Guidance', loaded_parameter_dict, results)
    get_str('refiner_swap_method', 'Refiner Swap Method', loaded_parameter_dict, results)
    get_number('adaptive_cfg', 'CFG Mimicking from TSNR', loaded_parameter_dict, results)
    get_number('clip_skip', 'CLIP Skip', loaded_parameter_dict, results, cast_type=int)
    get_str('base_model', 'Base Model', loaded_parameter_dict, results)
    get_str('refiner_model', 'Refiner Model', loaded_parameter_dict, results)
    get_number('refiner_switch', 'Refiner Switch', loaded_parameter_dict, results)
    get_str('sampler', 'Sampler', loaded_parameter_dict, results)
    get_str('scheduler', 'Scheduler', loaded_parameter_dict, results)
    get_str('vae', 'VAE', loaded_parameter_dict, results)
    get_seed('seed', 'Seed', loaded_parameter_dict, results)
    get_inpaint_engine_version('inpaint_engine_version', 'Inpaint Engine Version', loaded_parameter_dict, results, inpaint_mode)
    get_inpaint_method('inpaint_method', 'Inpaint Mode', loaded_parameter_dict, results)

    if is_generating:
        results.append(gr.update())
    else:
        results.append(gr.update(visible=True))

    results.append(gr.update(visible=False))

    get_freeu('freeu', 'FreeU', loaded_parameter_dict, results)

    # prevent performance LoRAs to be added twice, by performance and by lora
    performance_filename = None
    if performance is not None and performance in Performance.values():
        performance = Performance(performance)
        performance_filename = performance.lora_filename()

    for i in range(modules.config.default_max_lora_number):
        get_lora(f'lora_combined_{i + 1}', f'LoRA {i + 1}', loaded_parameter_dict, results, performance_filename)

    return results


def get_str(key: str, fallback: str | None, source_dict: dict, results: list, default=None) -> str | None:
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert isinstance(h, str)
        results.append(h)
        return h
    except:
        results.append(gr.update())
        return None


def get_list(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        h = eval(h)
        assert isinstance(h, list)
        results.append(h)
    except:
        results.append(gr.update())


def get_number(key: str, fallback: str | None, source_dict: dict, results: list, default=None, cast_type=float):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert h is not None
        h = cast_type(h)
        results.append(h)
    except:
        results.append(gr.update())


def get_image_number(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert h is not None
        h = int(h)
        h = min(h, modules.config.default_max_image_number)
        results.append(h)
    except:
        results.append(1)


def get_steps(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert h is not None
        h = int(h)
        # if not in steps or in steps and performance is not the same
        performance_name = source_dict.get('performance', '').replace(' ', '_').replace('-', '_').casefold()
        performance_candidates = [key for key in Steps.keys() if key.casefold() == performance_name and Steps[key] == h]
        if len(performance_candidates) == 0:
            results.append(h)
            return
        results.append(-1)
    except:
        results.append(-1)


def get_resolution(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        width, height = eval(h)
        formatted = modules.config.add_ratio(f'{width}*{height}')
        if formatted in modules.config.available_aspect_ratios_labels:
            results.append(formatted)
            results.append(-1)
            results.append(-1)
        else:
            results.append(gr.update())
            results.append(int(width))
            results.append(int(height))
    except:
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())


def get_seed(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert h is not None
        h = int(h)
        results.append(False)
        results.append(h)
    except:
        results.append(gr.update())
        results.append(gr.update())


def get_inpaint_engine_version(key: str, fallback: str | None, source_dict: dict, results: list, inpaint_mode: str, default=None) -> str | None:
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert isinstance(h, str) and h in modules.flags.inpaint_engine_versions
        if inpaint_mode != modules.flags.inpaint_option_detail:
            results.append(h)
        else:
            results.append(gr.update())
        results.append(h)
        return h
    except:
        results.append(gr.update())
        results.append('empty')
        return None


def get_inpaint_method(key: str, fallback: str | None, source_dict: dict, results: list, default=None) -> str | None:
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert isinstance(h, str) and h in modules.flags.inpaint_options
        results.append(h)
        for i in range(modules.config.default_enhance_tabs):
            results.append(h)
        return h
    except:
        results.append(gr.update())
        for i in range(modules.config.default_enhance_tabs):
            results.append(gr.update())


def get_adm_guidance(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        p, n, e = eval(h)
        results.append(float(p))
        results.append(float(n))
        results.append(float(e))
    except:
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())


def get_freeu(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        b1, b2, s1, s2 = eval(h)
        results.append(True)
        results.append(float(b1))
        results.append(float(b2))
        results.append(float(s1))
        results.append(float(s2))
    except:
        results.append(False)
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())


def get_lora(key: str, fallback: str | None, source_dict: dict, results: list, performance_filename: str | None):
    try:
        split_data = source_dict.get(key, source_dict.get(fallback)).split(' : ')
        enabled = True
        name = split_data[0]
        weight = split_data[1]

        if len(split_data) == 3:
            enabled = split_data[0] == 'True'
            name = split_data[1]
            weight = split_data[2]

        if name == performance_filename:
            raise Exception

        weight = float(weight)
        results.append(enabled)
        results.append(name)
        results.append(weight)
    except:
        results.append(True)
        results.append('None')
        results.append(1)


def parse_meta_from_preset(preset_content):
    assert isinstance(preset_content, dict)
    preset_prepared = {}
    items = preset_content

    for settings_key, meta_key in modules.config.possible_preset_keys.items():
        if settings_key == "default_loras":
            loras = getattr(modules.config, settings_key)
            if settings_key in items:
                loras = items[settings_key]
            for index, lora in enumerate(loras[:modules.config.default_max_lora_number]):
                preset_prepared[f'lora_combined_{index + 1}'] = ' : '.join(map(str, lora))
        elif settings_key == "default_aspect_ratio":
            if settings_key in items and items[settings_key] is not None:
                default_aspect_ratio = items[settings_key]
                width, height = default_aspect_ratio.split('*')
            else:
                default_aspect_ratio = getattr(modules.config, settings_key)
                width, height = default_aspect_ratio.split('×')
                height = height[:height.index(" ")]
            preset_prepared[meta_key] = (width, height)
        else:
            preset_prepared[meta_key] = items[settings_key] if settings_key in items and items[settings_key] is not None else getattr(modules.config, settings_key)

        if settings_key == "default_styles" or settings_key == "default_aspect_ratio":
            preset_prepared[meta_key] = str(preset_prepared[meta_key])

    return preset_prepared


class MetadataParser(ABC):
    def __init__(self):
        self.raw_prompt: str = ''
        self.full_prompt: str = ''
        self.raw_negative_prompt: str = ''
        self.full_negative_prompt: str = ''
        self.steps: int = Steps.SPEED.value
        self.base_model_name: str = ''
        self.base_model_hash: str = ''
        self.refiner_model_name: str = ''
        self.refiner_model_hash: str = ''
        self.loras: list = []
        self.vae_name: str = ''

    @abstractmethod
    def get_scheme(self) -> MetadataScheme:
        raise NotImplementedError

    @abstractmethod
    def to_json(self, metadata: dict | str) -> dict:
        raise NotImplementedError

    @abstractmethod
    def to_string(self, metadata: dict) -> str:
        raise NotImplementedError

    def set_data(self, raw_prompt, full_prompt, raw_negative_prompt, full_negative_prompt, steps, base_model_name,
                 refiner_model_name, loras, vae_name):
        self.raw_prompt = raw_prompt
        self.full_prompt = full_prompt
        self.raw_negative_prompt = raw_negative_prompt
        self.full_negative_prompt = full_negative_prompt
        self.steps = steps
        self.base_model_name = Path(base_model_name).stem

        base_model_path = get_file_from_folder_list(base_model_name, modules.config.paths_checkpoints)
        self.base_model_hash = sha256_from_cache(base_model_path)

        if refiner_model_name not in ['', 'None']:
            self.refiner_model_name = Path(refiner_model_name).stem
            refiner_model_path = get_file_from_folder_list(refiner_model_name, modules.config.paths_checkpoints)
            self.refiner_model_hash = sha256_from_cache(refiner_model_path)

        self.loras = []
        for (lora_name, lora_weight) in loras:
            if lora_name != 'None':
                lora_path = get_file_from_folder_list(lora_name, modules.config.paths_loras)
                lora_hash = sha256_from_cache(lora_path)
                self.loras.append((Path(lora_name).stem, lora_weight, lora_hash))
        self.vae_name = Path(vae_name).stem


class A1111MetadataParser(MetadataParser):
    def get_scheme(self) -> MetadataScheme:
        return MetadataScheme.A1111

    fooocus_to_a1111 = {
        'raw_prompt': 'Raw prompt',
        'raw_negative_prompt': 'Raw negative prompt',
        'negative_prompt': 'Negative prompt',
        'styles': 'Styles',
        'performance': 'Performance',
        'steps': 'Steps',
        'sampler': 'Sampler',
        'scheduler': 'Scheduler',
        'vae': 'VAE',
        'guidance_scale': 'CFG scale',
        'seed': 'Seed',
        'resolution': 'Size',
        'sharpness': 'Sharpness',
        'adm_guidance': 'ADM Guidance',
        'refiner_swap_method': 'Refiner Swap Method',
        'adaptive_cfg': 'Adaptive CFG',
        'clip_skip': 'Clip skip',
        'overwrite_switch': 'Overwrite Switch',
        'freeu': 'FreeU',
        'base_model': 'Model',
        'base_model_hash': 'Model hash',
        'refiner_model': 'Refiner',
        'refiner_model_hash': 'Refiner hash',
        'lora_hashes': 'Lora hashes',
        'lora_weights': 'Lora weights',
        'created_by': 'User',
        'version': 'Version'
    }

    def to_json(self, metadata: str) -> dict:
        metadata_prompt = ''
        metadata_negative_prompt = ''

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
                metadata_negative_prompt += ('' if metadata_negative_prompt == '' else "\n") + line
            else:
                metadata_prompt += ('' if metadata_prompt == '' else "\n") + line

        found_styles, prompt, negative_prompt = extract_styles_from_prompt(metadata_prompt, metadata_negative_prompt)

        data = {
            'prompt': prompt,
            'negative_prompt': negative_prompt
        }

        for k, v in re_param.findall(lastline):
            try:
                if v != '' and v[0] == '"' and v[-1] == '"':
                    v = unquote(v)

                m = re_imagesize.match(v)
                if m is not None:
                    data['resolution'] = str((m.group(1), m.group(2)))
                else:
                    data[list(self.fooocus_to_a1111.keys())[list(self.fooocus_to_a1111.values()).index(k)]] = v
            except Exception:
                print(f"Error parsing \"{k}: {v}\"")

        # workaround for multiline prompts
        if 'raw_prompt' in data:
            data['prompt'] = data['raw_prompt']
            raw_prompt = data['raw_prompt'].replace("\n", ', ')
            if metadata_prompt != raw_prompt and modules.sdxl_styles.fooocus_expansion not in found_styles:
                found_styles.append(modules.sdxl_styles.fooocus_expansion)

        if 'raw_negative_prompt' in data:
            data['negative_prompt'] = data['raw_negative_prompt']

        data['styles'] = str(found_styles)

        # try to load performance based on steps, fallback for direct A1111 imports
        if 'steps' in data and 'performance' in data is None:
            try:
                data['performance'] = Performance.by_steps(data['steps']).value
            except ValueError | KeyError:
                pass

        if 'sampler' in data:
            data['sampler'] = data['sampler'].replace(' Karras', '')
            # get key
            for k, v in SAMPLERS.items():
                if v == data['sampler']:
                    data['sampler'] = k
                    break

        for key in ['base_model', 'refiner_model', 'vae']:
            if key in data:
                if key == 'vae':
                    self.add_extension_to_filename(data, modules.config.vae_filenames, 'vae')
                else:
                    self.add_extension_to_filename(data, modules.config.model_filenames, key)

        lora_data = ''
        if 'lora_weights' in data and data['lora_weights'] != '':
            lora_data = data['lora_weights']
        elif 'lora_hashes' in data and data['lora_hashes'] != '' and data['lora_hashes'].split(', ')[0].count(':') == 2:
            lora_data = data['lora_hashes']

        if lora_data != '':
            for li, lora in enumerate(lora_data.split(', ')):
                lora_split = lora.split(': ')
                lora_name = lora_split[0]
                lora_weight = lora_split[2] if len(lora_split) == 3 else lora_split[1]
                for filename in modules.config.lora_filenames:
                    path = Path(filename)
                    if lora_name == path.stem:
                        data[f'lora_combined_{li + 1}'] = f'{filename} : {lora_weight}'
                        break

        return data

    def to_string(self, metadata: dict) -> str:
        data = {k: v for _, k, v in metadata}

        width, height = eval(data['resolution'])

        sampler = data['sampler']
        scheduler = data['scheduler']

        if sampler in SAMPLERS and SAMPLERS[sampler] != '':
            sampler = SAMPLERS[sampler]
            if sampler not in CIVITAI_NO_KARRAS and scheduler == 'karras':
                sampler += f' Karras'

        generation_params = {
            self.fooocus_to_a1111['steps']: self.steps,
            self.fooocus_to_a1111['sampler']: sampler,
            self.fooocus_to_a1111['seed']: data['seed'],
            self.fooocus_to_a1111['resolution']: f'{width}x{height}',
            self.fooocus_to_a1111['guidance_scale']: data['guidance_scale'],
            self.fooocus_to_a1111['sharpness']: data['sharpness'],
            self.fooocus_to_a1111['adm_guidance']: data['adm_guidance'],
            self.fooocus_to_a1111['base_model']: Path(data['base_model']).stem,
            self.fooocus_to_a1111['base_model_hash']: self.base_model_hash,

            self.fooocus_to_a1111['performance']: data['performance'],
            self.fooocus_to_a1111['scheduler']: scheduler,
            self.fooocus_to_a1111['vae']: Path(data['vae']).stem,
            # workaround for multiline prompts
            self.fooocus_to_a1111['raw_prompt']: self.raw_prompt,
            self.fooocus_to_a1111['raw_negative_prompt']: self.raw_negative_prompt,
        }

        if self.refiner_model_name not in ['', 'None']:
            generation_params |= {
                self.fooocus_to_a1111['refiner_model']: self.refiner_model_name,
                self.fooocus_to_a1111['refiner_model_hash']: self.refiner_model_hash
            }

        for key in ['adaptive_cfg', 'clip_skip', 'overwrite_switch', 'refiner_swap_method', 'freeu']:
            if key in data:
                generation_params[self.fooocus_to_a1111[key]] = data[key]

        if len(self.loras) > 0:
            lora_hashes = []
            lora_weights = []
            for index, (lora_name, lora_weight, lora_hash) in enumerate(self.loras):
                # workaround for Fooocus not knowing LoRA name in LoRA metadata
                lora_hashes.append(f'{lora_name}: {lora_hash}')
                lora_weights.append(f'{lora_name}: {lora_weight}')
            lora_hashes_string = ', '.join(lora_hashes)
            lora_weights_string = ', '.join(lora_weights)
            generation_params[self.fooocus_to_a1111['lora_hashes']] = lora_hashes_string
            generation_params[self.fooocus_to_a1111['lora_weights']] = lora_weights_string

        generation_params[self.fooocus_to_a1111['version']] = data['version']

        if modules.config.metadata_created_by != '':
            generation_params[self.fooocus_to_a1111['created_by']] = modules.config.metadata_created_by

        generation_params_text = ", ".join(
            [k if k == v else f'{k}: {quote(v)}' for k, v in generation_params.items() if
             v is not None])
        positive_prompt_resolved = ', '.join(self.full_prompt)
        negative_prompt_resolved = ', '.join(self.full_negative_prompt)
        negative_prompt_text = f"\nNegative prompt: {negative_prompt_resolved}" if negative_prompt_resolved else ""
        return f"{positive_prompt_resolved}{negative_prompt_text}\n{generation_params_text}".strip()

    @staticmethod
    def add_extension_to_filename(data, filenames, key):
        for filename in filenames:
            path = Path(filename)
            if data[key] == path.stem:
                data[key] = filename
                break


class FooocusMetadataParser(MetadataParser):
    def get_scheme(self) -> MetadataScheme:
        return MetadataScheme.FOOOCUS

    def to_json(self, metadata: dict) -> dict:
        for key, value in metadata.items():
            if value in ['', 'None']:
                continue
            if key in ['base_model', 'refiner_model']:
                metadata[key] = self.replace_value_with_filename(key, value, modules.config.model_filenames)
            elif key.startswith('lora_combined_'):
                metadata[key] = self.replace_value_with_filename(key, value, modules.config.lora_filenames)
            elif key == 'vae':
                metadata[key] = self.replace_value_with_filename(key, value, modules.config.vae_filenames)
            else:
                continue

        return metadata

    def to_string(self, metadata: list) -> str:
        for li, (label, key, value) in enumerate(metadata):
            # remove model folder paths from metadata
            if key.startswith('lora_combined_'):
                name, weight = value.split(' : ')
                name = Path(name).stem
                value = f'{name} : {weight}'
                metadata[li] = (label, key, value)

        res = {k: v for _, k, v in metadata}

        res['full_prompt'] = self.full_prompt
        res['full_negative_prompt'] = self.full_negative_prompt
        res['steps'] = self.steps
        res['base_model'] = self.base_model_name
        res['base_model_hash'] = self.base_model_hash

        if self.refiner_model_name not in ['', 'None']:
            res['refiner_model'] = self.refiner_model_name
            res['refiner_model_hash'] = self.refiner_model_hash

        res['vae'] = self.vae_name
        res['loras'] = self.loras

        if modules.config.metadata_created_by != '':
            res['created_by'] = modules.config.metadata_created_by

        return json.dumps(dict(sorted(res.items())))

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

        return None


def get_metadata_parser(metadata_scheme: MetadataScheme) -> MetadataParser:
    match metadata_scheme:
        case MetadataScheme.FOOOCUS:
            return FooocusMetadataParser()
        case MetadataScheme.A1111:
            return A1111MetadataParser()
        case _:
            raise NotImplementedError


def read_info_from_image(file) -> tuple[str | None, MetadataScheme | None]:
    items = (file.info or {}).copy()

    parameters = items.pop('parameters', None)
    metadata_scheme = items.pop('fooocus_scheme', None)
    exif = items.pop('exif', None)

    if parameters is not None and is_json(parameters):
        parameters = json.loads(parameters)
    elif exif is not None:
        exif = file.getexif()
        # 0x9286 = UserComment
        parameters = exif.get(0x9286, None)
        # 0x927C = MakerNote
        metadata_scheme = exif.get(0x927C, None)

        if is_json(parameters):
            parameters = json.loads(parameters)

    try:
        metadata_scheme = MetadataScheme(metadata_scheme)
    except ValueError:
        metadata_scheme = None

        # broad fallback
        if isinstance(parameters, dict):
            metadata_scheme = MetadataScheme.FOOOCUS

        if isinstance(parameters, str):
            metadata_scheme = MetadataScheme.A1111

    return parameters, metadata_scheme


def get_exif(metadata: str | None, metadata_scheme: str):
    exif = Image.Exif()
    # tags see see https://github.com/python-pillow/Pillow/blob/9.2.x/src/PIL/ExifTags.py
    # 0x9286 = UserComment
    exif[0x9286] = metadata
    # 0x0131 = Software
    exif[0x0131] = 'Fooocus v' + fooocus_version.version
    # 0x927C = MakerNote
    exif[0x927C] = metadata_scheme
    return exif
