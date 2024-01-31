import json

import gradio as gr

import modules.config
from modules.flags import lora_count, Steps


def load_parameter_button_click(raw_metadata: dict | str, is_generating: bool):
    loaded_parameter_dict = raw_metadata
    if isinstance(raw_metadata, str):
        loaded_parameter_dict = json.loads(raw_metadata)
    assert isinstance(loaded_parameter_dict, dict)

    results = [True, 1]

    get_str('prompt', 'Prompt', loaded_parameter_dict, results)
    get_str('negative_prompt', 'Negative Prompt', loaded_parameter_dict, results)
    get_list('styles', 'Styles', loaded_parameter_dict, results)
    get_str('performance', 'Performance', loaded_parameter_dict, results)
    get_steps('steps', 'Steps', loaded_parameter_dict, results)
    get_float('overwrite_switch', 'Overwrite Switch', loaded_parameter_dict, results)
    get_resolution('resolution', 'Resolution', loaded_parameter_dict, results)
    get_float('guidance_scale', 'Guidance Scale', loaded_parameter_dict, results)
    get_float('sharpness', 'Sharpness', loaded_parameter_dict, results)
    get_adm_guidance('adm_guidance', 'ADM Guidance', loaded_parameter_dict, results)
    get_str('refiner_swap_method', 'Refiner Swap Method', loaded_parameter_dict, results)
    get_str('adaptive_cfg', 'CFG Mimicking from TSNR', loaded_parameter_dict, results)
    get_str('base_model', 'Base Model', loaded_parameter_dict, results)
    get_str('refiner_model', 'Refiner Model', loaded_parameter_dict, results)
    get_float('refiner_switch', 'Refiner Switch', loaded_parameter_dict, results)
    get_str('sampler', 'Sampler', loaded_parameter_dict, results)
    get_str('scheduler', 'Scheduler', loaded_parameter_dict, results)
    get_seed('seed', 'Seed', loaded_parameter_dict, results)

    if is_generating:
        results.append(gr.update())
    else:
        results.append(gr.update(visible=True))

    results.append(gr.update(visible=False))

    get_freeu('freeu', 'FreeU', loaded_parameter_dict, results)

    for i in range(lora_count):
        get_lora(f'lora_combined_{i + 1}', f'LoRA {i + 1}', loaded_parameter_dict, results)

    return results


def get_str(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert isinstance(h, str)
        results.append(h)
    except:
        results.append(gr.update())


def get_list(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        h = eval(h)
        assert isinstance(h, list)
        results.append(h)
    except:
        results.append(gr.update())


def get_float(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert h is not None
        h = float(h)
        results.append(h)
    except:
        results.append(gr.update())


def get_steps(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, source_dict.get(fallback, default))
        assert h is not None
        h = int(h)
        if h not in set(item.value for item in Steps):
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
        if formatted in modules.config.available_aspect_ratios:
            results.append(formatted)
            results.append(-1)
            results.append(-1)
        else:
            results.append(gr.update())
            results.append(width)
            results.append(height)
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


def get_lora(key: str, fallback: str | None, source_dict: dict, results: list):
    try:
        n, w = source_dict.get(key, source_dict.get(fallback)).split(' : ')
        w = float(w)
        results.append(n)
        results.append(w)
    except:
        results.append('None')
        results.append(1)
