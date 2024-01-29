import json

import gradio as gr

import modules.config
from modules.flags import lora_count_with_lcm


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
    get_resolution('resolution', 'Resolution', loaded_parameter_dict, results)
    get_float('sharpness', 'Sharpness', loaded_parameter_dict, results)
    get_float('guidance_scale', 'Guidance Scale', loaded_parameter_dict, results)
    get_adm_guidance('adm_guidance', 'ADM Guidance', loaded_parameter_dict, results)
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

    for i in range(1, lora_count_with_lcm):
        try:
            n, w = loaded_parameter_dict.get(f'LoRA {i}').split(' : ')
            w = float(w)
            results.append(n)
            results.append(w)
        except:
            results.append(gr.update())
            results.append(gr.update())

    return results


def get_str(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, default)
        assert isinstance(h, str)
        results.append(h)
    except:
        if fallback is not None:
            get_str(fallback, None, source_dict, results, default)
            return
        results.append(gr.update())


def get_list(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, default)
        h = eval(h)
        assert isinstance(h, list)
        results.append(h)
    except:
        if fallback is not None:
            get_list(fallback, None, source_dict, results, default)
            return
        results.append(gr.update())


def get_float(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, default)
        assert h is not None
        h = float(h)
        results.append(h)
    except:
        if fallback is not None:
            get_float(fallback, None, source_dict, results, default)
            return
        results.append(gr.update())


def get_resolution(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, default)
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
        if fallback is not None:
            get_resolution(fallback, None, source_dict, results, default)
            return
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())


def get_seed(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, default)
        assert h is not None
        h = int(h)
        results.append(False)
        results.append(h)
    except:
        if fallback is not None:
            get_seed(fallback, None, source_dict, results, default)
            return
        results.append(gr.update())
        results.append(gr.update())


def get_adm_guidance(key: str, fallback: str | None, source_dict: dict, results: list, default=None):
    try:
        h = source_dict.get(key, default)
        p, n, e = eval(h)
        results.append(float(p))
        results.append(float(n))
        results.append(float(e))
    except:
        if fallback is not None:
            get_adm_guidance(fallback, None, source_dict, results, default)
            return
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())
