import json
import gradio as gr
import modules.config


def load_parameter_button_click(raw_prompt_txt, is_generating):
    loaded_parameter_dict = json.loads(raw_prompt_txt)
    assert isinstance(loaded_parameter_dict, dict)

    results = [True, 1]

    try:
        h = loaded_parameter_dict.get('Prompt', None)
        assert isinstance(h, str)
        results.append(h)
    except:
        results.append(gr.update())

    try:
        h = loaded_parameter_dict.get('Negative Prompt', None)
        assert isinstance(h, str)
        results.append(h)
    except:
        results.append(gr.update())

    try:
        h = loaded_parameter_dict.get('Styles', None)
        h = eval(h)
        assert isinstance(h, list)
        results.append(h)
    except:
        results.append(gr.update())

    try:
        h = loaded_parameter_dict.get('Performance', None)
        assert isinstance(h, str)
        results.append(h)
    except:
        results.append(gr.update())

    try:
        h = loaded_parameter_dict.get('Resolution', None)
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

    try:
        h = loaded_parameter_dict.get('Sharpness', None)
        assert h is not None
        h = float(h)
        results.append(h)
    except:
        results.append(gr.update())

    try:
        h = loaded_parameter_dict.get('Guidance Scale', None)
        assert h is not None
        h = float(h)
        results.append(h)
    except:
        results.append(gr.update())

    try:
        h = loaded_parameter_dict.get('ADM Guidance', None)
        p, n, e = eval(h)
        results.append(float(p))
        results.append(float(n))
        results.append(float(e))
    except:
        results.append(gr.update())
        results.append(gr.update())
        results.append(gr.update())

    try:
        h = loaded_parameter_dict.get('Base Model', None)
        assert isinstance(h, str)
        results.append(h)
    except:
        results.append(gr.update())

    try:
        h = loaded_parameter_dict.get('Refiner Model', None)
        assert isinstance(h, str)
        results.append(h)
    except:
        results.append(gr.update())

    try:
        h = loaded_parameter_dict.get('Refiner Switch', None)
        assert h is not None
        h = float(h)
        results.append(h)
    except:
        results.append(gr.update())

    try:
        h = loaded_parameter_dict.get('Sampler', None)
        assert isinstance(h, str)
        results.append(h)
    except:
        results.append(gr.update())

    try:
        h = loaded_parameter_dict.get('Scheduler', None)
        assert isinstance(h, str)
        results.append(h)
    except:
        results.append(gr.update())

    try:
        h = loaded_parameter_dict.get('Seed', None)
        assert h is not None
        h = int(h)
        results.append(False)
        results.append(h)
    except:
        results.append(gr.update())
        results.append(gr.update())

    if is_generating:
        results.append(gr.update())
    else:
        results.append(gr.update(visible=True))
    
    results.append(gr.update(visible=False))

    for i in range(1, 6):
        try:
            n, w = loaded_parameter_dict.get(f'LoRA {i}').split(' : ')
            w = float(w)
            results.append(n)
            results.append(w)
        except:
            results.append(gr.update())
            results.append(gr.update())

    return results
