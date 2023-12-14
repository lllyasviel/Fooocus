import os
import args_manager
import modules.config

from PIL import Image
from modules.util import generate_temp_filename


log_cache = {}


def get_current_html_path():
    date_string, local_temp_filename, only_name = generate_temp_filename(folder=modules.config.path_outputs,
                                                                         extension='png')
    html_name = os.path.join(os.path.dirname(local_temp_filename), 'log.html')
    return html_name


def log(img, dic, single_line_number=3):
    if args_manager.args.disable_image_log:
        return

    date_string, local_temp_filename, only_name = generate_temp_filename(folder=modules.config.path_outputs, extension='png')
    os.makedirs(os.path.dirname(local_temp_filename), exist_ok=True)
    Image.fromarray(img).save(local_temp_filename)
    html_name = os.path.join(os.path.dirname(local_temp_filename), 'log.html')

    css_styles = (
        "<style>"
        "body { background-color: #121212; color: #E0E0E0; } "
        "a { color: #BB86FC; } "
        ".metadata { border-collapse: collapse; width: 100%; } "
        ".metadata .key { width: 15%; } "
        ".metadata .value { width: 85%; font-weight: bold; } "
        ".metadata th, .metadata td { border: 1px solid #4d4d4d; padding: 4px; } "
        ".image-container img { height: auto; max-width: 512px; display: block; padding-right:10px; } "
        ".image-container div { text-align: center; padding: 4px; } "
        ".image-row { vertical-align: top; } "
        "</style>"
    )

    existing_log = log_cache.get(html_name, None)

    if existing_log is None:
        if os.path.exists(html_name):
            existing_log = open(html_name, 'r', encoding='utf-8').read()
        else:
            existing_log = f"<html><head>{css_styles}</head><body>"
            existing_log += f'\n<hr>\n<p>Fooocus Log {date_string} (private)</p>\n<p>All images do not contain any hidden data.</p>'

    div_name = only_name.replace('.', '_')
    item = f'<div id="{div_name}" class="image-container">\n'
    item += "<table><tr class='image-row'>"
    item += f"<td><a href='{only_name}'><img src='{only_name}' onerror=\"this.closest('.image-container').style.display='none';\" loading='lazy'></img></a><div>{only_name}</div></td>"
    item += "<td>"
    item += "<table class='metadata'>"

    if isinstance(dic, list):
        for item_tuple in dic:
            if len(item_tuple) == 2:
                key, value = item_tuple
                if key.startswith('LoRA [') and ']' in key:
                    lora_name = key[key.find('[') + 1 : key.find(']')]
                    rest_of_key = key[key.find(']') + 2:]
                    item += f"<tr><td class='key'>LoRA</td><td class='value'>{lora_name}: {value}</td></tr>\n"
                else:
                    item += f"<tr><td class='key'>{key}</td><td class='value'>{value}</td></tr>\n"

    item += "</table>"
    item += "</td>"
    item += "</tr></table></div>\n\n"
    existing_log = item + existing_log

    with open(html_name, 'w', encoding='utf-8') as f:
        f.write(existing_log)

    print(f'Image generated with private log at: {html_name}')

    log_cache[html_name] = existing_log

    return
