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

    existing_log = log_cache.get(html_name, None)

    if existing_log is None:
        if os.path.exists(html_name):
            existing_log = open(html_name, encoding='utf-8').read()
        else:
            existing_log = "<html><head><style>body { background-color: #121212; color: #E0E0E0; } a { color: #BB86FC; } table.metadata { border-collapse: collapse; } table.metadata th, table.metadata td { border: 1px solid #4d4d4d; } </style></head><body>"
            existing_log += f'<p>Fooocus Log {date_string} (private)</p>\n<p>All images do not contain any hidden data.</p>'

    div_name = only_name.replace('.', '_')
    item = f'<div id="{div_name}">\n'
    item += "<table><tr>"
    item += f"<td style='text-align: center;'><a href='{only_name}'><img src='{only_name}' width='auto' height='100%' loading='lazy' style='height:auto;max-width:512px; display:block;'></img></a><div style='text-align: center; padding: 4px'>{only_name}</div></td>"
    item += f"<td style='padding-left:10px;'>"
    item += "<table class='metadata'>"

    if isinstance(dic, list):
        for item_tuple in dic:
            if len(item_tuple) == 2:  # Ensure there is a key and a value
                key, value = item_tuple
                if key.startswith('LoRA [') and ']' in key:
                    lora_name = key[key.find('[') + 1 : key.find(']')]
                    rest_of_key = key[key.find(']') + 2:]
                    item += f"<tr><td style='padding: 4px; width: 15%;'>LoRA</td><td style='padding: 4px; width: 85%;'><b>{lora_name}: {value}</b></td></tr>"
                else:
                    item += f"<tr><td style='padding: 4px; width: 15%;'>{key}</td><td style='padding: 4px; width: 85%;'><b>{value}</b></td></tr>"

    item += "</table>"
    item += "</td>"
    item += "</tr></table><hr></div>\n"
    existing_log = item + existing_log

    existing_log += "</body></html>"

    with open(html_name, 'w', encoding='utf-8') as f:
        f.write(existing_log)

    print(f'Image generated with private log at: {html_name}')

    log_cache[html_name] = existing_log

    return
