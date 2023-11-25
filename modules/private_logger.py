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
            existing_log = f'<p>Fooocus Log {date_string} (private)</p>\n<p>All images do not contain any hidden data.</p>'

    div_name = only_name.replace('.', '_')
    item = f'<div id="{div_name}">\n'
    item += f"<p>{only_name}</p>\n"
    for i, (k, v) in enumerate(dic):
        if i < single_line_number:
            item += f"<p>{k}: <b>{v}</b> </p>\n"
        else:
            if (i - single_line_number) % 2 == 0:
                item += f"<p>{k}: <b>{v}</b>, "
            else:
                item += f"{k}: <b>{v}</b></p>\n"
    item += f"<p><img src=\"{only_name}\" width=auto height=100% loading=lazy style=\"height:auto;max-width:512px\" onerror=\"document.getElementById('{div_name}').style.display = 'none';\"></img></p><hr></div>\n"
    existing_log = item + existing_log

    with open(html_name, 'w', encoding='utf-8') as f:
        f.write(existing_log)

    print(f'Image generated with private log at: {html_name}')

    log_cache[html_name] = existing_log

    return
