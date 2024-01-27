# based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/v1.6.0/modules/ui_gradio_extensions.py

import os
import gradio as gr
import args_manager
import json
from modules.model_previewer import get_all_previews_for_checkpoints, get_all_previews_for_loras

from modules.localization import localization_js


GradioTemplateResponseOriginal = gr.routes.templates.TemplateResponse

modules_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.dirname(modules_path)


def webpath(fn):
    if fn.startswith(script_path):
        web_path = os.path.relpath(fn, script_path).replace('\\', '/')
    else:
        web_path = os.path.abspath(fn)

    return f'file={web_path}?{os.path.getmtime(fn)}'


def javascript_html():
    script_js_path = webpath('javascript/script.js')
    context_menus_js_path = webpath('javascript/contextMenus.js')
    localization_js_path = webpath('javascript/localization.js')
    zoom_js_path = webpath('javascript/zoom.js')
    edit_attention_js_path = webpath('javascript/edit-attention.js')
    viewer_js_path = webpath('javascript/viewer.js')
    image_viewer_js_path = webpath('javascript/imageviewer.js')
    samples_path = webpath(os.path.abspath('./sdxl_styles/samples/fooocus_v2.jpg'))
    head = f'<script type="text/javascript">{localization_js(args_manager.args.language)}</script>\n'
    head += f'<script type="text/javascript" src="{script_js_path}"></script>\n'
    head += f'<script type="text/javascript" src="{context_menus_js_path}"></script>\n'
    head += f'<script type="text/javascript" src="{localization_js_path}"></script>\n'
    head += f'<script type="text/javascript" src="{zoom_js_path}"></script>\n'
    head += f'<script type="text/javascript" src="{edit_attention_js_path}"></script>\n'
    head += f'<script type="text/javascript" src="{viewer_js_path}"></script>\n'
    head += f'<script type="text/javascript" src="{image_viewer_js_path}"></script>\n'
    head += f'<meta name="samples-path" content="{samples_path}"></meta>\n'
    
    js_code = get_js_code_from_updated_previews()
    head += f"<script>{js_code}</script>\n"

    if args_manager.args.theme:
        head += f'<script type="text/javascript">set_theme(\"{args_manager.args.theme}\");</script>\n'

    return head

def get_js_code_from_updated_previews():
    # Fetch the updated previews data
    updated_previews_checkpoint = get_all_previews_for_checkpoints()
    updated_previews_lora = get_all_previews_for_loras()

    # Convert to JSON strings
    updated_previews_checkpoint_json = json.dumps(updated_previews_checkpoint)
    updated_previews_lora_json = json.dumps(updated_previews_lora)

    # Inject updated data into JavaScript
    return f"""
        previewsCheckpoint = {updated_previews_checkpoint_json};
        previewsLora = {updated_previews_lora_json};
    """

def css_html():
    style_css_path = webpath('css/style.css')
    head = f'<link rel="stylesheet" property="stylesheet" href="{style_css_path}">'
    return head


def reload_javascript():
    js = javascript_html()
    css = css_html()

    def template_response(*args, **kwargs):
        res = GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'</head>', f'{js}</head>'.encode("utf8"))
        res.body = res.body.replace(b'</body>', f'{css}</body>'.encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response
