# based on https://github.com/AUTOMATIC1111/stable-diffusion-webui/blob/v1.6.0/modules/ui_gradio_extensions.py

import os
import gradio as gr
import args_manager

from modules.localization import localization_js


GradioTemplateResponseOriginal = gr.routes.templates.TemplateResponse

modules_path = os.path.dirname(os.path.realpath(__file__))
script_path = os.path.dirname(modules_path)


def webpath(fn):
    if fn.startswith(script_path):
        web_path = os.path.relpath(fn, script_path).replace('\\', '/')
    else:
        web_path = os.path.abspath(fn)

    return f'file={web_path}?{os.path.getmtime(fn)}', web_path


def javascript_html():
    head = f'<script type="text/javascript">{localization_js(args_manager.args.language)}</script>\n'
    allowed_paths = []
    
    for path in ['javascript/script.js', 'javascript/contextMenus.js', 'javascript/localization.js', \
            'javascript/zoom.js', 'javascript/edit-attention.js', 'javascript/viewer.js', \
                'javascript/imageviewer.js']:
        web_path, allowed_path = webpath(path)
        head += f'<script type="text/javascript" src="{web_path}"></script>\n'
        allowed_paths.append(allowed_path)

    return head, allowed_paths


def css_html():
    head = ''
    allowed_paths = []

    for path in ['css/style.css']:
        web_path, allowed_path = webpath(path)
        head += f'<link rel="stylesheet" property="stylesheet" href="{web_path}">'
        allowed_paths.append(allowed_path)
    
    return head, allowed_paths


def reload_javascript():
    js, js_allowed_paths = javascript_html()
    css, css_allowed_paths = css_html()
    
    def template_response(*args, **kwargs):
        res = GradioTemplateResponseOriginal(*args, **kwargs)
        res.body = res.body.replace(b'</head>', f'{js}</head>'.encode("utf8"))
        res.body = res.body.replace(b'</body>', f'{css}</body>'.encode("utf8"))
        res.init_headers()
        return res

    gr.routes.templates.TemplateResponse = template_response
    
    return js_allowed_paths + css_allowed_paths
