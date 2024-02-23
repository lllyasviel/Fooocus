import os
import args_manager
import modules.config
import json
import urllib.parse

from PIL import Image
from modules.util import generate_temp_filename

log_cache = {}

def get_current_html_path():
    date_string, local_temp_filename, only_name, logpath = generate_temp_filename(folder=modules.config.path_outputs, extension='png')
    html_name = os.path.join(os.path.dirname(local_temp_filename), 'log.html')
    return html_name


def log(img, dic, wildprompt=''):
    if args_manager.args.disable_image_log:
        return
    date_string, local_temp_filename, only_name, logpath = generate_temp_filename(folder=modules.config.path_outputs, extension='png', wildprompt=wildprompt)
    os.makedirs(os.path.dirname(local_temp_filename), exist_ok=True)
    Image.fromarray(img).save(local_temp_filename)
    html_name = os.path.join(os.path.dirname(logpath), 'log.html')

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
        "hr { border-color: gray; } "
        "button { background-color: black; color: white; border: 1px solid grey; border-radius: 5px; padding: 5px 10px; text-align: center; display: inline-block; font-size: 16px; cursor: pointer; }"
        "button:hover {background-color: grey; color: black;}"
        "#filters { display: flex; flex-wrap: wrap; gap: 2rem; padding: 2rem; }"
        ".filter-heading { font-weight: bold; font-size: 1.2rem; margin-bottom: 0.5em;}"
        "label { display: block; margin-bottom: 0.5em; cursor: pointer; }"
        "</style>"
    )

    js = (
        """
        <script>
            function to_clipboard(txt) { 
            txt = decodeURIComponent(txt);
            if (navigator.clipboard && navigator.permissions) {
                navigator.clipboard.writeText(txt)
            } else {
                const textArea = document.createElement('textArea')
                textArea.value = txt
                textArea.style.width = 0
                textArea.style.position = 'fixed'
                textArea.style.left = '-999px'
                textArea.style.top = '10px'
                textArea.setAttribute('readonly', 'readonly')
                document.body.appendChild(textArea)

                textArea.select()
                document.execCommand('copy')
                document.body.removeChild(textArea)
            }
            alert('Copied to Clipboard!\\nPaste to prompt area to load parameters.\\nCurrent clipboard content is:\\n\\n' + txt);
            }

            // Function to update visibility of log items based on filters
            function updateFilters() {
                var baseModelFilters = document.querySelectorAll('input[name="baseModelFilter"]:checked');
                var wildpromptFilters = document.querySelectorAll('input[name="wildpromptFilter"]:checked');
                
                // Loop through all log items
                var logItems = document.querySelectorAll('.image-container');
                logItems.forEach(function(item) {
                    var baseModel = item.getAttribute('data-model');
                    var wildpromptsAttr = item.getAttribute('data-wildprompts');
                    var wildprompts = [];
                    if (wildpromptsAttr && wildpromptsAttr != '[]') {
                        wildprompts = wildpromptsAttr.match(/'[^']+'/g);
                        if (wildprompts) {
                            wildprompts = wildprompts.map(function(item) {
                                return item.replace(/'/g, '');
                            });
                        }
                    }

                    var isVisible = true;
                    
                    // Check if base model filter is active
                    if (baseModelFilters.length > 0) {
                        var baseModelMatch = false;
                        baseModelFilters.forEach(function(filter) {
                            if (baseModel === filter.value) {
                                baseModelMatch = true;
                            }
                        });
                        if (!baseModelMatch) {
                            isVisible = false;
                        }
                    }
                    
                    // Check if wildprompt filter is active
                    if (wildpromptFilters.length > 0) {
                        var wildpromptMatch = false;
                        wildpromptFilters.forEach(function(filter) {
                            if (wildprompts.includes(filter.value)) {
                                wildpromptMatch = true;
                            }
                        });
                        if (!wildpromptMatch) {
                            isVisible = false;
                        }
                    }
                    
                    // Update visibility
                    if (isVisible) {
                        item.style.display = 'block';
                    } else {
                        item.style.display = 'none';
                    }
                });
            }
            
            // Function to initialize filters
            function initFilters() {
                // Base model filter
                var baseModels = {};
                var baseModelCheckboxes = document.getElementById('baseModelFilters');
                var baseModelOptions = document.querySelectorAll('.image-container');
                baseModelOptions.forEach(function(item) {
                    var baseModel = item.getAttribute('data-model');
                    if (!baseModels.hasOwnProperty(baseModel)) {
                        baseModels[baseModel] = 0;
                    }
                    baseModels[baseModel]++;
                });
                for (var model in baseModels) {
                    var checkbox = document.createElement('input');
                    checkbox.type = 'checkbox';
                    checkbox.name = 'baseModelFilter';
                    checkbox.value = model;
                    checkbox.addEventListener('change', updateFilters);
                    var label = document.createElement('label');
                    label.appendChild(checkbox);
                    label.appendChild(document.createTextNode(model + ' (' + baseModels[model] + ')'));
                    baseModelCheckboxes.appendChild(label);
                }
                
                // Wildprompt filter
                var wildpromptCheckboxes = document.getElementById('wildpromptFilters');
                var wildprompts = {};
                baseModelOptions.forEach(function(item) {
                    var wildpromptsAttr = item.getAttribute('data-wildprompts');
                    var prompts = [];
                    if (wildpromptsAttr && wildpromptsAttr != '[]') {
                        prompts = wildpromptsAttr.match(/'[^']+'/g).map(function(item) {
                            return item.replace(/'/g, '');
                        });
                    }
                    prompts.forEach(function(prompt) {
                        if (!wildprompts.hasOwnProperty(prompt)) {
                            wildprompts[prompt] = 0;
                        }
                        wildprompts[prompt]++;
                    });
                });
                for (var prompt in wildprompts) {
                    var checkbox = document.createElement('input');
                    checkbox.type = 'checkbox';
                    checkbox.name = 'wildpromptFilter';
                    checkbox.value = prompt;
                    checkbox.addEventListener('change', updateFilters);
                    var label = document.createElement('label');
                    label.appendChild(checkbox);
                    label.appendChild(document.createTextNode(prompt + ' (' + wildprompts[prompt] + ')'));
                    wildpromptCheckboxes.appendChild(label);
                }
            }
            
            // Initialize filters when the page is loaded
            window.addEventListener('load', initFilters);
        </script>
        """
    )

    begin_part = f"<!DOCTYPE html><html><head><title>Fooocus Log {date_string}</title>\n\n{css_styles}\n\n</head><body>\n\n{js}\n\n<div id=\"filters\"><div id=\"baseModelFilters\"><div class=\"filter-heading\">Base Model</div></div><div id=\"wildpromptFilters\"><div class=\"filter-heading\">Wildprompts</div></div>\n\n</div><!--fooocus-log-split-->\n\n"
    end_part = f'\n<!--fooocus-log-split--></body></html>'

    middle_part = log_cache.get(html_name, "")

    if middle_part == "":
        if os.path.exists(html_name):
            existing_split = open(html_name, 'r', encoding='utf-8').read().split('<!--fooocus-log-split-->')
            if len(existing_split) == 3:
                middle_part = existing_split[1]
            else:
                middle_part = existing_split[0]

    div_name = only_name.replace('.', '_')
    for key, value in dic:
        if key == 'Base Model':
            base_model = value
            break
    for key, value in dic:
        if key == 'Wildprompts':
            wildprompts = value
            break
    item = f"<div id=\"{div_name}\" class=\"image-container\" data-model=\"{base_model}\" data-wildprompts=\"{wildprompts}\"><hr><table><tr>\n"
    item += f"<td><a href=\"{only_name}\" target=\"_blank\"><img src='{only_name}' onerror=\"this.closest('.image-container').style.display='none';\" loading='lazy'/></a><div>{only_name}</div></td>"
    item += "<td><table class='metadata'>"
    for key, value in dic:
        value_txt = str(value).replace('\n', ' <br/> ')
        item += f"<tr><td class='key'>{key}</td><td class='value'>{value_txt}</td></tr>\n"
    item += "</table>"

    js_txt = urllib.parse.quote(json.dumps({k: v for k, v in dic}, indent=0), safe='')
    item += f"<br/><button onclick=\"to_clipboard('{js_txt}')\">Copy to Clipboard</button>"

    item += "</td>"
    item += "</tr></table></div>\n\n"

    middle_part = item + middle_part

    with open(html_name, 'w', encoding='utf-8') as f:
        f.write(begin_part + middle_part + end_part)

    print(f'Image generated with private log at: {html_name}')

    log_cache[html_name] = middle_part

    return
