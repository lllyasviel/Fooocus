import os
import json

from modules.util import get_files_from_folder


# cannot use modules.path - validators causing circular imports
styles_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../sdxl_styles/'))


default_styles_files = ['sdxl_styles_fooocus.json', 'sdxl_styles_sai.json', 'sdxl_styles_twri.json', 'sdxl_styles_diva.json', 'sdxl_styles_mre.json']


def normalize_key(k):
    k = k.replace('-', ' ')
    words = k.split(' ')
    words = [w[:1].upper() + w[1:].lower() for w in words]
    k = ' '.join(words)
    k = k.replace('3d', '3D')
    k = k.replace('Sai', 'SAI')
    k = k.replace('Mre', 'MRE')
    k = k.replace('(s', '(S')
    return k


styles = {}


def styles_list_to_styles_dict(styles_list=None, base_dict=None):
    styles_dict = {} if base_dict == None else base_dict
    if isinstance(styles_list, list) and len(styles_list) > 0:
        for entry in styles_list:
            name, prompt, negative_prompt = normalize_key(entry['name']), entry['prompt'], entry['negative_prompt']
            if name not in styles_dict:
                styles_dict |=  {name: (prompt, negative_prompt)}
    return styles_dict


def load_styles(filename=None, base_dict=None):
    styles_dict = {} if base_dict == None else base_dict
    full_path = os.path.join(styles_path, filename) if filename != None else None
    if full_path != None and os.path.exists(full_path):
        with open(full_path, encoding='utf-8') as styles_file:
            try:
                styles_obj = json.load(styles_file)
                styles_list_to_styles_dict(styles_obj, styles_dict)
            except Exception as e:
                print('load_styles, e: ' + str(e))
            finally:
                styles_file.close()
    return styles_dict


for styles_file in default_styles_files:
    styles = load_styles(styles_file, styles)


all_styles_files = get_files_from_folder(styles_path, ['.json'])
for styles_file in all_styles_files:
    if styles_file not in default_styles_files:
        styles = load_styles(styles_file, styles)


style_keys = list(styles.keys())
fooocus_expansion = "Fooocus V2"
legal_style_names = [fooocus_expansion] + style_keys


SD_XL_BASE_RATIOS = {
    "0.5": (704, 1408),
    "0.52": (704, 1344),
    "0.57": (768, 1344),
    "0.6": (768, 1280),
    "0.68": (832, 1216),
    "0.72": (832, 1152),
    "0.78": (896, 1152),
    "0.82": (896, 1088),
    "0.88": (960, 1088),
    "0.94": (960, 1024),
    "1.0": (1024, 1024),
    "1.07": (1024, 960),
    "1.13": (1088, 960),
    "1.21": (1088, 896),
    "1.29": (1152, 896),
    "1.38": (1152, 832),
    "1.46": (1216, 832),
    "1.67": (1280, 768),
    "1.75": (1344, 768),
    "1.91": (1344, 704),
    "2.0": (1408, 704),
    "2.09": (1472, 704),
    "2.4": (1536, 640),
    "2.5": (1600, 640),
    "2.89": (1664, 576),
    "3.0": (1728, 576),
}

aspect_ratios = {}

# import math

for k, (w, h) in SD_XL_BASE_RATIOS.items():
    txt = f'{w}×{h}'

    # gcd = math.gcd(w, h)
    # txt += f' {w//gcd}:{h//gcd}'
    
    aspect_ratios[txt] = (w, h)


def apply_style(style, positive):
    p, n = styles[style]
    return p.replace('{prompt}', positive), n
