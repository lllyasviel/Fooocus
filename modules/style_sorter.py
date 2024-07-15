import os
import gradio as gr
import modules.localization as localization
import json


all_styles = []


def try_load_sorted_styles(style_names, default_selected):
    global all_styles

    all_styles = style_names

    try:
        if os.path.exists('sorted_styles.json'):
            with open('sorted_styles.json', 'rt', encoding='utf-8') as fp:
                sorted_styles = []
                for x in json.load(fp):
                    if x in all_styles:
                        sorted_styles.append(x)
                for x in all_styles:
                    if x not in sorted_styles:
                        sorted_styles.append(x)
                all_styles = sorted_styles
    except Exception as e:
        print('Load style sorting failed.')
        print(e)

    unselected = [y for y in all_styles if y not in default_selected]
    all_styles = default_selected + unselected

    return


def sort_styles(selected):
    global all_styles
    unselected = [y for y in all_styles if y not in selected]
    sorted_styles = selected + unselected
    """
    try:
        with open('sorted_styles.json', 'wt', encoding='utf-8') as fp:
            json.dump(sorted_styles, fp, indent=4)
    except Exception as e:
        print('Write style sorting failed.')
        print(e)
    all_styles = sorted_styles
    """
    return gr.CheckboxGroup.update(choices=sorted_styles)


def localization_key(x):
    return x + localization.current_translation.get(x, '')


def search_styles(selected, query):
    unselected = [y for y in all_styles if y not in selected]
    matched = [y for y in unselected if query.lower() in localization_key(y).lower()] if len(query.replace(' ', '')) > 0 else []
    unmatched = [y for y in unselected if y not in matched]
    sorted_styles = matched + selected + unmatched
    return gr.CheckboxGroup.update(choices=sorted_styles)
