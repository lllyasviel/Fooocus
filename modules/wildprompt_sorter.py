import os
import gradio as gr
import modules.localization as localization
import json

all_wildprompts = []

def try_load_sorted_wildprompts(wildprompt_names, default_selected):
    global all_wildprompts

    all_wildprompts = wildprompt_names

    unselected = [y for y in all_wildprompts if y not in default_selected]
    all_wildprompts = default_selected + unselected

    return


def sort_wildprompts(selected):
    global all_wildprompts
    unselected = [y for y in all_wildprompts if y not in selected]
    sorted_wildprompts = selected + unselected
    try:
        with open('sorted_wildprompt.json', 'wt', encoding='utf-8') as fp:
            json.dump(sorted_wildprompts, fp, indent=4)
    except Exception as e:
        print('Write wildprompt sorting failed.')
        print(e)
    all_wildprompts = sorted_wildprompts
    return gr.CheckboxGroup.update(choices=sorted_wildprompts)


def localization_key(x):
    return x + localization.current_translation.get(x, '')


def search_wildprompts(selected, query):
    unselected = [y for y in all_wildprompts if y not in selected]
    matched = [y for y in unselected if query.lower() in localization_key(y).lower()] if len(query.replace(' ', '')) > 0 else []
    unmatched = [y for y in unselected if y not in matched]
    sorted_wildprompts = matched + selected + unmatched
    return gr.CheckboxGroup.update(choices=sorted_wildprompts)
