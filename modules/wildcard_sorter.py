import os
import gradio as gr
import modules.localization as localization
import json

all_wildcards = []

def try_load_sorted_wildcards(wildcard_names, default_selected):
    global all_wildcards

    all_wildcards = wildcard_names

    try:
        if os.path.exists('sorted_wildcards.json'):
            with open('sorted_wildcards.json', 'rt', encoding='utf-8') as fp:
                sorted_wildcards = []
                for x in json.load(fp):
                    if x in all_wildcards:
                        sorted_wildcards.append(x)
                for x in all_wildcards:
                    if x not in sorted_wildcards:
                        sorted_wildcards.append(x)
                all_wildcards = sorted_wildcards
    except Exception as e:
        print('Load wildcard sorting failed.')
        print(e)

    unselected = [y for y in all_wildcards if y not in default_selected]
    all_wildcards = default_selected + unselected

    return


def sort_wildcards(selected):
    global all_wildcards
    unselected = [y for y in all_wildcards if y not in selected]
    sorted_wildcards = selected + unselected
    try:
        with open('sorted_wildcards.json', 'wt', encoding='utf-8') as fp:
            json.dump(sorted_wildcards, fp, indent=4)
    except Exception as e:
        print('Write wildcard sorting failed.')
        print(e)
    all_wildcards = sorted_wildcards
    return gr.CheckboxGroup.update(choices=sorted_wildcards)


def localization_key(x):
    return x + localization.current_translation.get(x, '')


def search_wildcards(selected, query):
    unselected = [y for y in all_wildcards if y not in selected]
    matched = [y for y in unselected if query.lower() in localization_key(y).lower()] if len(query.replace(' ', '')) > 0 else []
    unmatched = [y for y in unselected if y not in matched]
    sorted_wildcards = matched + selected + unmatched
    return gr.CheckboxGroup.update(choices=sorted_wildcards)
