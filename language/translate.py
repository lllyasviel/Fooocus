from google.cloud import translate_v2 as translate
import json
import os


def translate_text(text, target_language, translate_client):
    result = translate_client.translate(text, target_language=target_language)
    return result['translatedText']

def translate_json_file(input_file, output_file, target_lang):
    translate_client = translate.Client()

    with open(input_file, 'r', encoding='utf-8') as file:
        data = json.load(file)

    translated_data = {}
    for key, value in data.items():
        if isinstance(value, str):
            translated_data[key] = translate_text(value, target_lang, translate_client)
        else:
            translated_data[key] = value

    with open(output_file, 'w', encoding='utf-8') as file:
        json.dump(translated_data, file, ensure_ascii=False, indent=4)

translate_json_file('en.json', 'ar.json', 'ar')
