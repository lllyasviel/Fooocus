import translators
from functools import lru_cache

@lru_cache(maxsize=32, typed=False)
def translate2en(text, element):
    if not text:
        return text

    try:
        result = translators.translate_text(text,to_language='en')
        print(f'[Parameters] Translated {element}: {result}')
        return result
    except Exception as e:
        print(f'[Parameters] Error during translation of {element}: {e}')
        return text