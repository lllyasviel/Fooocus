import json
import os


localization_root = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'language')


def localization_js(filename):
    data = {}

    if isinstance(filename, str):
        full_name = os.path.abspath(os.path.join(localization_root, filename + '.json'))
        if os.path.exists(full_name):
            try:
                with open(full_name, encoding='utf-8') as f:
                    data = json.load(f)
                    assert isinstance(data, dict)
                    for k, v in data.items():
                        assert isinstance(k, str)
                        assert isinstance(v, str)
            except Exception as e:
                print(str(e))
                print(f'Failed to load localization file {full_name}')

    return f"window.localization = {json.dumps(data)}"
