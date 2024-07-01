import json
import os

from modules.util import sha256, HASH_SHA256_LENGTH

hash_cache_filename = 'hash_cache.txt'
hash_cache = {}


def sha256_from_cache(filepath):
    global hash_cache
    if filepath not in hash_cache:
        hash_value = sha256(filepath)
        hash_cache[filepath] = hash_value
        save_cache_to_file(filepath, hash_value)

    return hash_cache[filepath]


def load_cache_from_file():
    global hash_cache

    try:
        if os.path.exists(hash_cache_filename):
            with open(hash_cache_filename, 'rt', encoding='utf-8') as fp:
                for line in fp:
                    entry = json.loads(line)
                    for filepath, hash_value in entry.items():
                        if not os.path.exists(filepath) or not isinstance(hash_value, str) and len(hash_value) != HASH_SHA256_LENGTH:
                            print(f'[Cache] Skipping invalid cache entry: {filepath}')
                            continue
                        hash_cache[filepath] = hash_value
    except Exception as e:
        print(f'[Cache] Loading failed: {e}')


def save_cache_to_file(filename=None, hash_value=None):
    global hash_cache

    if filename is not None and hash_value is not None:
        items = [(filename, hash_value)]
        mode = 'at'
    else:
        items = sorted(hash_cache.items())
        mode = 'wt'

    try:
        with open(hash_cache_filename, mode, encoding='utf-8') as fp:
            for filepath, hash_value in items:
                json.dump({filepath: hash_value}, fp)
                fp.write('\n')
    except Exception as e:
        print(f'[Cache] Saving failed: {e}')
