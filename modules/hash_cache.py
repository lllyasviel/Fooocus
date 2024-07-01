from modules.util import sha256, HASH_SHA256_LENGTH
import os
import json
hash_cache_filename = 'hash_cache.json'
hash_cache = {}


def sha256_from_cache(filepath):
    global hash_cache
    if filepath not in hash_cache:
        hash_cache[filepath] = sha256(filepath)
        save_cache_to_file()

    return hash_cache[filepath]


def load_cache_from_file():
    global hash_cache

    try:
        if os.path.exists(hash_cache_filename):
            with open(hash_cache_filename, 'rt', encoding='utf-8') as fp:
                for filepath, hash in json.load(fp).items():
                    if not os.path.exists(filepath) or not isinstance(hash, str) and len(hash) != HASH_SHA256_LENGTH:
                        print(f'[Cache] Skipping invalid cache entry: {filepath}')
                        continue
                    hash_cache[filepath] = hash
            print(f'[Cache] Warmed cache from file')
    except Exception as e:
        print(f'[Cache] Warming failed: {e}')


def save_cache_to_file():
    global hash_cache

    try:
        with open(hash_cache_filename, 'wt', encoding='utf-8') as fp:
            json.dump(hash_cache, fp, indent=4)
        print(f'[Cache] Updated cache file')
    except Exception as e:
        print(f'[Cache] Saving failed: {e}')
