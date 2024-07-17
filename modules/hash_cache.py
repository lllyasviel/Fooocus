import json
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

import args_manager
from modules.util import sha256, HASH_SHA256_LENGTH, get_file_from_folder_list

hash_cache_filename = 'hash_cache.txt'
hash_cache = {}


def sha256_from_cache(filepath):
    global hash_cache
    if filepath not in hash_cache:
        print(f"[Cache] Calculating sha256 for {filepath}")
        hash_value = sha256(filepath)
        print(f"[Cache] sha256 for {filepath}: {hash_value}")
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


def init_cache(model_filenames, paths_checkpoints, lora_filenames, paths_loras):
    load_cache_from_file()

    if args_manager.args.rebuild_hash_cache:
        max_workers = args_manager.args.rebuild_hash_cache if args_manager.args.rebuild_hash_cache > 0 else cpu_count()
        rebuild_cache(lora_filenames, model_filenames, paths_checkpoints, paths_loras, max_workers)

    # write cache to file again for sorting and cleanup of invalid cache entries
    save_cache_to_file()


def rebuild_cache(lora_filenames, model_filenames, paths_checkpoints, paths_loras, max_workers=cpu_count()):
    def thread(filename, paths):
        filepath = get_file_from_folder_list(filename, paths)
        sha256_from_cache(filepath)

    print('[Cache] Rebuilding hash cache')
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for model_filename in model_filenames:
            executor.submit(thread, model_filename, paths_checkpoints)
        for lora_filename in lora_filenames:
            executor.submit(thread, lora_filename, paths_loras)
    print('[Cache] Done')
