import json
import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import cpu_count

import args_manager
from modules.util import sha256, HASH_SHA256_LENGTH, get_file_from_folder_list

HASH_CACHE_PATH = os.environ.get('HASH_CACHE_PATH')


def sha256_from_cache(filepath) -> dict:

    print(f"[Cache] Calculating sha256 for {filepath}")
    hash_value = sha256(filepath)
    print(f"[Cache] sha256 for {filepath}: {hash_value}")
    hash_cache_obj = {filepath: hash_value}

    return hash_cache_obj


def load_cache_from_file(hash_cache_path=HASH_CACHE_PATH) -> dict:

    try:
        if os.path.exists(hash_cache_path):
            with open(hash_cache_path, 'r') as fp:
                hash_cache = json.load(fp)
            
            for filepath, hash_value in hash_cache.items():
                if not os.path.exists(filepath) or not isinstance(hash_value, str) and len(hash_value) != HASH_SHA256_LENGTH:
                    print(f'[Cache] Skipping invalid cache entry: {filepath}')
                    continue
                hash_cache[filepath] = hash_value
        else:
            hash_cache = {}
            with open(hash_cache_path, 'w') as fp:
                json.dump({}, fp, indent=2, ensure_ascii=False)
                    
    except Exception as e:
        print(f'[Cache] Loading failed: {e}')

    return hash_cache

def overwrite_old_cache(new_cache, hash_cache_path=HASH_CACHE_PATH):
    try:
        with open(hash_cache_path, 'w') as fp:
            json.dump(new_cache, fp, indent=2, ensure_ascii=False)
    except Exception as e:
        print(f'[Cache] Overwriting cache failed: {e}')
        raise e
    


def save_cache_to_file(filename=None, hash_value=None) -> bool:
    hash_cache = load_cache_from_file()

    if filename is not None and hash_value is not None:
        items = [(filename, hash_value)]
    else:
        items = sorted(hash_cache.items())

    try:
        with open(HASH_CACHE_PATH, 'w') as fp:
            json.dump(dict(items), fp, indent=2, ensure_ascii=False)
        return True
    except Exception as e:
        print(f'[Cache] Saving failed: {e}')
        raise e


def init_cache(model_filenames, paths_checkpoints, lora_filenames, paths_loras):
    hash_cache = {}
    max_workers = args_manager.args.rebuild_hash_cache if args_manager.args.rebuild_hash_cache > 0 else cpu_count()
    result = rebuild_cache(lora_filenames, model_filenames, paths_checkpoints, paths_loras, max_workers)

    # write cache to file again for sorting and cleanup of invalid cache entries
    for cache in result:
        hash_cache.update(cache)
    overwrite_old_cache(hash_cache)
    return hash_cache


def rebuild_cache(lora_filenames, model_filenames, paths_checkpoints, paths_loras, max_workers=cpu_count()):
    empty_cache = {}
    
    def thread(filename, paths):
        filepath = get_file_from_folder_list(filename, paths)
        result_cache = sha256_from_cache(filepath) # just to calculate sha256
        return result_cache

    submits = []
    print('[Cache] Rebuilding hash cache')
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for model_filename in model_filenames:
            submits.append(executor.submit(thread, model_filename, paths_checkpoints))
        for lora_filename in lora_filenames:
            submits.append(executor.submit(thread, lora_filename, paths_loras))
        
        while not all([submit.done() for submit in submits]):
            pass

        results = [submit.result() for submit in submits]

    return results
