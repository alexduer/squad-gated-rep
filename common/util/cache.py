import hashlib
import os
import pickle
from typing import Any, Callable


def cache(f: Callable[[], Any], cache_dir: str, cache_key_seed: str) -> Any:
    cache_file_path = os.path.join(cache_dir, hash(cache_key_seed) + '.pckl')
    if not os.path.exists(cache_file_path):
        pickle.dump(f(), open(cache_file_path, 'wb'))

    return pickle.load(open(cache_file_path, 'rb'))


def hash(s: str) -> str:
    return hashlib.sha224(s.encode('UTF-8')).hexdigest()
