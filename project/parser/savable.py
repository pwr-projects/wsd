import os
import pickle
from functools import wraps

from .config import *

__all__ = [
    'save',
    'load',
    'savable'
]


def save(data, path: str):
    with open(path, 'wb') as fhnd:
        print('Saving to {}'.format(path), end='...')
        pickle.dump(data, fhnd)
        print('OK')
    return data


def load(path: str):
    with open(path, 'rb') as fhnd:
        print('Loading from {}'.format(path), end='...')
        data = pickle.load(fhnd, fix_imports=True)
        print('OK')
    return data


def savable(what: str):
    """
    If you want to override the save file simply pass arg:
        override=True 
    in the wrapped function :3
    """
    override_argname = 'override'
    save_load_path = os.path.join(TEMP_DIR, what)

    def real_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if not (override_argname in kwargs.keys() and kwargs.get(override_argname)) and os.path.exists(save_load_path):
                return load(save_load_path)
            return save(func(*args, **kwargs), save_load_path)
        return wrapper
    return real_decorator
