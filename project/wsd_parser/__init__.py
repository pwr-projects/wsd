from .parser import get_ft_word_embeddings, get_sense_embeddings
from .config import *
import os

__all__ = [
    'get_ft_word_embeddings',
    'get_sense_embeddings'
]

for dirname in TEMP_DIR, DATA_DIR:
    if not os.path.isdir(dirname):
        print('Creating {}'.format(dirname))
        os.mkdir(dirname)
