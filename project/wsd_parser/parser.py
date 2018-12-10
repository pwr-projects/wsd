
import os
import re

import gensim.models.fasttext as ft
import numpy as np
from pyunpack import Archive
from tqdm import tqdm

from .config import *
from .savable import savable
from .downloader import *
from .utils import wc

__all__ = ['get_ft_word_embeddings',
           'get_sense_embeddings']


def load_vectors(filepath: str, prefix: str = r'(?:syn:(\d+))'):
    with open(filepath, 'r') as f:
        rgx = re.compile(prefix + '|' + r'(?:\s([-.\d]+))')
        out_data = {}

        for line in tqdm(f, 'Loading vector',
                         total=wc(filepath),
                         leave=False):

            line = line.strip()
            matched = rgx.findall(line)

            if matched:
                matched = list(filter(len, np.reshape(matched, -1)))
                synset_id = matched[0]
                vector = np.array(matched[1:], dtype=np.float)
                out_data[int(synset_id)] = vector

    return out_data


def get_ft_word_embeddings():
    model_path = os.path.join(TEMP_DIR, FILENAME_FASTTEXT_MODEL)

    if not os.path.isfile(model_path):
        temp_archive_filename = os.path.join(DATA_DIR, 'we.7z')
        download(URL_FASTTEXT_WORD_EMBEDDINGS, temp_archive_filename)

        unpack_path = os.path.join(DATA_DIR, 'we')

        print('Unpacking to {}...'.format(unpack_path))
        if not os.path.isdir(unpack_path):
            Archive(temp_archive_filename).extractall(unpack_path, True)

        print('Converting model...')
        data = ft.FastText.load_fasttext_format(os.path.join(unpack_path,
                                                             'fastText-plWNC-skipgram-300-lemmas-mwe-minCount-50.bin'))

        print('Saving model...')
        data.save(model_path)

    print('Loading word embeddings...')
    return ft.FastTextKeyedVectors.load(model_path)


@savable(FILENAME_SENSE_EMBEDDINGS)
def get_sense_embeddings():
    temp_archive_filename = os.path.join(DATA_DIR, 'se.7z')

    download(URL_SENSE_EMBEDDINGS, temp_archive_filename)
    unpack_path = os.path.join(DATA_DIR, 'se')

    if not os.path.isdir(unpack_path):
        print('Unpacking to {}'.format(unpack_path))
        Archive(temp_archive_filename).extractall(unpack_path, True)

    return load_vectors(os.path.join(unpack_path, 'plwn-vectors-joined-rels.txt'))
