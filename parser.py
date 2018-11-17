#!/bin/python

# %%
import os
import pickle
import re
from subprocess import check_output

import numpy as np
from tqdm import tqdm
import requests


def download(url: str, out_path: str):
    r = requests.get(url, stream=True)
    with open(out_path, 'wb') as f:
        file_size = int(r.headers['Content-Length'])
        chunk, i = 1024, 0
        num_bars = file_size / chunk
        with tqdm(r.iter_content(), total=num_bars) as bar:
            for chunk in bar:
                f.write(chunk)
                bar.update(i)
                i += 1


def wc(filename):
    return int(check_output(["wc", "-l", filename]).split()[0])

# %%


def load_vectors(filepath: str, extension: str = 'txt', prefix: str =r'(?:syn:(\d+))'):
    if os.path.isfile(f'{filepath}.pkl'):
        print('Found pickle, so loading from it')
        with open(f'{filepath}.pkl', 'rb') as f:
            return pickle.load(f)

    with open(f'{filepath}.{extension}', 'r') as f:
        rgx = re.compile(prefix + '|' + r'(?:\s([-.\d]+))')
        synsets = {}
        for line in tqdm(f, 'Loading vector',
                         total=wc(f'{filepath}.{extension}'),
                         leave=False):
            line = line.strip()
            matched = rgx.findall(line)
            if matched:
                matched = list(filter(len, np.reshape(matched, -1)))
                synset_id = matched[0]
                vector = np.array(matched[1:], dtype=np.float)
                synsets[synset_id] = vector

    with open(f'{filepath}.pkl', 'wb') as f:
        print('Saving pickle')
        pickle.dump(synsets, f)
    return synsets


# load_vectors('./data/plwn-vectors-joined-rels')
# %%
vec = download('http://tools.clarin-pl.eu/share/embeddings/kgr10.plain.lemma.skipgram.dim300.neg10.vec', 'data/lemma.vec')
# %%

vec = load_vectors('./data/lemma', 'vec', r'(?:([\S.]+))')
vec
