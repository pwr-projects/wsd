from os import path
import pickle
import re

import fastText
import numpy as np
from tqdm import tqdm


def load_word_embeddings(filepath):
    return fastText.load_model(filepath)


def load_sense_embeddings(filepath):
    loaded_data = None
    with open(filepath, 'rb') as f:
        loaded_data = pickle.load(f)

    return loaded_data


# adapted from Grzechu's utils package... :)
def pickle_sense_embeddings(file_dir, file_name, prefix=r'(?:syn:(\d+))'):
    tmp_lines_num = 322420  # :)
    out_data = {}
    with open(path.join(file_dir, file_name), 'r') as f:
        rgx = re.compile(prefix + '|' + r'(?:\s([-.\d]+))')

        for line in tqdm(f, total=tmp_lines_num, dynamic_ncols=True):
            line = line.strip()
            matched = rgx.findall(line)

            if matched:
                matched = list(filter(len, np.reshape(matched, -1)))
                synset_id = matched[0]
                vector = np.array(matched[1:], dtype=np.float)
                out_data[int(synset_id)] = vector

    with open(path.join(file_dir, 'sense_embeddings.pkl'), 'wb') as p:
        pickle.dump(out_data, p)
