#!/bin/python
import os
import warnings
from functools import partial
from itertools import product, starmap

import numpy as np
import pandas as pd
from sklearn.exceptions import DataConversionWarning, UndefinedMetricWarning
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from tqdm.autonotebook import tqdm

import plwn
from parsers import *
from wsd import *

warnings.filterwarnings(action='ignore', category=DataConversionWarning)
warnings.filterwarnings(action='ignore', category=UndefinedMetricWarning)
warnings.filterwarnings(action='ignore', category=RuntimeWarning)
# %% CONFIG

stop_words_file = 'polish.stopwords.txt'

# parse_dataset(WSDParseType.wsd)
# parse_dataset(WSDParseType.kdb)
# parse_dataset(WSDParseType.kpwr)
# we = get_ft_word_embeddings()
# se = get_sense_embeddings()

we = LocalEmb(get_we)
se = LocalEmb(get_se)
sw = StopWord(stop_words_file)
wn = Wordnet(plwn.load_default())
skladnica = load_skladnica(WSDParseType.wsd)
kpwr = load_kpwr()

lemmas = create_lemmas(*skladnica[:, 1], *kpwr[:, 1], sw=sw)

wsd = WSD(we, se, sw, wn, lemmas, tqdm_disable=False)

# %% score per sentence generator


def score(idx, sentence, senses, use_related, write_to_scores, get_results_by_idx):
    results_from_file = get_results_by_idx(idx)
    if results_from_file is not None:
        return results_from_file

    try:
        wsd_out = wsd(sentence, use_related=use_related)
    except (ValueError, KeyError) as e:
        print('FAILED', idx, sentence)
        print('ERROR:', e)
        return [[[np.NaN, ] * 4, ] * 3] * 2

    def _scores(true, preds):
        acc = accuracy_score(true, preds)
        f1 = f1_score(true, preds, average='macro')
        prec = precision_score(true, preds, average='macro')
        rec = recall_score(true, preds, average='macro')
        return acc, f1, prec, rec

    real_senses = list(filter(lambda word_sense: word_sense[1], senses))

    filter_type = lambda type_name: list(filter(lambda word_sense: word_sense[2] == type_name, real_senses))
    create_type_dict = lambda data: {elem[0]: elem[1:] for elem in data}

    real_senses_verbs = create_type_dict(filter_type('verb'))
    real_senses_nouns = create_type_dict(filter_type('noun'))
    real_senses_others = create_type_dict(filter_type('other'))

    # mati version
    def mati_score(true_vals):
        true, preds = [], []
        for word, sens_id in wsd_out.items():
            true.append(sens_id[1])
            pred = true_vals.get(word)
            preds.append(int(pred[0]) if pred else 0)

        return _scores(true, preds)

    # grzech version
    def grzech_score(true_vals):
        true, preds = [], []
        for word, (sens_id, _) in true_vals.items():
            true.append(int(sens_id))
            pred = wsd_out.get(word)
            preds.append(pred[1] if pred else 0)

        return _scores(true, preds)

    real_senses = [real_senses_verbs, real_senses_nouns, real_senses_others]

    mati_verses = list(map(mati_score, real_senses))
    grzech_verses = list(map(grzech_score, real_senses))

    write_to_scores(','.join([str(idx),
                              *map(str, list(chain(*mati_verses))),
                              *map(str, list(chain(*grzech_verses)))]))

    return np.asarray([mati_verses, grzech_verses])


# %% Test dataset
def test(dataset_type: WSDParseType, use_related):
    if dataset_type == WSDParseType.kpwr:
        dataset = kpwr
        dataset_name = 'kpwr'
    else:
        dataset = skladnica
        dataset_name = 'skladnica'

    related_name = '_related' if use_related else ''

    scores_filename = 'scores_{}{}.csv'.format(dataset_name, related_name)

    def write_to_scores(text, filename=scores_filename):
        with open(filename, 'a+') as f:
            f.write(text + '\n')

    def get_results_by_idx(idx, filename=scores_filename):
        with open(filename, 'r') as f:
            for line in f.read().strip().split('\n'):
                scores = line.split(',')
                fidx = int(scores[0]) if scores[0] != 'idx' else -1
                if fidx == idx:
                    print('Found', idx, 'sentence', end='\r')
                    scores = list(map(float, scores[1:]))
                    mati_scores_verb = scores[:4]
                    mati_scores_noun = scores[4:8]
                    mati_scores_other = scores[8:12]
                    grzech_scores_verb = scores[12:16]
                    grzech_scores_noun = scores[16:20]
                    grzech_scores_other = scores[20:]

                    mati_scores = [mati_scores_verb, mati_scores_noun, mati_scores_other]
                    grzech_scores = [grzech_scores_verb, grzech_scores_noun, grzech_scores_other]
                    scores = np.asarray([mati_scores, grzech_scores])
                    assert len(mati_scores_verb) == len(mati_scores_noun) == len(mati_scores_other) == len(
                        grzech_scores_verb) == len(grzech_scores_noun) == len(grzech_scores_other)
                    return scores
            return None

    if not os.path.isfile(scores_filename):
        write_to_scores('idx,' + ','.join(['_'.join(data) for data in product(['mati', 'grzech'],
                                                                              ['verb', 'noun', 'other'],
                                                                              ['acc', 'f1', 'prec', 'rec'], )]))

    _score = partial(score,
                     use_related=True,
                     write_to_scores=write_to_scores,
                     get_results_by_idx=get_results_by_idx)

    ress = list(starmap(_score, tqdm(dataset[:5], 'Sentence', dynamic_ncols=True)))

    def print_scores(results):

        results = np.array(results)

        mati_vers, grzech_vers = results[:, 0, :], results[:, 1, :]

        def filter_nans(vers_results):
            return np.asarray(list(filter(lambda x: any(not np.isnan(z) for y in x for z in y), vers_results)))

        def _print_scores(version: str, results, pos_type):
            if pos_type == 'verb':
                pos_pos = 0
            elif pos_type == 'noun':
                pos_pos = 1
            else:
                pos_pos = 2

            print(pos_type)
            print('\tacc ', version, results[pos_pos][0])
            print('\tf1  ', version, results[pos_pos][1])
            print('\tprec', version, results[pos_pos][2])
            print('\trec ', version, results[pos_pos][3])

        mati_vers = filter_nans(mati_vers)
        grzech_vers = filter_nans(grzech_vers)

        mati_vers = np.average(mati_vers, axis=0)
        grzech_vers = np.average(grzech_vers, axis=0)

        _print_scores('mati  \t', mati_vers, 'verb')
        _print_scores('mati  \t', mati_vers, 'noun')
        _print_scores('mati  \t', mati_vers, 'other')

        _print_scores('grzech\t', grzech_vers, 'verb')
        _print_scores('grzech\t', grzech_vers, 'noun')
        _print_scores('grzech\t', grzech_vers, 'other')

    print_scores(ress)


test(WSDParseType.wsd, False)
test(WSDParseType.wsd, True)
test(WSDParseType.kpwr, False)
test(WSDParseType.kpwr, False)


# %%
def show_scores(filename):
    print(filename, ':')
    return pd.DataFrame(pd.read_csv('scores_{}.csv'.format(filename)).mean()).transpose()


# %%
show_scores('skladnica')
# %%
show_scores('skladnica_related')
# %%
show_scores('kpwr')
# %%
show_scores('kpwr_related')
