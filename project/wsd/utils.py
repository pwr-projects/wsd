import os
import pickle
import re
import string
from collections import OrderedDict
from itertools import chain

from parsers.config import TEMP_DIR
from wrappers import wcrft


def clean_text(text: str, sw):
    text = ' '.join(word for word in text.split() if not sw.is_stop_word(word))
    return text.translate(str.maketrans({key: None for key in string.punctuation}))


def get_lemmas_dict(tagger_out):
    return {v: k for k, v in tagger_out.items()}


def get_senses(word: str, wn):
    return dict([[x[1].id, x[2]] for x in wn.senses(word)])


def get_words_from_lemma_output(lemma_out):
    return list(map(lambda x: x[0], lemma_out.values()))


def get_gloss(word: str, synset_id: int, lemmas, sw, wn):
    gloss = get_senses(word, wn).get(synset_id)
    gloss = clean_text(gloss, sw) if gloss else None
    return get_words_from_lemma_output(lemma(gloss, lemmas=lemmas, sw=sw)) if gloss else None


def sorted_by_senses_count(lemmas, wn):
    sorted_words = {}

    for lemma in lemmas:
        count = len(wn.senses(lemma))

        if count:
            sorted_words[lemma] = count

    return OrderedDict(sorted(sorted_words.items(), key=lambda x: x[1])).keys()


def create_lemmas(*sentences, sw):
    cleaned_sentences = map(lambda text: clean_text(text, sw), sentences)
    lemmas = wcrft.tag(*cleaned_sentences)
    lemmas = {k.lower(): v for k, v in lemmas.items()}
    with open(os.path.join(TEMP_DIR, 'lemmas.pkl'), 'wb') as fhd:
        pickle.dump(lemmas, fhd)
    return lemmas


def update_lemmas(word_lemma, lemmas):
    lemmas = {**lemmas, **word_lemma}

    with open(os.path.join(TEMP_DIR, 'lemmas.pkl'), 'wb') as fhd:
        pickle.dump(lemmas, fhd)


def lemma(*text, lemmas, sw):
    to_lemmatize = []
    lemmatized = {}

    for word in clean_text(' '.join(chain(*[elem.split() for elem in text])), sw).split():
        lemma = lemmas.get(word)
        if lemma:
            lemmatized = {**lemmatized, word.lower(): (lemma[0].lower(), lemma[1])}
        else:
            to_lemmatize.append(word.lower())

    if to_lemmatize:
        new_lemmas = wcrft.tag(*to_lemmatize)
        update_lemmas(new_lemmas, lemmas)
        lemmatized = {**lemmatized, **new_lemmas}

    return lemmatized


def related_syn_ids(word, synset_id, wn, *relations):
    related = [related[3] for related in wn.senses_verbose(word) if related[1].id == synset_id]
    if related:
        related = chain(*[v for k, v in related[0].items()
                          for relation in relations if k.startswith(relation)])
    return [int(r) for r in related]


def glosses_from_syn_ids(syn_ids, wn, sw, lemmas):
    def only_polish_def(syn_id):
        defs = wn.orig.synset_by_id(syn_id)
        defs = defs.to_dict()['units'][0]['definition'] if defs.is_polish else None
        return defs

    valid_defs = [only_polish_def(syn_id) for syn_id in syn_ids if only_polish_def(syn_id) not in [None, '.']]

    valid_defs = list(map(lambda out: out[0].lower(),
                          lemma(clean_text(' '.join(valid_defs), sw),
                                lemmas=lemmas,
                                sw=sw).values()))

    return valid_defs


def is_verb(tag):
    prefixes = ['fin', 'aglt', 'praet', 'impt', 'imps', 'inf', 'pcon', 'pant', 'ger', 'pact', 'ppas']
    return any(map(tag.startswith, prefixes))


def is_noun(tag):
    prefixes = ['subst', 'dept']
    return any(map(tag.startswith, prefixes))


def tag_name(tag):
    if is_verb(tag):
        return 'verb'
    if is_noun(tag):
        return 'noun'
    return 'other'
