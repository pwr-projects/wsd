import string
from collections import OrderedDict
from itertools import chain

from wrappers import wcrft


def clean_text(text: str, sw):
    text = ' '.join(word for word in text.split() if not sw.is_stop_word(word))
    return text.translate(str.maketrans({key: None for key in string.punctuation}))


def get_lemmas_dict(tagger_out):
    return {v[0]: k for k, v in tagger_out.items()}


def get_senses(word: str, wn):
    return dict([[x[1].id, x[2]] for x in wn.senses(word)])


def get_gloss(word: str, synset_id: int, sw, wn):
    gloss = get_senses(word, wn).get(synset_id)
    gloss = clean_text(gloss, sw) if gloss else None
    return get_lemmas_dict(wcrft.tag(gloss)).keys() if gloss else None


def sorted_by_senses_count(lemmas, wn):
    sorted_words = {}

    for lemma in lemmas:
        count = len(wn.senses(lemma))

        if count:
            sorted_words[lemma] = count

    return OrderedDict(sorted(sorted_words.items(), key=lambda x: x[1])).keys()


def related_syn_ids(word, synset_id, wn, *relations):
    related = [related[3] for related in wn.senses_verbose(word) if related[1].id == synset_id]
    if related:
        related = chain(*[v for k, v in related[0].items()
                          for relation in relations if k.startswith(relation)])
    return [int(r) for r in related]


def glosses_from_syn_ids(syn_ids, wn, sw):
    def only_polish_def(syn_id):
        defs = wn.orig.synset_by_id(syn_id)
        defs = defs.to_dict()['units'][0]['definition'] if defs.is_polish else None
        return defs

    valid_defs = [only_polish_def(syn_id) for syn_id in syn_ids if only_polish_def(syn_id) not in [None, '.']]
    valid_defs = [list(get_lemmas_dict(wcrft.tag(clean_text(text, sw))).keys()) for text in valid_defs]
    valid_defs = [word.lower() for single_def in valid_defs for word in single_def]

    return list(valid_defs)
