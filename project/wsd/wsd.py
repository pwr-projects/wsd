import numpy as np
from scipy.spatial.distance import cosine
from tqdm.autonotebook import tqdm

from wrappers import wcrft

from .utils import *


class WSD:
    def __init__(self, we, se, sw, wn):
        self._we = we
        self._se = se
        self._sw = sw
        self._wn = wn

    def _c_w(self, W_wout_W, disambiguated):
        def _get_vector(w):
            return self._se.get_embedding(disambiguated[w]) if w in disambiguated.keys() else self._we.get_embedding(w)
        return np.average([_get_vector(w) for w in W_wout_W], axis=0)

    def _g_s(self, word, synset_id, use_related=True, *relations):
        gloss = get_gloss(word, synset_id, self._sw, self._wn)
        gloss = list(gloss) if gloss else []

        if use_related:
            synset_ids = related_syn_ids(word, synset_id, self._wn, *relations)
            glosses = glosses_from_syn_ids(synset_ids, self._wn, self._sw)
            gloss.extend(glosses)

        gloss = set(gloss)

        return np.average([self._we.get_embedding(w) for w in gloss], axis=0) if len(gloss) else 0.0

    def wsd(self, text, use_related=True, relations=['hiperonimia', 'synonimia']):
        usages = get_lemmas_dict(wcrft.tag(clean_text(text, self._sw)))
        usages = {k.lower(): v.lower() for k, v in usages.items()}

        W = usages.keys()

        if(len(W) <= 1):
            raise ValueError('Please provide a wider context to WSD!')

        best_senses = {}
        ordered_W = list(sorted_by_senses_count(W, self._wn))

        sum_scores = 0.0

        for w in tqdm(ordered_W, 'Processing word and sense embeddings', leave=False):
            word_score = {}
            best_sense, best_score = None, 0.0

            W_wout_w = ordered_W.copy()
            W_wout_w.remove(w)

            c_w = self._c_w(W_wout_w, best_senses)

            for word, synset, gloss in self._wn.senses(w):
                try:
                    sense_emb = self._se.get_embedding(synset.id)
                except ValueError:
                    continue

                g_s = self._g_s(word, synset.id, use_related, *relations)

                first_cos = 1 - cosine(g_s, c_w)
                second_cos = 1 - cosine(sense_emb, c_w)
                score = first_cos + second_cos

                word_score[synset.id] = score

                if score > best_score:
                    best_score = score
                    best_sense = synset.id

            if best_sense:
                best_senses[w] = best_sense

            sum_scores += best_score

        mapped_best_senses = {usages[k]: (k, v) for k, v in best_senses.items()}
        return mapped_best_senses
