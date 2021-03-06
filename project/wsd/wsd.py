import numpy as np
from scipy.spatial.distance import cosine
from tqdm.autonotebook import tqdm

from wrappers import wcrft

from .utils import *


class WSD:
    def __init__(self, we, se, sw, wn, lemmas, tqdm_disable=True):
        self._we = we
        self._se = se
        self._sw = sw
        self._wn = wn
        self._lemmas = lemmas
        self._tqdm_disable = tqdm_disable

    def _c_w(self, W_wout_W, disambiguated):
        def _get_vector(w):
            return self._se[disambiguated[w]] if w in disambiguated.keys() else self._we[w]
        return np.average(list(map(_get_vector, W_wout_W)), axis=0)

    def _g_s(self, word, synset_id, use_related=True, *relations):
        gloss = get_gloss(word, synset_id, self._lemmas, self._sw, self._wn)
        gloss = list(gloss) if gloss else []

        if use_related:
            synset_ids = related_syn_ids(word, synset_id, self._wn, *relations)
            glosses = glosses_from_syn_ids(synset_ids, self._wn, self._sw, self._lemmas)
            gloss.extend(glosses)

        gloss = set(gloss)

        return np.average([self._we[w] for w in gloss], axis=0) if len(gloss) else 0.0

    def __call__(self, text, **kwargs):
        return self.wsd(text, **kwargs)

    def wsd(self, text, use_related=False, relations=['hiperonimia', 'synonimia']):
        usages = lemma(clean_text(text, self._sw), lemmas=self._lemmas, sw=self._sw)
        W = get_words_from_lemma_output(usages)

        usages = {v[0]: (k, tag_name(v[1])) for k, v in usages.items()}
        # assert len(W) > 1, 'Please provide a wider context to WSD!'

        best_senses = {}
        sum_scores = 0.0
        ordered_W = list(sorted_by_senses_count(W, self._wn))

        for w in tqdm(ordered_W,
                      'Proc. word and sense emb.',
                      leave=False,
                      disable=self._tqdm_disable,
                      dynamic_ncols=True):

            best_sense, best_score, word_score = None, 0.0, {}
            W_wout_w = ordered_W.copy()
            W_wout_w.remove(w)

            c_w = self._c_w(W_wout_w, best_senses)

            for word, synset, gloss in self._wn.senses(w):
                try:
                    sense_emb = self._se[synset.id]
                except (KeyError, ValueError):
                    continue

                g_s = self._g_s(word, synset.id, use_related, *relations)

                first_cos = (1 - cosine(g_s, c_w)) if g_s is not None else 0.0
                second_cos = 1 - cosine(sense_emb, c_w)
                score = first_cos + second_cos

                word_score[synset.id] = score

                if score > best_score:
                    best_score = score
                    best_sense = synset.id

            if best_sense:
                best_senses[w] = best_sense

            sum_scores += best_score

        return {usages[k][0]: (k, v, usages[k][1]) for k, v in best_senses.items()}
