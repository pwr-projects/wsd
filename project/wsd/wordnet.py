import numpy as np
from tqdm.autonotebook import tqdm


class Wordnet:

    def __init__(self, wordnet):
        self._wordnet = wordnet
        self._lexical_units = self._valid_lexical_units()

    def _valid_lexical_units(self):
        return [(lu.lemma, lu.synset, lu.definition) for lu in
                tqdm(self._wordnet.lexical_units(), desc='Filtering polish', leave=False, dynamic_ncols=True)
                if lu.is_polish]

    def senses(self, word):
        return [lu for lu in self._lexical_units if lu[0] == word]

    def senses_verbose(self, word):
        def reshape_senses(senses):
            if senses:
                return list(np.asarray(senses)[:, 0].astype(int))
            return []

        def verbose_part(sense):
            return dict([[k, reshape_senses(v)]
                         for k, v in sense[1].to_dict()['related'].items()])

        return [(*sense, verbose_part(sense))
                for sense in self.senses(word)]

    @property
    def orig(self):
        return self._wordnet
