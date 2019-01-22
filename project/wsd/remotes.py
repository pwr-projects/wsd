import ast
from enum import Enum
from urllib.parse import quote
from emb_ws.start_emb import get_we, get_se

import urllib3


class RemoteWordEmbedding:

    def __init__(self, address):
        self._address = address
        self._http = urllib3.PoolManager()

    def __getitem__(self, word):
        word = quote(word)
        target = '{addr}/word_emb/{word}'.format(addr=self._address, word=word)
        data = self._http.request('GET', target)
        if data.status == 200:
            return ast.literal_eval(data.data.decode('ascii'))
        else:
            raise ValueError('A problem occured during getting an embedding for word: {word}...'.format(word=word))


class RemoteSenseEmbedding:

    def __init__(self, address):
        self._address = address
        self._http = urllib3.PoolManager()

    def __getitem__(self, synset_id):
        target = '{addr}/sense_emb/{synset_id}'.format(addr=self._address, synset_id=synset_id)
        data = self._http.request('GET', target)
        if data.status == 200:
            return ast.literal_eval(data.data.decode('ascii'))
        else:
            raise ValueError('A problem occured during getting an embedding for synset id: {synset_id}...'.format(
                synset_id=synset_id))


class LocalEmb:
    def __init__(self, fun):
        self._fun = fun

    def __getitem__(self, excel):
        return self._fun(excel)
