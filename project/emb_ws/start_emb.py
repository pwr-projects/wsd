from flask import Flask, make_response
from json import dumps

from .utils import loaders


word_embeddings = loaders.load_word_embeddings('emb_ws/embeddings/fastText-plWNC-skipgram-300-lemmas-mwe-minCount-50.bin')
sense_embeddings = loaders.load_sense_embeddings('emb_ws/embeddings/sense_embeddings.pkl')


def get_we(word):
    return word_embeddings.get_word_vector(word).tolist()


def get_se(sense_id):
    return sense_embeddings[int(sense_id)].tolist()


# @app.route('/word_emb/<word>')
# def get_word_embedding(word):
#     return make_response(dumps(get_we(word)))


# @app.route('/sense_emb/<sense_id>')
# def get_sense_embedding(sense_id):
#     return make_response(dumps(get_se(sense_id)))


# if __name__ == '__main__':
#     app = Flask(__name__)
#     app.run(host='0.0.0.0')
