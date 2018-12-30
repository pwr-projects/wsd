from flask import Flask, make_response
from json import dumps

from utils import loaders


app = Flask(__name__)
word_embeddings = loaders.load_word_embeddings('embeddings/fastText-plWNC-skipgram-300-lemmas-mwe-minCount-50.bin')
sense_embeddings = loaders.load_sense_embeddings('embeddings/sense_embeddings.pkl')


@app.route('/word_emb/<word>')
def get_word_embedding(word):
	word_emb = word_embeddings.get_word_vector(word).tolist()
	return make_response(dumps(word_emb))


@app.route('/sense_emb/<sense_id>')
def get_sense_embedding(sense_id):
	sense_emb = sense_embeddings[int(sense_id)].tolist()
	return make_response(dumps(sense_emb))


if __name__ == '__main__':
	app.run(host='0.0.0.0')
