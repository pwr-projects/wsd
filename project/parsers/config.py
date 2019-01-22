
TEMP_DIR = '.tmp'
DATA_DIR = 'data'
SKLADNICA_DIR = 'skladnica'
KPWR_DIR = 'kpwr'
KPWR_DOCS_DIR = 'documents'

FILENAME_FASTTEXT_NAME = 'kgr10.plain.lemma.skipgram.dim300.neg10.vec'

FILENAME_FASTTEXT_MODEL = 'ft-model'
FILENAME_SENSE_EMBEDDINGS = 'se.pkl'
FILENAME_KPWR_MAPPING = 'kpwr-slowosiec.txt'

URL_WORD_EMBEDDINGS = 'http://tools.clarin-pl.eu/share/embeddings/kgr10.plain.lemma.skipgram.dim300.neg10.vec'
URL_FASTTEXT_WORD_EMBEDDINGS = 'http://svn.clarin-pl.eu/svn/ijn_students/autoextend-we.7z'
URL_SENSE_EMBEDDINGS = 'http://svn.clarin-pl.eu/svn/ijn_students/autoextend.7z'

FILENAME_SKLADNICA = lambda wsd_type: 'skladnica_{}.npy'.format(wsd_type)
FILENAME_KPWR = 'kpwr.npy'    