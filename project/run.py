#!/usr/bin/env python
#%%
from parsers import *

parse_dataset(WSDParseType.wsd)
parse_dataset(WSDParseType.kdb)
parse_dataset(WSDParseType.kpwr)
# get_ft_word_embeddings()
# get_sense_embeddings()