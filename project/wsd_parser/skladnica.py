#!/bin/python

import os
import re
import xml.etree.ElementTree as ET
from os.path import join as pj

import numpy as np
from tqdm import tqdm

from .config import *

__all__ = [
    'parse_skladnica'
]

SKLADNICA_PATH = pj(DATA_DIR, SKLADNICA_DIR)


def parse_sentence(sentence: ET.Element):
    concat_sentence = []
    word_synset = []

    for word in sentence.iterfind('.//tok'):
        orth = word.findtext('.//orth')
        synset_id = word.findtext(".//prop[@key='sense:ukb:syns_id']")
        concat_sentence.append(orth)
        word_synset.append((orth, synset_id))
    return ' '.join(concat_sentence), word_synset


def parse_xml(*xml_files):
    contents = []

    for xml_file in tqdm(xml_files, 'xml file', leave=False):
        tree = ET.parse(pj(SKLADNICA_PATH, xml_file))

        for sentence in tree.iterfind('.//sentence'):
            contents.append(parse_sentence(sentence))

    return contents


def parse_skladnica():
    re_xml = re.compile(r'.*\.xml')
    xml_names = filter(re_xml.match, os.listdir(pj(SKLADNICA_PATH)))
    contents = np.asarray(parse_xml(*xml_names))

    with open(pj(DATA_DIR, FILENAME_SKLADNICA), 'wb') as f:
        np.save(f, contents)
