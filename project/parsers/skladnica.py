#!/bin/python

import os
import re
import xml.etree.ElementTree as ET
from enum import Enum
from os.path import join as pj

import numpy as np
from tqdm import tqdm

from .config import *

__all__ = [
    'parse_skladnica',
    'load_skladnica',
    'WSDParseType'
]

SKLADNICA_PATH = pj(DATA_DIR, SKLADNICA_DIR)


class WSDParseType(Enum):
    kdb = 0
    wsd = 1


def parse_sentence(sentence: ET.Element, wsd_type: WSDParseType):
    wsd_type = {WSDParseType.wsd: 'wsd:synset',
                WSDParseType.kdb: 'sense:ukb:syns_id'}[wsd_type]

    concat_sentence = []
    word_synset = []

    for word in sentence.iterfind('.//tok'):
        orth = word.findtext('.//orth')
        synset_id = word.findtext(".//prop[@key='{}']".format(wsd_type))
        concat_sentence.append(orth)
        word_synset.append((orth, synset_id))
    
    return ' '.join(concat_sentence), word_synset


def parse_xml(wsd_type: WSDParseType, *xml_files):
    contents = []

    for xml_file in tqdm(xml_files, 'xml file', leave=False):
        tree = ET.parse(pj(SKLADNICA_PATH, xml_file))

        for sentence in tree.iterfind('.//sentence'):
            parsed_sentence = parse_sentence(sentence, wsd_type)
            if len(list(filter(None, parsed_sentence[1]))) > 1:
                contents.append(parsed_sentence)
    contents = list(filter(lambda sense_map: any(sense[1] for sense in sense_map[1]), contents))
    return contents


def parse_skladnica(wsd_type: WSDParseType):
    re_xml = re.compile(r'.*\.xml')
    xml_names = filter(re_xml.match, os.listdir(pj(SKLADNICA_PATH)))
    contents = np.asarray(parse_xml(wsd_type, *xml_names))

    with open(pj(DATA_DIR, FILENAME_SKLADNICA(wsd_type)), 'wb') as f:
        np.save(f, [(idx, *data) for idx, data in enumerate(contents)])


def load_skladnica(wsd_type: WSDParseType):
    with open(pj(DATA_DIR, FILENAME_SKLADNICA(wsd_type)), 'rb') as f:
        return np.load(f)
