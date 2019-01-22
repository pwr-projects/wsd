#!/bin/python

import os
import re
import xml.etree.ElementTree as ET
from enum import Enum
from os.path import join as pj

import numpy as np
from tqdm import tqdm

from .config import *
from .savable import *
from wsd.utils import tag_name

__all__ = [
    'parse_dataset',
    'load_skladnica',
    'load_kpwr',
    'WSDParseType'
]

SKLADNICA_PATH = pj(DATA_DIR, SKLADNICA_DIR)
KPWR_DOCS_PATH = pj(DATA_DIR, KPWR_DIR, KPWR_DOCS_DIR)


class WSDParseType(Enum):
    kdb = 0
    wsd = 1
    kpwr = 3


def kpwr_mapping():
    with open(pj(DATA_DIR, FILENAME_KPWR_MAPPING), 'r') as fhd:
        lines = fhd.read().strip().split('\n')

    mapping = {}

    for line in tqdm(lines, 'kpwr mapping', dynamic_ncols=True):
        matched = re.match(r'([\w\d]+-*(?:[\d\w\+]+)*)\s(\d+)', line)
        tag, synset_id = matched.groups()
        mapping[tag] = int(synset_id)
    return mapping


def parse_sentence(sentence: ET.Element, wsd_type: WSDParseType, mapping):
    wsd_type_re = {WSDParseType.wsd: 'wsd:synset',
                   WSDParseType.kdb: 'sense:ukb:syns_id',
                   WSDParseType.kpwr: 'sense:wsd_'}[wsd_type]

    concat_sentence = []
    word_synset = []

    for word in sentence.iterfind('.//tok'):
        orth = word.findtext('.//orth')
        synset_id = None
        ctag = word.findtext('.//lex/ctag')

        if wsd_type == WSDParseType.kpwr:
            props = word.findall(".//prop")

            if props:
                for tag, text in [(tag.attrib['key'], tag.text) for tag in props]:
                    if tag.startswith('sense:'):
                        synset_id = mapping.get(text)
        else:
            synset_id = word.findtext(".//prop[@key='{}']".format(wsd_type_re))

        concat_sentence.append(orth)
        word_synset.append((orth, synset_id, tag_name(ctag)))

    return ' '.join(concat_sentence), word_synset


def parse_xml(wsd_type: WSDParseType, mapping, *xml_files):
    if wsd_type == WSDParseType.kpwr:
        path = KPWR_DOCS_PATH
    else:
        path = SKLADNICA_PATH

    contents = []

    for xml_file in tqdm(xml_files, 'xml file', leave=False, dynamic_ncols=True):
        tree = ET.parse(pj(path, xml_file))

        for sentence in tree.iterfind('.//sentence'):
            parsed_sentence = parse_sentence(sentence, wsd_type, mapping)
            if len(list(filter(None, parsed_sentence[1]))) > 1:
                contents.append(parsed_sentence)
    contents = list(filter(lambda sense_map: any(sense[1] for sense in sense_map[1]), contents))
    return contents


def parse_dataset(wsd_type: WSDParseType):
    path = KPWR_DOCS_PATH if wsd_type == WSDParseType.kpwr else SKLADNICA_PATH

    mapping = kpwr_mapping() if wsd_type == WSDParseType.kpwr else None
    re_xml = re.compile(r'.*\.xml')
    xml_names = filter(re_xml.match, os.listdir(path))
    contents = np.asarray(parse_xml(wsd_type, mapping, *xml_names))
    filename = {WSDParseType.kdb: FILENAME_SKLADNICA(wsd_type),
                WSDParseType.wsd: FILENAME_SKLADNICA(wsd_type),
                WSDParseType.kpwr: FILENAME_KPWR}[wsd_type]

    with open(pj(DATA_DIR, filename), 'wb') as f:
        np.save(f, [(idx, *data) for idx, data in enumerate(contents)])


# @savable('skladnica.pkl')
def load_skladnica(wsd_type: WSDParseType):
    print('Loading skladnica...')
    with open(pj(DATA_DIR, FILENAME_SKLADNICA(wsd_type)), 'rb') as f:
        return np.load(f)


def load_kpwr():
    print('Loading kpwr...')
    with open(pj(DATA_DIR, FILENAME_KPWR), 'rb') as f:
        return np.load(f)
