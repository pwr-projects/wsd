from .parser import *
from .config import *
from .skladnica import *
import os


for dirname in TEMP_DIR, DATA_DIR:
    if not os.path.isdir(dirname):
        print('Creating {}'.format(dirname))
        os.mkdir(dirname)
