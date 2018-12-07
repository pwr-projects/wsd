from subprocess import check_output

__all__ = [
    'wc'
]

import os

def wc(filename):
    return int(check_output(["wc", "-l", filename]).split()[0])
