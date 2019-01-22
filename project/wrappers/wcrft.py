import os
import tempfile
from subprocess import PIPE, Popen

from itertools import chain


def tag(*sentences: str):
    fd, path = tempfile.mkstemp()

    try:
        # print('\rUsing WCRFT with', len(sentences), 'items:', '; '.join(sentences[:2]), '...', end='')

        with os.fdopen(fd, 'w') as tmp:
            tmp.writelines(map(lambda text: text + '\n' * 3, sentences))

        command=[
            'wcrft',
            'nkjp_e2.ini',
            '-C',
            '-d',
            '/usr/local/lib/python2.7/dist-packages/wcrft-1.0.0-py2.7.egg/wcrft/model/model_nkjp10_wcrft_e2/',
            '-i',
            'txt',
            path,
            '-o',
            'iob-chan'
        ]

        proc=Popen(' '.join(command), shell=True, stdout=PIPE)
        output, _=proc.communicate()
        output=output.decode('utf-8').strip().split('\n' * 2)
        output=[entry.strip().split('\n') for entry in output]
        output=[entry.strip().split('\t') for entry in chain(*output)]
        output={u: (b, p) for u, b, p in output}
        return output

    finally:
        os.remove(path)
