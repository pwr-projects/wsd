import os

import requests
from tqdm import tqdm


WGET_AVAILABLE = True
try:
    import wget
except ImportError:
    WGET_AVAILABLE = False


def download(url: str, out_path: str):
    if os.path.isfile(out_path):
        print('File already exists.')
        return
    if WGET_AVAILABLE:
        wget.download(url, out=out_path)
    else:
        r = requests.get(url, stream=True)

        with open(out_path, 'wb') as f:
            file_size = int(r.headers['Content-Length'])
            chunk, i = 1024, 0
            num_bars = int(file_size / chunk)

            with tqdm(r.iter_content(chunk), total=num_bars, unit='kB', dynamic_ncols=True) as bar:
                for chunk in bar:
                    f.write(chunk)
                    bar.update()

    if not os.path.exists(out_path):
        raise 'Couldn\'t download {}'.format(out_path)
