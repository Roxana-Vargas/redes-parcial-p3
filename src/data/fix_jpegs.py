"""Prepara las imágenes de CCT para Ultralytics sin alterar sus píxeles.

Ultralytics considera "corrupto" todo JPEG que no termina con el marcador de fin de imagen (EOI, bytes FF D9)
y lo vuelve a guardar con calidad 100: el archivo queda varias veces más pesado y con píxeles distintos al
original. Muchas fotos de CCT tienen bytes de relleno después del EOI. Este script agrega el marcador al final
de esas fotos (2 bytes; la imagen decodificada no cambia) para que Ultralytics no las modifique.

Con --restore, antes vuelve a descargar desde LILA las imágenes que ya fueron modificadas después de la
extracción (las que tienen fecha posterior a data/raw/.images_ok; tar conserva la fecha original de cada foto).
"""

import argparse
import os
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

LILA = 'https://storage.googleapis.com/public-datasets-lila/caltech-unzipped/cct_images/'
EOI = b'\xff\xd9'


def restore(path, attempts=3):
    tmp = path + '.tmp'
    for i in range(attempts):
        try:
            urllib.request.urlretrieve(LILA + os.path.basename(path), tmp)
            os.replace(tmp, path)
            return
        except OSError:
            if i == attempts - 1:
                raise
            time.sleep(2)


def ends_with_eoi(path):
    with open(path, 'rb') as f:
        f.seek(-2, 2)
        return f.read() == EOI


def main(args):
    files = [os.path.join(args.images_dir, f) for f in os.listdir(args.images_dir) if f.endswith('.jpg')]

    if args.restore:
        flag = os.path.getmtime(args.flag)
        modified = [p for p in files if os.path.getmtime(p) > flag]
        print(f'Restaurando desde LILA {len(modified):,} imágenes modificadas después de la extracción...')
        with ThreadPoolExecutor(32) as ex:
            for i, _ in enumerate(ex.map(restore, modified), 1):
                if i % 5000 == 0:
                    print(f'  {i:,} / {len(modified):,}')

    missing = [p for p in files if not ends_with_eoi(p)]
    for p in missing:
        with open(p, 'ab') as f:
            f.write(EOI)
    print(f'Marcador EOI agregado a {len(missing):,} de {len(files):,} imágenes')

    # Desde ahora, cualquier archivo con fecha posterior a la marca fue modificado por otro proceso
    os.utime(args.flag)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--images-dir', default='data/raw/cct_images')
    p.add_argument('--flag', default='data/raw/.images_ok', help='marca creada al terminar la extracción')
    p.add_argument('--restore', action='store_true', help='restaura desde LILA las imágenes ya modificadas')
    main(p.parse_args())
