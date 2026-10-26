"""E4: muestra de cajas de full_test para anotar la oclusión a mano.

CCT no trae etiquetas de oclusión, así que se anota una muestra. Se eligen al azar (semilla fija) hasta
--per-class cajas de cada especie de full_test, para que las especies raras también estén representadas.
Las imágenes de esas cajas se extraen del tar original de LILA (se lee de corrido una sola vez y se
detiene cuando ya encontró todas).

Genera:
  annotations/occlusion_test.csv   una fila por caja, con la columna "oclusion" vacía para anotar
  data/occlusion/<imagen>.jpg      las imágenes que se van a mostrar al anotar

Ejemplo:
  python src/robustness/occlusion_sample.py --tar ~/Downloads/cct_images.tar.gz
"""

import argparse
import csv
import random
import shutil
import tarfile
from collections import defaultdict
from pathlib import Path

from size_eval import class_names, read_labels, read_split


def sample_boxes(yolo_dir, split, per_class, seed):
    by_class = defaultdict(list)
    for stem in read_split(yolo_dir, split):
        for k, (c, cx, cy, w, h) in enumerate(read_labels(yolo_dir, stem)):
            by_class[c].append((stem, k, c, cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2))
    rng = random.Random(seed)
    boxes = []
    for c in sorted(by_class):
        boxes += rng.sample(by_class[c], min(per_class, len(by_class[c])))
    return boxes


def extract(tar_path, stems, out_dir):
    out_dir.mkdir(parents=True, exist_ok=True)
    missing = {s for s in stems if not (out_dir / f'{s}.jpg').exists()}
    print(f'Extrayendo {len(missing)} imágenes de {tar_path} (lee el tar de corrido)...')
    with tarfile.open(tar_path, 'r|*') as tar:  # el archivo de LILA es un tar sin comprimir aunque diga .gz
        for m in tar:
            stem = Path(m.name).stem
            if m.isfile() and stem in missing:
                with tar.extractfile(m) as src, open(out_dir / f'{stem}.jpg', 'wb') as dst:
                    shutil.copyfileobj(src, dst)
                missing.discard(stem)
                if not missing:
                    break
    if missing:
        print(f'Aviso: no se encontraron {len(missing)} imágenes en el tar')


def main(args):
    names = class_names(args.yolo_dir)
    boxes = sample_boxes(args.yolo_dir, args.split, args.per_class, args.seed)
    out = Path(args.out)
    if out.exists() and not args.overwrite:
        print(f'{out} ya existe (puede tener anotaciones); no se sobrescribe. Usa --overwrite para rehacerlo.')
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w', newline='', encoding='utf-8') as f:
            w = csv.writer(f)
            w.writerow(['imagen', 'caja', 'especie', 'x1', 'y1', 'x2', 'y2', 'oclusion'])
            for stem, k, c, x1, y1, x2, y2 in boxes:
                w.writerow([stem, k, names[c], f'{x1:.5f}', f'{y1:.5f}', f'{x2:.5f}', f'{y2:.5f}', ''])
        print(f'{out}: {len(boxes)} cajas de {len({b[0] for b in boxes})} imágenes')
    extract(Path(args.tar).expanduser(), {b[0] for b in boxes}, Path(args.img_dir))


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--tar', required=True, help='cct_images.tar.gz descargado de LILA')
    p.add_argument('--split', default='full_test')
    p.add_argument('--per-class', type=int, default=20)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--out', default='annotations/occlusion_test.csv')
    p.add_argument('--img-dir', default='data/occlusion')
    p.add_argument('--yolo-dir', default='data/yolo')
    p.add_argument('--overwrite', action='store_true')
    main(p.parse_args())
