"""Ablación de vacías: crea un train con menos imágenes de fondo que full_train.

Conserva todas las imágenes con animal de full_train y elige al azar (semilla fija) una parte de las vacías.
Por defecto, tantas vacías como imágenes con animal (50 % de fondo en vez de 83 %). Validación y test no
cambian, para que el resultado sea comparable con el modelo principal.

Genera en --yolo-dir:
  full_train_bg<N>.txt     lista de imágenes del nuevo train
  data_full_bg<N>.yaml     igual que data_full.yaml, pero con ese train

Ejemplo:
  python src/data/subsample_background.py              # vacías = imágenes con animal
  python src/data/subsample_background.py --n 10000
"""

import argparse
import random
from pathlib import Path


def main(args):
    yolo = Path(args.yolo_dir)
    lines = [l.strip() for l in open(yolo / 'full_train.txt') if l.strip()]
    with_box, empty = [], []
    for line in lines:
        label = yolo / 'labels' / f'{Path(line).stem}.txt'
        (with_box if label.stat().st_size > 0 else empty).append(line)

    n = args.n if args.n is not None else len(with_box)
    kept = random.Random(args.seed).sample(empty, min(n, len(empty)))
    train = sorted(with_box + kept)
    name = f'full_train_bg{len(kept)}'
    with open(yolo / f'{name}.txt', 'w') as f:
        f.writelines(f'{l}\n' for l in train)

    cfg = open(yolo / 'data_full.yaml').read().replace('train: full_train.txt', f'train: {name}.txt')
    with open(yolo / f'data_full_bg{len(kept)}.yaml', 'w') as f:
        f.write(cfg)

    print(f'{name}: {len(train):,} imágenes = {len(with_box):,} con animal + {len(kept):,} vacías '
          f'({len(kept) / len(train):.0%} de fondo; full_train tiene {len(empty) / len(lines):.0%})')
    print(f'Dataset: {yolo / f"data_full_bg{len(kept)}.yaml"}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--n', type=int, default=None, help='número de vacías (por defecto, = imágenes con animal)')
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--yolo-dir', default='data/yolo')
    main(p.parse_args())
