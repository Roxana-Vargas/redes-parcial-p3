"""Convierte las anotaciones de Caltech Camera Traps (COCO) a formato YOLO.

Genera, dentro de --out-dir:
  labels/<id>.txt        una línea por caja: clase cx cy w h (normalizado); vacío = fondo
  images                 enlace simbólico a la carpeta de imágenes originales
  <split>.txt            listas de imágenes por partición
  data_full.yaml         dataset del modelo principal (CCT completo)
  data_eccv.yaml         dataset del modelo E3 (partición oficial ECCV 2018)

Y en --results-dir (versionado en el repo):
  dataset_summary.csv    imágenes, vacías y cajas por partición y clase
  dataset_exclusions.csv imágenes descartadas y motivo
"""

import argparse
import csv
import json
import os
import random
from collections import Counter, defaultdict
from pathlib import Path

# Clases sin cajas suficientes para entrenar o evaluar (pig: 0, cow: 0, bat: 1, insect: 7)
EXCLUDED = {'pig', 'cow', 'bat', 'insect'}
EMPTY = 'empty'
ECCV_FILES = ['train', 'cis_val', 'trans_val', 'cis_test', 'trans_test']
INTERNAL_VAL_FRAC = 0.10
SEED = 0


def load(path):
    with open(path, encoding='utf-8') as f:
        return json.load(f)


def to_yolo(bbox, width, height):
    x, y, w, h = bbox
    x0, y0 = max(x, 0), max(y, 0)
    x1, y1 = min(x + w, width), min(y + h, height)
    if x1 <= x0 or y1 <= y0:
        return None
    return ((x0 + x1) / 2 / width, (y0 + y1) / 2 / height, (x1 - x0) / width, (y1 - y0) / height)


def pick_internal_val(locs, boxes_per_loc, protected):
    """Separa cámaras de train como validación interna, hasta ~10 % de las cajas."""
    candidates = sorted(l for l in locs if boxes_per_loc[l] > 0 and l not in protected)
    random.Random(SEED).shuffle(candidates)
    target = INTERNAL_VAL_FRAC * sum(boxes_per_loc[l] for l in locs)
    chosen, total = set(), 0
    for loc in candidates:
        if total >= target:
            break
        chosen.add(loc)
        total += boxes_per_loc[loc]
    return chosen


def main(args):
    ann = Path(args.ann_dir)
    meta = load(ann / 'caltech_images_20210113.json')
    bboxes = load(ann / 'caltech_bboxes_20200316.json')
    splits = {k: {str(l) for l in v} for k, v in load(ann / 'CaltechCameraTrapsSplits_v0.json')['splits'].items()}
    eccv = {f: {im['id'] for im in load(ann / 'eccv_18_annotation_files' / f'{f}_annotations.json')['images']}
            for f in ECCV_FILES}

    cat_name = {c['id']: c['name'] for c in meta['categories']}
    classes = sorted(n for n in cat_name.values() if n != EMPTY and n not in EXCLUDED)
    class_idx = {n: i for i, n in enumerate(classes)}

    images = {im['id']: im for im in meta['images']}
    image_labels = defaultdict(set)
    for a in meta['annotations']:
        image_labels[a['image_id']].add(cat_name[a['category_id']])
    boxes = defaultdict(list)
    for a in bboxes['annotations']:
        boxes[a['image_id']].append(a)

    # Clasificar cada imagen: con cajas, fondo o descartada
    labels, excluded = {}, {}
    for img_id, im in images.items():
        species = image_labels[img_id] - {EMPTY}
        anns = [a for a in boxes[img_id] if cat_name[a['category_id']] != EMPTY]
        if not species and not anns:
            labels[img_id] = []
        elif species & EXCLUDED or any(cat_name[a['category_id']] in EXCLUDED for a in anns):
            excluded[img_id] = 'clase_excluida'
        elif not anns:
            excluded[img_id] = 'animal_sin_caja'
        elif species - {cat_name[a['category_id']] for a in anns}:
            excluded[img_id] = 'cajas_incompletas'
        else:
            rows = []
            for a in anns:
                box = to_yolo(a['bbox'], im['width'], im['height'])
                if box:
                    rows.append((class_idx[cat_name[a['category_id']]], *box))
            if rows:
                labels[img_id] = rows
            else:
                excluded[img_id] = 'caja_invalida'

    loc = {img_id: str(im['location']) for img_id, im in images.items()}
    usable = set(labels)

    # Modelo principal: cámaras de train (sin las fotos cis de ECCV) / cámaras de val como prueba
    cis_holdout = eccv['cis_val'] | eccv['cis_test']
    boxes_per_loc = Counter(loc[i] for i in usable if labels[i])
    internal_val = pick_internal_val(splits['train'], boxes_per_loc, protected=splits['eccv_train'])
    in_train_locs = {i for i in usable if loc[i] in splits['train'] and i not in cis_holdout}
    partitions = {
        'full_train': {i for i in in_train_locs if loc[i] not in internal_val},
        'full_val': {i for i in in_train_locs if loc[i] in internal_val},
        'full_test': {i for i in usable if loc[i] in splits['val']},
        # Modelo E3: partición oficial ECCV 2018 (cis = cámaras vistas, trans = cámaras nuevas)
        'eccv_train': eccv['train'] & usable,
        'eccv_cis_val': eccv['cis_val'] & usable,
        'eccv_cis_test': eccv['cis_test'] & usable,
        'eccv_trans_test': eccv['trans_test'] & usable,
    }

    out = Path(args.out_dir)
    (out / 'labels').mkdir(parents=True, exist_ok=True)
    for img_id, rows in labels.items():
        with open(out / 'labels' / f'{Path(images[img_id]["file_name"]).stem}.txt', 'w') as f:
            f.writelines(f'{c} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n' for c, cx, cy, w, h in rows)

    for name, ids in partitions.items():
        with open(out / f'{name}.txt', 'w') as f:
            f.writelines(f'./images/{images[i]["file_name"]}\n' for i in sorted(ids))

    link = out / 'images'
    if not link.exists():
        try:
            os.symlink(Path(args.images_dir).resolve(), link, target_is_directory=True)
        except OSError as e:
            print(f'Aviso: no se pudo crear el enlace {link} -> {args.images_dir} ({e})')

    names = '\n'.join(f'  {i}: {n}' for i, n in enumerate(classes))
    yamls = {
        'data_full.yaml': ('full_train', 'full_val', 'full_test'),
        'data_eccv.yaml': ('eccv_train', 'eccv_cis_val', 'eccv_cis_test'),
    }
    for fname, (tr, va, te) in yamls.items():
        with open(out / fname, 'w') as f:
            f.write(f'path: {out.resolve().as_posix()}\ntrain: {tr}.txt\nval: {va}.txt\ntest: {te}.txt\n'
                    f'names:\n{names}\n')

    results = Path(args.results_dir)
    results.mkdir(parents=True, exist_ok=True)
    with open(results / 'dataset_summary.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['particion', 'camaras', 'imagenes', 'vacias', 'con_caja', 'cajas'] + classes)
        for name, ids in partitions.items():
            per_class = Counter(classes[r[0]] for i in ids for r in labels[i])
            w.writerow([name, len({loc[i] for i in ids}), len(ids), sum(1 for i in ids if not labels[i]),
                        sum(1 for i in ids if labels[i]), sum(per_class.values())]
                       + [per_class[c] for c in classes])
    with open(results / 'dataset_exclusions.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['image_id', 'location', 'etiquetas', 'motivo'])
        for img_id, reason in sorted(excluded.items()):
            w.writerow([img_id, loc[img_id], '|'.join(sorted(image_labels[img_id])), reason])

    print(f'Clases ({len(classes)}): {", ".join(classes)}')
    print(f'Imágenes utilizables: {len(usable):,} (con caja {sum(1 for i in usable if labels[i]):,}, '
          f'vacías {sum(1 for i in usable if not labels[i]):,})')
    print('Descartadas:', dict(Counter(excluded.values())))
    print('Cámaras de validación interna:', sorted(internal_val, key=int))
    for name, ids in partitions.items():
        print(f'  {name:16s} {len(ids):7,} imágenes')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--ann-dir', default='data/raw/annotations')
    p.add_argument('--images-dir', default='data/raw/cct_images')
    p.add_argument('--out-dir', default='data/yolo')
    p.add_argument('--results-dir', default='results')
    main(p.parse_args())
