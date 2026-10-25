"""E4: mAP por tamaño relativo de la caja (pequeño / mediano / grande).

Usa las predicciones que guarda src/eval/evaluate.py (predictions.json), así que no vuelve a correr el modelo.
El tamaño relativo es el área de la caja dividida por el área de la imagen. Los límites de los grupos son los
terciles de las cajas de full_train, para que cada grupo tenga la misma cantidad de ejemplos de entrenamiento.

La evaluación usa pycocotools con rangos de área propios, igual que los grupos small/medium/large de COCO:
al evaluar un grupo, las cajas reales de los otros grupos se ignoran, y una predicción que no coincide con
ninguna caja real solo cuenta como falso positivo si su tamaño cae dentro del grupo. Para que el área sea
relativa, las cajas se pasan a un lienzo de 1000 x 1000 (el IoU no cambia al escalar cada eje).

Ejemplo:
  python src/robustness/size_eval.py --pred runs/eval/full_yolov8s_full_test/predictions.json \
      --split full_test --model full_yolov8s
"""

import argparse
import contextlib
import csv
import io
import json
from pathlib import Path

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

CANVAS = 1000


def read_split(yolo_dir, split):
    with open(Path(yolo_dir) / f'{split}.txt') as f:
        return [Path(line.strip()).stem for line in f if line.strip()]


def read_labels(yolo_dir, stem):
    rows = []
    with open(Path(yolo_dir) / 'labels' / f'{stem}.txt') as f:
        for line in f:
            c, cx, cy, w, h = line.split()
            rows.append((int(c), float(cx), float(cy), float(w), float(h)))
    return rows


def class_names(yolo_dir):
    names, inside = {}, False
    for line in open(Path(yolo_dir) / 'data_full.yaml'):
        if line.startswith('names:'):
            inside = True
        elif inside and ':' in line:
            i, n = line.split(':')
            names[int(i)] = n.strip()
    return names


def terciles(yolo_dir, split):
    areas = [w * h for stem in read_split(yolo_dir, split) for _, _, _, w, h in read_labels(yolo_dir, stem)]
    return np.percentile(areas, [100 / 3, 200 / 3])


def coco_eval(gt, dets, area_rng, labels):
    """Devuelve mAP50, mAP50-95 y AP por clase para cada rango de área."""
    with contextlib.redirect_stdout(io.StringIO()):
        coco_gt = COCO()
        coco_gt.dataset = gt
        coco_gt.createIndex()
        coco_dt = coco_gt.loadRes(dets)
        ev = COCOeval(coco_gt, coco_dt, 'bbox')
        ev.params.areaRng = area_rng
        ev.params.areaRngLbl = labels
        ev.params.maxDets = [300]
        ev.evaluate()
        ev.accumulate()
    prec = ev.eval['precision']  # [IoU, recall, clase, área, maxDets]
    out = {}
    for a, label in enumerate(labels):
        p = prec[:, :, :, a, 0]
        ap_class = {}
        for k, cat in enumerate(ev.params.catIds):
            pk = p[:, :, k]
            if (pk > -1).any():
                ap_class[cat] = (pk[0][pk[0] > -1].mean(), pk[pk > -1].mean())
        out[label] = ap_class
    return out


def main(args):
    names = class_names(args.yolo_dir)
    t1, t2 = terciles(args.yolo_dir, args.train_split)
    groups = [('todas', 0, 1.01), ('pequeña', 0, t1), ('mediana', t1, t2), ('grande', t2, 1.01)]
    print(f'Límites de tamaño (terciles de {args.train_split}): {t1:.2%} y {t2:.2%} del área de la imagen')

    stems = read_split(args.yolo_dir, args.split)
    stem_set = set(stems)
    images, annotations, counts = [], [], {g[0]: 0 for g in groups}
    for i, stem in enumerate(stems):
        images.append({'id': stem, 'width': CANVAS, 'height': CANVAS})
        for c, cx, cy, w, h in read_labels(args.yolo_dir, stem):
            annotations.append({'id': len(annotations) + 1, 'image_id': stem, 'category_id': c, 'iscrowd': 0,
                                'bbox': [(cx - w / 2) * CANVAS, (cy - h / 2) * CANVAS, w * CANVAS, h * CANVAS],
                                'area': w * h * CANVAS ** 2})
            for g, lo, hi in groups:
                counts[g] += lo <= w * h < hi
    gt = {'images': images, 'annotations': annotations,
          'categories': [{'id': i, 'name': n} for i, n in names.items()]}

    # Tamaño original de cada imagen, para normalizar las predicciones
    meta = json.load(open(Path(args.ann_dir) / 'caltech_images_20210113.json', encoding='utf-8'))
    size = {Path(im['file_name']).stem: (im['width'], im['height']) for im in meta['images']
            if Path(im['file_name']).stem in stem_set}
    dets = []
    for d in json.load(open(args.pred)):
        stem = str(d['image_id'])
        if stem not in size:
            continue
        W, H = size[stem]
        x, y, w, h = d['bbox']
        # Ultralytics numera las clases desde 1 en predictions.json (class_map = 1..nc) y desde 0 en los labels
        dets.append({'image_id': stem, 'category_id': int(d['category_id']) - 1, 'score': d['score'],
                     'bbox': [x / W * CANVAS, y / H * CANVAS, w / W * CANVAS, h / H * CANVAS]})
    print(f'{len(stems):,} imágenes, {len(annotations):,} cajas reales, {len(dets):,} predicciones')

    area_rng = [[lo * CANVAS ** 2, hi * CANVAS ** 2] for _, lo, hi in groups]
    res = coco_eval(gt, dets, area_rng, [g[0] for g in groups])

    results = Path(args.results_dir)
    with open(results / f'e4_tamano_{args.model}_{args.split}.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['grupo', 'area_min', 'area_max', 'cajas', 'mAP50', 'mAP50_95'])
        for g, lo, hi in groups:
            ap = list(res[g].values())
            m50 = np.mean([a[0] for a in ap]) if ap else float('nan')
            m = np.mean([a[1] for a in ap]) if ap else float('nan')
            w.writerow([g, f'{lo:.4f}', f'{hi:.4f}', counts[g], f'{m50:.4f}', f'{m:.4f}'])
            print(f'  {g:8s} {counts[g]:6,} cajas   mAP@0.5 = {m50:.3f}   mAP@0.5:0.95 = {m:.3f}')
    with open(results / f'e4_tamano_{args.model}_{args.split}_especie.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['especie', 'grupo', 'AP50', 'AP50_95'])
        for g, _, _ in groups:
            for cat, (a50, a) in sorted(res[g].items()):
                w.writerow([names[cat], g, f'{a50:.4f}', f'{a:.4f}'])


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--pred', required=True, help='predictions.json generado por src/eval/evaluate.py')
    p.add_argument('--split', default='full_test')
    p.add_argument('--model', default='full_yolov8s')
    p.add_argument('--train-split', default='full_train', help='partición de la que salen los terciles')
    p.add_argument('--yolo-dir', default='data/yolo')
    p.add_argument('--ann-dir', default='data/raw/annotations')
    p.add_argument('--results-dir', default='results')
    main(p.parse_args())
