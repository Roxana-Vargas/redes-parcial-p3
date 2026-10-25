"""Falsos positivos en fotos vacías y efecto del umbral de confianza (E3 y E6).

Usa los predictions.json que guarda src/eval/evaluate.py. Para cada umbral calcula:
  recall          cajas reales detectadas (IoU >= 0.5, misma especie) / cajas reales
  precision       detecciones correctas / detecciones
  vacias_con_fp   fracción de fotos vacías con al menos una detección (falsa alarma para un especialista)
  fp_por_vacia    detecciones promedio en cada foto vacía

E3: si el modelo memoriza el fondo de las cámaras vistas, en las fotos vacías de cámaras nuevas (trans)
debería generar más falsas alarmas que en las de cámaras vistas (cis).
E6: el umbral cambia falsos negativos por falsos positivos; la tabla muestra ese compromiso.

Ejemplo:
  python src/eval/threshold_analysis.py --eval-dir runs/eval
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

SPLITS = ['full_test', 'eccv_cis_test', 'eccv_trans_test']
THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70]


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def load_gt(yolo_dir, split):
    gt = {}
    for line in open(Path(yolo_dir) / f'{split}.txt'):
        stem = Path(line.strip()).stem
        boxes = []
        for row in open(Path(yolo_dir) / 'labels' / f'{stem}.txt'):
            c, cx, cy, w, h = row.split()
            cx, cy, w, h = map(float, (cx, cy, w, h))
            boxes.append((int(c), (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)))
        gt[stem] = boxes
    return gt


def load_preds(path, sizes):
    preds = defaultdict(list)
    for d in json.load(open(path)):
        stem = str(d['image_id'])
        W, H = sizes[stem]
        x, y, w, h = d['bbox']
        # Ultralytics numera las clases desde 1 en predictions.json
        preds[stem].append((d['score'], int(d['category_id']) - 1, (x / W, y / H, (x + w) / W, (y + h) / H)))
    return preds


def match(gt_boxes, preds, conf):
    """Emparejamiento voraz por confianza, como en COCO. Devuelve (verdaderos positivos, detecciones)."""
    dets = sorted((p for p in preds if p[0] >= conf), reverse=True)
    used, tp = set(), 0
    for _, c, box in dets:
        best, best_iou = None, 0.5
        for j, (gc, gbox) in enumerate(gt_boxes):
            if j in used or gc != c:
                continue
            v = iou(box, gbox)
            if v >= best_iou:
                best, best_iou = j, v
        if best is not None:
            used.add(best)
            tp += 1
    return tp, len(dets)


def analyze(gt, preds):
    rows = []
    n_gt = sum(len(b) for b in gt.values())
    empty = [s for s, b in gt.items() if not b]
    for conf in THRESHOLDS:
        tp = n_det = 0
        for stem, boxes in gt.items():
            t, n = match(boxes, preds.get(stem, []), conf)
            tp += t
            n_det += n
        fp_empty = [sum(1 for p in preds.get(s, []) if p[0] >= conf) for s in empty]
        rows.append({'umbral': conf, 'recall': tp / n_gt, 'precision': tp / n_det if n_det else np.nan,
                     'vacias': len(empty), 'vacias_con_fp': np.mean([n > 0 for n in fp_empty]) if empty else np.nan,
                     'fp_por_vacia': np.mean(fp_empty) if empty else np.nan})
    return rows


def main(args):
    meta = json.load(open(Path(args.ann_dir) / 'caltech_images_20210113.json', encoding='utf-8'))
    sizes = {Path(im['file_name']).stem: (im['width'], im['height']) for im in meta['images']}
    results = Path(args.results_dir)
    summary = []
    for folder in sorted(Path(args.eval_dir).iterdir()):
        split = next((s for s in SPLITS if folder.name.endswith('_' + s)), None)
        pred_path = folder / 'predictions.json'
        if split is None or not pred_path.exists():
            continue
        model = folder.name[:-len(split) - 1]
        rows = analyze(load_gt(args.yolo_dir, split), load_preds(pred_path, sizes))
        with open(results / f'umbral_{model}_{split}.csv', 'w', newline='') as f:
            w = csv.DictWriter(f, fieldnames=list(rows[0]))
            w.writeheader()
            for r in rows:
                w.writerow({k: f'{v:.4f}' if isinstance(v, float) else v for k, v in r.items()})
        r = next(r for r in rows if abs(r['umbral'] - args.conf) < 1e-9)
        summary.append({'modelo': model, 'particion': split, **{k: r[k] for k in r if k != 'umbral'}})
        print(f"{model:16s} {split:16s} umbral {args.conf}: recall {r['recall']:.3f}  precisión {r['precision']:.3f}  "
              f"vacías con falsa alarma {r['vacias_con_fp']:.1%} de {r['vacias']:,}  "
              f"({r['fp_por_vacia']:.3f} por foto)")

    with open(results / 'fp_vacias.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(summary[0]))
        w.writeheader()
        for r in summary:
            w.writerow({k: f'{v:.4f}' if isinstance(v, float) else v for k, v in r.items()})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--eval-dir', default='runs/eval', help='carpeta con <modelo>_<particion>/predictions.json')
    p.add_argument('--conf', type=float, default=0.25, help='umbral del resumen fp_vacias.csv')
    p.add_argument('--yolo-dir', default='data/yolo')
    p.add_argument('--ann-dir', default='data/raw/annotations')
    p.add_argument('--results-dir', default='results')
    main(p.parse_args())
