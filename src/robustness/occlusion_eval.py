"""E4: recall por nivel de oclusión (baja / media / alta) con las cajas anotadas a mano.

Usa annotations/occlusion_test.csv (anotado con annotate_occlusion.py) y los predictions.json de full_test.
En cada foto se emparejan las predicciones con todas sus cajas reales, igual que en threshold_analysis.py
(voraz por confianza, IoU >= 0.5); luego se mira si cada caja anotada fue detectada:
  recall           detectada con la especie correcta
  recall_sin_clase detectada aunque con otra especie (separa "no lo vio" de "lo confundió")
  conf_media       confianza media de las detecciones correctas
No se calcula mAP por nivel porque un falso positivo no se puede asignar a un nivel de oclusión.

Ejemplo:
  python src/robustness/occlusion_eval.py --eval-dir runs/eval --models full_yolov8s full_bg18786
"""

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from size_eval import read_labels

LEVELS = ['baja', 'media', 'alta']


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def match(gt, preds, conf, use_class):
    """Devuelve {índice de caja real: confianza de la predicción emparejada}."""
    found = {}
    for score, c, box in sorted((p for p in preds if p[0] >= conf), reverse=True):
        best, best_iou = None, 0.5
        for j, (gc, gbox) in enumerate(gt):
            if j in found or (use_class and gc != c):
                continue
            v = iou(box, gbox)
            if v >= best_iou:
                best, best_iou = j, v
        if best is not None:
            found[best] = score
    return found


def main(args):
    rows = [r for r in csv.DictReader(open(args.csv, encoding='utf-8')) if r['oclusion']]
    stems = {r['imagen'] for r in rows}
    meta = json.load(open(Path(args.ann_dir) / 'caltech_images_20210113.json', encoding='utf-8'))
    sizes = {Path(im['file_name']).stem: (im['width'], im['height']) for im in meta['images']}
    gt = {s: [(c, (cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)) for c, cx, cy, w, h in read_labels(args.yolo_dir, s)]
          for s in stems}

    out = []
    for model in args.models:
        preds = defaultdict(list)
        for d in json.load(open(Path(args.eval_dir) / f'{model}_{args.split}' / 'predictions.json')):
            s = str(d['image_id'])
            if s in stems:
                W, H = sizes[s]
                x, y, w, h = d['bbox']
                # Ultralytics numera las clases desde 1 en predictions.json
                preds[s].append((d['score'], int(d['category_id']) - 1, (x / W, y / H, (x + w) / W, (y + h) / H)))
        hit = {s: match(gt[s], preds[s], args.conf, True) for s in stems}
        loc = {s: match(gt[s], preds[s], args.conf, False) for s in stems}
        for level in ['todas'] + LEVELS:
            sel = [r for r in rows if level == 'todas' or r['oclusion'] == level]
            if not sel:
                continue
            ok = [int(r['caja']) in hit[r['imagen']] for r in sel]
            confs = [hit[r['imagen']][int(r['caja'])] for r in sel if int(r['caja']) in hit[r['imagen']]]
            out.append({'modelo': model, 'oclusion': level, 'cajas': len(sel), 'recall': np.mean(ok),
                        'recall_sin_clase': np.mean([int(r['caja']) in loc[r['imagen']] for r in sel]),
                        'conf_media': np.mean(confs) if confs else np.nan})
            print(f"{model:14s} {level:6s} {len(sel):4d} cajas  recall {out[-1]['recall']:.3f}  "
                  f"sin clase {out[-1]['recall_sin_clase']:.3f}  conf {out[-1]['conf_media']:.3f}")

    counts = defaultdict(lambda: defaultdict(int))
    for r in rows:
        counts[r['especie']][r['oclusion']] += 1
    print('\nCajas por especie y nivel:')
    for sp in sorted(counts):
        print(f"  {sp:10s} " + '  '.join(f'{lv} {counts[sp][lv]:2d}' for lv in LEVELS))

    with open(Path(args.results_dir) / f'e4_oclusion_{args.split}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(out[0]))
        w.writeheader()
        for r in out:
            w.writerow({k: f'{v:.4f}' if isinstance(v, float) else v for k, v in r.items()})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--models', nargs='+', default=['full_yolov8s'])
    p.add_argument('--split', default='full_test')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--csv', default='annotations/occlusion_test.csv')
    p.add_argument('--eval-dir', default='runs/eval')
    p.add_argument('--yolo-dir', default='data/yolo')
    p.add_argument('--ann-dir', default='data/raw/annotations')
    p.add_argument('--results-dir', default='results')
    main(p.parse_args())
