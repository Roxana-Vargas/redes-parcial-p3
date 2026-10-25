"""Evalúa un modelo YOLO entrenado sobre una partición y guarda las métricas globales y por especie.

Ejemplos:
  python src/eval/evaluate.py --weights runs/full_yolov8s/weights/best.pt --data data/yolo/data_full.yaml \
      --split full_test --model full_yolov8s
  python src/eval/evaluate.py --weights runs/eccv_yolov8s/weights/best.pt --data data/yolo/data_eccv.yaml \
      --split eccv_trans_test --model eccv_yolov8s

Salidas en --results-dir:
  eval_<modelo>_<particion>.csv   métricas por especie, con el nº de cajas en train y en la partición evaluada
  eval_summary.csv                una fila por evaluación (se actualiza si ya existe esa combinación)

Y en --project/<modelo>_<particion>/predictions.json todas las predicciones (formato COCO, coordenadas en
píxeles de la imagen original). E4 y el análisis del umbral de confianza las usan sin volver a correr el modelo.
"""

import argparse
import csv
import tempfile
from pathlib import Path

import yaml
from ultralytics import YOLO

SUMMARY_FIELDS = ['modelo', 'particion', 'imagenes', 'mAP50', 'mAP50_95', 'precision', 'recall']


def box_counts(results_dir, partition):
    with open(Path(results_dir) / 'dataset_summary.csv', newline='') as f:
        for row in csv.DictReader(f):
            if row['particion'] == partition:
                return row
    raise ValueError(f'No hay datos de la partición {partition} en dataset_summary.csv')


def main(args):
    with open(args.data) as f:
        cfg = yaml.safe_load(f)
    train_split = Path(cfg['train']).stem
    cfg['test'] = f'{args.split}.txt'
    with tempfile.NamedTemporaryFile('w', suffix='.yaml', delete=False) as f:
        yaml.safe_dump(cfg, f)
        data = f.name

    m = YOLO(args.weights).val(data=data, split='test', imgsz=args.imgsz, batch=args.batch,
                               project=args.project, name=f'{args.model}_{args.split}', exist_ok=True,
                               save_json=True)

    train_counts = box_counts(args.results_dir, train_split)
    test_counts = box_counts(args.results_dir, args.split)
    results = Path(args.results_dir)

    with open(results / f'eval_{args.model}_{args.split}.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['especie', 'cajas_train', 'cajas_eval', 'AP50', 'AP50_95', 'precision', 'recall'])
        for i, c in enumerate(m.ap_class_index):
            p, r, ap50, ap = m.box.class_result(i)
            name = m.names[int(c)]
            w.writerow([name, train_counts[name], test_counts[name],
                        f'{ap50:.4f}', f'{ap:.4f}', f'{p:.4f}', f'{r:.4f}'])

    summary_path = results / 'eval_summary.csv'
    rows = []
    if summary_path.exists():
        with open(summary_path, newline='') as f:
            rows = [r for r in csv.DictReader(f) if (r['modelo'], r['particion']) != (args.model, args.split)]
    rows.append({'modelo': args.model, 'particion': args.split, 'imagenes': test_counts['imagenes'],
                 'mAP50': f'{m.box.map50:.4f}', 'mAP50_95': f'{m.box.map:.4f}',
                 'precision': f'{m.box.mp:.4f}', 'recall': f'{m.box.mr:.4f}'})
    with open(summary_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(rows)

    print(f'{args.model} en {args.split}: mAP@0.5 = {m.box.map50:.3f} | mAP@0.5:0.95 = {m.box.map:.3f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--weights', required=True)
    p.add_argument('--data', required=True, help='data_full.yaml o data_eccv.yaml')
    p.add_argument('--split', required=True, help='nombre de la partición, p. ej. full_test o eccv_trans_test')
    p.add_argument('--model', required=True, help='nombre del modelo para los archivos de salida')
    p.add_argument('--imgsz', type=int, default=640)
    p.add_argument('--batch', type=int, default=32)
    p.add_argument('--project', default='runs/eval')
    p.add_argument('--results-dir', default='results')
    main(p.parse_args())
