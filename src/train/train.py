"""Entrena un detector YOLO sobre una de las particiones generadas por src/data/coco_to_yolo.py.

Ejemplos:
  python src/train/train.py --data data/yolo/data_full.yaml --name full_yolov8s
  python src/train/train.py --data data/yolo/data_eccv.yaml --name eccv_yolov8s
  python src/train/train.py --name full_yolov8s --resume
"""

import argparse
from pathlib import Path

from ultralytics import YOLO


def main(args):
    run_dir = Path(args.project) / args.name
    if args.resume:
        YOLO(run_dir / 'weights' / 'last.pt').train(resume=True)
        return

    batch = int(args.batch) if args.batch is not None and args.batch >= 1 else args.batch
    overrides = {k: v for k, v in {'epochs': args.epochs, 'imgsz': args.imgsz,
                                   'batch': batch, 'fraction': args.fraction}.items() if v is not None}
    model = YOLO(args.weights)
    model.train(data=args.data, cfg=args.cfg, project=args.project, name=args.name, exist_ok=True, **overrides)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--data', default='data/yolo/data_full.yaml')
    p.add_argument('--name', required=True, help='nombre de la corrida (carpeta dentro de --project)')
    p.add_argument('--cfg', default='configs/train_base.yaml')
    p.add_argument('--weights', default='yolov8s.pt', help='pesos iniciales (preentrenados en COCO)')
    p.add_argument('--project', default='runs')
    p.add_argument('--resume', action='store_true', help='continúa la corrida desde su último checkpoint')
    p.add_argument('--epochs', type=int)
    p.add_argument('--imgsz', type=int)
    p.add_argument('--batch', type=float, help='tamaño de batch (-1 = automático)')
    p.add_argument('--fraction', type=float, help='fracción del train a usar (para corridas de prueba)')
    main(p.parse_args())
