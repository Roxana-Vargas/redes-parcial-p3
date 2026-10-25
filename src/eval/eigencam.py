"""E3: mapas de activación (EigenCAM) del backbone para ver si el modelo mira al animal o al fondo.

EigenCAM (Muhammad y Yeasin, 2020) proyecta las activaciones de una capa sobre su primera componente
principal. No necesita gradientes ni elegir una clase, por eso es el método recomendado para detectores
como YOLO (Grad-CAM supone una sola salida de clasificación). Por defecto se usa la capa 9 (SPPF),
la última del backbone de YOLOv8.

Para cada imagen con animal se mide qué parte de la activación cae dentro de las cajas reales:
  energia_en_caja   fracción de la activación total que está dentro de las cajas
  area_caja         fracción de la imagen que ocupan las cajas
  concentracion     energia_en_caja / area_caja (1 = igual que al azar; > 1 = se concentra en el animal)
  max_en_caja       si el punto de máxima activación cae dentro de una caja (pointing game)
  detectado         si el modelo detecta alguna caja real (IoU >= 0.5, misma especie, confianza >= --conf)

Si el modelo aprende el fondo, en las cámaras nuevas (trans) la activación debería concentrarse menos en el
animal que en las cámaras vistas (cis), y menos aún en los animales que no detecta.

Ejemplo:
  python src/eval/eigencam.py --weights runs/eccv_yolov8s/weights/best.pt --model eccv_yolov8s
"""

import argparse
import csv
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
from ultralytics import YOLO

IMGSZ = 640
ORANGE = '#eb6834'
SUMMARY_FIELDS = ['modelo', 'particion', 'subconjunto', 'imagenes', 'energia_en_caja', 'area_caja',
                  'concentracion_mediana', 'max_en_caja']


def letterbox(img):
    h, w = img.shape[:2]
    r = min(IMGSZ / h, IMGSZ / w)
    nh, nw = round(h * r), round(w * r)
    top, left = (IMGSZ - nh) // 2, (IMGSZ - nw) // 2
    out = np.full((IMGSZ, IMGSZ, 3), 114, np.uint8)
    out[top:top + nh, left:left + nw] = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    return out, (top, left, nh, nw)


class EigenCAM:
    def __init__(self, yolo, layer):
        self.net = yolo.model.eval()
        self.acts = None
        self.net.model[layer].register_forward_hook(lambda m, i, o: setattr(self, 'acts', o.detach()))

    @torch.no_grad()
    def __call__(self, img_bgr):
        boxed, (top, left, nh, nw) = letterbox(img_bgr)
        x = torch.from_numpy(boxed[:, :, ::-1].copy()).permute(2, 0, 1).float().div(255)
        param = next(self.net.parameters())
        self.net(x[None].to(param.device, param.dtype))
        a = self.acts[0].float()
        c, h, w = a.shape
        flat = a.reshape(c, -1).T
        centered = flat - flat.mean(0)
        _, _, vt = torch.linalg.svd(centered, full_matrices=False)
        proj = centered @ vt[0]
        # El signo de la componente principal es arbitrario: se orienta como la activación media
        if torch.corrcoef(torch.stack([proj, flat.mean(1)]))[0, 1] < 0:
            proj = -proj
        cam = proj.clamp(min=0).reshape(h, w).cpu().numpy()
        cam = cv2.resize(cam, (IMGSZ, IMGSZ), interpolation=cv2.INTER_LINEAR)[top:top + nh, left:left + nw]
        return cam / cam.max() if cam.max() > 0 else cam


def read_labels(yolo_dir, stem):
    path = Path(yolo_dir) / 'labels' / f'{stem}.txt'
    return [(int(r[0]), *map(float, r[1:])) for r in (line.split() for line in open(path)) if r]


def box_mask(labels, shape):
    h, w = shape
    mask = np.zeros(shape, bool)
    for _, cx, cy, bw, bh in labels:
        x0, x1 = int((cx - bw / 2) * w), int(np.ceil((cx + bw / 2) * w))
        y0, y1 = int((cy - bh / 2) * h), int(np.ceil((cy + bh / 2) * h))
        mask[max(y0, 0):y1, max(x0, 0):x1] = True
    return mask


def iou(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter)


def detected(result, labels):
    preds = [(int(c), b) for c, b in zip(result.boxes.cls.tolist(), result.boxes.xyxyn.tolist())]
    for c, cx, cy, bw, bh in labels:
        gt = (cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2)
        if any(pc == c and iou(gt, pb) >= 0.5 for pc, pb in preds):
            return True
    return False


def summarize(rows, model, split, subset):
    if not rows:
        return None
    return {'modelo': model, 'particion': split, 'subconjunto': subset, 'imagenes': len(rows),
            'energia_en_caja': f"{np.mean([r['energia_en_caja'] for r in rows]):.4f}",
            'area_caja': f"{np.mean([r['area_caja'] for r in rows]):.4f}",
            'concentracion_mediana': f"{np.median([r['concentracion'] for r in rows]):.3f}",
            'max_en_caja': f"{np.mean([r['max_en_caja'] for r in rows]):.4f}"}


def plot_examples(examples, path, model):
    n_rows, n_cols = len(examples), max(len(v) for v in examples.values())
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 2.6 * n_rows), squeeze=False)
    for ax in axes.flat:
        ax.axis('off')
    for r, (split, items) in enumerate(examples.items()):
        for c, (img, cam, labels, result, names) in enumerate(items):
            ax = axes[r, c]
            h, w = img.shape[:2]
            ax.imshow(img[:, :, ::-1])
            ax.imshow(cam, cmap='inferno', alpha=0.5, extent=(0, w, h, 0))
            for _, cx, cy, bw, bh in labels:
                ax.add_patch(plt.Rectangle(((cx - bw / 2) * w, (cy - bh / 2) * h), bw * w, bh * h,
                                           fill=False, ec='white', lw=1.5))
            for b in result.boxes.xyxy.tolist():
                ax.add_patch(plt.Rectangle((b[0], b[1]), b[2] - b[0], b[3] - b[1], fill=False, ec=ORANGE,
                                           lw=1.5, ls='--'))
            title = ', '.join(sorted({names[l[0]] for l in labels})) or 'vacía'
            ax.set_title(title, fontsize=9)
        axes[r, 0].text(-0.04, 0.5, split.replace('eccv_', ''), transform=axes[r, 0].transAxes,
                        rotation=90, va='center', ha='right', fontsize=10)
    fig.suptitle(f'EigenCAM ({model}): caja real en blanco, predicción en naranja', fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


def main(args):
    yolo = YOLO(args.weights)
    cam_fn = EigenCAM(yolo, args.layer)
    names = yolo.names
    rng = random.Random(args.seed)
    yolo_dir = Path(args.yolo_dir)

    per_image, summary, examples = [], [], {}
    for split in args.splits:
        stems = [Path(line.strip()).name for line in open(yolo_dir / f'{split}.txt') if line.strip()]
        with_box = [s for s in stems if read_labels(yolo_dir, Path(s).stem)]
        box_set = set(with_box)
        empty = [s for s in stems if s not in box_set]
        sample = rng.sample(with_box, min(args.n, len(with_box)))
        rows = []
        for i, fname in enumerate(sample):
            img = cv2.imread(str(yolo_dir / 'images' / fname))
            labels = read_labels(yolo_dir, Path(fname).stem)
            cam = cam_fn(img)
            mask = box_mask(labels, cam.shape)
            energy = cam[mask].sum() / max(cam.sum(), 1e-9)
            area = mask.mean()
            peak = np.unravel_index(cam.argmax(), cam.shape)
            result = yolo.predict(img, conf=args.conf, imgsz=IMGSZ, verbose=False)[0]
            row = {'modelo': args.model, 'particion': split, 'imagen': fname,
                   'especies': '|'.join(sorted({names[l[0]] for l in labels})),
                   'energia_en_caja': energy, 'area_caja': area, 'concentracion': energy / max(area, 1e-9),
                   'max_en_caja': bool(mask[peak]), 'detectado': detected(result, labels)}
            rows.append(row)
            if i < args.n_examples - 1:
                examples.setdefault(split, []).append((img, cam, labels, result, names))
            if (i + 1) % 100 == 0:
                print(f'  {split}: {i + 1}/{len(sample)}')
        if empty:
            img = cv2.imread(str(yolo_dir / 'images' / rng.choice(empty)))
            examples.setdefault(split, []).append(
                (img, cam_fn(img), [], yolo.predict(img, conf=args.conf, imgsz=IMGSZ, verbose=False)[0], names))
        per_image += rows
        for subset, sel in [('todas', rows), ('detectadas', [r for r in rows if r['detectado']]),
                            ('no_detectadas', [r for r in rows if not r['detectado']])]:
            s = summarize(sel, args.model, split, subset)
            if s:
                summary.append(s)
                print(f"{split:16s} {subset:14s} n={s['imagenes']:4d}  energía en caja={s['energia_en_caja']}  "
                      f"área={s['area_caja']}  concentración={s['concentracion_mediana']}  "
                      f"máximo en caja={s['max_en_caja']}")

    results = Path(args.results_dir)
    with open(results / f'e3_eigencam_{args.model}.csv', 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=list(per_image[0]))
        w.writeheader()
        for r in per_image:
            w.writerow({k: f'{v:.4f}' if isinstance(v, float) else v for k, v in r.items()})

    summary_path = results / 'e3_eigencam_resumen.csv'
    old = []
    if summary_path.exists():
        with open(summary_path, newline='') as f:
            old = [r for r in csv.DictReader(f) if r['modelo'] != args.model]
    with open(summary_path, 'w', newline='') as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_FIELDS)
        w.writeheader()
        w.writerows(old + summary)

    fig_dir = Path(args.fig_dir)
    fig_dir.mkdir(parents=True, exist_ok=True)
    plot_examples(examples, fig_dir / f'e3_eigencam_{args.model}.png', args.model)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--weights', required=True)
    p.add_argument('--model', required=True, help='nombre del modelo para los archivos de salida')
    p.add_argument('--splits', nargs='+', default=['eccv_cis_test', 'eccv_trans_test'])
    p.add_argument('--n', type=int, default=500, help='imágenes con animal por partición')
    p.add_argument('--n-examples', type=int, default=5, help='columnas de la figura (la última es una vacía)')
    p.add_argument('--layer', type=int, default=9, help='índice de la capa de YOLOv8 (9 = SPPF)')
    p.add_argument('--conf', type=float, default=0.25)
    p.add_argument('--seed', type=int, default=0)
    p.add_argument('--yolo-dir', default='data/yolo')
    p.add_argument('--results-dir', default='results')
    p.add_argument('--fig-dir', default='reports/figures')
    main(p.parse_args())
