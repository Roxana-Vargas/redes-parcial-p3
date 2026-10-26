"""E4: figuras de la anotación de oclusión.

  reports/figures/e4_oclusion_ejemplos.png   ejemplos anotados de cada nivel (una fila por nivel)
  reports/figures/e4_oclusion_recall.png     recall por nivel de oclusión (de results/e4_oclusion_full_test.csv)

Ejemplo:
  python src/robustness/occlusion_figure.py
"""

import argparse
import csv
import random
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np

LEVELS = ['baja', 'media', 'alta']
BLUE, ORANGE, INK_2, GRID = '#2a78d6', '#eb6834', '#52514e', '#e5e4e0'
MODEL_LABELS = {'full_yolov8s': 'Principal (todas las vacías)', 'full_bg18786': 'Ablación (18,786 vacías)'}


def crop(img, r, pad=0.6):
    h, w = img.shape[:2]
    x1, y1, x2, y2 = (float(r[k]) for k in ('x1', 'y1', 'x2', 'y2'))
    bw, bh = x2 - x1, y2 - y1
    cx1, cy1 = max(0, x1 - pad * bw), max(0, y1 - pad * bh)
    cx2, cy2 = min(1, x2 + pad * bw), min(1, y2 + pad * bh)
    img = img.copy()
    cv2.rectangle(img, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 230, 0), max(2, w // 400))
    return cv2.cvtColor(img[int(cy1 * h):int(cy2 * h), int(cx1 * w):int(cx2 * w)], cv2.COLOR_BGR2RGB)


def examples(rows, img_dir, out, n, seed):
    rng = random.Random(seed)
    fig, axes = plt.subplots(len(LEVELS), n, figsize=(2.6 * n, 2.4 * len(LEVELS)))
    for i, level in enumerate(LEVELS):
        sel = [r for r in rows if r['oclusion'] == level]
        for j, r in enumerate(rng.sample(sel, min(n, len(sel)))):
            ax = axes[i, j]
            ax.imshow(crop(cv2.imread(str(Path(img_dir) / f"{r['imagen']}.jpg")), r))
            ax.set_title(r['especie'], fontsize=9, color=INK_2)
        for ax in axes[i]:
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_visible(False)
        axes[i, 0].set_ylabel(f'Oclusión {level}\n({len(sel)} cajas)', fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f'Figura: {out}')


def recall_plot(results_csv, out):
    rows = [r for r in csv.DictReader(open(results_csv)) if r['oclusion'] in LEVELS]
    models = list(dict.fromkeys(r['modelo'] for r in rows))
    x = np.arange(len(LEVELS))
    width = 0.8 / len(models)
    fig, ax = plt.subplots(figsize=(6, 3.6))
    for k, (model, color) in enumerate(zip(models, [BLUE, ORANGE])):
        vals = [float(next(r['recall'] for r in rows if r['modelo'] == model and r['oclusion'] == lv)) for lv in LEVELS]
        bars = ax.bar(x + (k - (len(models) - 1) / 2) * width, vals, width * 0.92, color=color,
                      label=MODEL_LABELS.get(model, model))
        ax.bar_label(bars, fmt='%.2f', fontsize=8, color=INK_2, padding=2)
    n = {lv: next(r['cajas'] for r in rows if r['oclusion'] == lv) for lv in LEVELS}
    ax.set_xticks(x, [f'{lv}\n(n = {n[lv]})' for lv in LEVELS])
    ax.set_xlabel('Oclusión (anotada a mano)')
    ax.set_ylabel('Recall (IoU ≥ 0.5, conf ≥ 0.25)')
    ax.set_ylim(0, 1)
    ax.grid(axis='y', color=GRID)
    ax.set_axisbelow(True)
    for s in ('top', 'right'):
        ax.spines[s].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    print(f'Figura: {out}')


def main(args):
    rows = [r for r in csv.DictReader(open(args.csv, encoding='utf-8')) if r['oclusion']]
    figs = Path(args.fig_dir)
    figs.mkdir(parents=True, exist_ok=True)
    examples(rows, args.img_dir, figs / 'e4_oclusion_ejemplos.png', args.n, args.seed)
    recall_plot(args.results, figs / 'e4_oclusion_recall.png')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', default='annotations/occlusion_test.csv')
    p.add_argument('--img-dir', default='data/occlusion')
    p.add_argument('--results', default='results/e4_oclusion_full_test.csv')
    p.add_argument('--fig-dir', default='reports/figures')
    p.add_argument('--n', type=int, default=5)
    p.add_argument('--seed', type=int, default=0)
    main(p.parse_args())
