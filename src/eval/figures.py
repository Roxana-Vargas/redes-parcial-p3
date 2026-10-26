"""Figuras del informe que salen solo de los CSV de results/ (no necesitan imágenes ni GPU).

  e2_ap_por_especie.png    AP@0.5 por especie en full_test frente a cajas de entrenamiento (desbalance)
  e3_cis_vs_trans.png      mAP@0.5 en cámaras vistas y nuevas, modelo E3 y modelo principal
  e3_brecha_especie.png    AP@0.5 por especie del modelo E3 en cis y trans
  e3_fp_vacias.png         fotos vacías con falsa alarma, cis vs trans
  e4_tamano.png            mAP por tamaño relativo de la caja
  umbral_compromiso.png    recall, precisión y falsas alarmas según el umbral (modelo principal y ablación)

Ejemplo:
  python src/eval/figures.py
"""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BLUE, ORANGE = '#2a78d6', '#eb6834'
INK, INK_2, GRID = '#0b0b0b', '#52514e', '#e5e4e0'
plt.rcParams.update({
    'savefig.dpi': 200, 'savefig.bbox': 'tight', 'font.size': 9, 'axes.titlesize': 10,
    'axes.titleweight': 'bold', 'axes.edgecolor': GRID, 'axes.labelcolor': INK_2, 'axes.titlecolor': INK,
    'xtick.color': INK_2, 'ytick.color': INK_2, 'text.color': INK, 'axes.grid': True, 'grid.color': GRID,
    'axes.axisbelow': True, 'axes.spines.top': False, 'axes.spines.right': False, 'legend.frameon': False,
})
MODELS = {'eccv_yolov8s': 'Modelo E3 (10 cámaras)', 'full_yolov8s': 'Principal (76 cámaras)',
          'full_bg18786': 'Ablación (18,786 vacías)'}


def bar_pair(ax, labels, a, b, label_a, label_b):
    x = np.arange(len(labels))
    w = 0.38
    for off, vals, color, label in [(-w / 2, a, BLUE, label_a), (w / 2, b, ORANGE, label_b)]:
        bars = ax.bar(x + off, vals, w * 0.94, color=color, label=label)
        ax.bar_label(bars, fmt='%.2f', fontsize=7, color=INK_2, padding=2)
    ax.set_xticks(x, labels)
    ax.grid(axis='x', visible=False)


def e2(res, out):
    df = pd.read_csv(res / 'eval_full_yolov8s_full_test.csv').sort_values('cajas_train')
    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    ax.scatter(df['cajas_train'], df['AP50'], s=40, color=BLUE, edgecolor='white', linewidth=1, zorder=3)
    for _, r in df.iterrows():
        dy = -9 if r['especie'] == 'squirrel' else 3  # evita que se encime con cat
        ax.annotate(r['especie'], (r['cajas_train'], r['AP50']), textcoords='offset points', xytext=(5, dy),
                    fontsize=7, color=INK_2)
    ax.set_xscale('log')
    ax.set_ylim(0, 1.02)
    ax.set_xlabel('Cajas de la especie en entrenamiento (escala log)')
    ax.set_ylabel('AP@0.5 en full_test')
    rho = df[['cajas_train', 'AP50']].corr(method='spearman').iloc[0, 1]
    ax.set_title(f'AP por especie frente a ejemplos de entrenamiento (Spearman ρ = {rho:.2f})')
    fig.savefig(out / 'e2_ap_por_especie.png')
    plt.close(fig)
    return rho


def e3(res, out):
    s = pd.read_csv(res / 'eval_summary.csv')
    models = ['eccv_yolov8s', 'full_yolov8s']
    get = lambda m, p, k: float(s[(s.modelo == m) & (s.particion == p)][k].iloc[0])
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.4), sharey=True)
    for ax, k, title in zip(axes, ['mAP50', 'mAP50_95'], ['mAP@0.5', 'mAP@0.5:0.95']):
        bar_pair(ax, [MODELS[m].replace(' (', '\n(') for m in models],
                 [get(m, 'eccv_cis_test', k) for m in models], [get(m, 'eccv_trans_test', k) for m in models],
                 'cis (cámaras vistas)', 'trans (cámaras nuevas)')
        ax.set_title(title)
        ax.set_ylim(0, 1)
    axes[0].legend(loc='upper right', fontsize=7)
    fig.tight_layout()
    fig.savefig(out / 'e3_cis_vs_trans.png')
    plt.close(fig)

    cis = pd.read_csv(res / 'eval_eccv_yolov8s_eccv_cis_test.csv').set_index('especie')
    trans = pd.read_csv(res / 'eval_eccv_yolov8s_eccv_trans_test.csv').set_index('especie')
    gap = pd.DataFrame({'cis': cis['AP50'], 'trans': trans['AP50']}).dropna()
    gap = gap.loc[(gap['cis'] - gap['trans']).sort_values(ascending=False).index]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    bar_pair(ax, gap.index, gap['cis'], gap['trans'], 'cis (cámaras vistas)', 'trans (cámaras nuevas)')
    ax.set_ylabel('AP@0.5')
    ax.set_ylim(0, 1.08)
    ax.set_title('Modelo E3: AP por especie en cámaras vistas y nuevas (ordenado por caída)')
    ax.tick_params(axis='x', labelsize=7)
    ax.legend(loc='upper right', fontsize=7, ncol=2)
    fig.savefig(out / 'e3_brecha_especie.png')
    plt.close(fig)

    fp = pd.read_csv(res / 'fp_vacias.csv')
    fpv = lambda m, p: float(fp[(fp.modelo == m) & (fp.particion == p)]['vacias_con_fp'].iloc[0])
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    bar_pair(ax, [MODELS[m].replace(' (', '\n(') for m in models], [fpv(m, 'eccv_cis_test') for m in models],
             [fpv(m, 'eccv_trans_test') for m in models], 'cis (cámaras vistas)', 'trans (cámaras nuevas)')
    ax.set_ylabel('Fotos vacías con falsa alarma')
    ax.set_ylim(0, 0.45)
    ax.set_title('Falsas alarmas en fotos vacías (conf ≥ 0.25)')
    ax.legend(loc='upper right', fontsize=7)
    fig.savefig(out / 'e3_fp_vacias.png')
    plt.close(fig)


def e4(res, out):
    df = pd.read_csv(res / 'e4_tamano_full_yolov8s_full_test.csv')
    df = df[df.grupo != 'todas']
    fig, ax = plt.subplots(figsize=(5, 3.2))
    labels = [f"{g}\n({lo:.1%}–{hi:.1%})" if hi <= 1 else f"{g}\n(> {lo:.1%})"
              for g, lo, hi in zip(df.grupo, df.area_min, df.area_max)]
    bar_pair(ax, labels, df['mAP50'], df['mAP50_95'], 'mAP@0.5', 'mAP@0.5:0.95')
    ax.set_xlabel('Tamaño de la caja (fracción del área de la imagen)')
    ax.set_ylim(0, 0.7)
    ax.set_title('Modelo principal en full_test por tamaño de caja')
    ax.legend(loc='upper left', fontsize=7)
    fig.savefig(out / 'e4_tamano.png')
    plt.close(fig)


def threshold(res, out):
    runs = [(m, res / f'umbral_{m}_full_test.csv') for m in ['full_yolov8s', 'full_bg18786']]
    runs = [(m, pd.read_csv(p)) for m, p in runs if p.exists()]
    fig, axes = plt.subplots(1, 3, figsize=(9, 2.9))
    for ax, k, title in zip(axes, ['recall', 'precision', 'vacias_con_fp'],
                            ['Recall', 'Precisión', 'Fotos vacías con falsa alarma']):
        for (m, df), color in zip(runs, [BLUE, ORANGE]):
            label = 'Principal (todas las vacías)' if m == 'full_yolov8s' else MODELS[m]
            ax.plot(df['umbral'], df[k], color=color, linewidth=2, marker='o', markersize=4, label=label)
        ax.axvline(0.25, color=INK_2, linewidth=0.8, linestyle='--')
        ax.set_title(title)
        ax.set_xlabel('Umbral de confianza')
    axes[0].legend(fontsize=7, loc='lower left')
    fig.tight_layout()
    fig.savefig(out / 'umbral_compromiso.png')
    plt.close(fig)


def main(args):
    res, out = Path(args.results_dir), Path(args.fig_dir)
    out.mkdir(parents=True, exist_ok=True)
    rho = e2(res, out)
    e3(res, out)
    e4(res, out)
    threshold(res, out)
    print(f'Figuras en {out} (Spearman cajas de train vs AP@0.5 en full_test: {rho:.2f})')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--results-dir', default='results')
    p.add_argument('--fig-dir', default='reports/figures')
    main(p.parse_args())
