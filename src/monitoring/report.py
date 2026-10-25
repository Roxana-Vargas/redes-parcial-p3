"""E5: demostración del módulo de monitoreo ecológico en una cámara no vista en entrenamiento.

Pasa por el modelo todas las fotos de la cámara (vacías incluidas, como en un despliegue real), aplica las
reglas de src/monitoring/rules.py y genera el reporte de actividad por especie y periodo. Como la cámara
tiene etiquetas, el reporte se compara con la actividad real.

Salidas en --results-dir (sufijo cam<N>):
  e5_detecciones   detecciones por fotograma (confianza >= --pred-conf); se reutilizan si ya existen
  e5_fotos         fotos con su secuencia, hora corregida y si son infrarrojas
  e5_eventos       un evento por secuencia y especie, detectado o real
  e5_reporte       eventos por especie y periodo del día (detectados vs reales)
  e5_mensual       eventos por especie y mes
  e5_resumen       totales: fotos, tasa de vacías, especies, aciertos por evento
Y la figura reports/figures/e5_actividad_cam<N>.png.

Ejemplo:
  python src/monitoring/report.py --weights runs/full_yolov8s/weights/best.pt --location 46
"""

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from src.monitoring.rules import PERIODS, RULES, corrected_time, empty_rate, is_infrared, period, \
    sequence_events  # noqa: E402

BLUE, ORANGE, INK_2, GRID = '#2a78d6', '#eb6834', '#52514e', '#e5e4e0'


def camera_frames(ann_dir, location):
    meta = json.load(open(Path(ann_dir) / 'caltech_images_20210113.json', encoding='utf-8'))
    cat = {c['id']: c['name'] for c in meta['categories']}
    labels = defaultdict(set)
    for a in meta['annotations']:
        labels[a['image_id']].add(cat[a['category_id']])
    ims = [im for im in meta['images'] if str(im['location']) == str(location)]
    for im in ims:
        im['real'] = sorted(labels[im['id']] - {'empty'})
    return sorted(ims, key=lambda im: (im['date_captured'], im['frame_num']))


def predict(weights, ims, images_dir, pred_conf, batch=32):
    from ultralytics import YOLO
    yolo = YOLO(weights)
    dets, infrared, missing = [], {}, 0
    for start in range(0, len(ims), batch):
        chunk, imgs = [], []
        for im in ims[start:start + batch]:
            img = cv2.imread(str(Path(images_dir) / im['file_name']))
            if img is None:
                missing += 1
                continue
            chunk.append(im)
            imgs.append(img)
            infrared[im['id']] = is_infrared(img)
        if not imgs:
            continue
        for im, r in zip(chunk, yolo.predict(imgs, conf=pred_conf, imgsz=640, verbose=False)):
            for c, conf, box in zip(r.boxes.cls.tolist(), r.boxes.conf.tolist(), r.boxes.xyxyn.tolist()):
                dets.append({'imagen': im['id'], 'especie': yolo.names[int(c)], 'conf': round(conf, 4),
                             'x0': round(box[0], 4), 'y0': round(box[1], 4),
                             'x1': round(box[2], 4), 'y1': round(box[3], 4)})
        if (start // batch) % 20 == 0:
            print(f'  {start + len(chunk):,} / {len(ims):,} fotos')
    if missing:
        print(f'Aviso: {missing} fotos no están en {images_dir}')
    return pd.DataFrame(dets, columns=['imagen', 'especie', 'conf', 'x0', 'y0', 'x1', 'y1']), infrared


def build_events(frames, dets_by_image, rules):
    """Un evento por secuencia y especie, detectado (reglas) y real (etiquetas)."""
    by_seq = defaultdict(list)
    for f in frames:
        by_seq[f['seq_id']].append(f)
    events = []
    for seq, fs in by_seq.items():
        t = min(f['hora'] for f in fs)
        detected = sequence_events([dets_by_image.get(f['imagen'], []) for f in fs], rules)
        real = {sp for f in fs for sp in f['real'].split('|') if sp}
        for sp in sorted(set(detected) | real):
            d = detected.get(sp)
            events.append({'secuencia': seq, 'inicio': t, 'periodo': period(t.hour), 'mes': t.strftime('%Y-%m'),
                           'hora': t.hour, 'especie': sp, 'detectado': d is not None, 'real': sp in real,
                           'individuos': d['individuos'] if d else 0, 'fotogramas': d['fotogramas'] if d else 0,
                           'conf_max': round(d['conf_max'], 3) if d else 0})
    return pd.DataFrame(events)


def plot_activity(events, path, location, top=6):
    species = events[events['real']]['especie'].value_counts().index[:top].tolist()
    extra = [s for s in events[events['detectado']]['especie'].value_counts().index if s not in species]
    species += extra[:max(0, top - len(species))]
    n_cols = 3
    n_rows = int(np.ceil(len(species) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 2.6 * n_rows), sharex=True, squeeze=False)
    hours = np.arange(24)
    for ax in axes.flat[len(species):]:
        ax.axis('off')
    for ax, sp in zip(axes.flat, species):
        e = events[events['especie'] == sp]
        det = e[e['detectado']].groupby('hora').size().reindex(hours, fill_value=0)
        real = e[e['real']].groupby('hora').size().reindex(hours, fill_value=0)
        ax.bar(hours, det.values, width=0.8, color=BLUE, label='Detectado por el modelo')
        ax.step(hours, real.values, where='mid', color=ORANGE, lw=2, label='Real (etiquetas)')
        ax.set_title(sp, fontsize=10, loc='left')
        ax.grid(axis='y', color=GRID, lw=0.8)
        ax.set_axisbelow(True)
        for s in ('top', 'right'):
            ax.spines[s].set_visible(False)
        ax.tick_params(colors=INK_2, labelsize=8)
        ax.set_xticks([0, 6, 12, 18, 23])
    for ax in axes[-1]:
        ax.set_xlabel('Hora del día (corregida)', fontsize=8, color=INK_2)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', ncol=2, frameon=False, fontsize=9)
    fig.suptitle(f'Eventos por hora en la cámara {location} (no vista en entrenamiento)', x=0.01, ha='left',
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(path, dpi=140, bbox_inches='tight')
    plt.close(fig)


def main(args):
    rules = dict(RULES, conf=args.conf if args.conf is not None else RULES['conf'])
    results, tag = Path(args.results_dir), f'cam{args.location}'
    det_path, frames_path = results / f'e5_detecciones_{tag}.csv', results / f'e5_fotos_{tag}.csv'
    ims = camera_frames(args.ann_dir, args.location)
    print(f'Cámara {args.location}: {len(ims):,} fotos')

    if det_path.exists() and frames_path.exists() and not args.rerun:
        print(f'Usando las detecciones guardadas en {det_path}')
        dets = pd.read_csv(det_path)
        frames = pd.read_csv(frames_path)
    else:
        dets, infrared = predict(args.weights, ims, args.images_dir, args.pred_conf)
        frames = pd.DataFrame([{'imagen': im['id'], 'seq_id': im['seq_id'], 'fotograma': im['frame_num'],
                                'fecha_metadatos': im['date_captured'], 'infrarroja': infrared[im['id']],
                                'real': '|'.join(im['real'])} for im in ims if im['id'] in infrared])
        dets.to_csv(det_path, index=False)
        frames.to_csv(frames_path, index=False)

    frames['real'] = frames['real'].fillna('')
    frames['hora'] = [corrected_time(d, ir) for d, ir in zip(frames['fecha_metadatos'], frames['infrarroja'])]
    n_fixed = sum(1 for d, h in zip(frames['fecha_metadatos'], frames['hora']) if int(d[11:13]) != h.hour)
    print(f'Hora corregida (12 h -> 0 h, fotos infrarrojas): {n_fixed} fotos')

    dets_by_image = defaultdict(list)
    for r in dets.itertuples():
        dets_by_image[r.imagen].append({'especie': r.especie, 'conf': r.conf})
    events = build_events(frames.to_dict('records'), dets_by_image, rules)
    events.to_csv(results / f'e5_eventos_{tag}.csv', index=False)

    order = [name for _, _, name in PERIODS]
    rows = []
    for (sp, per), e in events.groupby(['especie', 'periodo']):
        rows.append({'especie': sp, 'periodo': per, 'eventos_detectados': int(e['detectado'].sum()),
                     'individuos_detectados': int(e['individuos'].sum()), 'eventos_reales': int(e['real'].sum())})
    report = pd.DataFrame(rows)
    report['periodo'] = pd.Categorical(report['periodo'], order, ordered=True)
    report = report.sort_values(['especie', 'periodo'])
    report.to_csv(results / f'e5_reporte_{tag}.csv', index=False)

    monthly = events.groupby(['especie', 'mes']).agg(eventos_detectados=('detectado', 'sum'),
                                                     eventos_reales=('real', 'sum')).reset_index()
    monthly.to_csv(results / f'e5_mensual_{tag}.csv', index=False)

    frames_dets = [dets_by_image.get(i, []) for i in frames['imagen']]
    tp = int((events['detectado'] & events['real']).sum())
    n_det, n_real = int(events['detectado'].sum()), int(events['real'].sum())
    summary = {
        'camara': args.location,
        'fotos': len(frames),
        'secuencias': frames['seq_id'].nunique(),
        'periodo_registro': f"{min(frames['hora']):%Y-%m-%d} a {max(frames['hora']):%Y-%m-%d}",
        'umbral_confianza': rules['conf'],
        'fotos_vacias_detectadas': f'{empty_rate(frames_dets, rules["conf"]):.3f}',
        'fotos_vacias_reales': f"{(frames['real'] == '').mean():.3f}",
        'especies_detectadas': '|'.join(sorted(events[events['detectado']]['especie'].unique())),
        'especies_reales': '|'.join(sorted(events[events['real']]['especie'].unique())),
        'eventos_detectados': n_det,
        'eventos_reales': n_real,
        'eventos_correctos': tp,
        'precision_eventos': f'{tp / n_det:.3f}' if n_det else '',
        'recall_eventos': f'{tp / n_real:.3f}' if n_real else '',
        'horas_corregidas': n_fixed,
    }
    with open(results / f'e5_resumen_{tag}.csv', 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['metrica', 'valor'])
        w.writerows(summary.items())
    for k, v in summary.items():
        print(f'  {k:24s} {v}')

    table = report.pivot_table(index='especie', columns='periodo', values=['eventos_detectados', 'eventos_reales'],
                               aggfunc='sum', observed=False)
    print('\nEventos por especie y periodo (detectados / reales):')
    print(table.to_string())

    Path(args.fig_dir).mkdir(parents=True, exist_ok=True)
    plot_activity(events, Path(args.fig_dir) / f'e5_actividad_{tag}.png', args.location)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--weights', default='runs/full_yolov8s/weights/best.pt')
    p.add_argument('--location', default='46', help='cámara de full_test (no vista por el modelo principal)')
    p.add_argument('--conf', type=float, default=None, help=f'umbral de las reglas (por defecto {RULES["conf"]})')
    p.add_argument('--pred-conf', type=float, default=0.05, help='umbral mínimo al guardar las detecciones')
    p.add_argument('--rerun', action='store_true', help='vuelve a correr el modelo aunque haya detecciones')
    p.add_argument('--ann-dir', default='data/raw/annotations')
    p.add_argument('--images-dir', default='data/raw/cct_images')
    p.add_argument('--results-dir', default='results')
    p.add_argument('--fig-dir', default='reports/figures')
    main(p.parse_args())
