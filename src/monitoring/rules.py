"""Reglas del módulo de monitoreo ecológico (E5).

Convierten las detecciones de YOLO por fotograma en eventos de actividad por especie, como los usaría un
equipo de manejo de fauna. Todas las reglas son explícitas y sus parámetros están en RULES.
"""

from datetime import datetime

import cv2
import numpy as np

RULES = {
    # R1: una detección cuenta solo si su confianza supera este umbral
    'conf': 0.25,
    # R3: una especie se confirma en una secuencia si aparece en al menos min_frames fotogramas
    # o si alguna detección supera conf_alta (las secuencias de CCT tienen 1 a 5 fotos por disparo)
    'min_frames': 2,
    'conf_alta': 0.50,
}

# Periodos del día para el reporte
PERIODS = [(0, 6, 'madrugada (0-6 h)'), (6, 12, 'mañana (6-12 h)'),
           (12, 18, 'tarde (12-18 h)'), (18, 24, 'noche (18-24 h)')]


def is_infrared(img_bgr, tol=8):
    """True si la foto es infrarroja (sin color). Se ignoran los píxeles oscuros del fondo nocturno."""
    small = cv2.resize(img_bgr, (160, 120)).astype(float)
    chroma = small.max(axis=2) - small.min(axis=2)
    lit = small.max(axis=2) > 40
    return chroma[lit].mean() < tol if lit.any() else True


def corrected_time(date_captured, infrared):
    """R0: corrige el error de 12 AM / 12 PM de los metadatos de CCT.

    En los metadatos casi no hay fotos a las 00 h y hay un exceso a las 12 h: las fotos de medianoche quedaron
    registradas como mediodía. Una foto de las 12 h en infrarrojo se toma como de las 00 h. Las cámaras con
    flash blanco toman fotos nocturnas a color, que esta regla no corrige (limitación documentada).
    """
    t = datetime.strptime(date_captured, '%Y-%m-%d %H:%M:%S')
    if t.hour == 12 and infrared:
        t = t.replace(hour=0)
    return t


def period(hour):
    for lo, hi, name in PERIODS:
        if lo <= hour < hi:
            return name


def frame_species(dets, conf):
    """R1 y R2: especies de un fotograma y número de individuos de cada una (detecciones sobre el umbral)."""
    counts = {}
    for d in dets:
        if d['conf'] >= conf:
            counts[d['especie']] = counts.get(d['especie'], 0) + 1
    return counts


def sequence_events(frames, rules=RULES):
    """R3: agrega los fotogramas de una secuencia (un disparo de la cámara) en eventos por especie.

    frames: lista de listas de detecciones, una por fotograma.
    Devuelve {especie: {'individuos': máximo por fotograma, 'fotogramas': n, 'conf_max': c}} solo para las
    especies confirmadas. Se usa el máximo por fotograma, no la suma, para no contar dos veces al mismo animal.
    """
    stats = {}
    for dets in frames:
        for sp, n in frame_species(dets, rules['conf']).items():
            s = stats.setdefault(sp, {'individuos': 0, 'fotogramas': 0, 'conf_max': 0.0})
            s['individuos'] = max(s['individuos'], n)
            s['fotogramas'] += 1
            s['conf_max'] = max(s['conf_max'], max(d['conf'] for d in dets if d['especie'] == sp))
    return {sp: s for sp, s in stats.items()
            if s['fotogramas'] >= min(rules['min_frames'], len(frames)) or s['conf_max'] >= rules['conf_alta']}


def empty_rate(frames_dets, conf):
    """Fracción de fotogramas sin ninguna detección sobre el umbral."""
    empty = sum(1 for dets in frames_dets if not frame_species(dets, conf))
    return empty / len(frames_dets) if frames_dets else np.nan
