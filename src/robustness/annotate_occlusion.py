"""E4: herramienta para anotar a mano la oclusión de cada caja de annotations/occlusion_test.csv.

Muestra la imagen con la caja a anotar en verde (las otras cajas de la foto en gris) y se responde con el
teclado:
  1  baja    el animal se ve casi completo (menos de ~25 % oculto)
  2  media   falta una parte importante (~25-50 % oculto)
  3  alta    se ve menos de la mitad del animal
  b  volver a la caja anterior
  q  salir (se guarda después de cada respuesta, así que se puede seguir otro día)

Cuenta como oculto lo tapado por vegetación, troncos, otros animales o el borde de la foto.

Ejemplo:
  python src/robustness/annotate_occlusion.py
"""

import argparse
import csv
from pathlib import Path

import cv2

LEVELS = {ord('1'): 'baja', ord('2'): 'media', ord('3'): 'alta'}
MAX_W, MAX_H = 1400, 850


def load(path):
    with open(path, newline='', encoding='utf-8') as f:
        return list(csv.DictReader(f))


def save(path, rows):
    with open(path, 'w', newline='', encoding='utf-8') as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def draw(img, rows, i):
    h, w = img.shape[:2]
    s = min(MAX_W / w, MAX_H / h, 1.0)
    img = cv2.resize(img, (int(w * s), int(h * s)))
    h, w = img.shape[:2]
    for r in rows:
        if r['imagen'] == rows[i]['imagen'] and r is not rows[i]:
            p1 = (int(float(r['x1']) * w), int(float(r['y1']) * h))
            p2 = (int(float(r['x2']) * w), int(float(r['y2']) * h))
            cv2.rectangle(img, p1, p2, (160, 160, 160), 1)
    r = rows[i]
    p1 = (int(float(r['x1']) * w), int(float(r['y1']) * h))
    p2 = (int(float(r['x2']) * w), int(float(r['y2']) * h))
    cv2.rectangle(img, p1, p2, (0, 230, 0), 2)
    done = sum(1 for x in rows if x['oclusion'])
    text = f"{i + 1}/{len(rows)}  {r['especie']}  actual: {r['oclusion'] or '-'}  |  1 baja  2 media  3 alta  b atras  q salir"
    cv2.rectangle(img, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(img, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(img, f'anotadas {done}', (8, h - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)
    return img


def main(args):
    rows = load(args.csv)
    img_dir = Path(args.img_dir)
    i = next((k for k, r in enumerate(rows) if not r['oclusion']), len(rows) - 1)
    win = 'Oclusion (E4)'
    cv2.namedWindow(win, cv2.WINDOW_AUTOSIZE)
    while True:
        img = cv2.imread(str(img_dir / f"{rows[i]['imagen']}.jpg"))
        if img is None:
            print(f"No se encontró la imagen {rows[i]['imagen']}; se salta")
            if i == len(rows) - 1:
                break
            i += 1
            continue
        cv2.imshow(win, draw(img, rows, i))
        key = cv2.waitKey(0) & 0xFF
        if key == ord('q') or cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
        if key == ord('b'):
            i = max(i - 1, 0)
        elif key in LEVELS:
            rows[i]['oclusion'] = LEVELS[key]
            save(args.csv, rows)
            if all(r['oclusion'] for r in rows):
                print('Todas las cajas están anotadas.')
                break
            i = min(i + 1, len(rows) - 1)
    cv2.destroyAllWindows()
    print(f"Anotadas {sum(1 for r in rows if r['oclusion'])} de {len(rows)} cajas en {args.csv}")


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--csv', default='annotations/occlusion_test.csv')
    p.add_argument('--img-dir', default='data/occlusion')
    main(p.parse_args())
