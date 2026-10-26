# Monitoreo de fauna silvestre con cámaras trampa (YOLO)

Pregunta 3 del Examen Parcial de **Redes Neuronales y Aprendizaje Profundo** (UNI FIIS, 2026-II).

Detección y clasificación de especies en las fotos de **Caltech Camera Traps (CCT)** con YOLOv8s:
entrenamiento con las imágenes vacías como fondo, mAP por especie, generalización a cámaras no vistas
(particiones *cis* / *trans* de ECCV 2018) con EigenCAM, robustez por oclusión y tamaño de caja, y un
módulo de reglas que convierte las detecciones en un reporte de actividad por especie y periodo del día.

## Ejecución en un solo comando

```bash
pip install -r requirements.txt
bash scripts/run_all.sh            # con los pesos entrenados (release pesos-v1)
bash scripts/run_all.sh --train    # reentrenando los tres modelos (~17 h en una L4)
```

Requiere las imágenes y anotaciones de CCT en `data/raw/` (ver [Datos](#datos)) y una GPU con CUDA.
Con los pesos publicados, la ejecución completa tarda alrededor de 1 h en una L4. Todas las tablas
quedan en `results/` y las figuras en `reports/figures/`; los resultados de nuestra corrida ya están
versionados ahí.

En Google Colab: ejecutar `notebooks/colab_setup.ipynb` (descarga y extrae las imágenes, ~1.5 h) y
luego `!bash scripts/run_all.sh` en la misma sesión, o seguir `notebooks/evaluacion.ipynb` paso a paso.

## Resultados principales

| | Resultado |
|---|---|
| **E1** Modelo principal (`full_yolov8s`) | YOLOv8s, 640 px, 40 épocas, todas las vacías como fondo; mAP@0.5 = 0.556 en `full_val` |
| **E2** `full_test` (cámaras nunca vistas) | mAP@0.5 = 0.481, mAP@0.5:0.95 = 0.310; de 0.87 (opossum) a 0 (lizard, 32 cajas en train) |
| **E3** cis → trans (modelo E3, solo `eccv_train`) | mAP@0.5 0.746 → 0.420; falsas alarmas en fotos vacías 16 % → 38 % |
| **E4** tamaño (pequeña / mediana / grande) | mAP@0.5 0.37 / 0.51 / 0.49 |
| **E4** oclusión (baja / media / alta, 298 cajas anotadas) | recall 0.63 / 0.24 / 0.16 |
| **E5** cámara 46 (no vista) | precisión por evento 0.76, recall por evento 0.59; patrones nocturnos recuperados |
| **Ablación** 18,786 vacías en vez de todas | mismo mAP (0.485), pero 5.4 % vs 2.4 % de fotos vacías con falsa alarma |

Detalle de las decisiones de entrenamiento en [`docs/decisiones_entrenamiento.md`](docs/decisiones_entrenamiento.md).

## Modelos

| Modelo | Entrenamiento | Uso |
|---|---|---|
| `full_yolov8s` | CCT completo (`full_train`, 76 cámaras), todas las imágenes vacías | E1, E2, E4, E5 |
| `eccv_yolov8s` | Partición oficial ECCV 2018 (`eccv_train`, 10 cámaras) | E3 |
| `full_bg18786` | Como el principal, pero con 18,786 vacías al azar | Ablación de E1 |

Los pesos (`best.pt`) están en el release [`pesos-v1`](https://github.com/Roxana-Vargas/redes-parcial-p3/releases/tag/pesos-v1);
`run_all.sh` los descarga en `runs/<modelo>/weights/best.pt`. Las curvas, la configuración y el notebook
ejecutado del entrenamiento están en `results/entrenamiento/`.

## Datos

Caltech Camera Traps en LILA BC: <https://lila.science/datasets/caltech-camera-traps>

```
data/raw/
  cct_images/                          cct_images.tar.gz extraído (243,100 fotos, ~105 GB)
  annotations/
    caltech_images_20210113.json       metadatos de todas las fotos (cámara, fecha, secuencia)
    caltech_bboxes_20200316.json       cajas
    eccv_18_annotation_files/          partición oficial ECCV 2018 (train, cis/trans val y test)
```

`src/data/coco_to_yolo.py` convierte las anotaciones a formato YOLO y crea las particiones:

- `full_*`: todo CCT dividido **por cámara** (las cámaras de test no aparecen en train), con las fotos vacías
  como fondo (etiqueta vacía).
- `eccv_*`: la partición oficial de ECCV 2018 para E3 (`eccv_cis_test` = cámaras vistas, `eccv_trans_test` = cámaras nuevas).

Se excluyen las fotos con animal pero sin caja, las clases casi sin cajas (pig, cow, bat, insect) y car,
cuyas cajas ocupan la imagen completa; detalle en `results/dataset_exclusions.csv`. `src/data/fix_jpegs.py`
agrega el marcador de fin a los JPEG que no lo tienen, para que Ultralytics no los recomprima.

## Estructura

```
scripts/run_all.sh       comando único end-to-end
configs/train_base.yaml  hiperparámetros comunes (resolución, aumento de datos, épocas)
src/
  data/                  coco_to_yolo, fix_jpegs, subsample_background (ablación)
  train/train.py         entrenamiento con Ultralytics
  eval/                  evaluate (mAP por especie), threshold_analysis (falsas alarmas y umbral),
                         eigencam (E3), figures (figuras del informe)
  robustness/            size_eval (tamaño de caja), occlusion_sample / annotate_occlusion /
                         occlusion_eval / occlusion_figure (oclusión)
  monitoring/            rules (reglas de monitoreo) y report (demo en una cámara)
annotations/             occlusion_test.csv: oclusión anotada a mano en 298 cajas de full_test
notebooks/               colab_setup, eda, entrenamiento, evaluacion
results/                 tablas (CSV) y artefactos de entrenamiento
reports/figures/         figuras del EDA, la evaluación y el informe
docs/                    bitácora de decisiones de entrenamiento
```

## Módulo de monitoreo (E5)

`src/monitoring/rules.py` aplica reglas fijas sobre las detecciones:

- **R0** Corrige la hora de las fotos infrarrojas que la cámara registra como 12:00.
- **R1** Solo cuentan las detecciones con confianza ≥ 0.25.
- **R2** Individuos por foto = número de cajas de la especie.
- **R3** Una secuencia de disparos es un evento por especie, confirmado si la especie aparece en al menos 2 fotos
  de la secuencia (o en todas, si tiene menos) o con confianza ≥ 0.5; individuos del evento = máximo por foto.
- Periodos del día: madrugada (0–6 h), mañana (6–12 h), tarde (12–18 h), noche (18–24 h).

`src/monitoring/report.py` genera el reporte de una cámara: detecciones, especies distintas, tasa de
fotos vacías frente a fotos con fauna, y actividad por especie y periodo, comparada con las etiquetas reales.

```bash
python src/monitoring/report.py --weights runs/full_yolov8s/weights/best.pt --location 46
```

## Oclusión (E4)

CCT no trae etiquetas de oclusión. La muestra se generó con `occlusion_sample.py` (20 cajas por especie
de `full_test`, semilla 0) y se anotó a mano con `annotate_occlusion.py` (teclas 1/2/3 = baja/media/alta).
Para repetir la anotación desde cero:

```bash
python src/robustness/occlusion_sample.py --tar ruta/a/cct_images.tar.gz --overwrite
python src/robustness/annotate_occlusion.py
```

## Autores

Grupo de cinco estudiantes (nombres en el informe) — Examen Parcial 2026-II, Sección B.
