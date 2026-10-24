# Monitoreo de Fauna Silvestre con Cámaras Trampa (YOLO)

Pregunta 3 del Examen Parcial de **Redes Neuronales y Aprendizaje Profundo** (UNI FIIS, 2026-II).

Detección y clasificación de especies en imágenes de cámaras trampa con YOLOv8 sobre
**Caltech Camera Traps (CCT-20)**, con evaluación de generalización a cámaras no vistas
(splits *cis* / *trans* de ECCV 2018), análisis de robustez ante oclusión y tamaño, y un
módulo de posprocesamiento para monitoreo ecológico.

## Objetivos / entregables

| # | Entregable | Ubicación |
|---|---|---|
| E1 | YOLO ajustado con decisiones de entrenamiento documentadas | `src/train/`, `configs/`, `docs/decisiones_entrenamiento.md` |
| E2 | mAP@0.5 y mAP@0.5:0.95 por especie (efecto del desbalance) | `src/eval/`, `results/` |
| E3 | Generalización entre ubicaciones (cis vs trans) + Grad-CAM | `src/eval/`, `results/` |
| E4 | Robustez por oclusión y tamaño de bounding box | `src/robustness/`, `annotations/` |
| E5 | Módulo de monitoreo ecológico + demo en cámara no vista | `src/monitoring/` |
| E6 | Discusión sobre uso responsable de IA en conservación | `reports/informe/` |

## Estructura

```
configs/              data.yaml, train_base.yaml y variantes en ablations/
data/                 dataset local (NO versionado): raw/ y yolo/
annotations/          etiquetas manuales de oclusión del test (versionadas)
src/
  data/               conversión COCO -> YOLO, splits, EDA
  train/              entrenamiento YOLO
  eval/               mAP por especie, cis vs trans, Grad-CAM
  robustness/         análisis por oclusión y tamaño
  monitoring/         reglas y reporte de actividad por cámara / especie / periodo
scripts/              run_all.sh (comando único end-to-end)
notebooks/            setup de Colab, EDA y figuras del informe
results/              tablas de métricas (CSV)
docs/                 bitácora de decisiones de entrenamiento
reports/              informe (LaTeX NeurIPS), figuras y diapositivas
```

## Dataset

Caltech Camera Traps vía LILA BC: <https://lila.science/datasets/caltech-camera-traps>

Se usa el subconjunto **CCT-20** con las anotaciones ECCV 2018
(`train`, `cis_val`, `trans_val`, `cis_test`, `trans_test`).

La descarga es manual. Ubicar los archivos así:

```
data/raw/
  images/         imágenes de CCT-20
  annotations/    JSON de ECCV 2018
```

## Instalación

```bash
pip install -r requirements.txt
```

## Ejecución end-to-end

```bash
bash scripts/run_all.sh   # por implementar
```

## Equipo

Grupo de 5 estudiantes — por completar.
