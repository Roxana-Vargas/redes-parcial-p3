#!/usr/bin/env bash
# Ejecución completa de la Pregunta 3, de las imágenes de CCT al reporte de monitoreo.
#
# Requisitos (ver README):
#   data/raw/cct_images/      imágenes de Caltech Camera Traps (cct_images.tar.gz de LILA, extraído)
#   data/raw/annotations/     caltech_images_20210113.json, caltech_bboxes_20200316.json,
#                             CaltechCameraTrapsSplits_v0.json y eccv_18_annotation_files/
#   GPU con CUDA (la evaluación completa tarda ~1 h en una L4; el entrenamiento, ~17 h)
#
# Uso:
#   bash scripts/run_all.sh            # usa los pesos publicados en el release del repo
#   bash scripts/run_all.sh --train    # reentrena los tres modelos (principal, E3 y ablación)
#
# Las anotaciones de oclusión (annotations/occlusion_test.csv) son manuales y no se regeneran.

set -euo pipefail
cd "$(dirname "$0")/.."

TRAIN=0
[[ "${1:-}" == "--train" ]] && TRAIN=1
RELEASE=https://github.com/Roxana-Vargas/redes-parcial-p3/releases/download/pesos-v1
EVAL=runs/eval

step() { echo; echo "==> $*"; }

step "1. Etiquetas YOLO y particiones"
python src/data/coco_to_yolo.py
[[ -f data/raw/.images_ok ]] || touch data/raw/.images_ok   # marca de extracción que usa fix_jpegs.py
python src/data/fix_jpegs.py
python src/data/subsample_background.py

step "2. Modelos"
for m in full_yolov8s eccv_yolov8s full_bg18786; do
    w=runs/$m/weights/best.pt
    if [[ $TRAIN == 1 ]]; then
        case $m in
            full_yolov8s) data=data/yolo/data_full.yaml ;;
            eccv_yolov8s) data=data/yolo/data_eccv.yaml ;;
            full_bg18786) data=data/yolo/data_full_bg18786.yaml ;;
        esac
        python src/train/train.py --data $data --name $m
    elif [[ ! -f $w ]]; then
        mkdir -p "$(dirname $w)"
        curl -L --fail -o $w $RELEASE/$m.pt
    fi
done

step "3. Evaluación: E2 (por especie) y E3 (cis vs trans)"
python src/eval/evaluate.py --weights runs/full_yolov8s/weights/best.pt --data data/yolo/data_full.yaml \
    --split full_test --model full_yolov8s --project $EVAL
python src/eval/evaluate.py --weights runs/full_bg18786/weights/best.pt --data data/yolo/data_full.yaml \
    --split full_test --model full_bg18786 --project $EVAL
for split in eccv_cis_test eccv_trans_test; do
    python src/eval/evaluate.py --weights runs/eccv_yolov8s/weights/best.pt --data data/yolo/data_eccv.yaml \
        --split $split --model eccv_yolov8s --project $EVAL
    python src/eval/evaluate.py --weights runs/full_yolov8s/weights/best.pt --data data/yolo/data_full.yaml \
        --split $split --model full_yolov8s --project $EVAL
done
python src/eval/threshold_analysis.py --eval-dir $EVAL

step "4. E3: EigenCAM"
python src/eval/eigencam.py --weights runs/eccv_yolov8s/weights/best.pt --model eccv_yolov8s
python src/eval/eigencam.py --weights runs/full_yolov8s/weights/best.pt --model full_yolov8s
python src/eval/eigencam.py --weights runs/eccv_yolov8s/weights/best.pt --model eccv_yolov8s_capa15 --layer 15

step "5. E4: tamaño de caja y oclusión"
python src/robustness/size_eval.py --pred $EVAL/full_yolov8s_full_test/predictions.json \
    --split full_test --model full_yolov8s
python src/robustness/occlusion_eval.py --eval-dir $EVAL --models full_yolov8s full_bg18786
python src/robustness/occlusion_figure.py --img-dir data/raw/cct_images

step "6. E5: monitoreo en la cámara 46 (no vista)"
python src/monitoring/report.py --weights runs/full_yolov8s/weights/best.pt --location 46

step "7. Figuras del informe"
python src/eval/figures.py

echo
echo "Listo: tablas en results/ y figuras en reports/figures/"
