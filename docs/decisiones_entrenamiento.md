# Bitácora de decisiones de entrenamiento (E1)

Registro de cada decisión de entrenamiento, con su justificación. Los valores están en `configs/train_base.yaml` y son comunes a los dos modelos (principal y E3), para que sus resultados sean comparables.

## Datos

| Decisión | Valor | Justificación |
|---|---|---|
| Dataset | CCT completo, resolución original | Se usan las 243,100 imágenes sin reescalar; YOLO las redimensiona a `imgsz` al cargarlas |
| Clases | 16 especies | Se excluyen pig, cow, bat e insect (0–7 cajas) y car (sus cajas son la imagen completa `[0, 0, W, H]`) |
| Fotos con animal sin caja | Excluidas (49,801) | Sin caja, YOLO las aprendería como fondo; además la etiqueta es por secuencia y a veces el animal no aparece |
| Particiones | Por cámara | Evita que el modelo reconozca el fondo de una cámara vista en entrenamiento. Detalle en `results/dataset_summary.csv` |

## Manejo de imágenes vacías (fondo)

- **Se usan todas las vacías** de las cámaras de entrenamiento: 94,856 de 113,642 imágenes de `full_train` (83 %).
- Cada vacía tiene un archivo de etiqueta **vacío explícito** (`labels/<id>.txt` sin líneas). Para YOLO eso significa "imagen sin objetos": el modelo aprende a no detectar nada en ella y cualquier predicción ahí cuenta como falso positivo.
- Justificación: en el uso real la mayoría de las fotos de una cámara trampa están vacías (51.7 % en CCT), y un detector que genera falsos positivos en ellas hace perder tiempo a los especialistas. Entrenar con todas expone al modelo a la variedad real de fondos (viento, vegetación, cambios de luz).
- Riesgo: con 83 % de fondo el modelo podría volverse conservador (menos recall) y las épocas son más largas. La guía de Ultralytics sugiere 0–10 % de fondo; aquí se prioriza la representatividad. Si sobra tiempo de GPU, ablación con una fracción de las vacías.
- Modelo E3: `eccv_train` no tiene vacías (así viene el archivo oficial ECCV 2018).

## Modelo y pesos iniciales

- Arquitectura: **YOLOv8s** (Ultralytics), recomendada en el enunciado.
- Pesos iniciales: preentrenados en COCO (`yolov8s.pt`). El backbone ya sabe detectar animales genéricos (COCO incluye perro, gato, oso, pájaro, etc.); se ajusta a las 16 especies.
- Justificación del tamaño "s": equilibrio entre capacidad y tiempo de entrenamiento con 113k imágenes por época en una GPU de Colab. "n" es más rápido pero menos preciso; "m" duplica el costo.

## Resolución de entrada

- `imgsz = 640`.
- Justificación: es la resolución nativa de los pesos preentrenados y la estándar de YOLOv8. Las cajas son pequeñas (dos tercios ocupan menos del 4.3 % de la imagen), así que una resolución mayor podría ayudar a los animales pequeños a costa de unas 2.5 veces más cómputo. Si sobra tiempo de GPU, ablación con 1024.

## Aumento de datos

| Aumento | Valor | Justificación |
|---|---|---|
| Mosaic | 1.0 (se apaga en las últimas 10 épocas) | Combina 4 imágenes: más variedad de fondos y de tamaños por batch. Se apaga al final para ajustar con imágenes reales |
| HSV (tono, saturación, brillo) | 0.015 / 0.7 / 0.4 | Simula cambios de iluminación (amanecer, sombra, nubes). En las imágenes infrarrojas (sin color) solo actúa el brillo |
| Flip horizontal | 0.5 | Un animal puede ir hacia la izquierda o la derecha |
| Flip vertical | 0.0 | Los animales no aparecen de cabeza; las cámaras están fijas |
| Traslación / escala | 0.1 / 0.5 | Animales en distintas posiciones y distancias a la cámara |
| Rotación / shear / perspectiva | 0 | Las cámaras están fijas y niveladas |
| Mixup | 0.0 | Superponer imágenes generaría animales semitransparentes poco realistas |
| Albumentations (por defecto en Ultralytics) | Blur, MedianBlur, ToGray y CLAHE, cada uno con p = 0.01 | Variaciones muy suaves de nitidez y contraste; ToGray imita las fotos infrarrojas |

## Hiperparámetros

| Parámetro | Valor | Justificación |
|---|---|---|
| Épocas | 40 máximo | Con 113k imágenes por época, 40 épocas son suficientes para converger partiendo de pesos preentrenados |
| Early stopping | 10 épocas sin mejora | Detiene el entrenamiento si el mAP en validación se estanca; ahorra GPU |
| Batch | automático (`-1`) | Ultralytics elige el mayor batch que cabe en ~60 % de la VRAM; se adapta a la GPU asignada (L4 o A100) |
| Optimizador / LR | `auto` | Ultralytics elige SGD o AdamW y la tasa de aprendizaje según la duración del entrenamiento |
| Semilla | 0, determinista | Reproducibilidad |
| Selección del modelo | `best.pt` (mejor mAP en validación) | La validación es de cámaras distintas a las de entrenamiento |

## Hardware y tiempos

- GPU: NVIDIA L4 (22.5 GB), 12 CPU, 53 GB de RAM (Colab Pro).
- Optimizador elegido por `auto`: AdamW, lr = 0.0005, momentum 0.9.
- Batch elegido por autobatch: ~45 imágenes (9.1 GB de VRAM).
- Velocidad (corrida de prueba): 1.8 iteraciones/s ≈ 80 imágenes/s; validación de 12,580 imágenes en ~1.5 min.
- Tiempo real del modelo E3: ~88 s por época (11,201 imágenes de train + 2,573 de validación) ≈ 150 imágenes/s de entrenamiento. La GPU trabaja al 88–100 % y en su límite de potencia (72 W): el cuello de botella es la GPU, no la carga de imágenes.
- Tiempo estimado del modelo principal: ~14 min por época (113,642 + 12,580 imágenes) → ~9 h para 40 épocas en L4.
- Preparación de cada sesión de Colab: ~1.5 h para extraer las imágenes desde LILA.
- Épocas hasta early stopping: modelo E3 → mejor época 29, detenido en la 39 (0.96 h).
- Observación: en la corrida de prueba `cls_loss` fue ~191 (lo normal es 1–3). Con 83 % de imágenes vacías hay batches con muy pocos animales, y la pérdida de clasificación se normaliza por el número de objetos; el entrenamiento queda dominado por el fondo.

## Historial de experimentos

| Fecha | Corrida | Datos | Cambio respecto al base | mAP@0.5 (val) | mAP@0.5:0.95 (val) | Notas |
|---|---|---|---|---|---|---|
| 24 oct | prueba | data_full, 2 % (2,273 imágenes) | 1 época | 0.0001 | 0.00001 | Pipeline verificado de punta a punta; mAP ~0 esperado |
| | full_yolov8s | data_full | — | | | |
| 24 oct | eccv_yolov8s | data_eccv (val: eccv_cis_val) | — | 0.842 | 0.588 | Early stopping en la época 39 (mejor: 29). P = 0.91, R = 0.80. cls_loss de 3.3 a 0.39. Badger (1) y deer (2) con muy pocos ejemplos en validación |
