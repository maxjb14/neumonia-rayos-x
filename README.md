# Detección de neumonía en radiografías de tórax

Proyecto final de Redes Neuronales Artificiales y Aprendizaje Profundo (CUCEI, UDG).

Clasificación de radiografías de tórax pediátricas en NORMAL y PNEUMONIA comparando una CNN entrenada desde cero contra transfer learning con MobileNetV2.

## Datos

[Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia), Kermany et al. (2018). El notebook lo descarga con `kagglehub`.

| Conjunto | Imágenes | NORMAL | PNEUMONIA |
|---|---|---|---|
| Entrenamiento | 4 447 | 1 147 | 3 300 |
| Validación | 785 | 202 | 583 |
| Prueba | 624 | 234 | 390 |

La carpeta `val` original solo trae 16 imágenes, así que se juntó con `train` y se separó un 15 % estratificado para validación. `test` se usa tal cual.

## Modelos

- **CNN desde cero:** 4 bloques Conv2D + MaxPooling (32, 64, 128, 128 filtros), Dropout 0.5 y 0.3, Dense 128, salida sigmoide. 1 879 489 parámetros.
- **MobileNetV2 (ImageNet):** modelo base congelado + GlobalAveragePooling + Dropout 0.3 + salida sigmoide; después fine-tuning de las últimas 30 capas con tasa de aprendizaje 1e-5.

Los dos usan data augmentation (rotación, zoom y traslación), pesos por clase, Adam y Early Stopping (paciencia 5).

## Resultados en prueba

| Modelo | Exactitud | Precisión | Sensibilidad | F1 | AUC |
|---|---|---|---|---|---|
| CNN desde cero | 82.37 % | 0.7941 | 0.9692 | 0.8730 | 0.9151 |
| MobileNetV2 (TL + FT) | 86.22 % | 0.8363 | 0.9692 | 0.8979 | 0.9591 |

## Ejecución

```
pip install -r requirements.txt
jupyter notebook neumonia_cnn_vs_transfer.ipynb
```

Se ejecutó en CPU con Python 3.12 y TensorFlow 2.21. También corre en Google Colab sin cambios.
