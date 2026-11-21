# TP1 — Regresión Softmax sobre 20 Newsgroups (TNG)

Redes Neuronales / Deep Learning — Fernández San Juan

## Qué hace

Clasifica los posts de 20 Newsgroups (~18.800 posts de foros, 20 categorías) con una **regresión softmax**: una sola capa lineal sobre vectores TF-IDF, implementada en PyTorch.

- Preprocesamiento configurable: stopwords, stemming o lematización, n-gramas.
- Búsqueda de hiperparámetros de preprocesamiento y entrenamiento con **Optuna**.
- Seguimiento de experimentos con **MLflow** y **TensorBoard**; además, todas las curvas se grafican en la notebook con matplotlib.
- Entrenamiento final con train / validación / test separados y early stopping sobre validación. El test se usa una sola vez.
- Repetición del modelo final con 3 semillas y comparación de la entrada TF-IDF contra la estandarizada.

## Archivos

| Archivo | Contenido |
|---|---|
| `TP1_RegresionSoftmax_FernandezSanJuan.ipynb` | Notebook del trabajo (versión corregida), ejecutada con todas las salidas |
| `requirements.txt` | Librerías necesarias |
| `README.md` | Este archivo |

## Cómo correrlo

**En Google Colab (recomendado):** abrir la notebook, elegir *Entorno de ejecución → Cambiar tipo de entorno → T4 GPU* y luego *Ejecutar todo*. La primera celda instala lo que falte.

**Localmente** (Python 3.10 o superior):

```
pip install -r requirements.txt
jupyter notebook TP1_RegresionSoftmax_FernandezSanJuan.ipynb
```

El dataset se descarga solo con `sklearn.datasets.fetch_20newsgroups`. Las semillas están fijas (incluida la del sampler de Optuna), aunque en GPU puede haber diferencias mínimas entre corridas. Para ver las curvas en TensorBoard: `tensorboard --logdir runs`. Para ver los experimentos en MLflow: `mlflow ui`.

## Correcciones respecto de la primera entrega

- Se completaron las respuestas 9 y 10.
- El tiempo se mide en el código con `time.perf_counter()`.
- Las curvas finales se grafican en la notebook, además de TensorBoard.
- El test ya no se usa como validación: hay early stopping sobre un conjunto de validación separado y el test se usa una sola vez.
- El modelo final se repite con 3 semillas.
- Se compara la entrada TF-IDF contra la estandarizada.
- Una sola línea de instalación, y se agregan README y requirements.
