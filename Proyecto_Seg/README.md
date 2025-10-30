# Insurance EDA

Análisis exploratorio de datos (EDA) sobre el dataset `insurance.csv` incluido en este repositorio. El objetivo es preparar y guardar los conjuntos train/test que se usarán posteriormente para la modelización.

Contenido:
- analysis/insurance_eda.py : script reproducible que realiza toda la exploración, limpieza, visualizaciones, matriz de correlación y genera los archivos `data/train.csv` y `data/test.csv`.
- data/insurance.csv : dataset original (ya presente en el repo).
- docs/index.html : página simple para GitHub Pages con enlace al análisis.

Instrucciones de uso:
1. Clona el repo y sitúate en su raíz.
2. Asegúrate de tener Python 3.8+ y las librerías listadas abajo.
3. Ejecuta: python analysis/insurance_eda.py
4. Los resultados (figuras y data splits) se guardarán en `analysis/output/` y `data/train.csv`, `data/test.csv`.

Dependencias (pip):
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn

Motivación: Este EDA prepara los datos (tratamiento de outliers por IQR, análisis univariante y bivariante, correlaciones) y guarda splits estratificados sobre la variable objetivo binned para conservar la distribución de costos.
