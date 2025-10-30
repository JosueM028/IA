# IA-ULAT 🤖 | Portafolio de Proyectos

> **Hola, soy Josué Miranda.**
>
> Estudiante de Ingeniería Biomédica | Combino ciencia, tecnología e innovación.

## Sobre Mí

Soy estudiante de Ingeniería Biomédica en la **Universidad Latina de Panamá**, con interés en la aplicación de **Inteligencia Artificial al área de la salud**.

Me apasiona el desarrollo de soluciones tecnológicas que integren machine learning, análisis de datos y dispositivos médicos para mejorar la calidad de vida de las personas. Actualmente explorando clasificación, regresión y otras técnicas de IA aplicadas.

---

## Proyectos y Trabajos

Repositorio de tareas y laboratorios del curso de Inteligencia Artificial.

### 1. Análisis de Datos - Clasificación Binaria (Estrés)
* **Propósito:** Realizar un análisis y pre-procesamiento exhaustivo del dataset de patrones de sueño y estilo de vida. El **objetivo** fue preparar los datos para un modelo de clasificación binaria.
* **Proceso:** El análisis incluyó una **exploración de datos** (tratamiento de nulos, transformaciones), **análisis univariante** para entender las distribuciones de cada variable, y **filtrado de outliers** mediante el método de intercuartiles. Se **transformó la variable "nivel de estrés"** en una categoría binaria (Estrés Moderado vs. Estresado) para definir el objetivo del modelo. Finalmente, se realizó un **análisis bivariante** y una **matriz de correlación** para entender la relación de las variables con el estrés y se generaron los conjuntos de `train.csv` y `test.csv` (80/20) mediante una **división estratificada** para preservar la proporción de clases.
* **Enlace:** [Ver proyecto →](https://github.com/JosueM028/IA/tree/main/Proyecto_Sueno)

### 2. Modelado de Datos - Clasificación Binaria (Estrés)
* **Propósito:** Desarrollar y evaluar múltiples modelos de machine learning para **predecir el nivel de estrés** (clasificación binaria) usando los datos pre-procesados del proyecto anterior.
* **Proceso:** Se inició con un **modelo base (Regresión Logística)** y se comparó contra otros 3+ algoritmos. El rendimiento de cada modelo se evaluó rigurosamente usando un conjunto de métricas (Precision, Recall, F1-Score, Sensitivity, Specificity) y la **matriz de confusión**. Se seleccionó el modelo con mejor desempeño, justificando la elección de la métrica principal. El proyecto también incluye un **análisis de error** para comprender las fallas del modelo y una **interfaz de usuario interactiva** (Gradio/Streamlit) para demostraciones en vivo.
* **Enlace:** [Ver proyecto →](https://github.com/JosueM028/IA/tree/main/Proyecto_Sueno)

### 3. Análisis de Datos - Regresión (Costos de Seguro)
* **Propósito:** Realizar un análisis de datos exploratorio (EDA) sobre el dataset de costos de seguros médicos para **preparar los datos para un modelo de regresión**.
* **Proceso:** El proceso cubrió la **exploración inicial** y tratamiento de valores nulos, **análisis univariante** para estudiar las distribuciones de variables (como edad, IMC, etc.), y **filtrado de outliers** con el método IQR. Se realizó un **análisis bivariante** para examinar la relación entre cada característica y la variable objetivo (costos). Finalmente, se estudió la **matriz de correlación** para detectar multicolinealidad y se generaron los conjuntos de `train.csv` y `test.csv` (80/20) para el modelado.
* **Enlace:** [Ver proyecto →](https://github.com/JosueM028/IA/tree/main/Proyecto_Seg)

### 4. Modelado de Datos - Regresión (Costos de Seguro)
* **Propósito:** Implementar y comparar varios modelos de regresión para **predecir los costos de seguros médicos**.
* **Proceso:** Se estableció un **modelo base (Regresión Lineal)** y se comparó con 3+ algoritmos más avanzados. La evaluación se realizó utilizando métricas clave de regresión como **MAPE, MSE y RMSE**. Se seleccionó el modelo más preciso, justificando la elección de la métrica de evaluación (p.ej., RMSE por su sensibilidad a errores grandes). El proyecto concluye con un **análisis de error** de las predicciones y una **interfaz de usuario interactiva** (Gradio/Streamlit) para probar el modelo.
* **Enlace:** [Ver proyecto →](https://github.com/JosueM028/IA/tree/main/Proyecto_Seg)

---

## Sobre este Portafolio Web

La página `index.html` y los estilos de este repositorio son el portafolio web personal (creado con HTML, CSS y JavaScript) que se utiliza para mostrar visualmente los proyectos de IA contenidos en las carpetas.

### Contacto
* **GitHub**: [JosueM028/IA](https://github.com/JosueM028/IA)
* **Institución**: Universidad Latina de Panamá

© 2025 Josué Miranda