# --- INICIO DEL CÓDIGO PARA APP.PY ---

import gradio as gr
import pandas as pd
import numpy as np
import joblib
import warnings

warnings.filterwarnings('ignore')

# --- 1. Cargar el Pipeline (Modelo + Preprocesador) ---
try:
    pipeline = joblib.load('stress_model_pipeline.pkl')
    print("Modelo cargado exitosamente.")
except FileNotFoundError:
    print("Error: Archivo 'stress_model_pipeline.pkl' no encontrado.")
    pipeline = None

# --- 2. Definir las Listas de Características (DEBEN COINCIDIR CON EL ENTRENAMIENTO) ---
numeric_features = ['Age', 'Sleep Duration', 'Quality of Sleep', 
                    'Physical Activity Level', 'Heart Rate', 'Daily Steps',
                    'Systolic_BP', 'Diastolic_BP']

categorical_features = ['Gender', 'Occupation', 'BMI Category', 'Sleep Disorder']

# --- 3. Definir Opciones para los Desplegables (de tu EDA o train.csv) ---
gender_options = ['Male', 'Female']
occupation_options = ['Nurse', 'Doctor', 'Engineer', 'Lawyer', 'Teacher', 
                      'Accountant', 'Salesperson', 'Software Engineer', 
                      'Scientist', 'Sales Representative', 'Manager']
bmi_options = ['Normal', 'Overweight', 'Normal Weight', 'Obese']
disorder_options = ['None', 'Sleep Apnea', 'Insomnia']


# --- 4. Función para Preprocesar 'Blood Pressure' ---
def engineer_bp_input(bp_string):
    try:
        systolic, diastolic = map(float, bp_string.split('/'))
        return systolic, diastolic
    except:
        # Valores por defecto si la entrada es inválida
        return np.nan, np.nan

# --- 5. La Función de Predicción (el "cerebro" de la app) ---
def predict_stress(Age, Sleep_Duration, Quality_of_Sleep, Physical_Activity_Level, 
                   Heart_Rate, Daily_Steps, Blood_Pressure,
                   Gender, Occupation, BMI_Category, Sleep_Disorder):
    
    if pipeline is None:
        return "Error: Modelo no cargado.", {"Error": 1.0}

    # 1. Convertir la presión arterial
    Systolic_BP, Diastolic_BP = engineer_bp_input(Blood_Pressure)
    
    # 2. Crear un DataFrame con UNA fila (los inputs)
    # El orden de las columnas debe ser EXACTO al usado en el preprocessor
    input_data = pd.DataFrame(
        data=[[Age, Sleep_Duration, Quality_of_Sleep, Physical_Activity_Level, Heart_Rate, Daily_Steps,
               Systolic_BP, Diastolic_BP,
               Gender, Occupation, BMI_Category, Sleep_Disorder]],
        columns=numeric_features + categorical_features # Orden crucial
    )
    
    # 3. Realizar la predicción
    try:
        prediction = pipeline.predict(input_data)[0]
        probabilities = pipeline.predict_proba(input_data)[0]
        
        # 4. Formatear la salida
        if prediction == 1:
            result = "Resultado: ESTRESADO"
        else:
            result = "Resultado: ESTRÉS MODERADO"
            
        # Crear diccionario de confianza
        confidences = {'Estrés Moderado': float(probabilities[0]), 
                       'Estresado': float(probabilities[1])}
                       
    except Exception as e:
        result = f"Error en la predicción: {e}"
        confidences = {"Error": 1.0}

    return result, confidences

# --- 6. Definir los Componentes de la Interfaz de Gradio ---
inputs = [
    # --- Entradas Numéricas ---
    gr.Slider(minimum=18, maximum=100, value=40, label="Edad (Age)"),
    gr.Slider(minimum=4.0, maximum=10.0, step=0.1, value=7.0, label="Duración del Sueño (horas)"),
    gr.Slider(minimum=1, maximum=10, step=1, value=7, label="Calidad del Sueño (1-10)"),
    gr.Slider(minimum=0, maximum=100, step=5, value=50, label="Nivel Actividad Física (min/día)"),
    gr.Slider(minimum=50, maximum=100, step=1, value=70, label="Ritmo Cardíaco (ppm)"),
    gr.Slider(minimum=1000, maximum=15000, step=100, value=6000, label="Pasos Diarios"),
    gr.Textbox(label="Presión Arterial (ej: 120/80)", value="120/80"),
    
    # --- Entradas Categóricas ---
    gr.Dropdown(choices=gender_options, label="Género", value="Male"),
    gr.Dropdown(choices=occupation_options, label="Ocupación", value="Doctor"),
    gr.Dropdown(choices=bmi_options, label="Categoría BMI", value="Normal"),
    gr.Dropdown(choices=disorder_options, label="Trastorno del Sueño", value="None")
]

outputs = [
    gr.Textbox(label="Resultado de Predicción"),
    gr.Label(label="Probabilidades")
]

# --- 7. Crear y Lanzar la Interfaz ---
iface = gr.Interface(
    fn=predict_stress,
    inputs=inputs,
    outputs=outputs,
    title="🤖 Detector de Estrés basado en Hábitos de Sueño",
    description="Ingresa tus datos de estilo de vida para predecir tu nivel de estrés. (Basado en el modelo Gradient Boosting).",
    live=False,  # Poner True para predicción en tiempo real mientras se mueven los sliders
    allow_flagging='never'
)

if __name__ == "__main__":
    print("Iniciando la aplicación Gradio...")
    iface.launch()

# --- FIN DEL CÓDIGO PARA APP.PY ---