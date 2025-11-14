"""
Interfaz web con Gradio para predicción de costos de seguro médico
Proyecto: Modelado-regresion
"""

import gradio as gr
import pandas as pd
import joblib
import os
import texts  # Importar el módulo con los textos

# Verificar que existan los modelos
if not os.path.exists('models/best_model.pkl'):
    print("❌ ERROR: No se encontró el modelo entrenado.")
    print("   Por favor, ejecuta primero: python train_models.py")
    exit()

# Cargar modelo
print("🔄 Cargando modelo...")
model = joblib.load('models/best_model.pkl')
print("✅ Modelo cargado exitosamente\n")

def predict_insurance_cost(age, sex, bmi, children, smoker, region):
    """
    Predice el costo del seguro médico
    """
    
    # Crear DataFrame con los datos
    input_data = pd.DataFrame({
        'age': [age],
        'sex': [sex],
        'bmi': [bmi],
        'children': [children],
        'smoker': [smoker],
        'region': [region]
    })
    
    # Realizar predicción
    try:
        prediction = model.predict(input_data)[0]
        
        # Construir el resultado usando textos del módulo
        result_parts = [texts.PREDICTION_HEADER.format(prediction=prediction)]
        
        # Contexto adicional
        if smoker == 'yes':
            result_parts.append(texts.SMOKER_NOTE)
        
        if bmi > 30:
            result_parts.append(texts.BMI_OBESE_NOTE)
        elif bmi < 18.5:
            result_parts.append(texts.BMI_UNDERWEIGHT_NOTE)
        
        if age > 50:
            result_parts.append(texts.AGE_NOTE)
        
        result_parts.append(texts.ESTIMATED_RANGE.format(lower=prediction*0.85, upper=prediction*1.15))
        result_parts.append(texts.MODEL_CONFIDENCE)
        
        return "\n".join(result_parts)
        
    except Exception as e:
        return texts.PREDICTION_ERROR.format(error=e)

# Ejemplos predefinidos
examples = [
    [30, "male", 25.0, 0, "no", "southwest"],
    [45, "female", 28.5, 2, "no", "northeast"],
    [35, "male", 32.0, 1, "yes", "southeast"],
    [55, "female", 27.0, 3, "no", "northwest"],
    [25, "male", 23.5, 0, "yes", "southwest"]
]

# Crear interfaz
with gr.Blocks(title="Predictor de Costos de Seguro", theme=gr.themes.Soft()) as demo:
    
    gr.Markdown(texts.TITLE_MARKDOWN)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown(texts.PATIENT_DATA_MARKDOWN)
            
            age = gr.Slider(
                minimum=18, 
                maximum=64, 
                value=30, 
                step=1,
                label="Edad (años)"
            )
            
            sex = gr.Radio(
                choices=["male", "female"],
                value="male",
                label="Sexo"
            )
            
            bmi = gr.Slider(
                minimum=15.0,
                maximum=54.0,
                value=25.0,
                step=0.1,
                label="IMC - Índice de Masa Corporal"
            )
            
            children = gr.Slider(
                minimum=0,
                maximum=5,
                value=0,
                step=1,
                label="Número de Hijos/Dependientes"
            )
            
            smoker = gr.Radio(
                choices=["no", "yes"],
                value="no",
                label="¿Es Fumador?"
            )
            
            region = gr.Dropdown(
                choices=["southwest", "southeast", "northwest", "northeast"],
                value="southwest",
                label="Región Geográfica"
            )
            
            predict_btn = gr.Button("🔮 Calcular Costo del Seguro", variant="primary", size="lg")
        
        with gr.Column():
            gr.Markdown(texts.RESULT_MARKDOWN)
            output = gr.Markdown(
                value=texts.INITIAL_OUTPUT_VALUE,
            )
    
    gr.Markdown("---")
    gr.Markdown(texts.EXAMPLES_MARKDOWN)
    gr.Examples(
        examples=examples,
        inputs=[age, sex, bmi, children, smoker, region],
        outputs=output,
        fn=predict_insurance_cost,
        cache_examples=False
    )
    
    gr.Markdown(texts.MODEL_INFO_MARKDOWN)
    
    # Conectar el botón a la función
    predict_btn.click(
        fn=predict_insurance_cost,
        inputs=[age, sex, bmi, children, smoker, region],
        outputs=output
    )

# Lanzar la aplicación
if __name__ == "__main__":
    print(texts.APP_STARTUP_MESSAGE)
    demo.launch()