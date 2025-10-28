"""
Interfaz web con Gradio para predicción de costos de seguro médico
Proyecto: Modelado-regresion
"""

import gradio as gr
import pandas as pd
import joblib
import os

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
        
        # Formatear resultado
        result = f"💰 **Costo Estimado: ${prediction:,.2f}**\n\n"
        
        # Contexto adicional
        if smoker == 'yes':
            result += "⚠️ **Nota importante:** Fumar aumenta significativamente los costos (3-4x más).\n"
        
        if bmi > 30:
            result += "📊 Su IMC indica obesidad, lo cual incrementa los costos.\n"
        elif bmi < 18.5:
            result += "📊 Su IMC está por debajo del peso normal.\n"
        
        if age > 50:
            result += "👴 La edad avanzada incrementa los costos médicos esperados.\n"
        
        result += f"\n📈 **Rango estimado:** ${prediction*0.85:,.2f} - ${prediction*1.15:,.2f}"
        result += f"\n\n🎯 **Confianza del modelo:** ~87% (R² = 0.86)"
        
        return result
        
    except Exception as e:
        return f"❌ Error en la predicción: {str(e)}"

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
    
    gr.Markdown("""
    # 🏥 Predictor de Costos de Seguro Médico
    
    ### Modelo: Gradient Boosting Regressor
    - **RMSE Test:** ~$3,562
    - **MAPE Test:** ~33%
    - **R² Score:** ~0.86
    
    ---
    """)
    
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 📝 Datos del Paciente")
            
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
            gr.Markdown("### 💰 Resultado de la Predicción")
            output = gr.Markdown(
                value="Los resultados aparecerán aquí después de hacer clic en **Calcular Costo**...",
            )
    
    gr.Markdown("---")
    gr.Markdown("### 📋 Ejemplos Predefinidos (haz clic para cargar)")
    gr.Examples(
        examples=examples,
        inputs=[age, sex, bmi, children, smoker, region],
        outputs=output,
        fn=predict_insurance_cost,
        cache_examples=False
    )
    
    predict_btn.click(
        fn=predict_insurance_cost,
        inputs=[age, sex, bmi, children, smoker, region],
        outputs=output
    )
    
    gr.Markdown("""
    ---
    ### ℹ️ Información del Modelo
    
    **Variables de entrada:**
    - **age**: 18-64 años
    - **sex**: masculino, femenino
    - **bmi**: 15.0-54.0 (Índice de Masa Corporal)
    - **children**: 0-5 dependientes
    - **smoker**: sí/no (⚠️ factor más importante)
    - **region**: noreste, noroeste, sureste, suroeste
    
    **⚠️ Nota Importante:** Este modelo es una herramienta de estimación educativa. 
    Los costos reales pueden variar según factores adicionales no incluidos en el modelo 
    (historial médico completo, condiciones preexistentes, tipo de cobertura, etc.).
    
    ---
    
    **Proyecto:** Modelado de Regresión - Costos de Seguro Médico  
    **Modelo:** Gradient Boosting Regressor  
    **Dataset:** Insurance Cost Dataset
    """)

# Lanzar aplicación
if __name__ == "__main__":
    print("\n" + "="*60)
    print("🚀 Iniciando aplicación Gradio...")
    print("="*60)
    print("\n📱 La aplicación se abrirá automáticamente en tu navegador")
    print("🌐 URL: http://localhost:7860")
    print("\n💡 Presiona Ctrl+C para detener el servidor\n")
    
    demo.launch(
        share=False,
        server_name="127.0.0.1",
        server_port=7860,
        show_error=True
    )