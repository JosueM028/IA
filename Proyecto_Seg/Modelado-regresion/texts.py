"""
Módulo para almacenar los textos de la interfaz de Gradio.
"""

TITLE_MARKDOWN = """
# 🏥 Predictor de Costos de Seguro Médico

### Modelo: Gradient Boosting Regressor
- **RMSE Test:** ~$3,562
- **MAPE Test:** ~33%
- **R² Score:** ~0.86

---
"""

PATIENT_DATA_MARKDOWN = "### 📝 Datos del Paciente"

RESULT_MARKDOWN = "### 💰 Resultado de la Predicción"

INITIAL_OUTPUT_VALUE = "Los resultados aparecerán aquí después de hacer clic en **Calcular Costo**..."

EXAMPLES_MARKDOWN = "### 📋 Ejemplos Predefinidos (haz clic para cargar)"

MODEL_INFO_MARKDOWN = """
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
"""

# --- Textos para la función de predicción ---

PREDICTION_HEADER = "💰 **Costo Estimado: ${prediction:,.2f}**\n"

SMOKER_NOTE = "⚠️ **Nota importante:** Fumar aumenta significativamente los costos (3-4x más)."

BMI_OBESE_NOTE = "📊 Su IMC indica obesidad, lo cual incrementa los costos."

BMI_UNDERWEIGHT_NOTE = "📊 Su IMC está por debajo del peso normal."

AGE_NOTE = "👴 La edad avanzada incrementa los costos médicos esperados."

ESTIMATED_RANGE = "\n📈 **Rango estimado:** ${lower:,.2f} - ${upper:,.2f}"

MODEL_CONFIDENCE = "🎯 **Confianza del modelo:** ~87% (R² = 0.86)"

PREDICTION_ERROR = "❌ Error en la predicción: {error}"

# --- Textos para el inicio de la aplicación ---

APP_STARTUP_MESSAGE = """
============================================================
🚀 Iniciando aplicación Gradio...
============================================================

📱 La aplicación se abrirá automáticamente en tu navegador
🌐 URL: http://localhost:7860

💡 Presiona Ctrl+C para detener el servidor
"""