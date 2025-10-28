"""
Script para entrenar y comparar modelos de regresión - Proyecto Seguro Médico
Adaptado a la estructura existente de Proyecto_Seg
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, r2_score
import joblib
import os

# Crear carpetas si no existen
os.makedirs('models', exist_ok=True)
os.makedirs('results', exist_ok=True)

print("="*60)
print("ENTRENAMIENTO DE MODELOS - PREDICCIÓN DE COSTOS DE SEGURO")
print("="*60)

# --- 1. CARGAR DATOS ---
print("\n[1/6] Cargando datos...")
try:
    # Intentar cargar desde la ruta de tu estructura
    train_df = pd.read_csv("data/train.csv")
    test_df = pd.read_csv("data/test.csv")
    print(f"✓ Train: {train_df.shape[0]} registros")
    print(f"✓ Test: {test_df.shape[0]} registros")
except FileNotFoundError:
    print("❌ ERROR: No se encontraron los archivos train.csv y test.csv")
    print("   Verifica que estés ejecutando el script desde: Proyecto_Seg/Modelado-regresion/")
    exit()

# Verificar columnas
print(f"✓ Columnas encontradas: {list(train_df.columns)}")

# Separar características y objetivo
X_train = train_df.drop('charges', axis=1)
y_train = train_df['charges']
X_test = test_df.drop('charges', axis=1)
y_test = test_df['charges']

# --- 2. DEFINIR PREPROCESAMIENTO ---
print("\n[2/6] Configurando preprocesamiento...")
numeric_features = ['age', 'bmi', 'children']
categorical_features = ['sex', 'smoker', 'region']

preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

print(f"✓ Variables numéricas: {numeric_features}")
print(f"✓ Variables categóricas: {categorical_features}")

# --- 3. MODELO BASE (LINEAR REGRESSION) ---
print("\n[3/6] Entrenando modelo base (Linear Regression)...")
base_model = Pipeline([
    ('preprocessor', preprocessor),
    ('model', LinearRegression())
])

base_model.fit(X_train, y_train)
y_pred_train_base = base_model.predict(X_train)

# Métricas del modelo base
mape_train_base = mean_absolute_percentage_error(y_train, y_pred_train_base)
mse_train_base = mean_squared_error(y_train, y_pred_train_base)
rmse_train_base = np.sqrt(mse_train_base)

print(f"✓ MAPE Train: {mape_train_base:.4f} ({mape_train_base*100:.1f}%)")
print(f"✓ MSE Train:  {mse_train_base:.2f}")
print(f"✓ RMSE Train: ${rmse_train_base:.2f}")

# --- 4. ENTRENAR MÚLTIPLES MODELOS ---
print("\n[4/6] Entrenando y comparando 4 modelos...")

models = {
    "Linear Regression": LinearRegression(),
    "Ridge Regression": Ridge(alpha=1.0),
    "Random Forest": RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1),
    "Gradient Boosting": GradientBoostingRegressor(n_estimators=100, random_state=42)
}

results = []

for name, model in models.items():
    print(f"\n  Entrenando {name}...")
    
    # Crear pipeline
    pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('model', model)
    ])
    
    # Entrenar
    pipeline.fit(X_train, y_train)
    
    # Predicciones en Train
    y_pred_train = pipeline.predict(X_train)
    mape_train = mean_absolute_percentage_error(y_train, y_pred_train)
    mse_train = mean_squared_error(y_train, y_pred_train)
    rmse_train = np.sqrt(mse_train)
    r2_train = r2_score(y_train, y_pred_train)
    
    # Predicciones en Test
    y_pred_test = pipeline.predict(X_test)
    mape_test = mean_absolute_percentage_error(y_test, y_pred_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    rmse_test = np.sqrt(mse_test)
    r2_test = r2_score(y_test, y_pred_test)
    
    results.append({
        "Model": name,
        "Train_MAPE": f"{mape_train*100:.2f}%",
        "Train_MSE": f"{mse_train:.2f}",
        "Train_RMSE": f"${rmse_train:.2f}",
        "Train_R2": f"{r2_train:.4f}",
        "Test_MAPE": f"{mape_test*100:.2f}%",
        "Test_MSE": f"{mse_test:.2f}",
        "Test_RMSE": f"${rmse_test:.2f}",
        "Test_R2": f"{r2_test:.4f}",
        "Test_RMSE_raw": rmse_test  # Para ordenar
    })
    
    print(f"  ✓ Test RMSE: ${rmse_test:.2f} | Test R²: {r2_test:.4f}")

# --- 5. GUARDAR RESULTADOS ---
print("\n[5/6] Guardando resultados...")
results_df = pd.DataFrame(results)

# Guardar con valores formateados (sin la columna raw)
results_display = results_df.drop('Test_RMSE_raw', axis=1)
results_display.to_csv('results/metrics_comparison.csv', index=False)
print("✓ Métricas guardadas en: results/metrics_comparison.csv")

# Mostrar tabla
print("\n" + "="*100)
print("TABLA COMPARATIVA DE MÉTRICAS")
print("="*100)
print(results_display.to_string(index=False))
print("="*100)

# --- 6. GUARDAR MEJOR MODELO ---
print("\n[6/6] Guardando mejor modelo...")

# Identificar mejor modelo (menor RMSE en test)
best_idx = results_df['Test_RMSE_raw'].idxmin()
best_model_name = results_df.loc[best_idx, 'Model']
best_rmse = results_df.loc[best_idx, 'Test_RMSE']

print(f"✓ Mejor modelo: {best_model_name}")
print(f"✓ Test RMSE: {best_rmse}")

# Entrenar el mejor modelo
if "Gradient" in best_model_name:
    best_model = GradientBoostingRegressor(n_estimators=100, random_state=42)
elif "Random" in best_model_name:
    best_model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
elif "Ridge" in best_model_name:
    best_model = Ridge(alpha=1.0)
else:
    best_model = LinearRegression()

best_pipeline = Pipeline([
    ('preprocessor', preprocessor),
    ('model', best_model)
])

best_pipeline.fit(X_train, y_train)

# Guardar modelo y preprocesador
joblib.dump(best_pipeline, 'models/best_model.pkl')
joblib.dump(preprocessor, 'models/preprocessor.pkl')
print("✓ Modelo guardado en: models/best_model.pkl")
print("✓ Preprocesador guardado en: models/preprocessor.pkl")

# --- 7. ANÁLISIS DE ERRORES ---
print("\n[7/7] Realizando análisis de errores...")

# Obtener predicciones del mejor modelo
y_pred = best_pipeline.predict(X_test)
errors = y_test.values - y_pred
abs_errors = np.abs(errors)

# Identificar casos con mayores errores
test_with_errors = test_df.copy()
test_with_errors['predicted'] = y_pred
test_with_errors['error'] = errors
test_with_errors['abs_error'] = abs_errors

# Top 10 peores predicciones
worst_predictions = test_with_errors.nlargest(10, 'abs_error')[
    ['age', 'sex', 'bmi', 'children', 'smoker', 'region', 'charges', 'predicted', 'error', 'abs_error']
]

worst_predictions.to_csv('results/error_analysis.csv', index=False)
print("✓ Análisis de errores guardado en: results/error_analysis.csv")

print("\n📊 Top 5 casos con mayor error:")
print(worst_predictions.head()[['age', 'bmi', 'smoker', 'charges', 'predicted', 'abs_error']].to_string(index=False))

# Estadísticas de error
print(f"\n📈 Estadísticas de Error:")
print(f"   Error promedio: ${np.mean(abs_errors):.2f}")
print(f"   Error mediano: ${np.median(abs_errors):.2f}")
print(f"   Error máximo: ${np.max(abs_errors):.2f}")
print(f"   Error mínimo: ${np.min(abs_errors):.2f}")

print("\n" + "="*60)
print("✅ ENTRENAMIENTO COMPLETADO EXITOSAMENTE")
print("="*60)
print(f"\n📋 Próximos pasos:")
print(f"1. Revisar métricas en: results/metrics_comparison.csv")
print(f"2. Analizar errores en: results/error_analysis.csv")
print(f"3. Ejecutar interfaz Gradio: python app_gradio.py")
print(f"4. O ejecutar Streamlit: streamlit run app_streamlit.py")
print("="*60)