#!/usr/bin/env python3
"""
Insurance EDA
Script para realizar Exploración de Datos, tratamiento inicial, filtrado de outliers, análisis univariante y bivariante,
matriz de correlación, y división train/test estratificada por bins de la variable objetivo (charges).
Guarda figuras en analysis/output/figs y datos en data/.

Ejecutar: python analysis/insurance_eda.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split

# Config
ROOT = os.path.dirname(os.path.dirname(__file__)) if os.path.dirname(__file__) else '.'
DATA_PATH = os.path.join(ROOT, 'data', 'insurance.csv')
OUTPUT_DIR = os.path.join(ROOT, 'analysis', 'output')
FIGS_DIR = os.path.join(OUTPUT_DIR, 'figs')
os.makedirs(FIGS_DIR, exist_ok=True)
os.makedirs(os.path.join(ROOT, 'data'), exist_ok=True)

# Cargar datos
print('Cargando', DATA_PATH)
df = pd.read_csv(DATA_PATH)
print('Shape:', df.shape)

# Exploración inicial
print('\n--- Info del DataFrame ---')
print(df.info())
print('\n--- Descripción numérica ---')
print(df.describe())
print('\n--- Primeras filas ---')
print(df.head())

# Revisar valores faltantes
print('\n--- Valores faltantes por columna ---')
print(df.isnull().sum())

# Tipos de datos: convertir categóricas a 'category'
cat_cols = ['sex', 'smoker', 'region']
for c in cat_cols:
    df[c] = df[c].astype('category')

# Univariante - numéricas
num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print('\nNum columns:', num_cols)
for col in num_cols:
    plt.figure(figsize=(6,4))
    sns.histplot(df[col], kde=True)
    plt.title(f'Distribución de {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, f'univariate_{col}.png'))
    plt.close()

# Univariante - categóricas
for col in cat_cols + ['children']:
    plt.figure(figsize=(6,4))
    try:
        sns.countplot(data=df, x=col)
    except Exception:
        df[col] = df[col].astype('category')
        sns.countplot(data=df, x=col)
    plt.title(f'Conteo por {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, f'univariate_{col}.png'))
    plt.close()

# Deducciones (guardadas en archivo de texto)
with open(os.path.join(OUTPUT_DIR, 'deductions.txt'), 'w', encoding='utf-8') as f:
    f.write('Deducciones iniciales:\n')
    f.write('- Variable objetivo: charges (continua).\n')
    f.write('- Smoker suele incrementar fuertemente charges (hipótesis a comprobar).\n')
    f.write('- children es numérica discreta; revisar relación con charges.\n')

# Filtrado de outliers por IQR sobre 'charges' (y opcionalmente bmi)
def remove_outliers_iqr(df_in, col, k=1.5):
    q1 = df_in[col].quantile(0.25)
    q3 = df_in[col].quantile(0.75)
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr
    mask = (df_in[col] >= lower) & (df_in[col] <= upper)
    return df_in.loc[mask]

# Hacemos una copia filtrada por charges usando IQR k=1.5
print('\nFiltrando outliers por IQR en charges (k=1.5)')
df_filtered = remove_outliers_iqr(df, 'charges', k=1.5)
print('Shape antes:', df.shape, 'Shape después:', df_filtered.shape)
df_filtered.to_csv(os.path.join(OUTPUT_DIR, 'insurance_filtered.csv'), index=False)

# Decisión sobre la variable objetivo: mantener 'charges' continua para regresión
# También guardamos una versión transformada (log) por si se requiere modelado más adelante
if (df_filtered['charges'] <= 0).any():
    print('Advertencia: charges tiene valores no positivos; no se puede aplicar log sin ajuste')
else:
    df_filtered['log_charges'] = np.log(df_filtered['charges'])

# Análisis bivariante: numeric vs target (scatter), categorical vs target (boxplot)
for col in num_cols:
    if col == 'charges' or col == 'log_charges':
        continue
    plt.figure(figsize=(6,4))
    sns.scatterplot(data=df_filtered, x=col, y='charges', alpha=0.6)
    plt.title(f'Charges vs {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, f'bivariate_charges_vs_{col}.png'))
    plt.close()

for col in cat_cols + ['children']:
    plt.figure(figsize=(8,5))
    sns.boxplot(data=df_filtered, x=col, y='charges')
    plt.title(f'Charges por {col}')
    plt.tight_layout()
    plt.savefig(os.path.join(FIGS_DIR, f'bivariate_charges_by_{col}.png'))
    plt.close()

# Matriz de correlación para variables numéricas
corr = df_filtered.select_dtypes(include=[np.number]).corr()
plt.figure(figsize=(10,8))
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', square=True)
plt.title('Matriz de correlación (numéricas)')
plt.tight_layout()
plt.savefig(os.path.join(FIGS_DIR, 'correlation_matrix.png'))
plt.close()

# Reportar correlaciones con la variable objetivo
corr_target = corr['charges'].sort_values(ascending=False)
print('\nCorrelaciones con charges:')
print(corr_target)
with open(os.path.join(OUTPUT_DIR, 'correlations_with_target.txt'), 'w', encoding='utf-8') as f:
    f.write(corr_target.to_string())

# Eliminar variables no numéricas con poco sentido si se requiere
# Regla: si existe una pareja de variables numéricas con correlación absoluta > 0.95, elimino la menos correlacionada con charges
high_corr_pairs = []
numeric = df_filtered.select_dtypes(include=[np.number]).columns.tolist()
for i in range(len(numeric)):
    for j in range(i+1, len(numeric)):
        a, b = numeric[i], numeric[j]
        if abs(corr.loc[a, b]) > 0.95:
            high_corr_pairs.append((a,b,corr.loc[a,b]))

if high_corr_pairs:
    with open(os.path.join(OUTPUT_DIR, 'high_corr_pairs.txt'), 'w', encoding='utf-8') as f:
        for a,b,val in high_corr_pairs:
            f.write(f'{a} - {b} : {val}\n')
    # tomar decisiones (ejemplo): eliminar la variable de la pareja con menor correlación absoluta con charges
    to_drop = []
    for a,b,val in high_corr_pairs:
        if abs(corr_target[a]) >= abs(corr_target[b]):
            to_drop.append(b)
        else:
            to_drop.append(a)
    to_drop = list(set(to_drop))
    print('Variables a eliminar por alta correlación entre pares:', to_drop)
    df_model = df_filtered.drop(columns=to_drop)
else:
    df_model = df_filtered.copy()

# División train/test 80/20 estratificando por la variable objetivo binned
# Stratify requiere variable categórica: creamos bins por cuantiles
n_bins = 5
df_model['charges_bin'] = pd.qcut(df_model['charges'], q=n_bins, labels=False, duplicates='drop')

train_df, test_df = train_test_split(df_model, test_size=0.2, random_state=42, stratify=df_model['charges_bin'])

# Comprobar proporciones
prop_train = train_df['charges_bin'].value_counts(normalize=True).sort_index()
prop_test = test_df['charges_bin'].value_counts(normalize=True).sort_index()
print('\nProporciones por bin en train:\n', prop_train)
print('\nProporciones por bin en test:\n', prop_test)

# Eliminar columna auxiliar
train_df = train_df.drop(columns=['charges_bin'])
test_df = test_df.drop(columns=['charges_bin'])

# Guardar splits
DATA_OUT_DIR = os.path.join(ROOT, 'data')
train_df.to_csv(os.path.join(DATA_OUT_DIR, 'train.csv'), index=False)
test_df.to_csv(os.path.join(DATA_OUT_DIR, 'test.csv'), index=False)
print('\nGuardado train.csv y test.csv en', DATA_OUT_DIR)

# Guardar resumen
with open(os.path.join(OUTPUT_DIR, 'summary.txt'), 'w', encoding='utf-8') as f:
    f.write('Resumen del EDA y pasos realizados:\n')
    f.write(f'Shape original: {df.shape}\n')
    f.write(f'Shape después filtro IQR (charges): {df_filtered.shape}\n')
    f.write(f'Train shape: {train_df.shape}\n')
    f.write(f'Test shape: {test_df.shape}\n')

print('\nEDA completado. Revisa el directorio analysis/output para figuras y archivos de resumen.')
