"""
Script para evaluar el modelo LightGBM y calcular el threshold óptimo con costo 5 y 200.
"""

import numpy as np
import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.metrics import confusion_matrix
from tools import optimizar_threshold_costo_cv, evaluate_model, generar_thresholds

# Configuración
MODEL_PATH = 'models/lightgbm_fraude.pkl'
DATA_PATH = '../../data/processed/fraud_dataset_processed.csv'
COSTO_FP = 5.0
COSTO_FN = 200.0
RANDOM_STATE = 42
TEST_SIZE = 0.25

print("=" * 80)
print("EVALUACIÓN DEL MODELO LIGHTGBM")
print("=" * 80)

# 1. Cargar datos
print("\n1. Cargando datos...")
dataset = pd.read_csv(DATA_PATH)

# Preparar datos igual que en el notebook
numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()

# Excluir columnas que pueden causar data leakage
columns_to_exclude = ['fraude', 'row_id', 'Unnamed: 0', 'score']
if 'fraude' in numeric_cols:
    numeric_cols.remove('fraude')
for col in columns_to_exclude:
    if col in numeric_cols:
        numeric_cols.remove(col)

# Agregar 'fraude' al final
if 'fraude' in dataset.columns:
    numeric_cols.append('fraude')

dataset = dataset[numeric_cols]

print(f"   Total de columnas: {len(numeric_cols) - 1}")
print(f"   Total de registros: {len(dataset)}")

# 2. Definir X, y
print("\n2. Preparando variables...")
X = dataset.drop(columns=['fraude'])
y = dataset['fraude']

print(f"   Forma de X: {X.shape}")
print(f"   Proporción de fraude: {y.mean():.4f}")

# 3. Train/test split
print("\n3. Dividiendo en train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
)

print(f"   Train: {X_train.shape[0]} registros")
print(f"   Test: {X_test.shape[0]} registros")

# 4. Cargar modelo
print("\n4. Cargando modelo...")
model = joblib.load(MODEL_PATH)
print(f"   ✅ Modelo cargado desde: {MODEL_PATH}")

# 5. Obtener probabilidades en test
print("\n5. Obteniendo probabilidades en test...")
y_proba_test = model.predict_proba(X_test)[:, 1]
print(f"   Probabilidades obtenidas: {len(y_proba_test)}")

# 6. Optimizar threshold usando Cross-Validation en train
print("\n6. Optimizando threshold con Cross-Validation...")
print(f"   Costo FP: {COSTO_FP}")
print(f"   Costo FN: {COSTO_FN}")

# Configurar Cross-Validation
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

# Optimizar threshold
resultados_optimizacion = optimizar_threshold_costo_cv(
    model=model,
    X=X_train,
    y=y_train,
    cv=cv,
    costo_fp=COSTO_FP,
    costo_fn=COSTO_FN,
    model_name="LightGBM"
)

threshold_optimo = resultados_optimizacion['threshold_optimo']
costo_optimo = resultados_optimizacion['costo_optimo']

print(f"\n   ✅ Threshold óptimo: {threshold_optimo:.4f}")
print(f"   ✅ Costo mínimo por 1000 registros: {costo_optimo:.2f}")

# 7. Evaluar en test con threshold óptimo
print("\n7. Evaluando en test con threshold óptimo...")
y_pred_test = (y_proba_test >= threshold_optimo).astype(int)

# Calcular matriz de confusión
cm = confusion_matrix(y_test, y_pred_test)
if cm.size == 4:
    tn, fp, fn, tp = cm.ravel()
else:
    if len(np.unique(y_pred_test)) == 1:
        if y_pred_test[0] == 0:
            tn, fp, fn, tp = len(y_test) - y_test.sum(), 0, y_test.sum(), 0
        else:
            tn, fp, fn, tp = 0, (y_test == 0).sum(), 0, y_test.sum()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

print(f"\n   Matriz de Confusión:")
print(f"   {'':>15} {'Predicho 0':>15} {'Predicho 1':>15}")
print(f"   {'Real 0':>15} {tn:>15} {fp:>15}")
print(f"   {'Real 1':>15} {fn:>15} {tp:>15}")

# 8. Calcular costo total en test
print("\n8. Calculando costo total...")

# Calcular proporción original
prop_original = np.mean(y_train)
prop_negativos_original = 1 - prop_original
prop_positivos = np.mean(y_test)
prop_negativos_nueva = 1 - prop_positivos

# Calcular factores de escala
if prop_negativos_original > 0:
    factor_fp = prop_negativos_nueva / prop_negativos_original
else:
    factor_fp = 1.0

if prop_original > 0:
    factor_fn = prop_positivos / prop_original
else:
    factor_fn = 1.0

# Ajustar FP y FN según la proporción
fp_ajustado = fp * factor_fp
fn_ajustado = fn * factor_fn

# Calcular costo total
costo_total = fp_ajustado * COSTO_FP + fn_ajustado * COSTO_FN
costo_por_1000 = (costo_total / len(y_test)) * 1000

print(f"\n   📊 RESULTADOS EN TEST:")
print(f"   {'':>30} {'Valor':>15}")
print(f"   {'False Positives (FP)':>30} {fp:>15}")
print(f"   {'False Negatives (FN)':>30} {fn:>15}")
print(f"   {'FP ajustado':>30} {fp_ajustado:>15.2f}")
print(f"   {'FN ajustado':>30} {fn_ajustado:>15.2f}")
print(f"   {'Costo FP (ajustado × 5)':>30} {fp_ajustado * COSTO_FP:>15.2f}")
print(f"   {'Costo FN (ajustado × 200)':>30} {fn_ajustado * COSTO_FN:>15.2f}")
print(f"   {'COSTO TOTAL':>30} {costo_total:>15.2f}")
print(f"   {'Costo por 1000 registros':>30} {costo_por_1000:>15.2f}")

# 9. Evaluación completa con evaluate_model
print("\n9. Evaluación completa del modelo...")
metrics = evaluate_model(
    y_true=y_test,
    y_proba=y_proba_test,
    threshold=threshold_optimo,
    model_name="LightGBM - Test",
    costo_fp=COSTO_FP,
    costo_fn=COSTO_FN,
    show_plots=False
)

print("\n" + "=" * 80)
print("RESUMEN FINAL")
print("=" * 80)
print(f"Threshold óptimo: {threshold_optimo:.4f}")
print(f"Costo total en test: {costo_total:.2f}")
print(f"Costo por 1000 registros: {costo_por_1000:.2f}")
print(f"ROC AUC: {metrics.get('roc_auc', 'N/A'):.4f}")
print(f"F1 Score: {metrics.get('f1_score', 'N/A'):.4f}")
print(f"Precision: {metrics.get('precision', 'N/A'):.4f}")
print(f"Recall: {metrics.get('recall', 'N/A'):.4f}")
print("=" * 80)

