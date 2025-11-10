import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, 
    accuracy_score, 
    roc_auc_score, 
    confusion_matrix, 
    roc_curve,
    f1_score
)

# 1. Cargar datos
df = pd.read_csv("data/processed/fraud_dataset_processed.csv")
# Quitar columna "producto_nombre" si está en el DataFrame de entrada
# Mostrar variables categóricas presentes en el dataset
categorical_cols = df.select_dtypes(include=['object']).columns.tolist()

# Quitar columnas 'producto_nombre', 'p', 'fecha' si están presentes en el DataFrame
cols_to_drop = ['producto_nombre', 'p', 'fecha', 'row_id', 'score', 'Unnamed: 0', 'pais', 'categoria_id', 'o']
df = df.drop(columns=[col for col in cols_to_drop if col in df.columns])



# 2. Definir X, y
if "fraude" not in df.columns:
    raise ValueError("La columna 'fraude' no está en el dataset")
X = df.drop(columns=['fraude'])
y = df['fraude']

# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

# 4. Modelo RandomForest
rf = RandomForestClassifier(
    n_estimators=100, 
    random_state=42, 
    n_jobs=-1, 
    class_weight='balanced'
)
rf.fit(X_train, y_train)

# 5. Predicciones
y_pred = rf.predict(X_test)
y_proba = rf.predict_proba(X_test)[:, 1]

# 6. Resultados
print("=== Classification Report ===")
print(classification_report(y_test, y_pred, digits=4))
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba):.4f}")

# 7. Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(4,4))
sns.heatmap(cm, annot=True, fmt='d', cmap="Blues", cbar=False)
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión")
plt.show()

# 8. Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_proba)
plt.figure(figsize=(6,4))
plt.plot(fpr, tpr, label=f"ROC Curve (AUC={roc_auc_score(y_test, y_proba):.3f})")
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Curva ROC - Random Forest')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# 9. Importancia de variables (top 10)
importances = pd.Series(rf.feature_importances_, index=X.columns)
top_imp = importances.sort_values(ascending=False).head(10)
plt.figure(figsize=(8,4))
sns.barplot(x=top_imp.values, y=top_imp.index)
plt.title("Top 10 - Importancia de Características (Random Forest)")
plt.xlabel("Importancia")
plt.tight_layout()
plt.show()


import joblib

# Guardar el modelo entrenado como archivo .pkl
joblib.dump(rf, "models/modelo_random_forest.pkl")
print("Modelo Random Forest guardado en 'models/modelo_random_forest.pkl'")


# RED NEURONAL
# RED NEURONAL: Entrenar una red neuronal y comparar con Random Forest
# NOTA: MLPClassifier no acepta valores faltantes (NaN) en los datos de entrada. 
# Si tienes NaNs en tus datos, debes imputarlos antes de entrenar el modelo.
# Puedes utilizar SimpleImputer o pipelines de scikit-learn.
# Alternativamente, considera usar HistGradientBoostingClassifier que sí acepta NaNs nativamente.
# Más información: https://scikit-learn.org/stable/modules/impute.html

from sklearn.neural_network import MLPClassifier
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

# Crear un pipeline que primero imputa NaNs y luego entrena la red neuronal
mlp_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy='mean')),  # Puedes cambiar la estrategia de imputación
    ('mlp', MLPClassifier(hidden_layer_sizes=(64, 32),
                          activation='relu',
                          solver='adam',
                          max_iter=200,
                          random_state=42))
])

# 1. Entrenar el pipeline con X_train que puede contener NaNs
mlp_pipeline.fit(X_train, y_train)

# 2. Predicción y probabilidades
y_pred_nn = mlp_pipeline.predict(X_test)
mlp = mlp_pipeline.named_steps['mlp']
if hasattr(mlp, "predict_proba"):
    y_proba_nn = mlp_pipeline.predict_proba(X_test)[:, 1]
else:
    y_proba_nn = mlp_pipeline.decision_function(X_test)
    y_proba_nn = (y_proba_nn - y_proba_nn.min()) / (y_proba_nn.max() - y_proba_nn.min())

# 3. Resultados de la red neuronal
print("\n=== RESULTADOS RED NEURONAL (MLPClassifier) ===")
print(classification_report(y_test, y_pred_nn, digits=4))
print(f"Accuracy: {accuracy_score(y_test, y_pred_nn):.4f}")
print(f"F1 Score: {f1_score(y_test, y_pred_nn):.4f}")
print(f"ROC AUC: {roc_auc_score(y_test, y_proba_nn):.4f}")

# 4. Matriz de confusión
cm_nn = confusion_matrix(y_test, y_pred_nn)
plt.figure(figsize=(4,4))
sns.heatmap(cm_nn, annot=True, fmt='d', cmap="Purples", cbar=False)
plt.xlabel("Predicho")
plt.ylabel("Real")
plt.title("Matriz de Confusión - Red Neuronal")
plt.show()

# 5. Curva ROC
fpr_nn, tpr_nn, thresholds_nn = roc_curve(y_test, y_proba_nn)
plt.figure(figsize=(6,4))
plt.plot(fpr_nn, tpr_nn, label=f"ROC NN (AUC={roc_auc_score(y_test, y_proba_nn):.3f})", color='purple')
plt.plot(fpr, tpr, label=f"ROC RF (AUC={roc_auc_score(y_test, y_proba):.3f})", color='blue', linestyle='--')
plt.plot([0,1], [0,1], linestyle='--', color='gray')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('Curva ROC - Comparación (Neuronal vs Random Forest)')
plt.legend(loc='lower right')
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# 6. Comparar resultados en resumen
print("="*60)
print("COMPARACIÓN DE MÉTRICAS PRINCIPALES")
print("="*60)
print(f"{'':20s} {'Random Forest':>15s} {'Red Neuronal':>15s}")
print(f"{'Accuracy:':20s} {accuracy_score(y_test, y_pred):15.4f} {accuracy_score(y_test, y_pred_nn):15.4f}")
print(f"{'F1 Score:':20s} {f1_score(y_test, y_pred):15.4f} {f1_score(y_test, y_pred_nn):15.4f}")
print(f"{'ROC AUC:':20s} {roc_auc_score(y_test, y_proba):15.4f} {roc_auc_score(y_test, y_proba_nn):15.4f}")

