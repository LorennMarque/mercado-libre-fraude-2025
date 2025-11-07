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
cols_to_drop = ['producto_nombre', 'p', 'fecha', 'row_id', 'score', 'Unnamed: 0']
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
