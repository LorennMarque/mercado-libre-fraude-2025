"""
Función para cargar datos, entrenar/cargar modelo y evaluar Random Forest sobre detección de fraude.
Retorna métricas y resultados de análisis de umbrales.
Se puede llamar desde otros scripts.
"""

import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from tools import evaluate_model, evaluate_model_with_thresholds

def run_fraud_detection_evaluation(
    data_path="data/processed/fraud_dataset_processed.csv",
    model=None,
    save_plots=True,
    output_dir="plots/",
    show_plots=True,
    random_state=42,
    test_size=0.25
):
    """
    Evalúa un modelo de detección de fraude usando Random Forest.
    El modelo debe ser pasado como objeto de Python (no como ruta). 
    Si model==None, se entrena un RandomForestClassifier nuevo.
    """
    # Cargar datos procesados
    df = pd.read_csv(data_path)

    # Quitar columnas no numéricas o que no queremos usar para el modelo
    cols_to_drop = ['producto_nombre', 'p', 'fecha', 'row_id', 'score', 'Unnamed: 0', 'pais', 'categoria_id', 'o']
    df_model = df.drop(columns=[col for col in cols_to_drop if col in df.columns])

    # Definir X, y
    X = df_model.drop(columns=['fraude'])
    y = df_model['fraude']

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    # Usar modelo dado, o entrenar uno si model=None
    if model is None:
        print("Entrenando nuevo modelo RandomForestClassifier...")
        model = RandomForestClassifier(
            n_estimators=100, 
            random_state=random_state, 
            n_jobs=-1, 
            class_weight='balanced'
        )
        model.fit(X_train, y_train)
        print("Modelo entrenado.")
    else:
        print("Usando modelo proporcionado.")

    # Predicciones
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    # Evaluar modelo con la función de tools.py
    print("\n" + "="*80)
    print("EVALUACIÓN COMPLETA DEL MODELO")
    print("="*80)

    metrics = evaluate_model(
        y_true=y_test,
        y_pred=y_pred,
        y_proba=y_proba,
        model_name="Random Forest - Detección de Fraude",
        save_plots=save_plots,
        output_dir=output_dir,
        show_plots=show_plots
    )

    # Análisis de umbrales (opcional)
    print("\n" + "="*80)
    print("ANÁLISIS DE UMBRALES")
    print("="*80)

    threshold_results = evaluate_model_with_thresholds(
        y_true=y_test,
        y_proba=y_proba,
        model_name="Random Forest - Detección de Fraude"
    )

    print("\n✅ Evaluación completada!")
    print(f"   F1 Score: {metrics['f1_score']:.4f}")
    print(f"   ROC AUC: {metrics['roc_auc']:.4f}")
    print(f"   Precision: {metrics['precision']:.4f}")
    print(f"   Recall: {metrics['recall']:.4f}")

    return {
        "metrics": metrics,
        "threshold_results": threshold_results,
        "model": model,
        "X_test": X_test,
        "y_test": y_test,
        "y_pred": y_pred,
        "y_proba": y_proba
    }

