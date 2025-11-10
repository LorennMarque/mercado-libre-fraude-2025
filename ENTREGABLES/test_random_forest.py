"""
Archivo de prueba para entrenar un modelo Random Forest para predecir fraude.
Selecciona variables relevantes basadas en el análisis del EDA y el listado de columnas.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, f1_score
import warnings
warnings.filterwarnings('ignore')

# Intentar importar tools si está disponible
try:
    from tools import evaluate_model, evaluate_model_with_thresholds
    HAS_TOOLS = True
except ImportError:
    HAS_TOOLS = False
    print("⚠️  No se encontró el módulo 'tools'. Se usarán métricas básicas.")


def seleccionar_variables_relevantes(df):
    """
    Selecciona variables relevantes basadas en el análisis del EDA.
    
    Variables seleccionadas:
    - Variables numéricas con buen poder discriminante (KS > 0.15): f, l, s, m, n, q, d
    - Variables imputadas para evitar nulos: b_imputado, c_imputado, d_imputado, etc.
    - Features de feature engineering: target encoding, frequency encoding, features temporales
    - Variables categóricas codificadas: o_is_Y, o_is_N, o_is_NA
    - Otras variables importantes: a, e, h, k, r, monto
    
    NO incluye:
    - 'score': puede ser data leakage (es un score de un modelo existente)
    - Variables categóricas sin codificar: pais, categoria_id, producto_nombre, o, p
    - Variables de identificación: Unnamed: 0, row_id
    - Fecha (ya está procesada en features temporales)
    """
    
    # Columnas a excluir
    cols_to_exclude = [
        'fraude',  # Variable objetivo
        'score',  # Posible data leakage
        'Unnamed: 0', 'row_id',  # Identificadores
        'producto_nombre', 'pais', 'categoria_id', 'o', 'p', 'fecha',  # Categóricas sin codificar
    ]
    
    # Variables numéricas originales relevantes (según análisis KS)
    # Variables con KS > 0.15: f, l, s, m, n, q, d
    variables_originales = ['a', 'e', 'h', 'k', 'r', 'monto', 'f', 'l', 's', 'm', 'n', 'q', 'd']
    
    # Variables imputadas (preferir estas para evitar nulos)
    variables_imputadas = [
        'b_imputado', 'c_imputado', 'd_imputado', 'f_imputado', 
        'q_imputado', 'l_imputado', 'm_imputado'
    ]
    
    # Features de feature engineering
    # Dummies de la columna 'o'
    dummies_o = ['o_is_N', 'o_is_Y', 'o_is_NA']
    
    # Encoding de categoria_id
    encoding_categoria = ['categoria_id_target_enc', 'categoria_id_freq_enc']
    
    # Encoding de pais
    encoding_pais = ['pais_target_enc', 'pais_freq_enc']
    
    # Features de producto_nombre
    features_producto = [
        'producto_num_chars', 'producto_num_words', 'producto_num_special_chars',
        'producto_avg_word_len', 'producto_freq'
    ]
    
    # Features temporales
    features_temporales = [
        'hora', 'dia_semana', 'dia_mes', 'mes',
        'es_fin_de_semana', 'es_nocturno', 'es_horario_laboral',
        'hora_sin', 'hora_cos', 'dia_semana_sin', 'dia_semana_cos'
    ]
    
    # Combinar todas las variables relevantes
    variables_relevantes = (
        variables_originales +
        variables_imputadas +
        dummies_o +
        encoding_categoria +
        encoding_pais +
        features_producto +
        features_temporales
    )
    
    # Filtrar solo las que existen en el dataframe
    variables_disponibles = [col for col in variables_relevantes if col in df.columns]
    
    # Excluir las que están en cols_to_exclude
    variables_finales = [col for col in variables_disponibles if col not in cols_to_exclude]
    
    return variables_finales


def entrenar_random_forest(
    data_path="../data/processed/fraud_dataset_processed.csv",
    test_size=0.25,
    random_state=42,
    n_estimators=100,
    class_weight='balanced',
    usar_tools=True
):
    """
    Entrena un modelo Random Forest para predecir fraude.
    
    Parameters:
    -----------
    data_path : str
        Ruta al archivo CSV procesado
    test_size : float
        Proporción del dataset para test
    random_state : int
        Semilla para reproducibilidad
    n_estimators : int
        Número de árboles en el Random Forest
    class_weight : str or dict
        Peso de clases para manejar desbalance
    usar_tools : bool
        Si True, usa las funciones de tools.py para evaluación completa
    
    Returns:
    --------
    dict : Diccionario con modelo, métricas y datos de evaluación
    """
    
    print("=" * 80)
    print("ENTRENAMIENTO DE RANDOM FOREST PARA DETECCIÓN DE FRAUDE")
    print("=" * 80)
    
    # 1. Cargar datos
    print("\n📂 Cargando datos...")
    df = pd.read_csv(data_path)
    print(f"   Dataset cargado: {df.shape[0]} filas, {df.shape[1]} columnas")
    
    # 2. Seleccionar variables relevantes
    print("\n🔍 Seleccionando variables relevantes...")
    variables_seleccionadas = seleccionar_variables_relevantes(df)
    print(f"   Variables seleccionadas: {len(variables_seleccionadas)}")
    print(f"   Variables: {', '.join(variables_seleccionadas[:10])}...")
    if len(variables_seleccionadas) > 10:
        print(f"   ... y {len(variables_seleccionadas) - 10} más")
    
    # Verificar que tenemos la variable objetivo
    if 'fraude' not in df.columns:
        raise ValueError("❌ La columna 'fraude' no está en el dataset")
    
    # 3. Preparar X e y
    X = df[variables_seleccionadas].copy()
    y = df['fraude'].copy()
    
    # Verificar valores faltantes
    nulos_por_col = X.isnull().sum()
    if nulos_por_col.sum() > 0:
        print(f"\n⚠️  Advertencia: Se encontraron valores faltantes:")
        cols_con_nulos = nulos_por_col[nulos_por_col > 0]
        for col, n_nulos in cols_con_nulos.items():
            print(f"   {col}: {n_nulos} nulos ({100*n_nulos/len(X):.2f}%)")
        print("   Se imputarán con la mediana antes del entrenamiento.")
        # Imputar con mediana
        from sklearn.impute import SimpleImputer
        imputer = SimpleImputer(strategy='median')
        X = pd.DataFrame(
            imputer.fit_transform(X),
            columns=X.columns,
            index=X.index
        )
    
    # 4. Train/test split
    print(f"\n✂️  Dividiendo datos (test_size={test_size})...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=y
    )
    print(f"   Train: {X_train.shape[0]} muestras ({100*X_train.shape[0]/len(X):.1f}%)")
    print(f"   Test:  {X_test.shape[0]} muestras ({100*X_test.shape[0]/len(X):.1f}%)")
    print(f"   Fraude en train: {y_train.sum()} ({100*y_train.mean():.2f}%)")
    print(f"   Fraude en test:  {y_test.sum()} ({100*y_test.mean():.2f}%)")
    
    # 5. Entrenar modelo
    print(f"\n🌲 Entrenando Random Forest...")
    print(f"   Parámetros:")
    print(f"     - n_estimators: {n_estimators}")
    print(f"     - class_weight: {class_weight}")
    print(f"     - random_state: {random_state}")
    
    rf = RandomForestClassifier(
        n_estimators=n_estimators,
        random_state=random_state,
        n_jobs=-1,
        class_weight=class_weight,
        max_depth=None,
        min_samples_split=2,
        min_samples_leaf=1
    )
    
    rf.fit(X_train, y_train)
    print("   ✅ Modelo entrenado")
    
    # 6. Predicciones
    print("\n🔮 Generando predicciones...")
    y_pred = rf.predict(X_test)
    y_proba = rf.predict_proba(X_test)[:, 1]
    
    # 7. Evaluación
    print("\n" + "=" * 80)
    print("EVALUACIÓN DEL MODELO")
    print("=" * 80)
    
    if HAS_TOOLS and usar_tools:
        # Evaluación completa con tools.py
        metrics = evaluate_model(
            y_true=y_test,
            y_pred=y_pred,
            y_proba=y_proba,
            model_name="Random Forest - Variables Relevantes",
            save_plots=False,
            show_plots=True
        )
        
        # Análisis de umbrales
        print("\n" + "=" * 80)
        threshold_results = evaluate_model_with_thresholds(
            y_true=y_test,
            y_proba=y_proba,
            model_name="Random Forest - Variables Relevantes"
        )
    else:
        # Evaluación básica
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)
        
        print(f"\n📊 MÉTRICAS BÁSICAS:")
        print(f"   Accuracy:  {accuracy:.4f}")
        print(f"   F1 Score:  {f1:.4f}")
        print(f"   ROC AUC:   {roc_auc:.4f}")
        
        print(f"\n📋 CLASSIFICATION REPORT:")
        print(classification_report(y_test, y_pred, target_names=['No Fraude', 'Fraude']))
        
        metrics = {
            'accuracy': accuracy,
            'f1_score': f1,
            'roc_auc': roc_auc
        }
        threshold_results = None
    
    # 8. Importancia de características
    print("\n" + "=" * 80)
    print("IMPORTANCIA DE CARACTERÍSTICAS (Top 20)")
    print("=" * 80)
    
    feature_importance = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    print("\nTop 20 características más importantes:")
    for idx, row in feature_importance.head(20).iterrows():
        print(f"   {row['feature']:30s} {row['importance']:.4f}")
    
    # 9. Resumen
    print("\n" + "=" * 80)
    print("RESUMEN")
    print("=" * 80)
    print(f"✅ Modelo entrenado exitosamente")
    print(f"   Variables usadas: {len(variables_seleccionadas)}")
    print(f"   F1 Score: {metrics.get('f1_score', f1):.4f}")
    print(f"   ROC AUC: {metrics.get('roc_auc', roc_auc):.4f}")
    
    return {
        'model': rf,
        'metrics': metrics,
        'feature_importance': feature_importance,
        'variables_seleccionadas': variables_seleccionadas,
        'X_test': X_test,
        'y_test': y_test,
        'y_pred': y_pred,
        'y_proba': y_proba,
        'threshold_results': threshold_results
    }


if __name__ == "__main__":
    # Ejecutar entrenamiento
    resultados = entrenar_random_forest(
        data_path="../data/processed/fraud_dataset_processed.csv",
        test_size=0.25,
        random_state=42,
        n_estimators=100,
        class_weight='balanced',
        usar_tools=True
    )
    
    print("\n✅ Proceso completado!")

