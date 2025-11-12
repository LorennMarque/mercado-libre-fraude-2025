"""
Herramientas de evaluación para modelos de detección de fraude.
Incluye métricas, visualizaciones y análisis detallado para problemas de clasificación desbalanceada.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    cohen_kappa_score,
    matthews_corrcoef
)
from sklearn.calibration import calibration_curve
import plotly.graph_objects as go
from plotly.subplots import make_subplots
try:
    import ipywidgets as widgets
    from IPython.display import display
    IPYWIDGETS_AVAILABLE = True
except ImportError:
    IPYWIDGETS_AVAILABLE = False
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para los plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def evaluate_model(y_true, y_pred=None, y_proba=None, threshold=None, model_name="Modelo", 
                   save_plots=False, output_dir="plots/", show_plots=True,
                   costo_fp=None, costo_fn=None, prop_positivos=None):
    """
    Evalúa un modelo de clasificación binaria con métricas y visualizaciones completas.
    Optimizado para detección de fraude (clases desbalanceadas).
    
    Parameters:
    -----------
    y_true : array-like
        Valores reales (ground truth)
    y_pred : array-like, optional
        Predicciones binarias del modelo. Si es None y se proporciona threshold, 
        se calcula a partir de y_proba y threshold.
    y_proba : array-like, optional
        Probabilidades predichas (para clase positiva). Requerido si se especifica threshold.
    threshold : float, optional
        Umbral de decisión para convertir probabilidades en predicciones binarias.
        Si se proporciona, se usa y_proba para calcular y_pred aplicando este threshold.
        Si es None, se usa y_pred directamente (o se calcula a partir de y_proba con threshold=0.5).
    model_name : str, default="Modelo"
        Nombre del modelo para los títulos de los gráficos
    save_plots : bool, default=False
        Si True, guarda los plots en archivos
    output_dir : str, default="plots/"
        Directorio donde guardar los plots
    show_plots : bool, default=True
        Si True, muestra los plots
    costo_fp : float, optional
        Costo de un False Positive. Si se proporciona junto con costo_fn, se calcula el costo total.
    costo_fn : float, optional
        Costo de un False Negative. Si se proporciona junto con costo_fp, se calcula el costo total.
    prop_positivos : float, optional
        Proporción de positivos para ajustar el cálculo del costo. Si es None, se usa la proporción real.
    
    Returns:
    --------
    dict : Diccionario con todas las métricas calculadas
    """
    
    # Convertir a numpy arrays
    y_true = np.array(y_true)
    
    # Manejar threshold: si se proporciona, calcular y_pred a partir de y_proba
    if threshold is not None:
        if y_proba is None:
            raise ValueError("Si se especifica 'threshold', se debe proporcionar 'y_proba'")
        y_proba = np.array(y_proba)
        y_pred = (y_proba >= threshold).astype(int)
    else:
        # Si no hay threshold, usar y_pred directamente o calcularlo
        if y_pred is None:
            if y_proba is None:
                raise ValueError("Debe proporcionarse 'y_pred' o 'y_proba' (con threshold)")
            # Si solo hay y_proba sin threshold, usar threshold=0.5 por defecto
            y_proba = np.array(y_proba)
            y_pred = (y_proba >= 0.5).astype(int)
        else:
            y_pred = np.array(y_pred)
            # Si no se proporcionan probabilidades, crear array binario
            if y_proba is None:
                y_proba = y_pred.astype(float)
            else:
                y_proba = np.array(y_proba)
    
    # Calcular métricas básicas
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    roc_auc = roc_auc_score(y_true, y_proba) if len(np.unique(y_proba)) > 1 else 0.0
    kappa = cohen_kappa_score(y_true, y_pred)
    mcc = matthews_corrcoef(y_true, y_pred)
    
    # Calcular métricas adicionales para problemas desbalanceados
    try:
        avg_precision = average_precision_score(y_true, y_proba)
    except:
        avg_precision = 0.0
    
    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    # Métricas adicionales de la matriz de confusión
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0.0  # Negative Predictive Value
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0.0  # False Positive Rate
    
    # Calcular distribución de clases
    class_distribution = pd.Series(y_true).value_counts().sort_index()
    fraud_percentage = (y_true.sum() / len(y_true)) * 100
    
    # Determinar el threshold usado
    threshold_usado = threshold if threshold is not None else (0.5 if y_proba is not None else None)
    
    # Calcular costo si se proporcionan los parámetros
    costo_por_1000 = None
    if costo_fp is not None and costo_fn is not None:
        # Calcular proporción para ajuste
        if prop_positivos is None:
            prop_positivos = fraud_percentage / 100.0
        
        prop_original = fraud_percentage / 100.0
        prop_negativos_original = 1 - prop_original
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
        
        # Ajustar FP y FN
        fp_ajustado = fp * factor_fp
        fn_ajustado = fn * factor_fn
        
        # Calcular costo
        costo_total = fp_ajustado * costo_fp + fn_ajustado * costo_fn
        costo_por_1000 = (costo_total / len(y_true)) * 1000
    
    # Crear diccionario de métricas
    metrics = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'average_precision': avg_precision,
        'cohen_kappa': kappa,
        'matthews_corrcoef': mcc,
        'specificity': specificity,
        'npv': npv,
        'false_positive_rate': fpr,
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'fraud_percentage': fraud_percentage,
        'total_samples': len(y_true),
        'threshold': threshold_usado
    }
    
    # Agregar costo si se calculó
    if costo_por_1000 is not None:
        metrics['costo_por_1000'] = costo_por_1000
    
    # Imprimir métricas
    print("=" * 80)
    print(f"EVALUACIÓN DEL MODELO: {model_name}")
    print("=" * 80)
    if threshold_usado is not None:
        print(f"\n📌 Threshold usado: {threshold_usado:.4f}")
    print(f"\n📊 DISTRIBUCIÓN DE CLASES:")
    print(f"   Clase 0 (No Fraude): {class_distribution.get(0, 0):,} ({100-fraud_percentage:.2f}%)")
    print(f"   Clase 1 (Fraude):    {class_distribution.get(1, 0):,} ({fraud_percentage:.2f}%)")
    print(f"   Total:               {len(y_true):,}")
    
    print(f"\n🎯 MÉTRICAS PRINCIPALES:")
    print(f"   Accuracy:            {accuracy:.4f}")
    print(f"   F1 Score:            {f1:.4f} ⭐")
    print(f"   Precision:           {precision:.4f}")
    print(f"   Recall (Sensitivity): {recall:.4f}")
    print(f"   Specificity:         {specificity:.4f}")
    print(f"   ROC AUC:             {roc_auc:.4f}")
    print(f"   Average Precision:   {avg_precision:.4f}")
    
    print(f"\n📈 MÉTRICAS ADICIONALES:")
    print(f"   Cohen's Kappa:       {kappa:.4f}")
    print(f"   Matthews Corr Coef:  {mcc:.4f}")
    print(f"   NPV:                 {npv:.4f}")
    print(f"   False Positive Rate: {fpr:.4f}")
    
    # Mostrar costo si se calculó
    if costo_por_1000 is not None:
        print(f"\n💰 COSTO:")
        print(f"   Costo por 1000 registros: {costo_por_1000:.2f}")
        if costo_fp is not None and costo_fn is not None:
            print(f"   (Costo FP: {costo_fp:.1f}, Costo FN: {costo_fn:.1f})")
    
    print(f"\n🔢 MATRIZ DE CONFUSIÓN:")
    print(f"                    Predicción")
    print(f"                  No Fraude  Fraude")
    print(f"   Real No Fraude    {tn:6d}   {fp:6d}")
    print(f"   Real Fraude       {fn:6d}   {tp:6d}")
    
    print(f"\n📋 CLASSIFICATION REPORT:")
    print(classification_report(y_true, y_pred, target_names=['No Fraude', 'Fraude'], digits=4))
    
    # Crear visualizaciones
    if save_plots or show_plots:
        import os
        if save_plots and not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        # 1. Matriz de Confusión
        fig, axes = plt.subplots(2, 2, figsize=(16, 14))
        
        # Matriz de confusión (valores absolutos)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 0], 
                   cbar_kws={'label': 'Cantidad'})
        axes[0, 0].set_xlabel('Predicción', fontsize=12, fontweight='bold')
        axes[0, 0].set_ylabel('Real', fontsize=12, fontweight='bold')
        axes[0, 0].set_title('Matriz de Confusión (Valores Absolutos)', fontsize=14, fontweight='bold')
        axes[0, 0].set_xticklabels(['No Fraude', 'Fraude'])
        axes[0, 0].set_yticklabels(['No Fraude', 'Fraude'])
        
        # Matriz de confusión (normalizada)
        cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        sns.heatmap(cm_normalized, annot=True, fmt='.3f', cmap='Blues', ax=axes[0, 1],
                   cbar_kws={'label': 'Proporción'})
        axes[0, 1].set_xlabel('Predicción', fontsize=12, fontweight='bold')
        axes[0, 1].set_ylabel('Real', fontsize=12, fontweight='bold')
        axes[0, 1].set_title('Matriz de Confusión (Normalizada)', fontsize=14, fontweight='bold')
        axes[0, 1].set_xticklabels(['No Fraude', 'Fraude'])
        axes[0, 1].set_yticklabels(['No Fraude', 'Fraude'])
        
        # 2. Curva ROC
        if len(np.unique(y_proba)) > 1:
            fpr_roc, tpr_roc, _ = roc_curve(y_true, y_proba)
            axes[1, 0].plot(fpr_roc, tpr_roc, linewidth=2, label=f'ROC (AUC = {roc_auc:.3f})')
            axes[1, 0].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random Classifier')
            axes[1, 0].set_xlabel('False Positive Rate', fontsize=12, fontweight='bold')
            axes[1, 0].set_ylabel('True Positive Rate', fontsize=12, fontweight='bold')
            axes[1, 0].set_title('Curva ROC', fontsize=14, fontweight='bold')
            axes[1, 0].legend(loc='lower right', fontsize=10)
            axes[1, 0].grid(True, alpha=0.3)
        
        # 3. Curva Precision-Recall
        if len(np.unique(y_proba)) > 1:
            precision_curve, recall_curve, _ = precision_recall_curve(y_true, y_proba)
            axes[1, 1].plot(recall_curve, precision_curve, linewidth=2, 
                           label=f'PR Curve (AP = {avg_precision:.3f})')
            baseline = (y_true == 1).sum() / len(y_true)
            axes[1, 1].axhline(y=baseline, color='k', linestyle='--', linewidth=1, 
                              label=f'Baseline ({baseline:.3f})')
            axes[1, 1].set_xlabel('Recall', fontsize=12, fontweight='bold')
            axes[1, 1].set_ylabel('Precision', fontsize=12, fontweight='bold')
            axes[1, 1].set_title('Curva Precision-Recall', fontsize=14, fontweight='bold')
            axes[1, 1].legend(loc='lower left', fontsize=10)
            axes[1, 1].grid(True, alpha=0.3)
        
        plt.suptitle(f'Evaluación del Modelo: {model_name}', fontsize=16, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{output_dir}{model_name}_evaluation.png", dpi=300, bbox_inches='tight')
            print(f"\n💾 Gráficos guardados en: {output_dir}{model_name}_evaluation.png")
        
        if show_plots:
            plt.show()
        else:
            plt.close()
        
        # 4. Distribución de probabilidades
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histograma de probabilidades por clase
        axes[0].hist(y_proba[y_true == 0], bins=50, alpha=0.7, label='No Fraude', 
                    color='green', density=True)
        axes[0].hist(y_proba[y_true == 1], bins=50, alpha=0.7, label='Fraude', 
                    color='red', density=True)
        axes[0].axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Umbral 0.5')
        axes[0].set_xlabel('Probabilidad Predicha', fontsize=12, fontweight='bold')
        axes[0].set_ylabel('Densidad', fontsize=12, fontweight='bold')
        axes[0].set_title('Distribución de Probabilidades por Clase', fontsize=14, fontweight='bold')
        axes[0].legend(fontsize=10)
        axes[0].grid(True, alpha=0.3)
        
        # Boxplot de probabilidades por clase
        data_for_box = [y_proba[y_true == 0], y_proba[y_true == 1]]
        axes[1].boxplot(data_for_box, labels=['No Fraude', 'Fraude'])
        axes[1].axhline(y=0.5, color='black', linestyle='--', linewidth=2, label='Umbral 0.5')
        axes[1].set_ylabel('Probabilidad Predicha', fontsize=12, fontweight='bold')
        axes[1].set_title('Distribución de Probabilidades (Boxplot)', fontsize=14, fontweight='bold')
        axes[1].legend(fontsize=10)
        axes[1].grid(True, alpha=0.3, axis='y')
        
        plt.suptitle(f'Análisis de Probabilidades: {model_name}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_plots:
            plt.savefig(f"{output_dir}{model_name}_probabilities.png", dpi=300, bbox_inches='tight')
        
        if show_plots:
            plt.show()
        else:
            plt.close()
    
    return metrics


def evaluate_model_with_thresholds(y_true, y_proba, thresholds=None, model_name="Modelo"):
    """
    Evalúa el modelo con diferentes umbrales de decisión.
    Útil para encontrar el umbral óptimo en problemas desbalanceados.
    
    Parameters:
    -----------
    y_true : array-like
        Valores reales
    y_proba : array-like
        Probabilidades predichas
    thresholds : array-like, optional
        Lista de umbrales a evaluar. Si es None, usa umbrales de 0.1 a 0.9
    model_name : str, default="Modelo"
        Nombre del modelo
    
    Returns:
    --------
    pd.DataFrame : DataFrame con métricas para cada umbral
    """
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    
    if thresholds is None:
        thresholds = np.arange(0.1, 1.0, 0.05)
    
    results = []
    
    for threshold in thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        
        metrics = {
            'threshold': threshold,
            'f1_score': f1_score(y_true, y_pred_thresh, zero_division=0),
            'precision': precision_score(y_true, y_pred_thresh, zero_division=0),
            'recall': recall_score(y_true, y_pred_thresh, zero_division=0),
            'accuracy': accuracy_score(y_true, y_pred_thresh),
        }
        
        cm = confusion_matrix(y_true, y_pred_thresh)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
            metrics['true_positives'] = tp
            metrics['false_positives'] = fp
            metrics['true_negatives'] = tn
            metrics['false_negatives'] = fn
        
        results.append(metrics)
    
    df_results = pd.DataFrame(results)
    
    # Visualizar
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    axes[0, 0].plot(df_results['threshold'], df_results['f1_score'], 'o-', label='F1 Score', linewidth=2)
    axes[0, 0].set_xlabel('Umbral', fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel('F1 Score', fontsize=12, fontweight='bold')
    axes[0, 0].set_title('F1 Score vs Umbral', fontsize=14, fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    axes[0, 1].plot(df_results['threshold'], df_results['precision'], 'o-', label='Precision', linewidth=2)
    axes[0, 1].plot(df_results['threshold'], df_results['recall'], 'o-', label='Recall', linewidth=2)
    axes[0, 1].set_xlabel('Umbral', fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel('Métrica', fontsize=12, fontweight='bold')
    axes[0, 1].set_title('Precision y Recall vs Umbral', fontsize=14, fontweight='bold')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    axes[1, 0].plot(df_results['threshold'], df_results['true_positives'], 'o-', label='TP', linewidth=2)
    axes[1, 0].plot(df_results['threshold'], df_results['false_positives'], 'o-', label='FP', linewidth=2)
    axes[1, 0].set_xlabel('Umbral', fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel('Cantidad', fontsize=12, fontweight='bold')
    axes[1, 0].set_title('True Positives y False Positives vs Umbral', fontsize=14, fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    axes[1, 1].plot(df_results['threshold'], df_results['accuracy'], 'o-', label='Accuracy', linewidth=2)
    axes[1, 1].set_xlabel('Umbral', fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel('Accuracy', fontsize=12, fontweight='bold')
    axes[1, 1].set_title('Accuracy vs Umbral', fontsize=14, fontweight='bold')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.suptitle(f'Análisis de Umbrales: {model_name}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.show()
    
    # Encontrar umbral óptimo (máximo F1)
    optimal_idx = df_results['f1_score'].idxmax()
    optimal_threshold = df_results.loc[optimal_idx, 'threshold']
    optimal_f1 = df_results.loc[optimal_idx, 'f1_score']
    
    print(f"\n🎯 Umbral óptimo (máximo F1): {optimal_threshold:.3f}")
    print(f"   F1 Score en umbral óptimo: {optimal_f1:.4f}")
    
    return df_results


def generar_thresholds(y_proba, max_thresholds=200):
    """
    Genera thresholds adaptativos basados en las probabilidades únicas del modelo.
    
    Parameters:
    -----------
    y_proba : array-like
        Probabilidades predichas por el modelo
    max_thresholds : int, default=200
        Número máximo de thresholds a generar
    
    Returns:
    --------
    array
        Array de thresholds adaptativos
    """
    y_proba = np.array(y_proba)
    # Obtener valores únicos redondeados a 4 decimales
    vals = np.unique(np.round(y_proba, 4))
    
    if len(vals) > max_thresholds:
        # Subsamplear de forma uniforme si hay demasiados
        vals = np.linspace(vals.min(), vals.max(), max_thresholds)
    
    return vals


def optimizar_threshold_costo_cv(model, X, y, cv, costo_fp=5.0, costo_fn=100.0, 
                                 prop_positivos=None, thresholds=None, model_name="Modelo",
                                 y_proba_cv=None):
    """
    Optimiza el threshold basándose en la función de costo usando Cross-Validation.
    
    Parameters:
    -----------
    model : sklearn estimator
        Modelo que implementa predict_proba
    X : array-like
        Features para entrenamiento
    y : array-like
        Target (valores reales)
    cv : cross-validation splitter
        Estrategia de cross-validation (ej: StratifiedKFold)
    costo_fp : float, default=5.0
        Costo de un False Positive
    costo_fn : float, default=100.0
        Costo de un False Negative
    prop_positivos : float, optional
        Proporción de positivos. Si es None, se calcula de y
    thresholds : array-like, optional
        Lista de thresholds a evaluar. Si es None, usa np.linspace(0.01, 0.99, 100)
    model_name : str, default="Modelo"
        Nombre del modelo
    y_proba_cv : array-like, optional
        Probabilidades de CV ya calculadas. Si es None, se calculan manualmente por fold.
        Útil para evitar recalcular los folds cuando se optimizan múltiples métricas.
    
    Returns:
    --------
    dict : Diccionario con threshold óptimo, costo mínimo y resultados por fold
    """
    if prop_positivos is None:
        prop_positivos = np.mean(y)
    
    # Convertir a arrays numpy
    X = np.array(X)
    y = np.array(y)
    
    # Calcular probabilidades por fold manualmente para garantizar correspondencia exacta
    # Esto asegura que las métricas por fold correspondan a los folds reales del CV
    # Siempre calculamos y_proba_por_fold porque lo necesitamos para las métricas por fold
    y_proba_por_fold = {}  # Diccionario: índice -> probabilidad
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        # Entrenar modelo en el fold de entrenamiento
        model_fold = type(model)(**model.get_params())
        model_fold.fit(X[train_idx], y[train_idx])
        # Predecir probabilidades en el fold de test
        y_proba_fold = model_fold.predict_proba(X[test_idx])[:, 1]
        # Almacenar probabilidades con sus índices originales
        for idx, proba in zip(test_idx, y_proba_fold):
            y_proba_por_fold[idx] = proba
    
    # Construir y_proba_cv desde y_proba_por_fold (evita usar cross_val_predict)
    if y_proba_cv is None:
        # Construir array en el orden correcto desde el diccionario
        y_proba_cv = np.array([y_proba_por_fold[i] for i in range(len(y))])
    else:
        # Si se proporciona y_proba_cv, lo usamos directamente
        # (y_proba_por_fold ya se calculó arriba para las métricas por fold)
        y_proba_cv = np.array(y_proba_cv)
    
    # Generar thresholds adaptativos si no se proporcionan
    if thresholds is None:
        thresholds = generar_thresholds(y_proba_cv, max_thresholds=200)
    
    # Calcular proporción original
    prop_original = np.mean(y)
    prop_negativos_original = 1 - prop_original
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
    
    # PASO 1: Evaluar todos los thresholds sin métricas por fold (más rápido)
    resultados = []
    for threshold in thresholds:
        y_pred = (y_proba_cv >= threshold).astype(int)
        cm = confusion_matrix(y, y_pred)
        
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            if len(np.unique(y_pred)) == 1:
                if y_pred[0] == 0:
                    tn, fp, fn, tp = len(y) - y.sum(), 0, y.sum(), 0
                else:
                    tn, fp, fn, tp = 0, (y == 0).sum(), 0, y.sum()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
        
        # Ajustar FP y FN según la proporción
        fp_ajustado = fp * factor_fp
        fn_ajustado = fn * factor_fn
        
        # Calcular costo
        costo_total = fp_ajustado * costo_fp + fn_ajustado * costo_fn
        costo_por_1000 = (costo_total / len(y)) * 1000
        
        resultados.append({
            'threshold': threshold,
            'costo_por_1000': costo_por_1000,
            'fp': fp,
            'fn': fn,
            'fp_ajustado': fp_ajustado,
            'fn_ajustado': fn_ajustado,
            'costo_mean_folds': np.nan,  # Se calculará después para top-5
            'costo_std_folds': np.nan,
            'cv_coeficiente': np.nan
        })
    
    df_resultados = pd.DataFrame(resultados)
    
    # PASO 2: Identificar top-5 thresholds (menores costos)
    top_n = min(5, len(df_resultados))
    top_thresholds = df_resultados.nsmallest(top_n, 'costo_por_1000')['threshold'].values
    
    # PASO 3: Evaluar métricas por fold solo para los top-5 thresholds
    for threshold in top_thresholds:
        # Calcular métricas por fold para evaluar robustez
        costos_por_fold = []
        for train_idx, test_idx in cv.split(X, y):
            y_fold_true = y[test_idx]
            # Usar probabilidades del diccionario que garantiza correspondencia exacta
            y_fold_proba = np.array([y_proba_por_fold[idx] for idx in test_idx])
            y_fold_pred = (y_fold_proba >= threshold).astype(int)
            
            cm_fold = confusion_matrix(y_fold_true, y_fold_pred)
            if cm_fold.size == 4:
                tn_fold, fp_fold, fn_fold, tp_fold = cm_fold.ravel()
            else:
                if len(np.unique(y_fold_pred)) == 1:
                    if y_fold_pred[0] == 0:
                        tn_fold, fp_fold, fn_fold, tp_fold = len(y_fold_true) - y_fold_true.sum(), 0, y_fold_true.sum(), 0
                    else:
                        tn_fold, fp_fold, fn_fold, tp_fold = 0, (y_fold_true == 0).sum(), 0, y_fold_true.sum()
                else:
                    tn_fold, fp_fold, fn_fold, tp_fold = 0, 0, 0, 0
            
            fp_fold_ajustado = fp_fold * factor_fp
            fn_fold_ajustado = fn_fold * factor_fn
            costo_fold_total = fp_fold_ajustado * costo_fp + fn_fold_ajustado * costo_fn
            costo_fold_por_1000 = (costo_fold_total / len(y_fold_true)) * 1000
            costos_por_fold.append(costo_fold_por_1000)
        
        # Calcular coeficiente de variación (robustez)
        costos_por_fold = np.array(costos_por_fold)
        mean_costo = np.mean(costos_por_fold)
        std_costo = np.std(costos_por_fold)
        cv_coeficiente = std_costo / mean_costo if mean_costo > 0 else np.inf
        
        # Actualizar resultados para este threshold
        idx = df_resultados[df_resultados['threshold'] == threshold].index[0]
        df_resultados.loc[idx, 'costo_mean_folds'] = mean_costo
        df_resultados.loc[idx, 'costo_std_folds'] = std_costo
        df_resultados.loc[idx, 'cv_coeficiente'] = cv_coeficiente
    
    # Encontrar threshold óptimo (mínimo costo)
    idx_optimo = df_resultados['costo_por_1000'].idxmin()
    threshold_optimo = df_resultados.loc[idx_optimo, 'threshold']
    costo_optimo = df_resultados.loc[idx_optimo, 'costo_por_1000']
    
    # Si el threshold óptimo no está en el top-5, calcular sus métricas por fold
    if threshold_optimo not in top_thresholds:
        costos_por_fold = []
        for train_idx, test_idx in cv.split(X, y):
            y_fold_true = y[test_idx]
            y_fold_proba = np.array([y_proba_por_fold[idx] for idx in test_idx])
            y_fold_pred = (y_fold_proba >= threshold_optimo).astype(int)
            
            cm_fold = confusion_matrix(y_fold_true, y_fold_pred)
            if cm_fold.size == 4:
                tn_fold, fp_fold, fn_fold, tp_fold = cm_fold.ravel()
            else:
                if len(np.unique(y_fold_pred)) == 1:
                    if y_fold_pred[0] == 0:
                        tn_fold, fp_fold, fn_fold, tp_fold = len(y_fold_true) - y_fold_true.sum(), 0, y_fold_true.sum(), 0
                    else:
                        tn_fold, fp_fold, fn_fold, tp_fold = 0, (y_fold_true == 0).sum(), 0, y_fold_true.sum()
                else:
                    tn_fold, fp_fold, fn_fold, tp_fold = 0, 0, 0, 0
            
            fp_fold_ajustado = fp_fold * factor_fp
            fn_fold_ajustado = fn_fold * factor_fn
            costo_fold_total = fp_fold_ajustado * costo_fp + fn_fold_ajustado * costo_fn
            costo_fold_por_1000 = (costo_fold_total / len(y_fold_true)) * 1000
            costos_por_fold.append(costo_fold_por_1000)
        
        costos_por_fold = np.array(costos_por_fold)
        mean_costo = np.mean(costos_por_fold)
        std_costo = np.std(costos_por_fold)
        cv_coeficiente = std_costo / mean_costo if mean_costo > 0 else np.inf
        
        df_resultados.loc[idx_optimo, 'costo_mean_folds'] = mean_costo
        df_resultados.loc[idx_optimo, 'costo_std_folds'] = std_costo
        df_resultados.loc[idx_optimo, 'cv_coeficiente'] = cv_coeficiente
    
    cv_optimo = df_resultados.loc[idx_optimo, 'cv_coeficiente']
    
    # Evaluar robustez (manejar NaN si no se calculó)
    if pd.isna(cv_optimo) or cv_optimo == np.inf:
        robustez = "No calculado"
        emoji = "❓"
    elif cv_optimo < 0.1:
        robustez = "Muy robusto"
        emoji = "✅"
    elif cv_optimo < 0.2:
        robustez = "Robusto"
        emoji = "✓"
    else:
        robustez = "Poco robusto"
        emoji = "⚠️"
    
    print(f"\n🎯 OPTIMIZACIÓN DE THRESHOLD POR COSTO (CV):")
    print(f"   Threshold óptimo: {threshold_optimo:.4f}")
    print(f"   Costo mínimo por 1000 registros: {costo_optimo:.2f}")
    print(f"   FP ajustado: {df_resultados.loc[idx_optimo, 'fp_ajustado']:.0f}")
    print(f"   FN ajustado: {df_resultados.loc[idx_optimo, 'fn_ajustado']:.0f}")
    print(f"\n📊 EVALUACIÓN DE ROBUSTEZ:")
    print(f"   Coeficiente de variación (CV): {cv_optimo:.4f}")
    print(f"   Robustez: {robustez} {emoji}")
    print(f"   Costo medio por fold: {df_resultados.loc[idx_optimo, 'costo_mean_folds']:.2f} ± {df_resultados.loc[idx_optimo, 'costo_std_folds']:.2f}")
    
    return {
        'threshold_optimo': threshold_optimo,
        'costo_minimo': costo_optimo,
        'cv_coeficiente': cv_optimo,
        'robustez': robustez,
        'resultados': df_resultados
    }


def optimizar_threshold_f1_cv(model, X, y, cv, thresholds=None, model_name="Modelo",
                              y_proba_cv=None):
    """
    Optimiza el threshold basándose en F1 Score usando Cross-Validation.
    
    Parameters:
    -----------
    model : sklearn estimator
        Modelo que implementa predict_proba
    X : array-like
        Features para entrenamiento
    y : array-like
        Target (valores reales)
    cv : cross-validation splitter
        Estrategia de cross-validation (ej: StratifiedKFold)
    thresholds : array-like, optional
        Lista de thresholds a evaluar. Si es None, usa np.linspace(0.01, 0.99, 100)
    model_name : str, default="Modelo"
        Nombre del modelo
    y_proba_cv : array-like, optional
        Probabilidades de CV ya calculadas. Si es None, se calculan manualmente por fold.
        Útil para evitar recalcular los folds cuando se optimizan múltiples métricas.
    
    Returns:
    --------
    dict : Diccionario con threshold óptimo, F1 máximo y resultados por fold
    """
    # Convertir a arrays numpy
    X = np.array(X)
    y = np.array(y)
    
    # Calcular probabilidades por fold manualmente para garantizar correspondencia exacta
    # Esto asegura que las métricas por fold correspondan a los folds reales del CV
    # Siempre calculamos y_proba_por_fold porque lo necesitamos para las métricas por fold
    y_proba_por_fold = {}  # Diccionario: índice -> probabilidad
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y)):
        # Entrenar modelo en el fold de entrenamiento
        model_fold = type(model)(**model.get_params())
        model_fold.fit(X[train_idx], y[train_idx])
        # Predecir probabilidades en el fold de test
        y_proba_fold = model_fold.predict_proba(X[test_idx])[:, 1]
        # Almacenar probabilidades con sus índices originales
        for idx, proba in zip(test_idx, y_proba_fold):
            y_proba_por_fold[idx] = proba
    
    # Construir y_proba_cv desde y_proba_por_fold (evita usar cross_val_predict)
    if y_proba_cv is None:
        # Construir array en el orden correcto desde el diccionario
        y_proba_cv = np.array([y_proba_por_fold[i] for i in range(len(y))])
    else:
        # Si se proporciona y_proba_cv, lo usamos directamente
        # (y_proba_por_fold ya se calculó arriba para las métricas por fold)
        y_proba_cv = np.array(y_proba_cv)
    
    # Generar thresholds adaptativos si no se proporcionan
    if thresholds is None:
        thresholds = generar_thresholds(y_proba_cv, max_thresholds=200)
    
    # PASO 1: Evaluar todos los thresholds sin métricas por fold (más rápido)
    resultados = []
    for threshold in thresholds:
        y_pred = (y_proba_cv >= threshold).astype(int)
        
        f1 = f1_score(y, y_pred, zero_division=0)
        precision = precision_score(y, y_pred, zero_division=0)
        recall = recall_score(y, y_pred, zero_division=0)
        
        resultados.append({
            'threshold': threshold,
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'f1_mean_folds': np.nan,  # Se calculará después para top-5
            'f1_std_folds': np.nan,
            'cv_coeficiente': np.nan
        })
    
    df_resultados = pd.DataFrame(resultados)
    
    # PASO 2: Identificar top-5 thresholds (mayores F1 scores)
    top_n = min(5, len(df_resultados))
    top_thresholds = df_resultados.nlargest(top_n, 'f1_score')['threshold'].values
    
    # PASO 3: Evaluar métricas por fold solo para los top-5 thresholds
    for threshold in top_thresholds:
        # Calcular métricas por fold para evaluar robustez
        f1_por_fold = []
        precision_por_fold = []
        recall_por_fold = []
        
        for train_idx, test_idx in cv.split(X, y):
            y_fold_true = y[test_idx]
            # Usar probabilidades del diccionario que garantiza correspondencia exacta
            y_fold_proba = np.array([y_proba_por_fold[idx] for idx in test_idx])
            y_fold_pred = (y_fold_proba >= threshold).astype(int)
            
            f1_fold = f1_score(y_fold_true, y_fold_pred, zero_division=0)
            precision_fold = precision_score(y_fold_true, y_fold_pred, zero_division=0)
            recall_fold = recall_score(y_fold_true, y_fold_pred, zero_division=0)
            
            f1_por_fold.append(f1_fold)
            precision_por_fold.append(precision_fold)
            recall_por_fold.append(recall_fold)
        
        # Calcular coeficiente de variación (robustez) para F1
        f1_por_fold = np.array(f1_por_fold)
        mean_f1 = np.mean(f1_por_fold)
        std_f1 = np.std(f1_por_fold)
        cv_coeficiente_f1 = std_f1 / mean_f1 if mean_f1 > 0 else np.inf
        
        # Actualizar resultados para este threshold
        idx = df_resultados[df_resultados['threshold'] == threshold].index[0]
        df_resultados.loc[idx, 'f1_mean_folds'] = mean_f1
        df_resultados.loc[idx, 'f1_std_folds'] = std_f1
        df_resultados.loc[idx, 'cv_coeficiente'] = cv_coeficiente_f1
    
    # Encontrar threshold óptimo (máximo F1)
    idx_optimo = df_resultados['f1_score'].idxmax()
    threshold_optimo = df_resultados.loc[idx_optimo, 'threshold']
    f1_optimo = df_resultados.loc[idx_optimo, 'f1_score']
    
    # Si el threshold óptimo no está en el top-5, calcular sus métricas por fold
    if threshold_optimo not in top_thresholds:
        f1_por_fold = []
        for train_idx, test_idx in cv.split(X, y):
            y_fold_true = y[test_idx]
            y_fold_proba = np.array([y_proba_por_fold[idx] for idx in test_idx])
            y_fold_pred = (y_fold_proba >= threshold_optimo).astype(int)
            
            f1_fold = f1_score(y_fold_true, y_fold_pred, zero_division=0)
            f1_por_fold.append(f1_fold)
        
        f1_por_fold = np.array(f1_por_fold)
        mean_f1 = np.mean(f1_por_fold)
        std_f1 = np.std(f1_por_fold)
        cv_coeficiente_f1 = std_f1 / mean_f1 if mean_f1 > 0 else np.inf
        
        df_resultados.loc[idx_optimo, 'f1_mean_folds'] = mean_f1
        df_resultados.loc[idx_optimo, 'f1_std_folds'] = std_f1
        df_resultados.loc[idx_optimo, 'cv_coeficiente'] = cv_coeficiente_f1
    
    cv_optimo = df_resultados.loc[idx_optimo, 'cv_coeficiente']
    
    # Evaluar robustez (manejar NaN si no se calculó)
    if pd.isna(cv_optimo) or cv_optimo == np.inf:
        robustez = "No calculado"
        emoji = "❓"
    elif cv_optimo < 0.1:
        robustez = "Muy robusto"
        emoji = "✅"
    elif cv_optimo < 0.2:
        robustez = "Robusto"
        emoji = "✓"
    else:
        robustez = "Poco robusto"
        emoji = "⚠️"
    
    print(f"\n🎯 OPTIMIZACIÓN DE THRESHOLD POR F1 SCORE (CV):")
    print(f"   Threshold óptimo: {threshold_optimo:.4f}")
    print(f"   F1 Score máximo: {f1_optimo:.4f}")
    print(f"   Precision: {df_resultados.loc[idx_optimo, 'precision']:.4f}")
    print(f"   Recall: {df_resultados.loc[idx_optimo, 'recall']:.4f}")
    print(f"\n📊 EVALUACIÓN DE ROBUSTEZ:")
    print(f"   Coeficiente de variación (CV): {cv_optimo:.4f}")
    print(f"   Robustez: {robustez} {emoji}")
    print(f"   F1 Score medio por fold: {df_resultados.loc[idx_optimo, 'f1_mean_folds']:.4f} ± {df_resultados.loc[idx_optimo, 'f1_std_folds']:.4f}")
    
    return {
        'threshold_optimo': threshold_optimo,
        'f1_maximo': f1_optimo,
        'cv_coeficiente': cv_optimo,
        'robustez': robustez,
        'resultados': df_resultados
    }


def plot_costo_interactivo(y_true, y_proba, costo_fp_inicial=5.0, costo_fn_inicial=100.0, 
                           prop_positivos_inicial=None, model_name="Modelo"):
    """
    Crea un gráfico interactivo con Plotly que muestra el costo cada 1000 registros
    en función del threshold del modelo, con inputs interactivos para ajustar parámetros.
    
    La función calcula el costo como: (FP * costo_FP + FN * costo_FN) / total_samples * 1000
    y permite ajustar interactivamente los costos de FP y FN, así como la proporción de positivos.
    
    Parameters:
    -----------
    y_true : array-like
        Valores reales (ground truth)
    y_proba : array-like
        Probabilidades predichas por el modelo
    costo_fp_inicial : float, default=5.0
        Costo inicial de un False Positive
    costo_fn_inicial : float, default=100.0
        Costo inicial de un False Negative
    prop_positivos_inicial : float, optional
        Proporción inicial de positivos reales. Si es None, se calcula de y_true.
        Nota: Este parámetro afecta el cálculo del costo al ajustar la escala.
    model_name : str, default="Modelo"
        Nombre del modelo para el título del gráfico
    
    Returns:
    --------
    plotly.graph_objects.Figure o widgets interactivos
        Si ipywidgets está disponible, retorna widgets interactivos. 
        Si no, retorna la figura de Plotly estática.
    """
    y_true = np.array(y_true)
    y_proba = np.array(y_proba)
    
    # Calcular proporción inicial si no se proporciona
    if prop_positivos_inicial is None:
        prop_positivos_inicial = y_true.mean()
    
    # Generar rango de thresholds (asegurar que esté entre 0 y 1)
    thresholds = np.linspace(0.0, 1.0, 200)
    
    # Pre-calcular FP y FN para cada threshold (esto se hace una sola vez)
    fp_counts = []
    fn_counts = []
    
    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred)
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            # Caso especial cuando solo hay una clase
            if len(np.unique(y_pred)) == 1:
                if y_pred[0] == 0:
                    tn, fp, fn, tp = len(y_true) - y_true.sum(), 0, y_true.sum(), 0
                else:
                    tn, fp, fn, tp = 0, (y_true == 0).sum(), 0, y_true.sum()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0
        
        fp_counts.append(fp)
        fn_counts.append(fn)
    
    fp_counts = np.array(fp_counts)
    fn_counts = np.array(fn_counts)
    total_samples = len(y_true)
    
    # Calcular proporción original
    prop_original = y_true.mean()
    prop_negativos_original = 1 - prop_original
    
    # Función para calcular costo normalizado cada 1000 registros
    def calcular_costo(costo_fp, costo_fn, prop_positivos=None):
        # Si no se especifica proporción, usar la original
        if prop_positivos is None:
            prop_positivos = prop_original
        
        # Ajustar FP y FN según la nueva proporción
        # FP depende de los negativos (predice positivo pero es negativo)
        # FN depende de los positivos (predice negativo pero es positivo)
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
        
        # Ajustar FP y FN proporcionalmente
        fp_ajustados = fp_counts * factor_fp
        fn_ajustados = fn_counts * factor_fn
        
        # El costo total es: FP_ajustado * costo_FP + FN_ajustado * costo_FN
        costo_total = fp_ajustados * costo_fp + fn_ajustados * costo_fn
        
        # Normalizar a cada 1000 registros
        # Usar el total de muestras ajustado según la nueva proporción
        # (aunque el total sigue siendo el mismo, la distribución cambia)
        costo_por_1000 = (costo_total / total_samples) * 1000
        return costo_por_1000
    
    # Calcular costo inicial
    costo_inicial = calcular_costo(costo_fp_inicial, costo_fn_inicial, prop_positivos_inicial)
    
    # Encontrar threshold óptimo (mínimo costo)
    idx_optimo = np.argmin(costo_inicial)
    threshold_optimo = thresholds[idx_optimo]
    costo_optimo = costo_inicial[idx_optimo]
    
    # Si ipywidgets está disponible, crear inputs interactivos con FigureWidget
    if IPYWIDGETS_AVAILABLE:
        # Crear figura interactiva usando FigureWidget (permite actualización en tiempo real)
        fig = go.FigureWidget(
            data=[
                go.Scatter(
                    x=thresholds,
                    y=costo_inicial,
                    mode='lines',
                    name='Costo por 1000 registros',
                    line=dict(color='#1f77b4', width=3),
                    hovertemplate='<b>Threshold:</b> %{x:.3f}<br>' +
                                 '<b>Costo/1000:</b> %{y:.2f}<br>' +
                                 '<b>FP:</b> %{customdata[0]}<br>' +
                                 '<b>FN:</b> %{customdata[1]}<br>' +
                                 '<extra></extra>',
                    customdata=np.column_stack((fp_counts, fn_counts))
                ),
                go.Scatter(
                    x=[threshold_optimo],
                    y=[costo_optimo],
                    mode='markers',
                    name='Umbral Óptimo',
                    marker=dict(
                        size=10,
                        color='red',
                        line=dict(width=1, color='darkred')
                    ),
                    hovertemplate='<b>Threshold Óptimo:</b> %{x:.3f}<br>' +
                                 '<b>Costo Mínimo/1000:</b> %{y:.2f}<br>' +
                                 '<extra></extra>'
                )
            ]
        )
        
        # Configurar layout con eje x acotado entre 0 y 1
        fig.update_layout(
            title={
                'text': f'Análisis de Costo por Threshold: {model_name}<br>' +
                       f'<sub>Costo mínimo: {costo_optimo:.2f} por 1000 registros en threshold {threshold_optimo:.3f}</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            xaxis=dict(
                title='Threshold',
                titlefont=dict(size=14),
                gridcolor='lightgray',
                range=[0, 1]  # Acotar eje x entre 0 y 1
            ),
            yaxis=dict(
                title='Costo por 1000 registros',
                titlefont=dict(size=14),
                gridcolor='lightgray'
            ),
            hovermode='x unified',
            template='plotly_white',
            height=600,
            margin=dict(l=80, r=50, t=100, b=50),
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='black',
                borderwidth=1
            )
        )
        
        # Crear widgets de entrada
        input_fp = widgets.FloatText(
            value=costo_fp_inicial,
            description='Costo FP:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        
        input_fn = widgets.FloatText(
            value=costo_fn_inicial,
            description='Costo FN:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px')
        )
        
        input_prop = widgets.FloatText(
            value=prop_positivos_inicial,
            description='Prop. Positivos:',
            style={'description_width': 'initial'},
            layout=widgets.Layout(width='200px'),
            min=0.01,
            max=0.99,
            step=0.01
        )
        
        # Función para actualizar el gráfico
        def actualizar_grafico(change=None):
            costo_fp = input_fp.value
            costo_fn = input_fn.value
            prop_positivos = input_prop.value
            
            costo_calc = calcular_costo(costo_fp, costo_fn, prop_positivos)
            idx_opt = np.argmin(costo_calc)
            threshold_opt = thresholds[idx_opt]
            costo_opt = costo_calc[idx_opt]
            
            # Actualizar datos del gráfico (FigureWidget permite actualización directa)
            with fig.batch_update():
                fig.data[0].y = costo_calc
                fig.data[1].x = [threshold_opt]
                fig.data[1].y = [costo_opt]
                fig.layout.title.text = (
                    f'Análisis de Costo por Threshold: {model_name}<br>' +
                    f'<sub>Costo mínimo: {costo_opt:.2f} por 1000 registros en threshold {threshold_opt:.3f}</sub>'
                )
        
        # Conectar widgets a la función de actualización
        input_fp.observe(actualizar_grafico, names='value')
        input_fn.observe(actualizar_grafico, names='value')
        input_prop.observe(actualizar_grafico, names='value')
        
        # Crear contenedor con widgets y gráfico
        container = widgets.VBox([
            widgets.HBox([input_fp, input_fn, input_prop]),
            fig
        ])
        
        return container
    else:
        # Si no hay ipywidgets, crear figura estática
        fig = go.Figure(
            data=[
                go.Scatter(
                    x=thresholds,
                    y=costo_inicial,
                    mode='lines',
                    name='Costo por 1000 registros',
                    line=dict(color='#1f77b4', width=3),
                    hovertemplate='<b>Threshold:</b> %{x:.3f}<br>' +
                                 '<b>Costo/1000:</b> %{y:.2f}<br>' +
                                 '<b>FP:</b> %{customdata[0]}<br>' +
                                 '<b>FN:</b> %{customdata[1]}<br>' +
                                 '<extra></extra>',
                    customdata=np.column_stack((fp_counts, fn_counts))
                ),
                go.Scatter(
                    x=[threshold_optimo],
                    y=[costo_optimo],
                    mode='markers',
                    name='Umbral Óptimo',
                    marker=dict(
                        size=10,
                        color='red',
                        line=dict(width=1, color='darkred')
                    ),
                    hovertemplate='<b>Threshold Óptimo:</b> %{x:.3f}<br>' +
                                 '<b>Costo Mínimo/1000:</b> %{y:.2f}<br>' +
                                 '<extra></extra>'
                )
            ]
        )
        
        # Configurar layout con eje x acotado entre 0 y 1
        fig.update_layout(
            title={
                'text': f'Análisis de Costo por Threshold: {model_name}<br>' +
                       f'<sub>Costo mínimo: {costo_optimo:.2f} por 1000 registros en threshold {threshold_optimo:.3f}</sub>',
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': 16}
            },
            xaxis=dict(
                title='Threshold',
                titlefont=dict(size=14),
                gridcolor='lightgray',
                range=[0, 1]  # Acotar eje x entre 0 y 1
            ),
            yaxis=dict(
                title='Costo por 1000 registros',
                titlefont=dict(size=14),
                gridcolor='lightgray'
            ),
            hovermode='x unified',
            template='plotly_white',
            height=600,
            margin=dict(l=80, r=50, t=100, b=50),
            showlegend=True,
            legend=dict(
                x=0.02,
                y=0.98,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='black',
                borderwidth=1
            )
        )
        
        print("⚠️ ipywidgets no está disponible. Instala con: pip install ipywidgets")
        print("   Para interactividad, usa: fig.show() y modifica los parámetros manualmente.")
        return fig

def seleccionar_variables(
    df, 
    o_dummies=False, 
    categoria_encoding=False, 
    pais_encoding=False, 
    producto_nombre_encoding=False, 
    fecha_encoding=0,
    fecha_categoricas=False,
    usar_imputadas=False,
    remove_r=False
):
    """
    Selecciona variables del dataset procesado basándose en los parámetros especificados.
    
    Parámetros:
    -----------
    df : pandas.DataFrame
        DataFrame procesado con todas las columnas disponibles
        
    o_dummies : bool, default=False
        Si True, incluye las dummies de la columna 'o' (o_is_Y, o_is_N, o_is_NA) y 
        excluye la columna original 'o'. Si False, mantiene la columna original 'o'.
        
    categoria_encoding : bool, default=False
        Si True, incluye encoding de categoria_id (categoria_id_target_enc, categoria_id_freq_enc) 
        y excluye la columna original 'categoria_id'. Si False, mantiene la columna original 'categoria_id'.
        
    pais_encoding : bool, default=False
        Si True, incluye encoding de pais (pais_target_enc, pais_freq_enc) y excluye la columna 
        original 'pais'. Si False, mantiene la columna original 'pais'.
        
    producto_nombre_encoding : bool, default=False
        Si True, incluye features de producto_nombre (producto_num_chars, producto_num_words, 
        producto_num_special_chars, producto_avg_word_len, producto_freq) y excluye la columna 
        original 'producto_nombre'. Si False, mantiene la columna original 'producto_nombre'.
        
    fecha_encoding : int o str, default=0
        Controla qué variables temporales incluir:
        - 0 o 'none': Sin encoding de hora/fecha
        - 1 o 'normal': Solo variables normales (hora, dia_semana, dia_mes, mes)
        - 2 o 'ciclico': Solo seno/coseno (hora_sin, hora_cos, dia_semana_sin, dia_semana_cos, 
                        dia_mes_sin, dia_mes_cos)
        - 3 o 'both': Ambos (normales + cíclicas)
        
    fecha_categoricas : bool, default=False
        Si True, incluye variables categóricas temporales (es_fin_de_semana, es_nocturno, 
        es_horario_laboral). Nota: Se solapa con fecha_encoding='normal' o 'both'
        
    usar_imputadas : bool, default=False
        Si True, reemplaza las columnas con valores faltantes (NAs) por sus versiones 
        imputadas (columnas con sufijo '_imputado'). Si False, mantiene las columnas originales.
    
    remove_r : bool, default=False
        Si True, excluye la columna 'r' de las variables seleccionadas. Si False, incluye 'r'.
    
    Returns:
    --------
    pandas.DataFrame
        DataFrame con las columnas seleccionadas
    """
    
    # 1. Siempre incluir variables originales sin modificaciones
    # Columnas numéricas originales que no fueron imputadas ni encodificadas
    columnas_originales = ['a', 'b', 'c', 'd', 'e', 'f', 'h', 'k', 'l', 'm', 'n', 'q', 'r', 's', 'monto', 'score', 'fraude']
    
    # Filtrar solo las que existen en el dataframe
    columnas_seleccionadas = [col for col in columnas_originales if col in df.columns]
    
    # Remover 'r' si remove_r=True
    if remove_r and 'r' in columnas_seleccionadas:
        columnas_seleccionadas.remove('r')
    
    # Agregar row_id si existe (es una columna de identificación)
    if 'row_id' in df.columns:
        columnas_seleccionadas.append('row_id')
    
    # 2. Manejar columna 'o': si o_dummies=True, usar dummies y excluir original; si False, mantener original
    if o_dummies:
        o_cols = ['o_is_Y', 'o_is_N', 'o_is_NA']
        for col in o_cols:
            if col in df.columns:
                columnas_seleccionadas.append(col)
    else:
        # Si o_dummies=False, mantener la columna original 'o'
        if 'o' in df.columns:
            columnas_seleccionadas.append('o')
    
    # 3. Manejar categoria_id: si categoria_encoding=True, usar encodings y excluir original; si False, mantener original
    if categoria_encoding:
        categoria_cols = ['categoria_id_target_enc', 'categoria_id_freq_enc']
        for col in categoria_cols:
            if col in df.columns:
                columnas_seleccionadas.append(col)
    else:
        # Si categoria_encoding=False, mantener la columna original 'categoria_id'
        if 'categoria_id' in df.columns:
            columnas_seleccionadas.append('categoria_id')
    
    # 4. Manejar pais: si pais_encoding=True, usar encodings y excluir original; si False, mantener original
    if pais_encoding:
        pais_cols = ['pais_target_enc', 'pais_freq_enc']
        for col in pais_cols:
            if col in df.columns:
                columnas_seleccionadas.append(col)
    else:
        # Si pais_encoding=False, mantener la columna original 'pais'
        if 'pais' in df.columns:
            columnas_seleccionadas.append('pais')
    
    # 5. Manejar producto_nombre: si producto_nombre_encoding=True, usar features y excluir original; si False, mantener original
    if producto_nombre_encoding:
        producto_cols = ['producto_num_chars', 'producto_num_words', 'producto_num_special_chars', 
                        'producto_avg_word_len', 'producto_freq']
        for col in producto_cols:
            if col in df.columns:
                columnas_seleccionadas.append(col)
    else:
        # Si producto_nombre_encoding=False, mantener la columna original 'producto_nombre'
        if 'producto_nombre' in df.columns:
            columnas_seleccionadas.append('producto_nombre')
    
    # 6. Agregar variables temporales según fecha_encoding
    # Normalizar el parámetro fecha_encoding
    if isinstance(fecha_encoding, str):
        fecha_encoding = fecha_encoding.lower()
        if fecha_encoding in ['none', '0']:
            fecha_encoding = 0
        elif fecha_encoding in ['normal', '1']:
            fecha_encoding = 1
        elif fecha_encoding in ['ciclico', 'ciclic', '2']:
            fecha_encoding = 2
        elif fecha_encoding in ['both', 'ambos', '3']:
            fecha_encoding = 3
    
    # Variables temporales normales
    fecha_normales = ['hora', 'dia_semana', 'dia_mes', 'mes']
    
    # Variables temporales cíclicas
    fecha_ciclicas = ['hora_sin', 'hora_cos', 'dia_semana_sin', 'dia_semana_cos', 'dia_mes_sin', 'dia_mes_cos']
    
    if fecha_encoding == 1 or fecha_encoding == 'normal':
        # Solo variables normales
        for col in fecha_normales:
            if col in df.columns:
                columnas_seleccionadas.append(col)
    elif fecha_encoding == 2 or fecha_encoding == 'ciclico':
        # Solo variables cíclicas
        for col in fecha_ciclicas:
            if col in df.columns:
                columnas_seleccionadas.append(col)
    elif fecha_encoding == 3 or fecha_encoding == 'both':
        # Ambas (normales + cíclicas)
        for col in fecha_normales + fecha_ciclicas:
            if col in df.columns:
                columnas_seleccionadas.append(col)
    # Si fecha_encoding == 0 o 'none', no agregamos ninguna variable temporal
    
    # 7. Agregar fecha_categoricas si se solicita (solo si no están ya incluidas)
    if fecha_categoricas:
        fecha_cat_cols = ['es_fin_de_semana', 'es_nocturno', 'es_horario_laboral']
        for col in fecha_cat_cols:
            if col in df.columns and col not in columnas_seleccionadas:
                columnas_seleccionadas.append(col)
    
    # Eliminar duplicados manteniendo el orden
    columnas_seleccionadas = list(dict.fromkeys(columnas_seleccionadas))
    
    # Retornar solo las columnas que existen en el dataframe
    columnas_finales = [col for col in columnas_seleccionadas if col in df.columns]
    
    # 8. Si usar_imputadas=True, reemplazar columnas con NAs por sus versiones imputadas
    if usar_imputadas:
        columnas_reemplazadas = []
        columnas_finales_actualizadas = []
        for col in columnas_finales:
            col_imputada = f'{col}_imputado'
            # Si existe la versión imputada y la columna original tiene NAs, usar la imputada
            if col_imputada in df.columns and col in df.columns:
                if df[col].isnull().sum() > 0:
                    # Usar la versión imputada en lugar de la original
                    columnas_finales_actualizadas.append(col_imputada)
                    columnas_reemplazadas.append(col)
                else:
                    # La columna no tiene NAs, mantener la original
                    columnas_finales_actualizadas.append(col)
            else:
                # No hay versión imputada disponible, mantener la original
                columnas_finales_actualizadas.append(col)
        
        columnas_finales = columnas_finales_actualizadas
        if columnas_reemplazadas:
            print(f"Columnas reemplazadas por versiones imputadas: {', '.join(columnas_reemplazadas)}")
    
    return df[columnas_finales].copy()
