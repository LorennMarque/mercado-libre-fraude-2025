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
import warnings
warnings.filterwarnings('ignore')

# Configuración de estilo para los plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


def evaluate_model(y_true, y_pred, y_proba=None, model_name="Modelo", 
                   save_plots=False, output_dir="plots/", show_plots=True):
    """
    Evalúa un modelo de clasificación binaria con métricas y visualizaciones completas.
    Optimizado para detección de fraude (clases desbalanceadas).
    
    Parameters:
    -----------
    y_true : array-like
        Valores reales (ground truth)
    y_pred : array-like
        Predicciones binarias del modelo
    y_proba : array-like, optional
        Probabilidades predichas (para clase positiva). Si es None, se calcula a partir de y_pred
    model_name : str, default="Modelo"
        Nombre del modelo para los títulos de los gráficos
    save_plots : bool, default=False
        Si True, guarda los plots en archivos
    output_dir : str, default="plots/"
        Directorio donde guardar los plots
    show_plots : bool, default=True
        Si True, muestra los plots
    
    Returns:
    --------
    dict : Diccionario con todas las métricas calculadas
    """
    
    # Convertir a numpy arrays
    y_true = np.array(y_true)
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
        'total_samples': len(y_true)
    }
    
    # Imprimir métricas
    print("=" * 80)
    print(f"EVALUACIÓN DEL MODELO: {model_name}")
    print("=" * 80)
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

def seleccionar_variables(
    df, 
    o_dummies=False, 
    categoria_encoding=False, 
    pais_encoding=False, 
    producto_nombre_encoding=False, 
    fecha_encoding=0,
    fecha_categoricas=False
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
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame con las columnas seleccionadas
    """
    
    # 1. Siempre incluir variables originales sin modificaciones
    # Columnas numéricas originales que no fueron imputadas ni encodificadas
    columnas_originales = ['a', 'b', 'c', 'd', 'e', 'f', 'h', 'k', 'l', 'm', 'n', 'q', 'r', 's', 'monto', 'score']
    
    # Filtrar solo las que existen en el dataframe
    columnas_seleccionadas = [col for col in columnas_originales if col in df.columns]
    
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
    
    return df[columnas_finales].copy()
