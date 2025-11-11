import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, roc_curve
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import re

# Cargar datos
fraud_df = pd.read_csv('data/raw/fraud_dataset_v2.csv')

############################################
# LIMPIEZA DE DATOS Y NOMBRE DE COLUMNAS   #
############################################

# Eliminar columna "Unnamed: 0" si existe
if 'Unnamed: 0' in fraud_df.columns:
    fraud_df = fraud_df.drop(columns=['Unnamed: 0'])
    print("Columna 'Unnamed: 0' eliminada")

# Renombramos Columnas ---------------------#
fraud_df = fraud_df.rename(columns={
    "g": "pais",
    "i": "producto_nombre",
    "j": "categoria_id"
    })

print("Columnas renombradas")

# Quitar columnas con correlacion spearman cerca de 1 (o -1)

# Tratar valores faltantes ------------------#

# Imputar "BR" en todos los valores nulos de la columna "pais"
if "pais" in fraud_df.columns:
    fraud_df["pais"] = fraud_df["pais"].fillna("BR")


# Procesamiento de columna "o" creando variables dummies limpias
# MANTENEMOS la columna original 'o' y agregamos las dummies

o_dummies = pd.get_dummies(fraud_df['o'], prefix='o', dummy_na=True)

o_dummies = o_dummies.rename(columns={
    'o_Y': 'o_is_Y',
    'o_N': 'o_is_N',
    'o_nan': 'o_is_NA'
}).astype(int)

# NO eliminamos 'o', solo agregamos las dummies
fraud_df = pd.concat([fraud_df, o_dummies], axis=1)

print("Columna 'o' procesada, dummies creados (columna original 'o' mantenida)")

# Convertir columnas 'o' y 'p' a numéricas (Y=1, N=0, manteniendo NA en 'o')
if 'o' in fraud_df.columns:
    fraud_df['o'] = fraud_df['o'].map({'Y': 1, 'N': 0}).astype('Int64')  # Int64 permite NA
    print("Columna 'o' convertida a numérica (Y=1, N=0, NA mantenidos)")

if 'p' in fraud_df.columns:
    fraud_df['p'] = fraud_df['p'].map({'Y': 1, 'N': 0}).astype('Int64')  # Int64 permite NA
    print("Columna 'p' convertida a numérica (Y=1, N=0)")

# MANTENEMOS la columna 'r' - NO la eliminamos
# fraud_df = fraud_df.drop(columns=['r'])  # COMENTADO: mantenemos todas las columnas
print("Columna 'r' mantenida (no eliminada)")

# Imputación de columnas 'b', 'c', 'd', 'f', 'q', 'l', 'm' usando IterativeImputer (MICE)
# Crearemos columnas nuevas con sufijo '_imputado' para mantener las originales
cols_imputar = ['b', 'c', 'd', 'e', 'f', 'h', 'monto', 'q', 'l', 'm']
cols_existentes = [col for col in cols_imputar if col in fraud_df.columns]

# Solo imputar columnas que tienen valores faltantes
cols_con_nulos = [col for col in cols_existentes if fraud_df[col].isnull().sum() > 0]
print(f"Columnas a imputar (con valores faltantes): {cols_con_nulos}")

# Inicializar lista de columnas imputadas
cols_imputadas_creadas = []

if len(cols_con_nulos) > 0:
    # Incluir también columnas sin nulos que pueden ayudar en la imputación
    df_imputar = fraud_df[cols_existentes].copy()
    
    imputer = IterativeImputer(
        max_iter=10,
        random_state=42,
        initial_strategy="median"
    )
    imputed = imputer.fit_transform(df_imputar)
    df_imputado = pd.DataFrame(imputed, columns=cols_existentes, index=fraud_df.index)
    
    # Crear nuevas columnas con sufijo '_imputado' para las versiones imputadas
    # Mantenemos las columnas originales (con nulos) y agregamos las imputadas
    for col in cols_con_nulos:
        col_imputada = f'{col}_imputado'
        fraud_df[col_imputada] = df_imputado[col]
        cols_imputadas_creadas.append(col_imputada)
    
    print(f"Columnas imputadas creadas con sufijo '_imputado': {cols_imputadas_creadas}")
else:
    print("No hay columnas con valores faltantes para imputar")

# FEATURE ENGINEERING ----------------------------------#
# Función auxiliar para aplicar feature engineering completo
def aplicar_feature_engineering(df):
    """Aplica todo el feature engineering a un dataframe"""
    df = df.copy()
    
    # Encoding de categoria_id
    if 'categoria_id' in df.columns:
        target_mean = df.groupby('categoria_id')['fraude'].mean()
        df['categoria_id_target_enc'] = df['categoria_id'].map(target_mean).fillna(0.0).astype(float)
        freq = df['categoria_id'].value_counts()
        df['categoria_id_freq_enc'] = df['categoria_id'].map(freq).fillna(0).astype(int)
    
    # Encoding de pais
    if 'pais' in df.columns:
        g_target_mean = df.groupby('pais')['fraude'].mean()
        df['pais_target_enc'] = df['pais'].map(g_target_mean).fillna(0.0).astype(float)
        g_freq = df['pais'].value_counts()
        df['pais_freq_enc'] = df['pais'].map(g_freq).fillna(0).astype(int)
    
    # Features de producto_nombre
    if 'producto_nombre' in df.columns:
        df['producto_num_chars'] = df['producto_nombre'].astype(str).apply(len).astype(int)
        df['producto_num_words'] = df['producto_nombre'].astype(str).apply(lambda x: len(x.split())).astype(int)
        df['producto_num_special_chars'] = df['producto_nombre'].astype(str).apply(lambda x: len(re.findall(r'[^a-zA-Z0-9\s]', x))).astype(int)
        def avg_word_length(s):
            words = str(s).split()
            if len(words) == 0:
                return 0.0
            return float(sum(len(word) for word in words) / len(words))
        df['producto_avg_word_len'] = df['producto_nombre'].apply(avg_word_length).astype(float)
        producto_freq = df['producto_nombre'].value_counts()
        df['producto_freq'] = df['producto_nombre'].map(producto_freq).fillna(0).astype(int)
    
    # Features temporales
    if 'fecha' in df.columns:
        df['fecha'] = pd.to_datetime(df['fecha'], errors='coerce')
        df['hora'] = df['fecha'].dt.hour.fillna(0).astype(int)
        df['dia_semana'] = df['fecha'].dt.dayofweek.fillna(0).astype(int)
        df['dia_mes'] = df['fecha'].dt.day.fillna(1).astype(int)
        df['mes'] = df['fecha'].dt.month.fillna(1).astype(int)
        df['es_fin_de_semana'] = df['dia_semana'].isin([5, 6]).astype(int)
        df['es_nocturno'] = df['hora'].apply(lambda x: 1 if (x >= 22 or x < 6) else 0).astype(int)
        df['es_horario_laboral'] = df['hora'].apply(lambda x: 1 if (x >= 9 and x < 18) else 0).astype(int)
        df['hora_sin'] = np.sin(2 * np.pi * df['hora']/24).astype(float)
        df['hora_cos'] = np.cos(2 * np.pi * df['hora']/24).astype(float)
        df['dia_semana_sin'] = np.sin(2 * np.pi * df['dia_semana']/7).astype(float)
        df['dia_semana_cos'] = np.cos(2 * np.pi * df['dia_semana']/7).astype(float)
        df['dia_mes_sin'] = np.sin(2 * np.pi * df['dia_mes']/31).astype(float)
        df['dia_mes_cos'] = np.cos(2 * np.pi * df['dia_mes']/31).astype(float)
    
    return df

# Función auxiliar para normalizar columnas numéricas
def normalizar_columnas(df):
    """Normaliza todas las columnas numéricas entre 0 y 1, excepto las variables cíclicas (seno/coseno)"""
    df = df.copy()
    numerical_cols = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    # Excluir variables objetivo, identificadores y variables cíclicas (seno/coseno)
    excluir = ['fraude', 'row_id', 'hora_sin', 'hora_cos', 'dia_semana_sin', 'dia_semana_cos', 
               'dia_mes_sin', 'dia_mes_cos']
    for excl in excluir:
        if excl in numerical_cols:
            numerical_cols.remove(excl)
    
    for col in numerical_cols:
        min_val = df[col].min()
        max_val = df[col].max()
        if min_val != max_val:
            df[col] = (df[col] - min_val) / (max_val - min_val)
        else:
            df[col] = 0.0
    
    return df

# Aplicar feature engineering completo
print("\nAplicando feature engineering completo...")
fraud_df = aplicar_feature_engineering(fraud_df)

# Normalizar columnas numéricas (después del feature engineering)
# Esto normalizará tanto las columnas originales como las imputadas
print("Normalizando columnas numéricas...")
fraud_df = normalizar_columnas(fraud_df)

numerical_cols = fraud_df.select_dtypes(include=['float64', 'int64']).columns.tolist()
for excl in ['fraude', 'row_id']:
    if excl in numerical_cols:
        numerical_cols.remove(excl)
print(f"Total de columnas numéricas normalizadas: {len(numerical_cols)} columnas")

# Identificar columnas imputadas para el reporte
cols_imputadas = [col for col in fraud_df.columns if col.endswith('_imputado')]

# Guardar dataset único con ambas versiones (originales e imputadas)
fraud_df.to_csv("data/processed/fraud_dataset_processed.csv", index=False)
print("\n✅ Datos procesados guardados en data/processed/fraud_dataset_processed.csv")
print(f"   - Columnas originales (pueden tener nulos): {len([c for c in fraud_df.columns if not c.endswith('_imputado')])}")
print(f"   - Columnas imputadas (sin nulos): {len(cols_imputadas)}")
print(f"   - Total de columnas: {len(fraud_df.columns)}")
if len(cols_imputadas) > 0:
    print(f"   - Columnas imputadas: {', '.join(cols_imputadas)}")