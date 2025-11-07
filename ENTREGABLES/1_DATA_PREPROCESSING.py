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

# Renombramos Columnas ---------------------#
fraud_df = fraud_df.rename(columns={
    "g": "pais",
    "i": "producto_nombre",
    "j": "categoria_id"
    })

print("Columnas renombradas")

# Quitar columnas con correlacion spearman cerca de 1 (o -1)

# Tratar valores faltantes ------------------#

# Procesamiento de columna "o" creando variables dummies limpias

o_dummies = pd.get_dummies(fraud_df['o'], prefix='o', dummy_na=True)

o_dummies = o_dummies.rename(columns={
    'o_Y': 'o_is_Y',
    'o_N': 'o_is_N',
    'o_nan': 'o_is_NA',
    'Unnamed: 0': 'row_id'
}).astype(int)

fraud_df = pd.concat([fraud_df.drop(columns=['o']), o_dummies], axis=1)

print("Columna 'o' procesada, dummies creados")


# Eliminamos r, imputacion de valores deb y c con MICE. 
fraud_df = fraud_df.drop(columns=['r'])
print("Columna 'r' eliminada")

# Imputación de columnas 'b' y 'c' usando IterativeImputer (MICE)
cols_imputar = ['b', 'c', 'd', 'e', 'f', 'h', 'monto']
cols_existentes = [col for col in cols_imputar if col in fraud_df.columns]

df_imputar = fraud_df[cols_existentes].copy()

imputer = IterativeImputer(
    max_iter=10,
    random_state=42,
    initial_strategy="median"
)
imputed = imputer.fit_transform(df_imputar)
df_imputado = pd.DataFrame(imputed, columns=cols_existentes, index=fraud_df.index)

fraud_df['b'] = df_imputado['b']
fraud_df['c'] = df_imputado['c']

print("Columnas 'b' y 'c' imputadas")

# Normalizamos las columnas numéricas ------------------#
# Obtener columnas numéricas automáticamente
numerical_cols = fraud_df.select_dtypes(include=['float64', 'int64']).columns.tolist()

# Excluir la variable objetivo 'fraude' y 'row_id' si estuvieran entre ellas
for excl in ['fraude', 'row_id']:
    if excl in numerical_cols:
        numerical_cols.remove(excl)

# Normalizar todas las variables numéricas entre 0 y 1
for col in numerical_cols:
    min_val = fraud_df[col].min()
    max_val = fraud_df[col].max()
    # Evitar división por cero
    if min_val != max_val:
        fraud_df[col] = (fraud_df[col] - min_val) / (max_val - min_val)
    else:
        fraud_df[col] = 0.0

print(f"Columnas numéricas normalizadas entre 0 y 1: {numerical_cols}")


# FEATURE ENGINEERING ----------------------------------#

#categoricas:
# para categoria hacemos => Target Encoding (peso) + Frecuency Encoding

# Encoding de variable categórica 'j' (categoría) usando Target Encoding y Frequency Encoding

# Nos aseguramos de que 'j' existe
if 'categoria_id' in fraud_df.columns:
    # Target Encoding de 'categoria_id'
    target_mean = fraud_df.groupby('categoria_id')['fraude'].mean()
    fraud_df['categoria_id_target_enc'] = fraud_df['categoria_id'].map(target_mean)
    
    # Frequency Encoding de 'categoria_id'
    freq = fraud_df['categoria_id'].value_counts()
    fraud_df['categoria_id_freq_enc'] = fraud_df['categoria_id'].map(freq)
    
    # Quitamos la columna original 'categoria_id'
    fraud_df = fraud_df.drop(columns=['categoria_id'])
    print("Columna 'categoria_id' reemplazada por target encoding y frequency encoding.")
else:
    print("La columna 'categoria_id' no existe en el dataframe.")

# para paises "g" => Target Encoding (peso) + Frecuency Encoding

# Encoding de variable categórica 'pais' (país) usando Target Encoding y Frequency Encoding
if 'pais' in fraud_df.columns:
    # Target Encoding de 'pais'
    g_target_mean = fraud_df.groupby('pais')['fraude'].mean()
    fraud_df['pais_target_enc'] = fraud_df['pais'].map(g_target_mean)

    # Frequency Encoding de 'pais'
    g_freq = fraud_df['pais'].value_counts()
    fraud_df['pais_freq_enc'] = fraud_df['pais'].map(g_freq)

    # Quitamos la columna original 'pais'
    fraud_df = fraud_df.drop(columns=['pais'])
    print("Columna 'pais' reemplazada por target encoding y frequency encoding.")
else:
    print("La columna 'pais' no existe en el dataframe.")

# Para nombre de producto "i" => Sacamos y creamos: # de caracteres,# de palabras, # de caracteres especiales, promedio longitud palabra,
if 'producto_nombre' in fraud_df.columns:
    # Número de caracteres
    fraud_df['producto_num_chars'] = fraud_df['producto_nombre'].astype(str).apply(len)
    
    # Número de palabras
    fraud_df['producto_num_words'] = fraud_df['producto_nombre'].astype(str).apply(lambda x: len(x.split()))
    
    # Número de caracteres especiales (caracteres que no son alfanuméricos ni espacios)
    fraud_df['producto_num_special_chars'] = fraud_df['producto_nombre'].astype(str).apply(lambda x: len(re.findall(r'[^a-zA-Z0-9\s]', x)))
    
    # Promedio longitud de palabra
    def avg_word_length(s):
        words = str(s).split()
        if len(words) == 0:
            return 0
        return sum(len(word) for word in words) / len(words)
    fraud_df['producto_avg_word_len'] = fraud_df['producto_nombre'].apply(avg_word_length)

    print("Features creadas para 'producto_nombre': número de caracteres, palabras, caracteres especiales y promedio longitud palabra.")
else:
    print("La columna 'producto_nombre' no existe en el dataframe.")

# se_ha_comprado_antes => N comprados
if 'producto_nombre' in fraud_df.columns:
    # Frecuencia del nombre de producto 'producto_nombre'
    producto_freq = fraud_df['producto_nombre'].value_counts()
    fraud_df['producto_freq'] = fraud_df['producto_nombre'].map(producto_freq)
    print("Feature creada para 'producto_nombre': frecuencia de nombre de producto.")
else:
    print("La columna 'producto_nombre' no existe en el dataframe.")


# Fechas y frecuencias:
# hora, dia_semana, dia_mes, mes => 1 si es fin de semana, 0 si no
# es_madrugada, es_noche, es_horario_laboral => 1 si es madrugada, noche o horario laboral, 0 si no
# "Ciclico, sen y coseno"
if 'fecha' in fraud_df.columns:
    # Convertimos fecha a datetime si no lo está
    fraud_df['fecha'] = pd.to_datetime(fraud_df['fecha'], errors='coerce')
    
    # Hora del día
    fraud_df['hora'] = fraud_df['fecha'].dt.hour
    
    # Día de la semana (0=lunes, 6=domingo)
    fraud_df['dia_semana'] = fraud_df['fecha'].dt.dayofweek
    
    # Día del mes
    fraud_df['dia_mes'] = fraud_df['fecha'].dt.day
    
    # Mes
    fraud_df['mes'] = fraud_df['fecha'].dt.month

    # Es fin de semana (sábado = 5, domingo = 6)
    fraud_df['es_fin_de_semana'] = fraud_df['dia_semana'].isin([5, 6]).astype(int)
    
    # Es horario nocturno (ejemplo: entre las 22 y las 6)
    fraud_df['es_nocturno'] = fraud_df['hora'].apply(lambda x: 1 if (x >= 22 or x < 6) else 0)
    
    # Es horario laboral (ejemplo: 9 a 18)
    fraud_df['es_horario_laboral'] = fraud_df['hora'].apply(lambda x: 1 if (x >= 9 and x < 18) else 0)
    
    # Codificación cíclica de hora del día
    fraud_df['hora_sin'] = np.sin(2 * np.pi * fraud_df['hora']/24)
    fraud_df['hora_cos'] = np.cos(2 * np.pi * fraud_df['hora']/24)
    
    # Codificación cíclica de día de la semana
    fraud_df['dia_semana_sin'] = np.sin(2 * np.pi * fraud_df['dia_semana']/7)
    fraud_df['dia_semana_cos'] = np.cos(2 * np.pi * fraud_df['dia_semana']/7)

    print("Columnas de fecha y frecuencia creadas: hora, dia_semana, mes, dia_mes, es_fin_de_semana, es_nocturno, es_horario_laboral, codificaciones cíclicas.")
else:
    print("No existe columna 'fecha' en el dataframe.")

# Guardar el DataFrame procesado como CSV
fraud_df.to_csv("data/processed/fraud_dataset_processed.csv", index=False)
print("Datos procesados guardados en data/processed/fraud_dataset_processed.csv")