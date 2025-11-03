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

# Cargar datos
fraud_df = pd.read_csv('01_datos/fraud_dataset_v2.csv')

# LIMPIEZA DE DATOS Y NOMBRE DE COLUMNAS

# Renombramos
# Columna G es pais
# Columna "i" es producto
# Columna "j" es categoria id
fraud_df = fraud_df.rename(columns={"g": "pais", "i": "producto", "j": "categoria_id"})

# Quitar columnas con correlacion spearman cerca de 1 (o -1)

# Tratar valores faltantes
# Borramos "o" y lo pasamos a one hot encoding (1, 0) y otro de "is null" (0, 1)

# Eliminamos r y usamos MICE para imputar NAs de b y c. IMPUTAMOS TODO CON MICE
# Los NAs de C y b no son una caracteristica o relacion del producto. Son errores de medicion. (que pasaron mucho la S17 de 2020)

# Normalizamos las columnas numéricas

# FEATURE ENGINEERING

#categoricas:
# para categoria hacemos => Target Encoding (peso) + Frecuency Encoding
# para paises "g" => Target Encoding (peso) + Frecuency Encoding
# Para nombre de producto "i" => Sacamos y creamos: # de caracteres,# de palabras, # de caracteres especiales, promedio longitud palabra,
# se_ha_comprado_antes => N comprados

# Fechas y frecuencias:
# hora, dia_semana, dia_mes, mes => 1 si es fin de semana, 0 si no
# es_madrugada, es_noche, es_horario_laboral => 1 si es madrugada, noche o horario laboral, 0 si no
# "Ciclico, sen y coseno"

# Split train/test