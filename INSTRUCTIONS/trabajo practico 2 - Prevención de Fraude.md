# 🎓 TRABAJO PRÁCTICO 2
## Detección de Fraude con Machine Learning

**Asignatura**: Taller de Resolución de Problemas II  
**Tema**: Análisis y Modelado de Datos Desbalanceados  
**Modalidad**: Trabajo grupal 
**Fecha de entrega**: 14 de noviembre a presentar en la oficina de Mercado Libre

---

## 📋 ÍNDICE

1. [Introducción](#introducción)
2. [Objetivos de Aprendizaje](#objetivos-de-aprendizaje)
3. [Contexto del Problema](#contexto-del-problema)
4. [Dataset Proporcionado](#dataset-proporcionado)
5. [Consignas del Trabajo](#consignas-del-trabajo)
6. [Criterios de Evaluación](#criterios-de-evaluación)
7. [Entregables](#entregables)
8. [Recursos Disponibles](#recursos-disponibles)
9. [Cronograma Sugerido](#cronograma-sugerido)
10. [Preguntas Frecuentes](#preguntas-frecuentes)

---

## 🎯 INTRODUCCIÓN

El fraude en transacciones financieras representa uno de los desafíos más importantes para las empresas de comercio electrónico y fintech. Detectar transacciones fraudulentas en tiempo real es crucial para:

- **Proteger a los clientes** de cargos no autorizados
- **Minimizar pérdidas económicas** de la empresa
- **Mantener la confianza** en la plataforma

Sin embargo, este problema presenta un desafío técnico significativo: **los datasets de fraude son altamente desbalanceados**. En un escenario típico, solo el 1-5% de las transacciones son fraudulentas, mientras que el 95-99% son legítimas.

Este trabajo práctico te desafía a desarrollar un sistema de detección de fraude utilizando técnicas de machine learning, enfrentando los desafíos reales que encontrarías en la industria.

---

## 🎓 OBJETIVOS DE APRENDIZAJE

Al completar este trabajo práctico, serás capaz de:

### Objetivos Técnicos
- ✅ Identificar y analizar datasets desbalanceados
- ✅ Aplicar técnicas de preprocesamiento de datos (limpieza, encoding, normalización)
- ✅ Implementar técnicas de balanceo (SMOTE, class weights, undersampling)
- ✅ Desarrollar modelos de clasificación supervisada
- ✅ Evaluar modelos con métricas apropiadas para datos desbalanceados
- ✅ Optimizar umbrales de decisión según objetivos de negocio
- ✅ Realizar feature engineering para mejorar el rendimiento

### Objetivos de Negocio
- ✅ Interpretar resultados en contexto real
- ✅ Calcular y minimizar costos de negocio (FP vs FN)
- ✅ Comunicar hallazgos de manera clara y profesional
- ✅ Tomar decisiones basadas en datos

---

## 🏢 CONTEXTO DEL PROBLEMA

### Escenario

Trabajas como Data Scientist en **Mercado Libre**, una empresa de e-commerce en Latinoamérica. La empresa procesa aproximadamente muchas transacciones, pero enfrenta un de fraude (por el volumen de transacciones y su atractivo para ser atacada).

### Tu Misión

El equipo directivo te ha encomendado desarrollar un **modelo de machine learning** que:

1. **Maximice la detección de fraudes** (recall alto)
2. **Minimice las falsas alarmas** (precision razonable)
3. **Reduzca los costos totales** del sistema (FP + FN)
4. **Sea interpretable y justificable** para el equipo de negocio
5. **Que sugieras cualquier otra mejora** para enriquecer la detección de fraude

### Costos de Negocio

```
False Positive (FP): Bloquear transacción legítima
├── Llamada de verificación al cliente
├── Tiempo de atención al cliente
├── Posible pérdida de venta
└── Costo estimado: $5 por FP (si solo tenemos en cuenta el costo del contacto y no los efetos de marca o reputacionales, el churn, etc.)

False Negative (FN): Fraude no detectado
├── Pérdida del monto de transacción
├── Cargo de vuelta (chargeback)
├── Tarifa de procesamiento del banco
├── Investigación del caso
└── Costo estimado: $200 por FN

Ratio de costo: FN:FP = 40:1 (con los assumptions anteriores)
```

**Objetivo de negocio**: Minimizar el costo total mensual de errores.

---

## 📊 DATASET PROPORCIONADO

### Descripción General

**Archivo**: `01_datos/fraud_dataset_v2.csv`

- **Registros**: 250,000 transacciones
- **Período**: 4 meses (Marzo - Abril 2020)
- **Región**: Principalmente Brasil (74%) y Argentina (21%)
- **Desbalance**: 97% no fraude (242,498) vs **3% fraude (7,502)**
- **Ratio**: 32:1 (clase mayoritaria:minoritaria)

### Variables del Dataset

#### Variable Objetivo
- **`fraude`**: {0 = No Fraude, 1 = Fraude}

#### Variables de Entrada (Features)

**Variables Numéricas** (16 columnas):
| Variable | Tipo | Descripción | Rango |
|----------|------|-------------|-------|
| `monto` | float | Monto de la transacción | $0.02 - $3,696 |
| `score` | int | Score de riesgo del sistema actual | 0-100 |
| `a`, `b`, `c`, `d`, `e`, `f`, `h`, `k`, `l`, `m`, `n`, `q`, `r`, `s` | float/int | Variables anonimizadas de comportamiento | Variado |

**Variables Categóricas** (7 columnas):
| Variable | Tipo | Descripción | Valores Únicos |
|----------|------|-------------|----------------|
| `g` | string | País de la transacción | 51 países |
| `i` | string | ID de producto | 127,804 productos |
| `j` | string | Categoría de producto | 8,324 categorías |
| `o` | string | Variable binaria Y/N | 2 valores |
| `p` | string | Variable binaria Y/N | 2 valores |
| `fecha` | datetime | Timestamp de transacción | 145,813 valores únicos |

**⚠️ Nota**: Algunas variables contienen valores faltantes (NaN) que deberás manejar apropiadamente.

---

## 📝 CONSIGNAS DEL TRABAJO

El trabajo está dividido en **6 secciones obligatorias** y **1 sección opcional** para destacarse.

---

### **PARTE 1: Análisis Exploratorio de Datos (EDA)** - 15 puntos

#### Objetivos
Comprender en profundidad el dataset y el problema de desbalance.

#### Tareas Requeridas

**1.1. Exploración Básica**
- Cargar el dataset y mostrar información general (shape, tipos de datos, valores faltantes)
- Calcular estadísticas descriptivas de variables numéricas
- Identificar y cuantificar el desbalance de clases
- Visualizar la distribución de la variable objetivo

**1.2. Análisis de Variables**
- Analizar distribución de variables numéricas (histogramas, boxplots)
- Analizar variables categóricas (frecuencias, top valores)
- Identificar outliers en variables numéricas
- Analizar correlaciones entre variables numéricas

**1.3. Análisis del Fraude**
- Comparar características de transacciones fraudulentas vs normales
- Identificar variables con mayor diferencia entre clases
- Analizar patrones temporales (hora del día, día de semana)
- Analizar distribución geográfica del fraude

**Entregable**: Notebook con visualizaciones y conclusiones del EDA.

---

### **PARTE 2: Preprocesamiento de Datos**

#### Objetivos
Preparar el dataset para el modelado, manejando apropiadamente los desafíos presentes.

#### Tareas Requeridas

**2.1. Manejo de Valores Faltantes**
- Analizar el patrón de valores faltantes
- Implementar estrategia de imputación justificada
- Documentar decisiones tomadas

**2.2. Feature Engineering**
- Extraer features temporales de la variable `fecha`:
  - Hora del día
  - Día de la semana
  - Es fin de semana (binaria)
  - Es horario nocturno (binaria)
  - Día del mes
- Crear features adicionales (ratios, agregaciones, etc.)

**2.3. Encoding de Variables Categóricas**
- Implementar encoding apropiado para variables categóricas
- Manejar variables de alta cardinalidad (`i`, `j`)
- Justificar la elección de técnica de encoding

**2.4. Split Train/Test**
- Dividir dataset en train (80%) y test (20%)
- Verificar que el desbalance se mantiene en ambos sets

**Entregable**: Código documentado de preprocesamiento y dataset procesado.

---

### **PARTE 3: Modelo Baseline**

#### Objetivos
Desarrollar un modelo baseline que sirva como punto de comparación.

#### Tareas Requeridas

**3.1. Entrenamiento del Modelo Baseline**
- Entrenar un modelo de clasificación **sin técnicas de balanceo avanzadas**
- Algoritmos sugeridos: Random Forest, Logistic Regression, o XGBoost
- Usar `class_weight='balanced'` (o equivalente) como técnica básica

**3.2. Evaluación con Métricas Apropiadas**
- **NO usar Accuracy** como métrica principal
- Calcular y reportar:
  - **Confusion Matrix** (interpretar cada cuadrante)
  - **Recall** (métrica principal para fraude)
  - **Precision**
  - **F1-Score**
  - **AUC-ROC**
  - **AUC-PR** (Precision-Recall Curve)
- Visualizar curvas ROC y Precision-Recall

**3.3. Interpretación de Resultados**
- Explicar qué significan los resultados en contexto de negocio
- Identificar el principal problema del modelo baseline
- Calcular el costo total de errores (FP × $5 + FN × $200)

**Entregable**: Modelo baseline entrenado, métricas calculadas e interpretadas.

---

### **PARTE 4: Técnicas de Balanceo**

#### Objetivos
Implementar y comparar técnicas avanzadas para manejar el desbalance.

#### Tareas Requeridas

**4.1. SMOTE (Synthetic Minority Over-sampling Technique)**
- Implementar SMOTE en el **conjunto de train** únicamente
- ⚠️ **IMPORTANTE**: Aplicar SMOTE **DESPUÉS** del train/test split (evitar data leakage)
- Verificar el nuevo balance de clases
- Entrenar modelo con datos balanceados
- Evaluar con las mismas métricas que el baseline

**4.2. Técnica Adicional de Balanceo**
Implementar **al menos una** de las siguientes:
- **Undersampling** de la clase mayoritaria
- **Combinación** de SMOTE + Undersampling (SMOTETomek, SMOTEENN)
- **Ensemble con balanceo**: BalancedRandomForest
- **Ajuste de class_weight** optimizado

**4.3. Comparación de Técnicas**
- Crear tabla comparativa de resultados:
- Analizar trade-offs (precision vs recall)
- Justificar cuál técnica es más apropiada para el negocio

**Entregable**: Modelos con diferentes técnicas de balanceo, comparación de resultados.

---

### **PARTE 5: Optimización de Threshold**

#### Objetivos
Optimizar el umbral de decisión para maximizar el objetivo de negocio.

#### Tareas Requeridas

**5.1. Búsqueda de Threshold Óptimo**
- Probar diferentes valores de threshold (0.1, 0.15, 0.2, ..., 0.9)
- Para cada threshold, calcular:
  - Confusion matrix
  - Precision, Recall, F1-Score
  - Costo total de negocio
- Visualizar cómo varían las métricas según el threshold

**5.2. Selección del Threshold Óptimo**
Justificar la elección del threshold óptimo según **dos criterios**:
1. **Maximizar F1-Score** (balance precision-recall)
2. **Minimizar costo total de negocio**

Nota: Ambos criterios pueden dar thresholds diferentes. Discutir las implicaciones.


**Entregable**: Análisis de threshold optimization con visualizaciones y recomendación final.

---

### **PARTE 6: Presentar todo**

#### Objetivos
Poder comunicar lo que hicieron

#### Tareas Requeridas

**6.1 Preparar una presentación de todo lo que hicieron que se entrega en vivo el 14 de noviembre (lo presentan)**. Tener en cuenta:
- Contar lo que hicieron en forma más sintetica que con los notebooks
- Tener el material técnico a mano para acceder
- Usar recursos visuales que ayuden y ejemplos que lo bajen a tierra
- Contar que cosas extras hubiesen hecho

**Entregable**: Presentación y Sección de conclusiones y recomendaciones en el informe final.

### Recursos Externos Recomendados

#### Documentación Oficial
- [Scikit-learn: Imbalanced Data](https://scikit-learn.org/stable/modules/cross_validation.html#stratification)
- [Imbalanced-learn Documentation](https://imbalanced-learn.org/stable/)
- [SMOTE Paper Original](https://arxiv.org/abs/1106.1813)

#### Tutoriales
- [Machine Learning Mastery: Imbalanced Classification](https://machinelearningmastery.com/what-is-imbalanced-classification/)
- [Towards Data Science: Dealing with Imbalanced Data](https://towardsdatascience.com/methods-for-dealing-with-imbalanced-data-5b761be45a18)
