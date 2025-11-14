# 📊 RÚBRICA DE EVALUACIÓN - TRABAJO PRÁCTICO FINAL
## Detección de Fraude con Machine Learning

**Uso**: Este documento es para el docente. Permite evaluar de manera objetiva y consistente todos los trabajos.

---

## 🎯 DISTRIBUCIÓN DE PUNTAJE

| Sección | Puntaje | % del Total |
|---------|---------|-------------|
| Parte 1: EDA | 15 | 15% |
| Parte 2: Preprocesamiento | 15 | 15% |
| Parte 3: Modelo Baseline | 15 | 15% |
| Parte 4: Técnicas de Balanceo | 20 | 20% |
| Parte 5: Threshold Optimization | 15 | 15% |
| Parte 6: Conclusiones | 15 | 15% |
| Presentación y Formato | 5 | 5% |
| **TOTAL OBLIGATORIO** | **100** | **100%** |
| Parte 7: Trabajo Destacado (opcional) | +10 | Bonus |

---

## 📋 PARTE 1: ANÁLISIS EXPLORATORIO (15 puntos)

### 1.1 Exploración Básica (5 puntos)

| Puntaje | Criterio |
|---------|----------|
| **5** | • Carga correcta del dataset<br>• Info completa (shape, dtypes, missing values)<br>• Estadísticas descriptivas de todas las variables numéricas<br>• Identifica claramente el desbalance (97%-3%)<br>• Visualización de la distribución de clases |
| **4** | • Todo lo anterior pero falta alguna estadística menor<br>• Visualización básica presente |
| **3** | • Exploración incompleta<br>• Falta análisis de valores faltantes o estadísticas |
| **2** | • Exploración muy básica<br>• Solo muestra head() y shape |
| **0-1** | • No realiza exploración o tiene errores graves |

### 1.2 Análisis de Variables (5 puntos)

| Puntaje | Criterio |
|---------|----------|
| **5** | • Histogramas/boxplots de variables numéricas principales<br>• Análisis de variables categóricas (value_counts)<br>• Identifica y analiza outliers<br>• Matriz de correlación o heatmap<br>• Interpretación de cada visualización |
| **4** | • La mayoría de lo anterior presente<br>• Visualizaciones básicas correctas<br>• Interpretación mínima |
| **3** | • Algunas visualizaciones presentes<br>• Falta análisis de outliers o correlaciones |
| **2** | • Visualizaciones muy básicas<br>• Sin interpretación |
| **0-1** | • Casi sin análisis de variables |

### 1.3 Análisis del Fraude (5 puntos)

| Puntaje | Criterio |
|---------|----------|
| **5** | • Comparación clara fraude vs no fraude en múltiples variables<br>• Identifica variables discriminativas<br>• Análisis temporal (hora, día) con visualizaciones<br>• Análisis geográfico (países)<br>• Insights relevantes extraídos |
| **4** | • Comparación fraude vs no fraude presente<br>• Análisis temporal básico<br>• Algunos insights identificados |
| **3** | • Comparación parcial fraude vs no fraude<br>• Análisis temporal o geográfico ausente |
| **2** | • Comparación muy superficial<br>• Pocos insights |
| **0-1** | • No realiza análisis específico del fraude |

**SUBTOTAL PARTE 1**: _____ / 15

---

## 📋 PARTE 2: PREPROCESAMIENTO (15 puntos)

### 2.1 Manejo de Valores Faltantes (4 puntos)

| Puntaje | Criterio |
|---------|----------|
| **4** | • Analiza el patrón de valores faltantes<br>• Estrategia justificada (imputación, eliminación)<br>• Implementación correcta<br>• Verifica que no quedan NaN |
| **3** | • Estrategia razonable implementada<br>• Justificación básica |
| **2** | • Imputación simple sin justificar<br>• Funciona pero no es óptimo |
| **0-1** | • No maneja NaN o lo hace incorrectamente |

### 2.2 Feature Engineering (6 puntos)

| Puntaje | Criterio |
|---------|----------|
| **6** | • Extrae TODAS las features temporales obligatorias:<br>  - hora, día_semana, es_fin_semana, es_noche, día_mes<br>• Crea features adicionales creativas (ratios, agregaciones)<br>• Justifica cada feature creada |
| **5** | • Extrae todas las features temporales obligatorias<br>• 1-2 features adicionales |
| **4** | • Extrae la mayoría de features temporales<br>• Sin features adicionales |
| **3** | • Extrae solo algunas features temporales (hora, día) |
| **0-2** | • Feature engineering mínimo o ausente |

### 2.3 Encoding de Categóricas (3 puntos)

| Puntaje | Criterio |
|---------|----------|
| **3** | • Encoding apropiado para cada variable (label, one-hot, frequency)<br>• Maneja correctamente variables de alta cardinalidad (i, j)<br>• Justifica la elección de técnica |
| **2** | • Encoding básico funcional (label encoding)<br>• Manejo razonable de alta cardinalidad |
| **1** | • Encoding básico con problemas menores |
| **0** | • No hace encoding o es incorrecto |

### 2.4 Split Train/Test (2 puntos)

| Puntaje | Criterio |
|---------|----------|
| **2** | • Split 80/20 o 70/30<br>• **Usa stratify=y**<br>• Verifica que el desbalance se mantiene<br>• Usa random_state fijo |
| **1** | • Split correcto pero sin verificar stratification |
| **0** | • Split sin stratify o incorrecto |

**SUBTOTAL PARTE 2**: _____ / 15

---

## 📋 PARTE 3: MODELO BASELINE (15 puntos)

### 3.1 Entrenamiento (5 puntos)

| Puntaje | Criterio |
|---------|----------|
| **5** | • Modelo de clasificación apropiado (RF, LR, XGB)<br>• Usa `class_weight='balanced'`<br>• Entrenamiento correcto en train set<br>• Código limpio y reproducible (random_state) |
| **4** | • Modelo funcional con class_weight<br>• Entrenamiento correcto |
| **3** | • Modelo funcional sin class_weight<br>• Entrenamiento básico |
| **0-2** | • Modelo con errores o no entrena |

### 3.2 Evaluación (7 puntos)

| Puntaje | Criterio |
|---------|----------|
| **7** | • Calcula TODAS las métricas requeridas:<br>  - Confusion Matrix (interpretada)<br>  - Recall, Precision, F1-Score<br>  - AUC-ROC, AUC-PR<br>• Visualiza curvas ROC y PR<br>• **NO usa Accuracy** como métrica principal |
| **5-6** | • Calcula la mayoría de métricas<br>• Visualizaciones presentes<br>• Interpretación básica |
| **3-4** | • Calcula métricas principales (Recall, Precision, F1)<br>• Falta AUC-PR o visualizaciones |
| **0-2** | • Métricas incompletas<br>• Usa Accuracy como principal |

### 3.3 Interpretación (3 puntos)

| Puntaje | Criterio |
|---------|----------|
| **3** | • Interpreta resultados en contexto de negocio<br>• Identifica el problema (recall bajo, muchos FN)<br>• Calcula costo total: `FP × $5 + FN × $200` |
| **2** | • Interpretación básica correcta<br>• Menciona el problema<br>• Calcula costo |
| **1** | • Interpretación superficial<br>• No calcula costo o está mal |
| **0** | • Sin interpretación |

**SUBTOTAL PARTE 3**: _____ / 15

---

## 📋 PARTE 4: TÉCNICAS DE BALANCEO (20 puntos)

### 4.1 SMOTE (8 puntos)

| Puntaje | Criterio |
|---------|----------|
| **8** | • Implementa SMOTE **DESPUÉS** del train/test split<br>• Aplica SOLO en train set<br>• Verifica el nuevo balance (aprox 50-50)<br>• Entrena modelo con datos balanceados<br>• Evalúa en test set (sin SMOTE)<br>• Compara con baseline |
| **6-7** | • SMOTE correctamente aplicado post-split<br>• Evaluación correcta<br>• Comparación presente |
| **4-5** | • SMOTE aplicado pero con errores menores<br>• Evaluación básica |
| **0-3** | • SMOTE aplicado antes del split (❌ data leakage)<br>• Evaluación incorrecta |

**⚠️ CRITERIO CRÍTICO**: Si aplica SMOTE antes del split, máximo 3 puntos.

### 4.2 Técnica Adicional (7 puntos)

| Puntaje | Criterio |
|---------|----------|
| **7** | • Implementa 2+ técnicas adicionales:<br>  - Undersampling<br>  - SMOTETomek / SMOTEENN<br>  - BalancedRandomForest<br>  - Class weight optimizado<br>• Correctamente aplicadas<br>• Evaluadas con las mismas métricas |
| **5-6** | • Implementa 1 técnica adicional correctamente<br>• Evaluación completa |
| **3-4** | • Implementa 1 técnica con problemas menores |
| **0-2** | • No implementa técnica adicional o es incorrecta |

### 4.3 Comparación (5 puntos)

| Puntaje | Criterio |
|---------|----------|
| **5** | • Tabla comparativa completa con:<br>  - Baseline, SMOTE, Técnica 2 (y 3 si aplica)<br>  - Recall, Precision, F1, AUC-PR, Costo Total<br>• Analiza trade-offs (precision vs recall)<br>• Justifica cuál técnica es mejor para el negocio<br>• Visualización comparativa (gráfico de barras) |
| **4** | • Tabla comparativa presente<br>• Análisis básico de trade-offs<br>• Justificación razonable |
| **3** | • Comparación parcial<br>• Análisis superficial |
| **0-2** | • Comparación ausente o muy incompleta |

**SUBTOTAL PARTE 4**: _____ / 20

---

## 📋 PARTE 5: THRESHOLD OPTIMIZATION (15 puntos)

### 5.1 Búsqueda de Threshold (7 puntos)

| Puntaje | Criterio |
|---------|----------|
| **7** | • Prueba rango amplio de thresholds (0.1 a 0.9, pasos 0.05)<br>• Para cada threshold calcula:<br>  - Confusion matrix<br>  - Precision, Recall, F1<br>  - Costo total<br>• Visualiza cómo varían las métricas (gráfico lineal)<br>• Código limpio y eficiente (loop o función) |
| **5-6** | • Prueba múltiples thresholds<br>• Calcula métricas principales<br>• Visualización presente |
| **3-4** | • Prueba algunos thresholds<br>• Métricas incompletas<br>• Visualización básica |
| **0-2** | • Búsqueda muy limitada o incorrecta |

### 5.2 Selección del Threshold (5 puntos)

| Puntaje | Criterio |
|---------|----------|
| **5** | • Identifica threshold óptimo según **DOS criterios**:<br>  1. Maximizar F1-Score<br>  2. Minimizar Costo Total<br>• Discute si son diferentes y por qué<br>• Justifica cuál elegir según objetivos de negocio<br>• Elección fundamentada |
| **4** | • Identifica threshold óptimo según 1 criterio<br>• Justificación razonable |
| **3** | • Identifica threshold pero justificación débil |
| **0-2** | • No justifica la elección o es incorrecta |

### 5.3 Evaluación Final (3 puntos)

| Puntaje | Criterio |
|---------|----------|
| **3** | • Re-evalúa modelo con threshold óptimo<br>• Compara con baseline (% de mejora en recall)<br>• Cuantifica mejora en costo total<br>• Interpreta confusion matrix final |
| **2** | • Re-evaluación presente<br>• Comparación básica con baseline |
| **1** | • Re-evaluación parcial |
| **0** | • No realiza evaluación final |

**SUBTOTAL PARTE 5**: _____ / 15

---

## 📋 PARTE 6: FEATURE IMPORTANCE Y CONCLUSIONES (15 puntos)

### 6.1 Feature Importance (7 puntos)

| Puntaje | Criterio |
|---------|----------|
| **7** | • Calcula feature importance del modelo final<br>• Visualiza top 10-15 features (gráfico de barras)<br>• Interpreta qué variables son más importantes<br>• Analiza si features temporales aportan valor<br>• Insights sobre qué caracteriza al fraude |
| **5-6** | • Feature importance calculada y visualizada<br>• Interpretación básica correcta |
| **3-4** | • Feature importance presente<br>• Poca interpretación |
| **0-2** | • Feature importance ausente o incorrecta |

### 6.2 Conclusiones Técnicas (4 puntos)

| Puntaje | Criterio |
|---------|----------|
| **4** | • Resume logros principales con números concretos:<br>  - Mejora en recall (X% → Y%, +Z%)<br>  - Fraudes adicionales detectados<br>  - Técnica de balanceo más efectiva<br>  - Impacto del threshold tuning<br>• Síntesis clara y cuantificada |
| **3** | • Resumen de logros presente<br>• Algunos números cuantificados |
| **2** | • Resumen básico<br>• Pocos números específicos |
| **0-1** | • Conclusiones vagas o ausentes |

### 6.3 Recomendaciones de Negocio (4 puntos)

| Puntaje | Criterio |
|---------|----------|
| **4** | • Traduce resultados a lenguaje de negocio<br>• Calcula impacto económico mensual/anual<br>• Propone implementación en producción<br>• Sugiere métricas de monitoreo<br>• Identifica limitaciones y riesgos<br>• Accionable para stakeholders no técnicos |
| **3** | • Recomendaciones presentes<br>• Impacto económico calculado<br>• Orientadas a negocio |
| **2** | • Recomendaciones básicas<br>• Algo orientadas a negocio |
| **0-1** | • Recomendaciones ausentes o solo técnicas |

**SUBTOTAL PARTE 6**: _____ / 15

---

## 📋 PRESENTACIÓN Y FORMATO (5 puntos)

### Organización del Notebook (2 puntos)

| Puntaje | Criterio |
|---------|----------|
| **2** | • Estructura clara con secciones bien definidas<br>• Tabla de contenidos<br>• Flujo lógico de análisis<br>• Fácil de seguir |
| **1** | • Organización básica funcional<br>• Algo difícil de seguir |
| **0** | • Desorganizado o confuso |

### Código (2 puntos)

| Puntaje | Criterio |
|---------|----------|
| **2** | • Código limpio y legible<br>• Comentarios apropiados<br>• Uso de funciones cuando corresponde<br>• Variables con nombres descriptivos<br>• Reproducible (random_state fijos) |
| **1** | • Código funcional pero mejorable<br>• Algunos comentarios |
| **0** | • Código difícil de leer o sin comentarios |

### Markdown y Explicaciones (1 punto)

| Puntaje | Criterio |
|---------|----------|
| **1** | • Explicaciones claras entre secciones de código<br>• Interpreta cada resultado<br>• Usa markdown apropiadamente (títulos, listas, etc.) |
| **0.5** | • Explicaciones mínimas presentes |
| **0** | • Sin explicaciones en markdown |

**SUBTOTAL PRESENTACIÓN**: _____ / 5

---

## 📋 INFORME EJECUTIVO PDF (Incluido en puntaje general)

### Checklist de Contenido

- [ ] **Resumen Ejecutivo** (1 párrafo - problema, solución, resultado)
- [ ] **Problema de Negocio** (media página - contexto y costos)
- [ ] **Solución Propuesta** (1 página - enfoque técnico en lenguaje simple)
- [ ] **Resultados** (1-1.5 páginas - métricas, mejoras, impacto económico)
- [ ] **Recomendaciones** (media página - implementación, próximos pasos, limitaciones)
- [ ] **Anexo** (tabla comparativa de modelos, gráficos clave)

### Evaluación del Informe

| Aspecto | Peso en Parte 6 |
|---------|----------------|
| Claridad para no técnicos | 30% |
| Impacto económico cuantificado | 30% |
| Recomendaciones accionables | 25% |
| Visualizaciones efectivas | 15% |

---

## 🎁 PARTE 7: TRABAJO DESTACADO (Hasta +10 puntos extra)

### Opciones (2+ para puntos extra)

| Elemento | Puntaje | Criterio |
|----------|---------|----------|
| **Cross-Validation** | +5 | • K-Fold estratificado implementado<br>• Reporta mean y std de métricas<br>• Analiza variabilidad |
| **Hyperparameter Tuning** | +5 | • GridSearchCV o RandomizedSearchCV<br>• Espacio de búsqueda razonable<br>• Mejora demostrada |
| **Ensemble** | +5 | • 3+ algoritmos entrenados<br>• Voting/stacking implementado<br>• Mejora sobre individuales |
| **Análisis de Costos** | +5 | • Múltiples escenarios simulados<br>• Visualización de trade-offs<br>• Análisis de sensibilidad |
| **Feature Engineering Avanzado** | +5 | • Features de agregación<br>• Feature selection automático<br>• Mejora demostrada |
| **Deep Learning** | +8 | • Red neuronal implementada<br>• Arquitectura justificada<br>• Comparación con ML tradicional |
| **Dashboard Interactivo** | +8 | • Streamlit/Dash funcional<br>• Threshold ajustable dinámicamente<br>• Visualización de impacto |

**SUBTOTAL PARTE 7 (opcional)**: _____ / 10 (extra)

---

## 📊 RESUMEN DE EVALUACIÓN

### Cálculo de Puntaje Final

| Sección | Puntaje Obtenido | Puntaje Máximo |
|---------|------------------|----------------|
| Parte 1: EDA | _____ | 15 |
| Parte 2: Preprocesamiento | _____ | 15 |
| Parte 3: Baseline | _____ | 15 |
| Parte 4: Balanceo | _____ | 20 |
| Parte 5: Threshold | _____ | 15 |
| Parte 6: Conclusiones | _____ | 15 |
| Presentación | _____ | 5 |
| **SUBTOTAL** | **_____** | **100** |
| Parte 7: Extra (opcional) | _____ | +10 |
| **TOTAL** | **_____** | **110** |

### Escala de Calificación

| Puntaje | Nota | Calificación |
|---------|------|--------------|
| 90-110 | 10-9 | Excelente ⭐⭐⭐⭐⭐ |
| 80-89 | 8-9 | Muy Bueno ⭐⭐⭐⭐ |
| 70-79 | 7-8 | Bueno ⭐⭐⭐ |
| 60-69 | 6-7 | Suficiente ⭐⭐ |
| < 60 | < 6 | Insuficiente ⭐ |

---

## 📝 OBSERVACIONES DEL DOCENTE

### Fortalezas del Trabajo

```
[Escribe aquí las fortalezas principales identificadas]
```

### Áreas de Mejora

```
[Escribe aquí los aspectos que el estudiante debe mejorar]
```

### Comentarios Adicionales

```
[Comentarios generales sobre el trabajo]
```

---

## 🚨 CRITERIOS DE PENALIZACIÓN

| Problema | Penalización |
|----------|--------------|
| **Data leakage** (SMOTE antes del split) | -10 puntos |
| Usa Accuracy como métrica principal | -5 puntos |
| No estratifica train/test split | -3 puntos |
| Evalúa en train en lugar de test | -5 puntos |
| Código no reproducible (sin random_state) | -2 puntos |
| Notebook no ejecuta completo | -10 puntos |
| Entrega fuera de plazo (por día) | -5 puntos |
| Formato de entrega incorrecto | -3 puntos |
| Sin informe ejecutivo PDF | -10 puntos |

---

## 🎯 CRITERIOS DE EXCELENCIA

Para obtener 90+ puntos, el trabajo debe:

- ✅ **Análisis profundo**: No solo muestra resultados, interpreta y explica
- ✅ **Múltiples técnicas**: Compara 3+ enfoques de balanceo
- ✅ **Visualizaciones impactantes**: Gráficos claros, informativos y profesionales
- ✅ **Enfoque de negocio**: Constantemente traduce resultados a impacto económico
- ✅ **Código limpio**: Bien estructurado, documentado y reproducible
- ✅ **Conclusiones accionables**: Recomendaciones claras para implementación
- ✅ **Trabajo extra**: Al menos 2 elementos de la Parte 7

---

## 📋 CHECKLIST DE REVISIÓN RÁPIDA

### Antes de evaluar
- [ ] Descargué y descomprimí el archivo correctamente
- [ ] Verifiqué que el notebook ejecuta sin errores
- [ ] Tengo la rúbrica impresa o en pantalla

### Durante la evaluación
- [ ] Evalúo cada sección según los criterios
- [ ] Registro comentarios específicos por sección
- [ ] Verifico criterios de penalización
- [ ] Reviso el informe ejecutivo PDF

### Después de evaluar
- [ ] Sumo todos los puntajes parciales
- [ ] Aplico penalizaciones si corresponde
- [ ] Escribo feedback constructivo
- [ ] Registro la nota final

---

**Tiempo estimado de evaluación por trabajo**: 45-60 minutos

---

*Rúbrica de evaluación v1.0 - Taller de Resolución de Problemas II*
