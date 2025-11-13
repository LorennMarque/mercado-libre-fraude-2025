# Listado de Columnas del Dataset Procesado

**Total de columnas: 58**

## Columnas Originales (24)

1. **Unnamed: 0** - int64 - Índice original (0 nulos)
2. **a** - int64 - Variable numérica (0 nulos)
3. **b** - float64 - Variable numérica (21,474 nulos - 8.59%)
4. **c** - float64 - Variable numérica (21,474 nulos - 8.59%)
5. **d** - float64 - Variable numérica (594 nulos - 0.24%)
6. **e** - float64 - Variable numérica (0 nulos)
7. **f** - float64 - Variable numérica (15 nulos - 0.01%)
8. **pais** (originalmente 'g') - object - País (0 nulos)
9. **h** - int64 - Variable numérica (0 nulos)
10. **producto_nombre** (originalmente 'i') - object - Nombre del producto (0 nulos)
11. **categoria_id** (originalmente 'j') - object - ID de categoría (0 nulos)
12. **k** - float64 - Variable numérica (0 nulos)
13. **l** - float64 - Variable numérica (15 nulos - 0.01%)
14. **m** - float64 - Variable numérica (594 nulos - 0.24%)
15. **n** - int64 - Variable numérica (0 nulos)
16. **o** - Int64 - Variable numérica binaria (Y=1, N=0) (183,628 nulos - 73.45%)
17. **p** - Int64 - Variable numérica binaria (Y=1, N=0) (0 nulos)
18. **q** - float64 - Variable numérica (594 nulos - 0.24%)
19. **r** - int64 - Variable numérica (0 nulos)
20. **s** - int64 - Variable numérica (0 nulos)
21. **fecha** - datetime64[ns] - Fecha de la transacción (0 nulos)
22. **monto** - float64 - Monto de la transacción (0 nulos)
23. **score** - float64 - Score (0 nulos)
24. **fraude** - int64 - Variable objetivo (0=No fraude, 1=Fraude) (0 nulos)

## Columnas Imputadas (7 columnas)

Estas columnas contienen las versiones imputadas de las columnas originales que tenían valores faltantes. Se crearon usando IterativeImputer (MICE) y tienen el sufijo `_imputado`. **Todas estas columnas tienen 0 nulos**.

25. **b_imputado** - float64 - Versión imputada de 'b' (0 nulos)
26. **c_imputado** - float64 - Versión imputada de 'c' (0 nulos)
27. **d_imputado** - float64 - Versión imputada de 'd' (0 nulos)
28. **f_imputado** - float64 - Versión imputada de 'f' (0 nulos)
29. **q_imputado** - float64 - Versión imputada de 'q' (0 nulos)
30. **l_imputado** - float64 - Versión imputada de 'l' (0 nulos)
31. **m_imputado** - float64 - Versión imputada de 'm' (0 nulos)

## Columnas Creadas por Feature Engineering (25 columnas)

### Dummies de la columna 'o' (3 columnas)
32. **o_is_N** - int64 - Dummy: o == 'N' (0 nulos)
33. **o_is_Y** - int64 - Dummy: o == 'Y' (0 nulos)
34. **o_is_NA** - int64 - Dummy: o es nulo (0 nulos)

### Encoding de categoria_id (2 columnas)
35. **categoria_id_target_enc** - float64 - Target encoding de categoria_id (0 nulos)
36. **categoria_id_freq_enc** - int64 - Frequency encoding de categoria_id (0 nulos)

### Encoding de pais (2 columnas)
37. **pais_target_enc** - float64 - Target encoding de pais (324 nulos - 0.13%)
38. **pais_freq_enc** - float64 - Frequency encoding de pais (324 nulos - 0.13%)

### Indicadores binarios de país (2 columnas)
39. **is_brasil** - int64 - Indicador binario: 1 si pais == 'BR', 0 en caso contrario (0 nulos)
40. **is_arg** - int64 - Indicador binario: 1 si pais == 'AR', 0 en caso contrario (0 nulos)

### Features de producto_nombre (5 columnas)
41. **producto_num_chars** - int64 - Número de caracteres en el nombre del producto (0 nulos)
42. **producto_num_words** - int64 - Número de palabras en el nombre del producto (0 nulos)
43. **producto_num_special_chars** - int64 - Número de caracteres especiales (0 nulos)
44. **producto_avg_word_len** - float64 - Promedio de longitud de palabras (0 nulos)
45. **producto_freq** - int64 - Frecuencia del nombre del producto (0 nulos)

### Features temporales de fecha (13 columnas)
46. **hora** - int32 - Hora del día (0-23) (0 nulos)
47. **dia_semana** - int32 - Día de la semana (0=Lunes, 6=Domingo) (0 nulos)
48. **dia_mes** - int32 - Día del mes (1-31) (0 nulos)
49. **mes** - int32 - Mes (1-12) (0 nulos)
50. **es_fin_de_semana** - int64 - 1 si es fin de semana, 0 si no (0 nulos)
51. **es_nocturno** - int64 - 1 si es horario nocturno (22:00-06:00), 0 si no (0 nulos)
52. **es_horario_laboral** - int64 - 1 si es horario laboral (09:00-18:00), 0 si no (0 nulos)
53. **hora_sin** - float64 - Codificación cíclica seno de la hora (0 nulos)
54. **hora_cos** - float64 - Codificación cíclica coseno de la hora (0 nulos)
55. **dia_semana_sin** - float64 - Codificación cíclica seno del día de la semana (0 nulos)
56. **dia_semana_cos** - float64 - Codificación cíclica coseno del día de la semana (0 nulos)
57. **dia_mes_sin** - float64 - Codificación cíclica seno del día del mes (0 nulos)
58. **dia_mes_cos** - float64 - Codificación cíclica coseno del día del mes (0 nulos)

## Resumen

- **Total de columnas**: 58
- **Columnas originales**: 24 (algunas con nulos)
- **Columnas imputadas**: 7 (todas sin nulos)
- **Columnas creadas por feature engineering**: 27

## Notas Importantes

1. **Todas las columnas numéricas** (excepto 'fraude', 'row_id' y las variables cíclicas de seno/coseno) han sido **normalizadas entre 0 y 1** en el dataset final. Las variables cíclicas (hora_sin, hora_cos, dia_semana_sin, dia_semana_cos, dia_mes_sin, dia_mes_cos) mantienen sus valores originales entre -1 y 1.

2. **Columnas con nulos en originales**:
   - `b`: 21,474 nulos (8.59%)
   - `c`: 21,474 nulos (8.59%)
   - `d`: 594 nulos (0.24%)
   - `f`: 15 nulos (0.01%)
   - `l`: 15 nulos (0.01%)
   - `m`: 594 nulos (0.24%)
   - `q`: 594 nulos (0.24%)
   - `pais`: 324 nulos (0.13%)
   - `o`: 183,628 nulos (73.45%)

3. **Columnas imputadas**: Las columnas con sufijo `_imputado` contienen valores imputados usando IterativeImputer (MICE) y **no tienen nulos**. Puedes elegir usar las columnas originales o las imputadas según tu necesidad.

4. **Mantenimiento de originales**: Todas las columnas originales se mantienen en el dataset, incluyendo `r`, `categoria_id`, `pais`, `producto_nombre` y `o`, además de sus versiones procesadas.

5. **Feature Engineering**: Se crearon features adicionales mediante:
   - Target Encoding y Frequency Encoding para variables categóricas
   - Indicadores binarios de país (is_brasil, is_arg)
   - Extracción de características de texto del nombre del producto
   - Extracción de características temporales de la fecha
   - Codificación cíclica para variables temporales
