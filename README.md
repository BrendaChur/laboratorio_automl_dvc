# Laboratorio AutoML con DVC 

**Autora:** Brenda Guiselle Chur Chinchilla  
**Curso:** Product Development  
**Objetivo:** Diseñar un pipeline automatizado inspirado en AutoML, utilizando **DVC (Data Version Control)** y **Git** para gestionar datasets, modelos y experimentos de forma reproducible.

## Descripción general

Este laboratorio implementa un **pipeline reproducible** de aprendizaje automático controlado con DVC y Git, capaz de:

- Preprocesar datos automáticamente según su tipo.  
- Entrenar y comparar distintos modelos (Regresión Lineal, Random Forest, Gradient Boosting).  
- Evaluar métricas (R² y RMSE) y registrar resultados con versionado de datos y configuraciones.  
- Analizar cómo los cambios en el dataset afectan el rendimiento del modelo.
  
## Versiones del Dataset y resultados
- **V1:** Dataset original sin modificaciones. Se estableció el pipeline base (preprocess -> train -> evaluate). R2=0.6468 RMSE=0.68032
- **V2:** Limpieza, eliminación de filas duplicadas y valores nulos. R2=0.6468 RMSE=0.68032
- **V3:** Ampliación, se agregaron nuevas observaciones (~20% más registros). R2=0.65647 RMSE=0.67852
- **V4:** Transformación. creación de nuevas variables (RoomsPerPerson, BedroomsPerRoom) y eliminación de las columnas originales. R2=0.65092 RMSE=0.68399
- **V5:** Tratamiento de outliers, eliminación de valores extremos en MedHouseVal. R2=0.64799 RMSE=0.67767

## Interpretación de resultados
- v1 -> v2: No hubo cambios en el rendimiento, indicando que el dataset original estaba limpio.
- v2 -> v3: Se observó una mejora leve al aumentar el tamaño del conjunto de entrenamiento.
- v3 -> v4: Las transformaciones de variables redujeron ligeramente el desempeño del modelo.
- v4 -> v5: El tratamiento de outliers redujo el error promedio (RMSE), mejorando la estabilidad general.
- v1 -> v5: En conjunto, se evidencia una pequeña mejora global en el modelo final.