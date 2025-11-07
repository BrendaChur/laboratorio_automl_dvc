# Reporte de Resultados – Laboratorio AutoML con DVC

**Autora:** Brenda Guiselle Chur Chinchilla  
**Curso:** Product Development  

---

## 1. Objetivo

Diseñar un pipeline automatizado inspirado en AutoML, utilizando **DVC (Data Version Control)** y **Git** para gestionar datasets, modelos y experimentos de forma reproducible.

---

## 2. Descripción general

Este laboratorio implementa un pipeline reproducible de aprendizaje automático controlado con DVC y Git, capaz de:

- Preprocesar datos automáticamente según su tipo.  
- Entrenar y comparar distintos modelos (Regresión Lineal y Random Forest).  
- Evaluar métricas (R² y RMSE) y registrar resultados con versionado de datos y configuraciones.  
- Analizar cómo los cambios en el dataset afectan el rendimiento del modelo.

---

## 3. Versiones del dataset y resultados

| Versión | Descripción                                                                                   | R²      | RMSE     |
|:-------:|:----------------------------------------------------------------------------------------------|--------:|---------:|
| **V1** | Dataset original sin modificaciones. Se estableció el pipeline base (preprocess → train → evaluate). | 0.6468  | 0.68032  |
| **V2** | Limpieza, eliminación de filas duplicadas y valores nulos.                                    | 0.6468  | 0.68032  |
| **V3** | Ampliación, se agregaron nuevas observaciones (~20% más registros).                           | 0.65647 | 0.67852  |
| **V4** | Transformación: creación de nuevas variables (*RoomsPerPerson*, *BedroomsPerRoom*) y eliminación de las columnas originales. | 0.65092 | 0.68399  |
| **V5** | Tratamiento de outliers, eliminación de valores extremos en *MedHouseVal*.                    | 0.64799 | 0.67767  |

> Las métricas se registraron automáticamente en `metrics.json` y fueron comparadas con los comandos:
> ```bash
> dvc metrics show
> dvc metrics diff v1_data v5_data
> ```

---

## 4. Modelo ganador y parámetros finales

A partir de la versión final del dataset (V5), el modelo con mejor rendimiento fue **Random Forest**, determinado mediante el RMSE en el conjunto de entrenamiento.

**Modelo ganador:** Random Forest  
**Métrica principal:** RMSE  
**Métricas en test (V5):**
- RMSE = 0.67767  
- R² = 0.64799  

**Hiperparámetros finales del modelo:**
- `n_estimators = 100`  
- `max_depth = 5`  
- `random_state = 42`

Los archivos generados en el pipeline fueron:
- `models/best_model.joblib` → modelo entrenado y seleccionado automáticamente.  
- `models/best_model_info.json` → registro de resultados de todos los modelos.  
- `metrics.json` → métricas finales del modelo ganador.

---

## 5. Comparación y análisis de resultados

Los resultados muestran cómo los cambios progresivos en el dataset afectaron el rendimiento del modelo:

- **V1 → V2 (Limpieza de duplicados y nulos):**  
  No hubo cambios en las métricas, lo que indica que el dataset original ya estaba limpio y sin valores redundantes.

- **V2 → V3 (Ampliación de datos):**  
  Se observó una mejora leve en el rendimiento (R² aumentó y RMSE disminuyó).  
  El incremento del tamaño del dataset (~20%) permitió al modelo generalizar mejor.

- **V3 → V4 (Nuevas variables derivadas):**  
  Las transformaciones no mejoraron el desempeño; el RMSE aumentó ligeramente.  
  Esto sugiere que las variables creadas no aportaron información predictiva adicional o eliminaron relaciones relevantes.

- **V4 → V5 (Tratamiento de outliers):**  
  El rendimiento mejoró nuevamente, con una reducción del RMSE y un R² estable.  
  La eliminación de valores extremos ayudó a estabilizar el modelo y reducir la influencia de observaciones atípicas.

- **V1 → V5 (Comparación global):**  
  El modelo final presentó una pequeña mejora global respecto a la versión inicial, destacando el impacto positivo de la ampliación del dataset y la corrección de outliers.

---

## 6. Conclusiones

1. El uso de **DVC** permitió controlar la evolución del pipeline, versionar datos, métricas y modelos de forma reproducible.  
2. Las comparaciones con `dvc metrics diff` evidenciaron cómo la calidad y cantidad de datos influyen directamente en el rendimiento del modelo.  
3. Las transformaciones de variables deben validarse empíricamente, ya que no todas contribuyen a una mejora en el desempeño.  
4. El **Random Forest** se consolidó como el mejor modelo para este conjunto de datos, combinando bajo error (RMSE) con buena capacidad de generalización.  
5. En conjunto, las mejoras aplicadas al dataset lograron estabilizar y optimizar el rendimiento del pipeline automatizado de aprendizaje automático.

---

📄 **Comandos clave del análisis:**
```bash
dvc repro
dvc metrics show
dvc metrics diff v1_data v5_data


