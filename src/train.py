import os
import json
import yaml
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error
import joblib


def load_params(path: str = "params.yaml") -> dict:
    """Carga parámetros del archivo params.yaml."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_model(model_type: str, params: dict):
    """
    Construye un modelo de sklearn según el tipo indicado.

    Args:
        model_type: tipo de modelo (linear_regression, random_forest, gradient_boosting).
        params: hiperparámetros del modelo.

    Returns:
        Modelo de sklearn instanciado.
    """
    if model_type == "linear_regression":
        return LinearRegression(**params)
    if model_type == "random_forest":
        return RandomForestRegressor(**params)
    if model_type == "gradient_boosting":
        return GradientBoostingRegressor(**params)
    raise ValueError(f"Modelo no soportado: {model_type}")


def main() -> None:
    """
    Entrena todos los modelos definidos en params.yaml,
    calcula el RMSE en train y guarda:
    - models/best_model.joblib: modelo ganador
    - models/best_model_info.json: info de todos los modelos y el mejor
    """
    params = load_params()
    primary_metric = params["metrics"]["primary"]

    # Cargar datos procesados
    X_train = pd.read_csv("data/processed/X_train.csv")
    y_train = pd.read_csv("data/processed/y_train.csv").iloc[:, 0]

    resultados = []
    mejor_score = np.inf
    mejor_nombre = None
    mejor_modelo = None

    for name, conf in params["models"].items():
        model_type = conf["type"]
        model_params = conf.get("params", {})

        model = build_model(model_type, model_params)
        model.fit(X_train, y_train)

        # Métrica sobre train (podrías cambiar a validación cruzada si quisieras)
        preds = model.predict(X_train)
        rmse = mean_squared_error(y_train, preds) ** 0.5


        resultados.append(
            {
                "model_name": name,
                "model_type": model_type,
                "rmse_train": rmse,
                "params": model_params,
            }
        )

        if rmse < mejor_score:
            mejor_score = rmse
            mejor_nombre = name
            mejor_modelo = model

    os.makedirs("models", exist_ok=True)
    joblib.dump(mejor_modelo, "models/best_model.joblib")

    info = {
        "primary_metric": primary_metric,
        "best_model_name": mejor_nombre,
        "best_rmse_train": float(mejor_score),
        "all_models": resultados,
    }

    with open("models/best_model_info.json", "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2)


if __name__ == "__main__":
    main()

