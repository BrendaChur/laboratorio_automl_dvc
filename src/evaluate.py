import json
import yaml
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
import joblib


def load_params(path: str = "params.yaml") -> dict:
    """Carga parámetros del archivo params.yaml."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """
    Evalúa el mejor modelo (guardado en models/best_model.joblib)
    sobre el conjunto de prueba y escribe las métricas en metrics.json.
    """
    params = load_params()
    metrics_cfg = params["metrics"]
    primary = metrics_cfg["primary"]

    # Cargar datos de prueba
    X_test = pd.read_csv("data/processed/X_test.csv")
    y_test = pd.read_csv("data/processed/y_test.csv").iloc[:, 0]

    # Cargar modelo ganador
    model = joblib.load("models/best_model.joblib")

    # Predicciones y métricas
    preds = model.predict(X_test)
    rmse = mean_squared_error(y_test, preds) ** 0.5
    r2 = r2_score(y_test, preds)

    # Info adicional del mejor modelo
    with open("models/best_model_info.json", "r", encoding="utf-8") as f:
        best_info = json.load(f)

    metrics = {
        "primary_metric": primary,
        "rmse": float(rmse),
        "r2": float(r2),
        "best_model_name": best_info["best_model_name"],
    }

    with open("metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()

