import os
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import yaml


def load_params(path: str = "params.yaml") -> dict:
    """Carga parámetros del archivo params.yaml."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    """
    Preprocesa el dataset:
    - Lee el CSV definido en params.yaml
    - Separa X (features) e y (target = MedHouseVal)
    - Detecta columnas numéricas y categóricas
    - Aplica escalado a numéricas y one-hot a categóricas
    - Genera X_train, X_test, y_train, y_test en data/processed/
    """
    params = load_params()
    data_path = params["data"]["path"]
    target_col = params["data"]["target"]
    test_size = params["data"]["test_size"]
    random_state = params["data"]["random_state"]

    # Cargar datos
    df = pd.read_csv(data_path)

    # Separar variables explicativas y objetivo
    X = df.drop(columns=[target_col])
    y = df[target_col]

    # En tu dataset, todas son numéricas, pero dejamos la lógica genérica
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), num_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols),
        ]
    )

    # Split train / test
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    # Ajustar transformador en train y aplicar en test
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Si el resultado es matriz dispersa, convertir a densa
    if hasattr(X_train_proc, "toarray"):
        X_train_proc = X_train_proc.toarray()
        X_test_proc = X_test_proc.toarray()

    X_train_df = pd.DataFrame(X_train_proc)
    X_test_df = pd.DataFrame(X_test_proc)

    os.makedirs("data/processed", exist_ok=True)
    X_train_df.to_csv("data/processed/X_train.csv", index=False)
    X_test_df.to_csv("data/processed/X_test.csv", index=False)
    y_train.to_csv("data/processed/y_train.csv", index=False)
    y_test.to_csv("data/processed/y_test.csv", index=False)


if __name__ == "__main__":
    main()

