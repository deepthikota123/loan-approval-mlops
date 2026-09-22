import os
import numpy as np
import pandas as pd
import mlflow
from src.training.train import load_data, train_and_log, TARGET_COL


def _make_synthetic_features(n=60):
    rng = np.random.default_rng(42)
    df = pd.DataFrame({
        "Gender": rng.integers(0, 2, n),
        "Married": rng.integers(0, 2, n),
        "Dependents": rng.integers(0, 4, n),
        "Education": rng.integers(0, 2, n),
        "Self_Employed": rng.integers(0, 2, n),
        "Credit_History": rng.choice([0.0, 1.0], n, p=[0.2, 0.8]),
        "Property_Area": rng.integers(0, 3, n),
        "LoanAmount_log": rng.normal(4.8, 0.3, n),
        "TotalIncome_log": rng.normal(8.5, 0.3, n),
    })
    df[TARGET_COL] = (df["Credit_History"] == 1.0).astype(int)
    return df


def test_load_data_splits_correctly(tmp_path):
    df = _make_synthetic_features(60)
    path = tmp_path / "features.csv"
    df.to_csv(path, index=False)

    X_train, X_test, y_train, y_test = load_data(path=str(path))

    assert len(X_train) + len(X_test) == 60
    assert TARGET_COL not in X_train.columns
    assert TARGET_COL not in X_test.columns
    assert set(y_train.unique()).issubset({0, 1})


def test_train_and_log_returns_best_model(tmp_path, monkeypatch):
    mlflow_dir = tmp_path / "mlruns"
    monkeypatch.chdir(tmp_path)
    mlflow.set_tracking_uri(f"sqlite:///{tmp_path / 'mlflow_test.db'}") 
    df = _make_synthetic_features(60)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]

    from sklearn.model_selection import train_test_split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    best_model, best_model_name, best_score = train_and_log(
        X_train, X_test, y_train, y_test
    )

    assert best_model_name in ("logistic_regression", "random_forest")
    assert 0.0 <= best_score <= 1.0
    assert hasattr(best_model, "predict")

    preds = best_model.predict(X_test)
    assert len(preds) == len(X_test)
