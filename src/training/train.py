import pandas as pd
import json
import os
import joblib
import mlflow
import mlflow.sklearn

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score


TARGET_COL = "Loan_Status"


def load_data(path="data/processed/features.csv"):
    df = pd.read_csv(path)
    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    return train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)


def train_and_log(X_train, X_test, y_train, y_test):
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("loan-approval-prediction")

    models = {
        "logistic_regression": LogisticRegression(max_iter=1000),
        "random_forest": RandomForestClassifier(n_estimators=200, random_state=42),
    }

    best_model = None
    best_model_name = None
    best_score = -1

    for name, model in models.items():
        with mlflow.start_run(run_name=name):
            model.fit(X_train, y_train)
            preds = model.predict(X_test)

            acc = accuracy_score(y_test, preds)
            f1 = f1_score(y_test, preds)
            prec = precision_score(y_test, preds)
            rec = recall_score(y_test, preds)

            mlflow.log_param("model_type", name)
            if name == "random_forest":
                mlflow.log_param("n_estimators", 200)

            mlflow.log_metric("accuracy", acc)
            mlflow.log_metric("f1_score", f1)
            mlflow.log_metric("precision", prec)
            mlflow.log_metric("recall", rec)

            mlflow.sklearn.log_model(model, "model", serialization_format="pickle")

            print(f"[{name}] accuracy={acc:.4f} f1={f1:.4f} "
                  f"precision={prec:.4f} recall={rec:.4f}")

            if acc > best_score:
                best_score = acc
                best_model = model
                best_model_name = name

    return best_model, best_model_name, best_score


if __name__ == "__main__":
    os.makedirs("models", exist_ok=True)

    X_train, X_test, y_train, y_test = load_data()
    print(f"Train size: {len(X_train)}, Test size: {len(X_test)}")

    best_model, best_model_name, best_score = train_and_log(
        X_train, X_test, y_train, y_test
    )

    joblib.dump(best_model, "models/model.pkl")

    with open("metrics.json", "w") as f:
        json.dump({"best_model": best_model_name, "accuracy": best_score}, f, indent=2)

    print(f"\nBest model: {best_model_name} (accuracy={best_score:.4f})")
    print("Saved best model to models/model.pkl")
    print("Saved metrics to metrics.json")
