"""Train, evaluate, compare and track two classification pipelines."""
import json
import warnings
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from joblib import dump
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "heart_disease.csv"
ARTIFACTS = ROOT / "artifacts"
TARGET = "target"
CATEGORICAL = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]


def make_pipeline(X: pd.DataFrame, estimator) -> Pipeline:
    cat_cols = [c for c in CATEGORICAL if c in X.columns]
    num_cols = [c for c in X.columns if c not in cat_cols]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    prep = ColumnTransformer([
        ("numeric", numeric_pipe, num_cols),
        ("categorical", categorical_pipe, cat_cols),
    ])
    return Pipeline([("preprocess", prep), ("model", estimator)])


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError("Run `python -m src.data` first.")
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(DATA_PATH)
    df = df.dropna(subset=[TARGET])
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    mlflow.set_tracking_uri((ROOT / "mlruns").as_uri())
    mlflow.set_experiment("heart-disease-classification")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = {
        "logistic_regression": LogisticRegression(max_iter=2000, class_weight="balanced"),
        "random_forest": RandomForestClassifier(
            n_estimators=300, random_state=42, class_weight="balanced"
        ),
    }
    results = []
    best_name, best_model, best_auc = None, None, -1.0
    for name, estimator in models.items():
        pipeline = make_pipeline(X_train, estimator)
        scores = cross_validate(
            pipeline, X_train, y_train, cv=cv,
            scoring={"accuracy": "accuracy", "precision": "precision",
                     "recall": "recall", "roc_auc": "roc_auc"},
            error_score="raise"
        )
        pipeline.fit(X_train, y_train)
        pred = pipeline.predict(X_test)
        prob = pipeline.predict_proba(X_test)[:, 1]
        metrics = {
            "cv_accuracy_mean": float(np.mean(scores["test_accuracy"])),
            "cv_precision_mean": float(np.mean(scores["test_precision"])),
            "cv_recall_mean": float(np.mean(scores["test_recall"])),
            "cv_roc_auc_mean": float(np.mean(scores["test_roc_auc"])),
            "test_accuracy": float(accuracy_score(y_test, pred)),
            "test_precision": float(precision_score(y_test, pred, zero_division=0)),
            "test_recall": float(recall_score(y_test, pred, zero_division=0)),
            "test_roc_auc": float(roc_auc_score(y_test, prob)),
        }
        with mlflow.start_run(run_name=name):
            mlflow.log_param("model", name)
            mlflow.log_param("test_size", 0.2)
            mlflow.log_param("random_state", 42)
            mlflow.log_param("cv_folds", 5)
            mlflow.log_metrics(metrics)
            mlflow.sklearn.log_model(
    pipeline,
    name="model",
    skops_trusted_types=[
        "numpy.dtype",
        "sklearn.tree._tree.Tree",
    ],
)
            report = classification_report(y_test, pred, output_dict=True, zero_division=0)
            report_path = ARTIFACTS / f"{name}_classification_report.json"
            report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
            mlflow.log_artifact(str(report_path))
        results.append({"model": name, **metrics})
        if metrics["cv_roc_auc_mean"] > best_auc:
            best_name, best_model, best_auc = name, pipeline, metrics["cv_roc_auc_mean"]

    dump(best_model, ARTIFACTS / "model.joblib")
    summary = {"selected_model": best_name, "selection_metric": "mean 5-fold CV ROC-AUC", "results": results}
    (ARTIFACTS / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    print("Saved final model to artifacts/model.joblib")
    print("Important: interpret metrics and explain the selection in your report.")


if __name__ == "__main__":
    with warnings.catch_warnings():
        warnings.simplefilter("default")
        main()
