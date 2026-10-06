"""Train a customer churn baseline using the prepared training data."""

import json
from pathlib import Path
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "cleaned"
PREDICTION_THRESHOLD = 0.75


def predict_churn(model, inputs, threshold=PREDICTION_THRESHOLD):
    """Apply our custom threshold instead of the estimator's default predict rule."""
    if not 0 < threshold < 1:
        raise ValueError("The prediction threshold must be between 0 and 1.")
    churn_column = np.flatnonzero(model.classes_ == 1)
    if len(churn_column) != 1:
        raise ValueError("The model must contain the churn class 1.")
    probabilities = model.predict_proba(inputs)[:, churn_column[0]]
    return (probabilities >= threshold).astype(int)


def main():
    X_train = pd.read_csv(DATA_DIR / "X_train_encoded.csv")
    target_data = pd.read_csv(DATA_DIR / "y_train.csv")
    with (DATA_DIR / "preprocessing.json").open(encoding="utf-8") as file:
        preprocessing = json.load(file)

    if X_train.columns.tolist() != preprocessing["feature_columns"]:
        raise ValueError("Training feature columns do not match the preprocessing metadata.")
    excluded = {"Customer ID", "Customer Status", "Churn Category", "Churn Reason"}
    if excluded.intersection(X_train.columns):
        raise ValueError("Identifiers or outcome columns are present in the inputs.")
    if target_data.columns.tolist() != ["Customer Status"]:
        raise ValueError("Expected Customer Status as the only target column.")
    if X_train.empty or len(X_train) != len(target_data):
        raise ValueError("Training inputs and outcomes must have matching nonzero row counts.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X_train.dtypes):
        raise ValueError("All model inputs must be numeric.")
    if not np.isfinite(X_train.to_numpy()).all():
        raise ValueError("Training inputs contain missing or infinite values.")
    y_train = target_data["Customer Status"]
    if y_train.isna().any() or set(y_train.unique()) != {0, 1}:
        raise ValueError("Training outcomes must contain both 0 (Stayed) and 1 (Churned).")

    print(f"Training customers: {len(X_train)}")
    print(f"Input features: {X_train.shape[1]}")
    print(f"Stayed: {y_train.eq(0).sum()}, Churned: {y_train.eq(1).sum()}")

    # Scaling and encoding are already fitted; train only on the prepared inputs.
    model = LogisticRegression(solver="lbfgs", max_iter=2000, random_state=42)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        model.fit(X_train, y_train)

    model_dir = PROJECT_DIR / "models"
    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "customer_churn_model.joblib"
    joblib.dump(model, model_path)

    summary = {
        "model": "LogisticRegression",
        "training_rows": len(X_train),
        "feature_columns": X_train.columns.tolist(),
        "target_mapping": preprocessing["target_mapping"],
        "prediction_threshold": PREDICTION_THRESHOLD,
        "prediction_rule": "Use predict_churn with predict_proba; estimator.predict still uses its default threshold.",
        "iterations": int(model.n_iter_.max()),
        "test_data_used": False,
        "pre_churn_availability_verified": preprocessing["pre_churn_availability_verified"],
        "input_requirement": "Numeric inputs prepared with the saved training preprocessing, in feature_columns order.",
    }
    with (model_dir / "training_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)

    print(f"Training finished in {summary['iterations']} iterations.")
    print(f"Saved model to: {model_path}")
    print(f"Churn prediction threshold: {PREDICTION_THRESHOLD:.0%} (use predict_churn)")
    print("Test data was not loaded. Predictive performance has not been evaluated yet.")


if __name__ == "__main__":
    main()
