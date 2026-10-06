"""Evaluate correlation-based feature selection without selecting on test data."""

import json
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "cleaned"


class CorrelationSelector(TransformerMixin, BaseEstimator):
    def __init__(self, threshold=0.0):
        self.threshold = threshold

    def fit(self, X, y):
        values = np.asarray(X, dtype=float)
        centered = values - values.mean(axis=0)
        outcome = np.asarray(y, dtype=float)
        outcome = outcome - outcome.mean()
        denominator = np.sqrt((centered ** 2).sum(axis=0) * (outcome ** 2).sum())
        self.correlations_ = np.divide(
            centered.T @ outcome, denominator,
            out=np.zeros(values.shape[1]), where=denominator > 0,
        )
        self.support_ = np.abs(self.correlations_) >= self.threshold
        if not self.support_.any():
            raise ValueError("The correlation cutoff removed all features.")
        return self

    def transform(self, X):
        return np.asarray(X)[:, self.support_]


def main():
    with (DATA_DIR / "preprocessing.json").open(encoding="utf-8") as file:
        metadata = json.load(file)
    customers = pd.read_csv(
        DATA_DIR / "telecom_customer_churn.csv", keep_default_na=False,
        na_values=[""], dtype={"Customer ID": "string", "Zip Code": "string"},
    ).set_index("Customer ID", verify_integrity=True)
    numeric = metadata["scaled_columns"]
    binary = metadata["binary_columns"] + ["Gender"]
    categorical = metadata["categorical_columns"]

    def load_split(name):
        saved = pd.read_csv(DATA_DIR / f"customers_{name}.csv", dtype={"Customer ID": "string"})
        if saved["Customer ID"].duplicated().any():
            raise ValueError("Duplicate split IDs.")
        records = customers.loc[saved["Customer ID"]].copy()
        y = records["Customer Status"].map(metadata["target_mapping"])
        if y.isna().any() or not np.array_equal(y.to_numpy(), saved["Customer Status"].to_numpy()):
            raise ValueError("Saved split outcomes do not match the customer records.")
        for column in metadata["binary_columns"]:
            records[column] = records[column].map(metadata["binary_mapping"])
        records["Gender"] = records["Gender"].map(metadata["gender_mapping"])
        return records[numeric + binary + categorical], y

    X_train, y_train = load_split("train")
    prepare = ColumnTransformer([
        ("measurements", Pipeline([
            ("fill", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric),
        ("binary", SimpleImputer(strategy="most_frequent"), binary),
        ("categories", Pipeline([
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical),
    ], verbose_feature_names_out=False)
    pipeline = Pipeline([
        ("prepare", prepare),
        ("select", CorrelationSelector()),
        ("model", LogisticRegression(C=1, max_iter=2000, random_state=42)),
    ])
    search = GridSearchCV(
        pipeline, {"select__threshold": [0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]},
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
        scoring={"accuracy": "accuracy", "recall": "recall"},
        refit="accuracy", n_jobs=1, error_score="raise",
    )
    print("Comparing correlation cutoffs with fold-fitted preprocessing and selection.", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        search.fit(X_train, y_train)

    best = search.best_estimator_
    names = best.named_steps["prepare"].get_feature_names_out()
    selector = best.named_steps["select"]
    ranking = pd.DataFrame({
        "feature": names, "correlation_with_churn": selector.correlations_,
        "absolute_correlation": np.abs(selector.correlations_),
        "selected": selector.support_,
    }).sort_values("absolute_correlation", ascending=False)
    results = pd.DataFrame({
        "cutoff": search.cv_results_["param_select__threshold"].astype(float),
        "mean_validation_accuracy": search.cv_results_["mean_test_accuracy"],
        "validation_accuracy_std": search.cv_results_["std_test_accuracy"],
        "mean_validation_churn_recall": search.cv_results_["mean_test_recall"],
    })
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    ranking.to_csv(results_dir / "feature_churn_correlations.csv", index=False)
    results.to_csv(results_dir / "correlation_cutoff_comparison.csv", index=False)
    train_encoded = pd.DataFrame(best.named_steps["prepare"].transform(X_train), columns=names)
    train_encoded.corr().to_csv(results_dir / "feature_correlation_matrix.csv", index_label="feature")

    # The cutoff is fixed using training CV before loading the held-out test split.
    X_test, y_test = load_split("test")
    if not X_train.index.intersection(X_test.index).empty:
        raise ValueError("Training and test customers overlap.")
    predictions = best.predict(X_test)
    summary = {
        "best_cutoff": float(search.best_params_["select__threshold"]),
        "mean_validation_accuracy": float(search.best_score_),
        "selected_features": names[selector.support_].tolist(),
        "test_accuracy": float(accuracy_score(y_test, predictions)),
        "test_precision": float(precision_score(y_test, predictions, zero_division=0)),
        "test_churn_recall": float(recall_score(y_test, predictions, zero_division=0)),
        "test_correct": int((predictions == y_test.to_numpy()).sum()),
        "test_rows": len(y_test),
        "C": 1,
        "note": "Test set previously evaluated; this is a comparison, not a new untouched final evaluation.",
    }
    with (results_dir / "correlation_selection_summary.json").open("w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2)
    print(results.to_string(index=False))
    print("\nStrongest training correlations:")
    print(ranking.head(10).to_string(index=False))
    print(f"\nChosen cutoff: {summary['best_cutoff']}; selected features: {len(summary['selected_features'])}/{len(names)}")
    print(f"Test accuracy: {summary['test_accuracy']:.2%}; churn recall: {summary['test_churn_recall']:.2%}")
    print(f"Correct test predictions: {summary['test_correct']}/{summary['test_rows']}")
    print("The existing saved baseline model has not been replaced.")


if __name__ == "__main__":
    main()
