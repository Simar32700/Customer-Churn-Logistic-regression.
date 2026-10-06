"""Compare logistic regression C values using training-only cross-validation."""

import json
import argparse
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "cleaned"


def main(compare_penalties=False):
    with (DATA_DIR / "preprocessing.json").open(encoding="utf-8") as file:
        metadata = json.load(file)

    # Use saved training IDs, but take values before learned filling/encoding.
    customers = pd.read_csv(
        DATA_DIR / "telecom_customer_churn.csv",
        keep_default_na=False, na_values=[""],
        dtype={"Customer ID": "string", "Zip Code": "string"},
    )
    saved_train = pd.read_csv(DATA_DIR / "customers_train.csv", dtype={"Customer ID": "string"})
    if customers["Customer ID"].duplicated().any() or saved_train["Customer ID"].duplicated().any():
        raise ValueError("Customer IDs must be unique.")
    train = customers.set_index("Customer ID").loc[saved_train["Customer ID"]].copy()
    y = train["Customer Status"].map(metadata["target_mapping"])
    if y.isna().any() or not np.array_equal(y.to_numpy(), saved_train["Customer Status"].to_numpy()):
        raise ValueError("Training IDs and outcomes do not match the saved split.")

    binary_columns = metadata["binary_columns"] + ["Gender"]
    for column in metadata["binary_columns"]:
        train[column] = train[column].map(metadata["binary_mapping"])
    train["Gender"] = train["Gender"].map(metadata["gender_mapping"])
    numeric_columns = metadata["scaled_columns"]
    categorical_columns = metadata["categorical_columns"]
    X = train[numeric_columns + binary_columns + categorical_columns]

    preprocessor = ColumnTransformer([
        ("measurements", Pipeline([
            ("fill", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]), numeric_columns),
        ("binary", SimpleImputer(strategy="most_frequent"), binary_columns),
        ("categories", Pipeline([
            ("fill", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]), categorical_columns),
    ])
    pipeline = Pipeline([
        ("prepare", preprocessor),
        ("model", LogisticRegression(
            solver="liblinear" if compare_penalties else "lbfgs",
            max_iter=2000, random_state=42,
        )),
    ])
    folds = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    parameters = {"model__C": [0.01, 0.1, 1, 10, 100]}
    if compare_penalties:
        parameters["model__penalty"] = ["l1", "l2"]
    search = GridSearchCV(
        pipeline,
        parameters,
        scoring={"accuracy": "accuracy", "churn_recall": "recall"},
        cv=folds, refit=False, n_jobs=1, error_score="raise",
    )
    print(f"Comparing 5 C values on {len(X)} training customers, using 5 folds.", flush=True)
    if compare_penalties:
        print("Comparing L1 and L2 with the same liblinear solver (50 total fits).", flush=True)
    print("All learned preprocessing is fitted inside each fold. No test files are loaded.", flush=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        search.fit(X, y)

    results = pd.DataFrame({
        "C": [float(value) for value in search.cv_results_["param_model__C"]],
        "mean_accuracy": search.cv_results_["mean_test_accuracy"],
        "accuracy_std": search.cv_results_["std_test_accuracy"],
        "mean_churn_recall": search.cv_results_["mean_test_churn_recall"],
    })
    results_dir = PROJECT_DIR / "results"
    results_dir.mkdir(exist_ok=True)
    if compare_penalties:
        results.insert(0, "penalty", search.cv_results_["param_model__penalty"].tolist())
    filename = "logistic_penalty_comparison.csv" if compare_penalties else "logistic_C_comparison.csv"
    results.to_csv(results_dir / filename, index=False)
    display = results.copy()
    for column in ["mean_accuracy", "accuracy_std", "mean_churn_recall"]:
        display[column] = display[column].map(lambda value: f"{value:.2%}")
    print(display.to_string(index=False))
    best = results.loc[results["mean_accuracy"].idxmax()]
    baseline_mask = results["C"].eq(1)
    if compare_penalties:
        baseline_mask = baseline_mask & results["penalty"].eq("l2")
        print(f"Best penalty: {best['penalty'].upper()}")
    baseline = results.loc[baseline_mask, "mean_accuracy"].iloc[0]
    print(f"Best mean validation accuracy: C={best['C']:g}, accuracy={best['mean_accuracy']:.2%}")
    print(f"Difference from C=1: {(best['mean_accuracy'] - baseline) * 100:.3f} percentage points")
    print("These are validation results, not test scores. The saved baseline model is unchanged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare-penalties", action="store_true")
    main(compare_penalties=parser.parse_args().compare_penalties)
