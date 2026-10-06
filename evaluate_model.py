"""Generate test metrics, plots, predictions, and model details."""

import json
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibrationDisplay
from sklearn.metrics import (
    ConfusionMatrixDisplay, PrecisionRecallDisplay, RocCurveDisplay,
    accuracy_score, auc, average_precision_score, balanced_accuracy_score,
    brier_score_loss, classification_report, cohen_kappa_score, confusion_matrix,
    f1_score, log_loss, matthews_corrcoef, precision_recall_curve,
    precision_score, recall_score, roc_auc_score,
)

from model_train import predict_churn


PROJECT_DIR = Path(__file__).resolve().parent
DATA_DIR = PROJECT_DIR / "data" / "cleaned"
RESULTS_DIR = PROJECT_DIR / "results" / "evaluation"


def threshold_metrics(y, probabilities, threshold):
    predictions = (probabilities >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, predictions, labels=[0, 1]).ravel()

    def ratio(numerator, denominator):
        return float(numerator / denominator) if denominator else None

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y, predictions)),
        "precision_churn": float(precision_score(y, predictions, zero_division=0)),
        "recall_churn": float(recall_score(y, predictions, zero_division=0)),
        "f1_churn": float(f1_score(y, predictions, zero_division=0)),
        "specificity": ratio(tn, tn + fp),
        "negative_predictive_value": ratio(tn, tn + fn),
        "false_positive_rate": ratio(fp, tn + fp),
        "false_negative_rate": ratio(fn, fn + tp),
        "balanced_accuracy": float(balanced_accuracy_score(y, predictions)),
        "matthews_correlation_coefficient": float(matthews_corrcoef(y, predictions)),
        "cohen_kappa": float(cohen_kappa_score(y, predictions)),
        "true_negatives": int(tn), "false_positives": int(fp),
        "false_negatives": int(fn), "true_positives": int(tp),
    }


def save_plot(display, filename, title):
    display.ax_.set_title(title)
    display.figure_.tight_layout()
    display.figure_.savefig(RESULTS_DIR / filename, dpi=160, bbox_inches="tight")
    plt.close(display.figure_)


def main():
    model = joblib.load(PROJECT_DIR / "models" / "customer_churn_model.joblib")
    with (PROJECT_DIR / "models" / "training_summary.json").open(encoding="utf-8") as file:
        training = json.load(file)
    with (DATA_DIR / "preprocessing.json").open(encoding="utf-8") as file:
        preprocessing = json.load(file)
    X = pd.read_csv(DATA_DIR / "X_test_encoded.csv")
    target = pd.read_csv(DATA_DIR / "y_test.csv")
    customers = pd.read_csv(DATA_DIR / "customers_test.csv", dtype={"Customer ID": "string"})
    if X.columns.tolist() != model.feature_names_in_.tolist():
        raise ValueError("Test feature names/order do not match the model.")
    if target.columns.tolist() != ["Customer Status"] or len(X) != len(target) or len(X) != len(customers):
        raise ValueError("Test inputs, targets, and customer records must align.")
    if not all(pd.api.types.is_numeric_dtype(dtype) for dtype in X.dtypes) or not np.isfinite(X.to_numpy()).all():
        raise ValueError("Test features must be numeric and finite.")
    y = target["Customer Status"]
    if y.isna().any() or set(y.unique()) != {0, 1}:
        raise ValueError("Test labels must contain both Stayed (0) and Churned (1).")
    if not np.array_equal(y.to_numpy(), customers["Customer Status"].to_numpy()):
        raise ValueError("Test customer rows and targets do not match.")

    threshold = training["prediction_threshold"]
    probabilities = model.predict_proba(X)[:, np.flatnonzero(model.classes_ == 1)[0]]
    predictions = predict_churn(model, X, threshold=threshold)
    metrics = threshold_metrics(y, probabilities, threshold)
    pr_precision, pr_recall, _ = precision_recall_curve(y, probabilities)
    metrics.update({
        "roc_auc": float(roc_auc_score(y, probabilities)),
        "average_precision": float(average_precision_score(y, probabilities)),
        "pr_auc_trapezoidal": float(auc(pr_recall, pr_precision)),
        "log_loss": float(log_loss(y, probabilities, labels=[0, 1])),
        "brier_score": float(brier_score_loss(y, probabilities)),
        "test_rows": len(y), "input_features": X.shape[1],
        "actual_churners": int(y.eq(1).sum()),
        "actual_stayers": int(y.eq(0).sum()),
        "churn_prevalence": float(y.mean()),
        "always_stayed_baseline_accuracy": float(y.eq(0).mean()),
        "majority_class_baseline_accuracy": float(y.value_counts(normalize=True).max()),
    })
    report_dict = classification_report(
        y, predictions, labels=[0, 1], target_names=["Stayed", "Churned"],
        output_dict=True, zero_division=0,
    )
    report_text = classification_report(
        y, predictions, labels=[0, 1], target_names=["Stayed", "Churned"],
        digits=4, zero_division=0,
    )
    parameters = model.get_params()
    limitations = [
        "The test set has been evaluated previously; this is not a new untouched final test.",
        "Pre-churn feature availability is unverified. Scores concern recorded outcomes, not proven future churn forecasts.",
        "ROC AUC and probability losses use probabilities, not thresholded Yes/No predictions.",
        "Higher log loss and Brier score are worse; higher accuracy, F1, ROC AUC and average precision are better.",
        "Average precision and trapezoidal PR AUC are different summaries of the precision-recall curve.",
        "Undefined classification precision/recall/F1 is reported as zero; undefined rates are null.",
    ]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    evaluation = {
        "metrics": metrics, "classification_report": report_dict,
        "model_parameters": parameters, "training_details": training,
        "preprocessing_details": preprocessing, "limitations": limitations,
    }
    with (RESULTS_DIR / "evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(evaluation, file, indent=2, allow_nan=False)

    comparisons = pd.DataFrame([
        threshold_metrics(y, probabilities, value) for value in sorted({0.5, float(threshold)})
    ])
    comparisons.to_csv(RESULTS_DIR / "threshold_comparison.csv", index=False)
    pd.DataFrame([metrics]).to_csv(RESULTS_DIR / "metrics.csv", index=False)
    pd.DataFrame({
        "Customer ID": customers["Customer ID"], "actual_churn": y,
        "predicted_churn": predictions, "churn_probability": probabilities,
        "answer": np.where(predictions == 1, "Yes", "No"),
        "correct": predictions == y.to_numpy(),
    }).to_csv(RESULTS_DIR / "test_predictions.csv", index=False)
    matrix = confusion_matrix(y, predictions, labels=[0, 1])
    pd.DataFrame(matrix, index=["Actual Stayed", "Actual Churned"], columns=["Predicted Stayed", "Predicted Churned"]).to_csv(
        RESULTS_DIR / "confusion_matrix.csv", index_label="Actual / Predicted"
    )
    pd.DataFrame({
        "feature": model.feature_names_in_, "beta": model.coef_[0],
        "absolute_beta": np.abs(model.coef_[0]),
    }).sort_values("absolute_beta", ascending=False).to_csv(RESULTS_DIR / "model_coefficients.csv", index=False)

    save_plot(ConfusionMatrixDisplay.from_predictions(
        y, predictions, labels=[0, 1], display_labels=["Stayed", "Churned"], cmap="Blues",
    ), "confusion_matrix.png", f"Confusion Matrix - Threshold {threshold:.2f}")
    roc = RocCurveDisplay.from_predictions(y, probabilities, name="Logistic regression")
    roc.ax_.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Random ranking")
    roc.ax_.legend(loc="lower right")
    save_plot(roc, "roc_curve.png", f"ROC Curve - AUC {metrics['roc_auc']:.4f}")
    pr = PrecisionRecallDisplay.from_predictions(y, probabilities, name="Logistic regression")
    pr.ax_.axhline(y.mean(), linestyle="--", color="gray", label="Churn prevalence")
    pr.ax_.legend(loc="lower left")
    save_plot(pr, "precision_recall_curve.png", f"Precision-Recall - AP {metrics['average_precision']:.4f}")
    save_plot(CalibrationDisplay.from_predictions(
        y, probabilities, n_bins=10, strategy="quantile", name="Logistic regression",
    ), "calibration_curve.png", "Probability Calibration")

    lines = ["CUSTOMER CHURN MODEL EVALUATION", f"Decision threshold: {threshold:.2f}", "", "METRICS"]
    lines += [f"{key}: {value:.6f}" if isinstance(value, float) else f"{key}: {value}" for key, value in metrics.items()]
    lines += ["", "CLASSIFICATION REPORT", report_text, "CONFUSION MATRIX", "Rows: actual; columns: predicted. Order: Stayed, Churned.", str(matrix)]
    lines += ["", "THRESHOLD COMPARISON", comparisons.to_string(index=False), "", "MODEL PARAMETERS", json.dumps(parameters, indent=2)]
    lines += ["", f"Intercept (beta_0): {model.intercept_.tolist()}", f"Training iterations: {model.n_iter_.tolist()}", "", "PREPROCESSING AND TRAINING DETAILS", json.dumps({"training": training, "preprocessing": preprocessing}, indent=2), "", "LIMITATIONS"]
    lines += limitations
    (RESULTS_DIR / "results_report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Accuracy: {metrics['accuracy']:.2%}; Precision: {metrics['precision_churn']:.2%}; Recall: {metrics['recall_churn']:.2%}")
    print(f"F1: {metrics['f1_churn']:.4f}; ROC AUC: {metrics['roc_auc']:.4f}; Average precision: {metrics['average_precision']:.4f}")
    print(f"All reports and plots saved to: {RESULTS_DIR}")


if __name__ == "__main__":
    main()
