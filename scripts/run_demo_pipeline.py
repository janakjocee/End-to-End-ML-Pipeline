"""Run a deterministic, local end-to-end churn training pipeline."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from scripts.generate_sample_data import generate_churn_dataset


def build_pipeline(data: pd.DataFrame) -> Pipeline:
    """Build a preprocessing and classification pipeline."""
    features = data.drop(columns=["customer_id", "churn"])
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = features.select_dtypes(exclude="number").columns.tolist()

    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1000,
                    random_state=42,
                    solver="liblinear",
                ),
            ),
        ]
    )


def evaluate(model: Pipeline, data: pd.DataFrame) -> dict:
    """Evaluate a fitted model on a labeled dataset."""
    features = data.drop(columns=["customer_id", "churn"])
    target = data["churn"]
    predictions = model.predict(features)
    probabilities = model.predict_proba(features)[:, 1]
    return {
        "samples": int(len(data)),
        "churn_rate": round(float(target.mean()), 4),
        "accuracy": round(float(accuracy_score(target, predictions)), 4),
        "precision": round(float(precision_score(target, predictions, zero_division=0)), 4),
        "recall": round(float(recall_score(target, predictions, zero_division=0)), 4),
        "f1": round(float(f1_score(target, predictions, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(target, probabilities)), 4),
    }


def render_summary(metrics: dict, model: Pipeline, test_data: pd.DataFrame, output_path: Path) -> None:
    """Render a result summary suitable for README documentation."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    names = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    values = [metrics["test"][name] for name in names]
    axes[0].bar(names, values, color=["#2563eb", "#0891b2", "#059669", "#7c3aed", "#ea580c"])
    axes[0].set_ylim(0, 1)
    axes[0].set_title("Holdout metrics")
    axes[0].set_ylabel("Score")
    axes[0].tick_params(axis="x", rotation=25)
    for index, value in enumerate(values):
        axes[0].text(index, value + 0.02, f"{value:.3f}", ha="center", fontsize=9)

    features = test_data.drop(columns=["customer_id", "churn"])
    ConfusionMatrixDisplay.from_estimator(
        model,
        features,
        test_data["churn"],
        ax=axes[1],
        colorbar=False,
        cmap="Blues",
    )
    axes[1].set_title("Holdout confusion matrix")
    figure.suptitle("End-to-End ML Pipeline: Verified Demo Run", fontsize=15, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_pipeline(samples: int, output_dir: Path, screenshot_path: Path) -> dict:
    """Generate data, train, evaluate, and persist artifacts."""
    started = perf_counter()
    data = generate_churn_dataset(n_samples=samples, random_state=42)
    drift_data = generate_churn_dataset(n_samples=max(500, samples // 2), random_state=43, drift=True)
    train_data, test_data = train_test_split(
        data,
        test_size=0.25,
        random_state=42,
        stratify=data["churn"],
    )

    model = build_pipeline(train_data)
    model.fit(train_data.drop(columns=["customer_id", "churn"]), train_data["churn"])
    metrics = {
        "test": evaluate(model, test_data),
        "drift": evaluate(model, drift_data),
    }
    metrics["roc_auc_change_under_drift"] = round(
        metrics["drift"]["roc_auc"] - metrics["test"]["roc_auc"], 4
    )
    metrics["duration_seconds"] = round(perf_counter() - started, 3)

    output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "churn_pipeline.joblib")
    test_data.to_csv(output_dir / "holdout.csv", index=False)
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    render_summary(metrics, model, test_data, screenshot_path)
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=5000)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/demo"))
    parser.add_argument(
        "--screenshot",
        type=Path,
        default=Path("docs/assets/demo-pipeline-results.png"),
    )
    args = parser.parse_args()
    metrics = run_pipeline(args.samples, args.output_dir, args.screenshot)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
