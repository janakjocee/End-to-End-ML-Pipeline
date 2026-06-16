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

from shared.churn_business import score_customer, summarize_portfolio
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


def export_scoring_bundle(model: Pipeline, training_data: pd.DataFrame, metrics: dict) -> dict:
    """Export the fitted model into a lightweight JSON scoring bundle."""
    features = training_data.drop(columns=["customer_id", "churn"])
    preprocessor = model.named_steps["preprocessor"]
    classifier = model.named_steps["classifier"]
    numeric_features = preprocessor.transformers_[0][2]
    categorical_features = preprocessor.transformers_[1][2]
    scaler = preprocessor.named_transformers_["numeric"]
    encoder = preprocessor.named_transformers_["categorical"]
    coefficients = classifier.coef_[0]

    schema = {}
    coefficient_index = 0
    numeric_weights = {}
    categorical_weights = {}

    for index, feature in enumerate(numeric_features):
        schema[feature] = {
            "type": "numeric",
            "default": round(float(features[feature].median()), 4),
            "mean": round(float(scaler.mean_[index]), 8),
            "scale": round(float(scaler.scale_[index]), 8),
            "min": round(float(features[feature].min()), 4),
            "max": round(float(features[feature].max()), 4),
        }
        numeric_weights[feature] = round(float(coefficients[coefficient_index]), 10)
        coefficient_index += 1

    for feature, categories in zip(categorical_features, encoder.categories_):
        values = [str(value) for value in categories]
        schema[feature] = {
            "type": "categorical",
            "default": str(features[feature].mode().iloc[0]),
            "values": values,
        }
        categorical_weights[feature] = {}
        for value in values:
            categorical_weights[feature][value] = round(float(coefficients[coefficient_index]), 10)
            coefficient_index += 1

    return {
        "model_id": "churn-command-center-logreg-v1",
        "problem": "Predict customer churn risk and recommend customer-success actions.",
        "decision_thresholds": {"low": 0.0, "medium": 0.4, "high": 0.7},
        "metrics": metrics,
        "schema": schema,
        "weights": {
            "intercept": round(float(classifier.intercept_[0]), 10),
            "numeric": numeric_weights,
            "categorical": categorical_weights,
        },
    }


def export_customer_portfolio(bundle: dict, data: pd.DataFrame, output_dir: Path) -> dict:
    """Export scored customers and business summary for the web app."""
    scored_customers = []
    sample = data.sort_values(["monthly_charges", "tenure"], ascending=[False, True]).head(250)
    for record in sample.to_dict(orient="records"):
        features = {key: value for key, value in record.items() if key not in {"churn"}}
        score = score_customer(bundle, features)
        scored_customers.append(
            {
                "customer_id": record["customer_id"],
                "features": features,
                "actual_churn": int(record["churn"]),
                "score": score,
            }
        )

    scored_customers.sort(key=lambda item: item["score"]["revenue_at_risk"], reverse=True)
    summary = summarize_portfolio(scored_customers)
    (output_dir / "customer_scores.json").write_text(
        json.dumps(scored_customers, indent=2),
        encoding="utf-8",
    )
    (output_dir / "business_summary.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )
    return summary


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


def render_drift_summary(reference: pd.DataFrame, drift: pd.DataFrame, output_path: Path) -> None:
    """Render the main distribution shifts introduced by the generator."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.5))
    axes[0].hist(reference["monthly_charges"], bins=25, alpha=0.7, label="reference")
    axes[0].hist(drift["monthly_charges"], bins=25, alpha=0.7, label="drift")
    axes[0].set_title("Monthly charges")
    axes[0].set_xlabel("Charge")
    axes[0].legend()

    contract_order = ["Month-to-month", "One year", "Two year"]
    contract_rates = pd.DataFrame(
        {
            "reference": reference["contract"].value_counts(normalize=True),
            "drift": drift["contract"].value_counts(normalize=True),
        }
    ).reindex(contract_order)
    contract_rates.plot.bar(ax=axes[1], color=["#2563eb", "#ea580c"])
    axes[1].set_title("Contract distribution")
    axes[1].set_ylabel("Share")
    axes[1].tick_params(axis="x", rotation=25)

    axes[2].bar(
        ["reference", "drift"],
        [reference["churn"].mean(), drift["churn"].mean()],
        color=["#2563eb", "#ea580c"],
    )
    axes[2].set_ylim(0, 0.5)
    axes[2].set_title("Churn rate")
    axes[2].set_ylabel("Share")
    figure.suptitle("Generated Drift Scenario", fontsize=15, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_pipeline(
    samples: int,
    output_dir: Path,
    screenshot_path: Path,
    drift_screenshot_path: Path = Path("docs/assets/data-drift-comparison.png"),
    web_public_dir: Path = Path("public"),
) -> dict:
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
    web_public_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, output_dir / "churn_pipeline.joblib")
    test_data.to_csv(output_dir / "holdout.csv", index=False)
    bundle = export_scoring_bundle(model, train_data, metrics)
    business_summary = export_customer_portfolio(bundle, test_data, output_dir)
    bundle["business_summary"] = business_summary
    with (output_dir / "metrics.json").open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)
    for path in [output_dir / "model_bundle.json", web_public_dir / "model_bundle.json"]:
        path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    for name in ["customer_scores.json", "business_summary.json"]:
        (web_public_dir / name).write_text((output_dir / name).read_text(encoding="utf-8"), encoding="utf-8")
    render_summary(metrics, model, test_data, screenshot_path)
    render_drift_summary(data, drift_data, drift_screenshot_path)
    for image_path in [screenshot_path, drift_screenshot_path]:
        target = web_public_dir / image_path.name
        target.write_bytes(image_path.read_bytes())
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
    parser.add_argument(
        "--drift-screenshot",
        type=Path,
        default=Path("docs/assets/data-drift-comparison.png"),
    )
    parser.add_argument("--web-public-dir", type=Path, default=Path("public"))
    args = parser.parse_args()
    metrics = run_pipeline(
        args.samples,
        args.output_dir,
        args.screenshot,
        args.drift_screenshot,
        args.web_public_dir,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
