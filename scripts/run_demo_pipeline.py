"""Run a deterministic, local end-to-end churn training pipeline."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import joblib
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    brier_score_loss,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from shared.churn_business import score_customer, summarize_portfolio
from scripts.generate_sample_data import generate_churn_dataset


def build_preprocessor(data: pd.DataFrame) -> ColumnTransformer:
    """Build feature preprocessing from the training dataframe."""
    features = data.drop(columns=["customer_id", "churn"])
    numeric = features.select_dtypes(include="number").columns.tolist()
    categorical = features.select_dtypes(exclude="number").columns.tolist()

    return ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), numeric),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                categorical,
            ),
        ]
    )


def build_pipeline(data: pd.DataFrame) -> Pipeline:
    """Build the production preprocessing and classification pipeline."""
    preprocessor = build_preprocessor(data)
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


def build_candidate_models(data: pd.DataFrame) -> dict[str, Pipeline]:
    """Build candidate models for model selection reporting."""
    logistic = build_pipeline(data)
    calibrated = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(data)),
            (
                "classifier",
                CalibratedClassifierCV(
                    estimator=LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=42,
                        solver="liblinear",
                    ),
                    cv=3,
                    method="sigmoid",
                ),
            ),
        ]
    )
    forest = Pipeline(
        steps=[
            ("preprocessor", build_preprocessor(data)),
            (
                "classifier",
                RandomForestClassifier(
                    class_weight="balanced_subsample",
                    max_depth=8,
                    min_samples_leaf=20,
                    n_estimators=160,
                    n_jobs=-1,
                    random_state=42,
                ),
            ),
        ]
    )
    return {
        "production_logistic": logistic,
        "calibrated_logistic": calibrated,
        "random_forest": forest,
    }


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
        "brier": round(float(brier_score_loss(target, probabilities)), 4),
    }


def evaluate_model_candidates(train_data: pd.DataFrame, test_data: pd.DataFrame) -> tuple[dict, Pipeline]:
    """Run cross-validation and holdout evaluation for candidate models."""
    features = train_data.drop(columns=["customer_id", "churn"])
    target = train_data["churn"]
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {
        "roc_auc": "roc_auc",
        "recall": "recall",
        "precision": "precision",
        "f1": "f1",
        "neg_brier": "neg_brier_score",
    }
    candidates = build_candidate_models(train_data)
    report = {}

    for name, candidate in candidates.items():
        scores = cross_validate(candidate, features, target, cv=cv, scoring=scoring, n_jobs=None)
        candidate.fit(features, target)
        holdout = evaluate(candidate, test_data)
        report[name] = {
            "cross_validation": {
                "roc_auc_mean": round(float(scores["test_roc_auc"].mean()), 4),
                "roc_auc_std": round(float(scores["test_roc_auc"].std()), 4),
                "recall_mean": round(float(scores["test_recall"].mean()), 4),
                "precision_mean": round(float(scores["test_precision"].mean()), 4),
                "f1_mean": round(float(scores["test_f1"].mean()), 4),
                "brier_mean": round(float((-scores["test_neg_brier"]).mean()), 4),
            },
            "holdout": holdout,
        }

    return report, candidates["production_logistic"]


def tune_business_thresholds(model: Pipeline, validation_data: pd.DataFrame) -> dict:
    """Choose decision thresholds by expected net value, not only ROC-AUC."""
    features = validation_data.drop(columns=["customer_id", "churn"])
    probabilities = model.predict_proba(features)[:, 1]
    rows = validation_data.to_dict(orient="records")
    options = []
    for threshold in [round(value / 100, 2) for value in range(30, 86, 5)]:
        expected_net_value = 0.0
        interventions = 0
        captured_churners = 0
        for probability, row in zip(probabilities, rows):
            if probability < threshold:
                continue
            interventions += 1
            annual_revenue = float(row["monthly_charges"]) * 12
            if row["churn"]:
                captured_churners += 1
            expected_net_value += annual_revenue * probability * 0.18 - 22
        options.append(
            {
                "threshold": threshold,
                "interventions": interventions,
                "captured_churners": captured_churners,
                "expected_net_value": round(float(expected_net_value), 2),
            }
        )
    best = max(options, key=lambda item: item["expected_net_value"])
    return {
        "objective": "maximize expected retention net value on holdout data",
        "assumptions": {
            "average_intervention_cost": 22,
            "average_save_lift": 0.18,
        },
        "recommended_threshold": best["threshold"],
        "options": options,
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


def render_model_selection_report(model_report: dict, threshold_policy: dict, output_path: Path) -> None:
    """Render model selection and business threshold evidence."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    names = list(model_report)
    roc_auc = [model_report[name]["cross_validation"]["roc_auc_mean"] for name in names]
    brier = [model_report[name]["cross_validation"]["brier_mean"] for name in names]
    threshold_rows = threshold_policy["options"]

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    axes[0].bar(names, roc_auc, color="#2563eb")
    axes[0].set_ylim(0, 1)
    axes[0].set_title("5-fold ROC-AUC")
    axes[0].tick_params(axis="x", rotation=20)

    axes[1].bar(names, brier, color="#7c3aed")
    axes[1].set_title("Calibration loss (Brier)")
    axes[1].tick_params(axis="x", rotation=20)

    axes[2].plot(
        [row["threshold"] for row in threshold_rows],
        [row["expected_net_value"] for row in threshold_rows],
        marker="o",
        color="#059669",
    )
    axes[2].axvline(threshold_policy["recommended_threshold"], color="#ef4444", linestyle="--")
    axes[2].set_title("Business threshold tuning")
    axes[2].set_xlabel("Action threshold")
    axes[2].set_ylabel("Expected net value")

    figure.suptitle("Model Selection and Decision Policy", fontsize=15, fontweight="bold")
    figure.tight_layout()
    figure.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(figure)


def run_pipeline(
    samples: int,
    output_dir: Path,
    screenshot_path: Path,
    drift_screenshot_path: Path = Path("docs/assets/data-drift-comparison.png"),
    model_report_path: Path = Path("docs/assets/model-selection-report.png"),
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

    model_report, model = evaluate_model_candidates(train_data, test_data)
    metrics = {
        "test": evaluate(model, test_data),
        "drift": evaluate(model, drift_data),
        "model_selection": model_report,
    }
    threshold_policy = tune_business_thresholds(model, test_data)
    metrics["threshold_policy"] = threshold_policy
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
    (output_dir / "model_selection.json").write_text(json.dumps(model_report, indent=2), encoding="utf-8")
    (output_dir / "threshold_policy.json").write_text(json.dumps(threshold_policy, indent=2), encoding="utf-8")
    for path in [output_dir / "model_bundle.json", web_public_dir / "model_bundle.json"]:
        path.write_text(json.dumps(bundle, indent=2), encoding="utf-8")
    for name in ["customer_scores.json", "business_summary.json", "model_selection.json", "threshold_policy.json"]:
        (web_public_dir / name).write_text((output_dir / name).read_text(encoding="utf-8"), encoding="utf-8")
    render_summary(metrics, model, test_data, screenshot_path)
    render_drift_summary(data, drift_data, drift_screenshot_path)
    render_model_selection_report(model_report, threshold_policy, model_report_path)
    for image_path in [screenshot_path, drift_screenshot_path, model_report_path]:
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
    parser.add_argument(
        "--model-report-screenshot",
        type=Path,
        default=Path("docs/assets/model-selection-report.png"),
    )
    parser.add_argument("--web-public-dir", type=Path, default=Path("public"))
    args = parser.parse_args()
    metrics = run_pipeline(
        args.samples,
        args.output_dir,
        args.screenshot,
        args.drift_screenshot,
        args.model_report_screenshot,
        args.web_public_dir,
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
