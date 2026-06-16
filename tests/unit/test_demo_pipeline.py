"""Tests for the locally runnable end-to-end demo."""

import json

import joblib

from shared.churn_business import score_customer
from scripts.run_demo_pipeline import run_pipeline


def test_demo_pipeline_creates_loadable_artifacts(tmp_path):
    output_dir = tmp_path / "artifacts"
    screenshot = tmp_path / "results.png"
    drift_screenshot = tmp_path / "drift.png"

    metrics = run_pipeline(500, output_dir, screenshot, drift_screenshot, tmp_path / "public")

    assert metrics["test"]["roc_auc"] >= 0.6
    assert metrics["drift"]["samples"] == 500
    assert screenshot.exists() and screenshot.stat().st_size > 0
    assert drift_screenshot.exists() and drift_screenshot.stat().st_size > 0
    assert joblib.load(output_dir / "churn_pipeline.joblib")
    assert json.loads((output_dir / "metrics.json").read_text()) == metrics


def test_exported_scoring_bundle_scores_customer(tmp_path):
    output_dir = tmp_path / "artifacts"
    run_pipeline(
        500,
        output_dir,
        tmp_path / "results.png",
        tmp_path / "drift.png",
        tmp_path / "public",
    )
    bundle = json.loads((output_dir / "model_bundle.json").read_text())
    customer = json.loads((output_dir / "customer_scores.json").read_text())[0]

    score = score_customer(bundle, customer["features"])

    assert 0 <= score["churn_probability"] <= 1
    assert score["risk_band"] in {"low", "medium", "high"}
    assert score["recommended_action"]
    assert len(score["top_drivers"]) == 5
