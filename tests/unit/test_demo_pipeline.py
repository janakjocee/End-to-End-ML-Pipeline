"""Tests for the locally runnable end-to-end demo."""

import json

import joblib

from scripts.run_demo_pipeline import run_pipeline


def test_demo_pipeline_creates_loadable_artifacts(tmp_path):
    output_dir = tmp_path / "artifacts"
    screenshot = tmp_path / "results.png"

    metrics = run_pipeline(500, output_dir, screenshot)

    assert metrics["test"]["roc_auc"] >= 0.6
    assert metrics["drift"]["samples"] == 500
    assert screenshot.exists() and screenshot.stat().st_size > 0
    assert joblib.load(output_dir / "churn_pipeline.joblib")
    assert json.loads((output_dir / "metrics.json").read_text()) == metrics
