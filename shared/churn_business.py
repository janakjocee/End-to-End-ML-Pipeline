"""Business-facing churn scoring utilities.

This module intentionally uses only the Python standard library so the same
model bundle can power lightweight serverless demos without scikit-learn.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any


LOW_THRESHOLD = 0.4
MEDIUM_THRESHOLD = 0.7


@dataclass(frozen=True)
class RetentionAction:
    name: str
    priority: str
    cost: float
    expected_lift: float
    description: str


RETENTION_ACTIONS = {
    "save_offer": RetentionAction(
        name="Targeted save offer",
        priority="high",
        cost=55.0,
        expected_lift=0.28,
        description="Offer a tailored discount or plan credit to high-value customers with flexible contracts.",
    ),
    "success_call": RetentionAction(
        name="Customer success call",
        priority="high",
        cost=22.0,
        expected_lift=0.18,
        description="Route the customer to a retention specialist for plan review and objection handling.",
    ),
    "support_bundle": RetentionAction(
        name="Support bundle",
        priority="medium",
        cost=14.0,
        expected_lift=0.12,
        description="Bundle technical support or online security for customers lacking service add-ons.",
    ),
    "education_nudge": RetentionAction(
        name="Education nudge",
        priority="medium",
        cost=3.0,
        expected_lift=0.05,
        description="Send onboarding, product education, and billing-transparency messaging.",
    ),
    "monitor": RetentionAction(
        name="Monitor only",
        priority="low",
        cost=0.0,
        expected_lift=0.0,
        description="No intervention needed now; keep the account in the next scoring cycle.",
    ),
}


def risk_band(probability: float) -> str:
    """Map a churn probability to an operations-friendly risk band."""
    if probability >= MEDIUM_THRESHOLD:
        return "high"
    if probability >= LOW_THRESHOLD:
        return "medium"
    return "low"


def sigmoid(value: float) -> float:
    """Numerically stable logistic function."""
    if value >= 0:
        z = math.exp(-value)
        return 1 / (1 + z)
    z = math.exp(value)
    return z / (1 + z)


def default_features(bundle: dict[str, Any]) -> dict[str, Any]:
    """Return deploy-safe defaults for the scoring form."""
    return {name: spec["default"] for name, spec in bundle["schema"].items()}


def score_customer(bundle: dict[str, Any], features: dict[str, Any]) -> dict[str, Any]:
    """Score one customer using an exported model bundle."""
    schema = bundle["schema"]
    weights = bundle["weights"]
    contributions: list[dict[str, Any]] = []
    logit = float(weights["intercept"])

    for name, spec in schema.items():
        value = features.get(name, spec["default"])
        if spec["type"] == "numeric":
            numeric_value = float(value)
            scaled = (numeric_value - spec["mean"]) / spec["scale"] if spec["scale"] else 0.0
            contribution = scaled * weights["numeric"].get(name, 0.0)
            contributions.append(
                {
                    "feature": name,
                    "value": numeric_value,
                    "contribution": contribution,
                    "direction": "raises risk" if contribution > 0 else "lowers risk",
                }
            )
            logit += contribution
        else:
            text_value = str(value)
            contribution = weights["categorical"].get(name, {}).get(text_value, 0.0)
            contributions.append(
                {
                    "feature": name,
                    "value": text_value,
                    "contribution": contribution,
                    "direction": "raises risk" if contribution > 0 else "lowers risk",
                }
            )
            logit += contribution

    probability = sigmoid(logit)
    band = risk_band(probability)
    action = choose_action(features, probability)
    monthly_revenue = float(features.get("monthly_charges", 0.0))
    annual_revenue = monthly_revenue * 12
    revenue_at_risk = annual_revenue * probability
    expected_saved = revenue_at_risk * action.expected_lift
    net_value = expected_saved - action.cost

    contributions.sort(key=lambda item: abs(item["contribution"]), reverse=True)
    return {
        "churn_probability": round(probability, 4),
        "risk_band": band,
        "recommended_action": action.name,
        "action_priority": action.priority,
        "action_description": action.description,
        "estimated_action_cost": round(action.cost, 2),
        "expected_save_lift": action.expected_lift,
        "annual_revenue": round(annual_revenue, 2),
        "revenue_at_risk": round(revenue_at_risk, 2),
        "expected_saved_revenue": round(expected_saved, 2),
        "expected_net_value": round(net_value, 2),
        "top_drivers": contributions[:5],
    }


def choose_action(features: dict[str, Any], probability: float) -> RetentionAction:
    """Select a practical next-best action for customer success teams."""
    monthly = float(features.get("monthly_charges", 0.0))
    contract = str(features.get("contract", ""))
    tech_support = str(features.get("tech_support", ""))
    online_security = str(features.get("online_security", ""))

    if probability >= MEDIUM_THRESHOLD and monthly >= 70 and contract == "Month-to-month":
        return RETENTION_ACTIONS["save_offer"]
    if probability >= MEDIUM_THRESHOLD:
        return RETENTION_ACTIONS["success_call"]
    if probability >= LOW_THRESHOLD and (tech_support == "No" or online_security == "No"):
        return RETENTION_ACTIONS["support_bundle"]
    if probability >= LOW_THRESHOLD:
        return RETENTION_ACTIONS["education_nudge"]
    return RETENTION_ACTIONS["monitor"]


def summarize_portfolio(scored_customers: list[dict[str, Any]]) -> dict[str, Any]:
    """Summarize a scored customer portfolio for dashboard cards."""
    total = len(scored_customers)
    risk_counts = {"high": 0, "medium": 0, "low": 0}
    action_counts: dict[str, int] = {}
    revenue_at_risk = 0.0
    expected_net_value = 0.0

    for customer in scored_customers:
        score = customer["score"]
        risk_counts[score["risk_band"]] += 1
        action_counts[score["recommended_action"]] = action_counts.get(score["recommended_action"], 0) + 1
        revenue_at_risk += score["revenue_at_risk"]
        expected_net_value += score["expected_net_value"]

    return {
        "customers_scored": total,
        "risk_counts": risk_counts,
        "high_risk_share": round(risk_counts["high"] / total, 4) if total else 0.0,
        "revenue_at_risk": round(revenue_at_risk, 2),
        "expected_net_value": round(expected_net_value, 2),
        "action_counts": dict(sorted(action_counts.items(), key=lambda item: item[1], reverse=True)),
    }
