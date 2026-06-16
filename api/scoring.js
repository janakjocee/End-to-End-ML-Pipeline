const fs = require("node:fs");
const path = require("node:path");

const bundlePath = path.join(process.cwd(), "public", "model_bundle.json");
const bundle = JSON.parse(fs.readFileSync(bundlePath, "utf8"));

function sigmoid(value) {
  if (value >= 0) {
    const z = Math.exp(-value);
    return 1 / (1 + z);
  }
  const z = Math.exp(value);
  return z / (1 + z);
}

function riskBand(probability) {
  if (probability >= 0.7) return "high";
  if (probability >= 0.4) return "medium";
  return "low";
}

function chooseAction(features, probability) {
  const monthly = Number(features.monthly_charges || 0);
  if (probability >= 0.7 && monthly >= 70 && features.contract === "Month-to-month") {
    return ["Targeted save offer", "high", 55, 0.28];
  }
  if (probability >= 0.7) return ["Customer success call", "high", 22, 0.18];
  if (probability >= 0.4 && (features.tech_support === "No" || features.online_security === "No")) {
    return ["Support bundle", "medium", 14, 0.12];
  }
  if (probability >= 0.4) return ["Education nudge", "medium", 3, 0.05];
  return ["Monitor only", "low", 0, 0];
}

function normalizeFeatures(rawFeatures) {
  const features = {};
  for (const [name, spec] of Object.entries(bundle.schema)) {
    const value = rawFeatures[name] ?? spec.default;
    features[name] = spec.type === "numeric" ? Number(value) : String(value);
  }
  return features;
}

function scoreCustomer(rawFeatures) {
  const features = normalizeFeatures(rawFeatures);
  let logit = bundle.weights.intercept;
  const drivers = [];
  Object.entries(bundle.schema).forEach(([name, spec]) => {
    const value = features[name] ?? spec.default;
    let contribution = 0;
    if (spec.type === "numeric") {
      const scaled = spec.scale ? (Number(value) - spec.mean) / spec.scale : 0;
      contribution = scaled * (bundle.weights.numeric[name] || 0);
    } else {
      contribution = bundle.weights.categorical[name]?.[String(value)] || 0;
    }
    logit += contribution;
    drivers.push({ feature: name, value, contribution });
  });
  const probability = sigmoid(logit);
  const [action, priority, cost, lift] = chooseAction(features, probability);
  const annualRevenue = Number(features.monthly_charges || 0) * 12;
  const revenueAtRisk = annualRevenue * probability;
  return {
    churn_probability: Number(probability.toFixed(4)),
    risk_band: riskBand(probability),
    recommended_action: action,
    action_priority: priority,
    estimated_action_cost: cost,
    expected_save_lift: lift,
    annual_revenue: Number(annualRevenue.toFixed(2)),
    revenue_at_risk: Number(revenueAtRisk.toFixed(2)),
    expected_net_value: Number((revenueAtRisk * lift - cost).toFixed(2)),
    top_drivers: drivers.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)).slice(0, 5),
  };
}

function summarizeScores(scoredRows) {
  const risk_counts = { high: 0, medium: 0, low: 0 };
  const action_counts = {};
  let revenue_at_risk = 0;
  let expected_net_value = 0;

  for (const row of scoredRows) {
    const score = row.score;
    risk_counts[score.risk_band] += 1;
    action_counts[score.recommended_action] = (action_counts[score.recommended_action] || 0) + 1;
    revenue_at_risk += score.revenue_at_risk;
    expected_net_value += score.expected_net_value;
  }

  return {
    customers_scored: scoredRows.length,
    risk_counts,
    high_risk_share: scoredRows.length ? Number((risk_counts.high / scoredRows.length).toFixed(4)) : 0,
    revenue_at_risk: Number(revenue_at_risk.toFixed(2)),
    expected_net_value: Number(expected_net_value.toFixed(2)),
    action_counts,
  };
}

function scoreBatch(records) {
  const scored = records.map((record, index) => {
    const customer_id = record.customer_id || `uploaded-${index + 1}`;
    const features = normalizeFeatures(record);
    return {
      customer_id,
      features,
      score: scoreCustomer(features),
    };
  });
  scored.sort((a, b) => b.score.revenue_at_risk - a.score.revenue_at_risk);
  return {
    summary: summarizeScores(scored),
    scored,
  };
}

module.exports = {
  bundle,
  scoreCustomer,
  scoreBatch,
};
