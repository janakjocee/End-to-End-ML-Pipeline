import fs from "node:fs";
import path from "node:path";

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

function score(features) {
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

export default async function handler(request) {
  if (request.method !== "POST") {
    return Response.json({ error: "Use POST with a JSON body." }, { status: 405, headers: { Allow: "POST" } });
  }
  const body = await request.json().catch(() => ({}));
  const features = body.features || body || {};
  return Response.json(score(features));
}
