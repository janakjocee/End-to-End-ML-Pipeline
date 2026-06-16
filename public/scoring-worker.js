self.addEventListener("message", (event) => {
  const { bundle, records, inference } = event.data;
  const scored = records
    .map((record, index) => {
      const features = {};
      Object.entries(bundle.schema).forEach(([name, spec]) => {
        const sourceColumn = inference.mapping[name];
        const rawValue = sourceColumn ? record[sourceColumn] : spec.default;
        if (spec.type === "numeric") {
          const numeric = Number(rawValue);
          features[name] = Number.isFinite(numeric) ? numeric : Number(spec.default);
        } else {
          features[name] = normalizeCategory(rawValue, spec);
        }
      });
      return {
        customer_id: inference.customerId ? record[inference.customerId] : `uploaded-${index + 1}`,
        features,
        score: scoreCustomer(bundle, features),
        action_status: "Not started",
        outcome: "Pending",
      };
    })
    .sort((a, b) => b.score.revenue_at_risk - a.score.revenue_at_risk);
  self.postMessage({ scored });
});

function canonicalize(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "");
}

function normalizeCategory(value, spec) {
  const text = String(value ?? "").trim();
  if (!text) return spec.default;
  const exact = spec.values.find((candidate) => candidate.toLowerCase() === text.toLowerCase());
  if (exact) return exact;
  const compact = canonicalize(text);
  const loose = spec.values.find((candidate) => canonicalize(candidate) === compact);
  if (loose) return loose;
  if (["true", "1", "y"].includes(compact) && spec.values.includes("Yes")) return "Yes";
  if (["false", "0", "n"].includes(compact) && spec.values.includes("No")) return "No";
  return spec.default;
}

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

function scoreCustomer(bundle, features) {
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
    drivers.push({
      feature: name,
      value,
      contribution,
      direction: contribution > 0 ? "raises risk" : "lowers risk",
    });
  });
  const probability = sigmoid(logit);
  const [action, priority, cost, lift] = chooseAction(features, probability);
  const annualRevenue = Number(features.monthly_charges || 0) * 12;
  const revenueAtRisk = annualRevenue * probability;
  const expectedSaved = revenueAtRisk * lift;
  return {
    churn_probability: Number(probability.toFixed(4)),
    risk_band: riskBand(probability),
    recommended_action: action,
    action_priority: priority,
    estimated_action_cost: cost,
    expected_save_lift: lift,
    annual_revenue: Number(annualRevenue.toFixed(2)),
    revenue_at_risk: Number(revenueAtRisk.toFixed(2)),
    expected_saved_revenue: Number(expectedSaved.toFixed(2)),
    expected_net_value: Number((expectedSaved - cost).toFixed(2)),
    top_drivers: drivers.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)).slice(0, 5),
  };
}
