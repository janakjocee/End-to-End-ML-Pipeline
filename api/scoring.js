const fs = require("node:fs");
const path = require("node:path");

const bundlePath = path.join(process.cwd(), "public", "model_bundle.json");
const bundle = JSON.parse(fs.readFileSync(bundlePath, "utf8"));

const columnAliases = {
  customer_id: ["customerid", "customer", "id", "accountid", "accountnumber", "clientid", "subscriberid"],
  senior_citizen: ["seniorcitizen", "senior", "is_senior", "seniorflag", "age65plus"],
  tenure: ["tenure", "tenuremonths", "monthsactive", "monthssubscribed", "customerage", "months"],
  monthly_charges: ["monthlycharges", "monthlycharge", "monthlyfee", "monthlyamount", "monthlybill", "mrr", "arpu"],
  total_charges: ["totalcharges", "totalcharge", "lifetimevalue", "ltv", "totalspend", "totalbilled"],
  gender: ["gender", "sex"],
  partner: ["partner", "haspartner", "married", "spouse"],
  dependents: ["dependents", "hasdependents", "children"],
  phone_service: ["phoneservice", "phone", "hasphone", "voice"],
  multiple_lines: ["multiplelines", "multilines", "additionallines", "lines"],
  internet_service: ["internetservice", "internet", "internettype", "broadband", "serviceinternet"],
  online_security: ["onlinesecurity", "security", "cybersecurity", "securityaddon"],
  online_backup: ["onlinebackup", "backup", "cloudbackup"],
  device_protection: ["deviceprotection", "devicecover", "protection", "insurance"],
  tech_support: ["techsupport", "technicalsupport", "support", "premiumsupport"],
  streaming_tv: ["streamingtv", "tvstreaming", "tv"],
  streaming_movies: ["streamingmovies", "moviestreaming", "movies"],
  contract: ["contract", "contracttype", "plan", "plantype", "subscriptiontype", "term"],
  paperless_billing: ["paperlessbilling", "paperless", "ebilling", "digitalbilling"],
  payment_method: ["paymentmethod", "payment", "paymethod", "billingmethod", "methodofpayment"],
};

function canonicalize(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[^a-z0-9]/g, "");
}

function inferColumnMapping(record) {
  const normalizedHeaders = new Map(Object.keys(record || {}).map((header) => [canonicalize(header), header]));
  const mapping = {};
  const defaults = [];

  for (const feature of Object.keys(bundle.schema)) {
    const candidates = [feature, ...(columnAliases[feature] || [])].map(canonicalize);
    const matched = candidates.map((candidate) => normalizedHeaders.get(candidate)).find(Boolean);
    if (matched) {
      mapping[feature] = matched;
    } else {
      defaults.push(feature);
    }
  }

  const customerId = ["customer_id", ...(columnAliases.customer_id || [])]
    .map(canonicalize)
    .map((candidate) => normalizedHeaders.get(candidate))
    .find(Boolean);
  return { mapping, defaults, customerId };
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

function normalizeFeatures(rawFeatures) {
  const features = {};
  for (const [name, spec] of Object.entries(bundle.schema)) {
    const value = rawFeatures[name] ?? spec.default;
    if (spec.type === "numeric") {
      const numeric = Number(value);
      features[name] = Number.isFinite(numeric) ? numeric : Number(spec.default);
    } else {
      features[name] = normalizeCategory(value, spec);
    }
  }
  return features;
}

function coerceMappedFeatures(record, mapping) {
  const features = {};
  for (const [name, spec] of Object.entries(bundle.schema)) {
    const sourceColumn = mapping[name];
    const value = sourceColumn ? record[sourceColumn] : spec.default;
    if (spec.type === "numeric") {
      const numeric = Number(value);
      features[name] = Number.isFinite(numeric) ? numeric : Number(spec.default);
    } else {
      features[name] = normalizeCategory(value, spec);
    }
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
  const inference = inferColumnMapping(records[0]);
  const scored = records.map((record, index) => {
    const customer_id = inference.customerId ? record[inference.customerId] : `uploaded-${index + 1}`;
    const features = coerceMappedFeatures(record, inference.mapping);
    return {
      customer_id,
      features,
      score: scoreCustomer(features),
    };
  });
  scored.sort((a, b) => b.score.revenue_at_risk - a.score.revenue_at_risk);
  return {
    summary: summarizeScores(scored),
    mapping: {
      detected_fields: inference.mapping,
      defaults_used: inference.defaults,
      customer_id_column: inference.customerId || null,
    },
    scored,
  };
}

module.exports = {
  bundle,
  scoreCustomer,
  scoreBatch,
  inferColumnMapping,
};
