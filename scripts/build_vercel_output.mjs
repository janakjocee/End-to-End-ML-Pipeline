import fs from "node:fs";
import path from "node:path";

const root = process.cwd();
const outputRoot = path.join(root, ".vercel", "output");
const staticRoot = path.join(outputRoot, "static");
const functionRoot = path.join(outputRoot, "functions", "api", "score.func");

function copyDirectory(source, destination) {
  fs.mkdirSync(destination, { recursive: true });
  for (const entry of fs.readdirSync(source, { withFileTypes: true })) {
    const sourcePath = path.join(source, entry.name);
    const destinationPath = path.join(destination, entry.name);
    if (entry.isDirectory()) {
      copyDirectory(sourcePath, destinationPath);
    } else {
      fs.copyFileSync(sourcePath, destinationPath);
    }
  }
}

fs.rmSync(outputRoot, { recursive: true, force: true });
copyDirectory(path.join(root, "public"), staticRoot);
fs.mkdirSync(functionRoot, { recursive: true });

const bundle = fs.readFileSync(path.join(root, "public", "model_bundle.json"), "utf8");
fs.writeFileSync(path.join(functionRoot, "model_bundle.json"), bundle);

fs.writeFileSync(
  path.join(functionRoot, ".vc-config.json"),
  JSON.stringify(
    {
      runtime: "nodejs22.x",
      handler: "index.js",
      launcherType: "Nodejs",
      shouldAddHelpers: true,
      maxDuration: 10,
    },
    null,
    2,
  ),
);

fs.writeFileSync(
  path.join(functionRoot, "index.js"),
  String.raw`const fs = require("node:fs");
const path = require("node:path");

const bundle = JSON.parse(fs.readFileSync(path.join(__dirname, "model_bundle.json"), "utf8"));

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

module.exports = (request, response) => {
  if (request.method !== "POST") {
    response.setHeader("Allow", "POST");
    return response.status(405).json({ error: "Use POST with a JSON body." });
  }
  const features = request.body?.features || request.body || {};
  return response.status(200).json(score(features));
};
`,
);

fs.writeFileSync(
  path.join(outputRoot, "config.json"),
  JSON.stringify(
    {
      version: 3,
      routes: [
        {
          src: "^(?:/(.*))$",
          headers: {
            "X-Content-Type-Options": "nosniff",
            "X-Frame-Options": "DENY",
          },
          continue: true,
        },
        { handle: "filesystem" },
      ],
    },
    null,
    2,
  ),
);

console.log("Created Vercel Build Output API package with static assets and /api/score function.");
