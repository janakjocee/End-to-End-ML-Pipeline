import fs from "node:fs";
import path from "node:path";

const root = process.cwd();
const required = [
  "public/index.html",
  "public/app.js",
  "public/styles.css",
  "public/model_bundle.json",
  "public/customer_scores.json",
  "public/business_summary.json",
  "public/model_selection.json",
  "public/threshold_policy.json",
  "public/sample-customers.csv",
  "public/sample-flexible-customers.csv",
  "public/telco-sample.csv",
  "public/scoring-worker.js",
  "public/demo-pipeline-results.png",
  "public/data-drift-comparison.png",
  "public/model-selection-report.png",
  "api/score.js",
  "api/batch-score.js",
  "api/batches.js",
  "api/database-status.js",
  "api/lib/db.js",
  "api/lib/schema.js",
  "api/scoring.js",
  "database/schema.sql",
];

for (const relative of required) {
  const fullPath = path.join(root, relative);
  if (!fs.existsSync(fullPath)) {
    throw new Error(`Missing web asset: ${relative}`);
  }
}

const bundle = JSON.parse(fs.readFileSync(path.join(root, "public/model_bundle.json"), "utf8"));
const customers = JSON.parse(fs.readFileSync(path.join(root, "public/customer_scores.json"), "utf8"));
const summary = JSON.parse(fs.readFileSync(path.join(root, "public/business_summary.json"), "utf8"));

if (!bundle.schema || !bundle.weights || !bundle.metrics) {
  throw new Error("Model bundle is missing schema, weights, or metrics.");
}

if (customers.length < 50) {
  throw new Error("Expected at least 50 scored customers for the dashboard.");
}

if (summary.revenue_at_risk <= 0 || summary.customers_scored !== customers.length) {
  throw new Error("Business summary does not match scored customer artifacts.");
}

const { scoreCustomer, scoreBatch } = await import("../api/scoring.js");
const batchesApi = await import("../api/batches.js");
const databaseStatusApi = await import("../api/database-status.js");
const sample = {
  tenure: 12,
  monthly_charges: 95,
  total_charges: 1140,
  contract: "Month-to-month",
  payment_method: "Electronic check",
  internet_service: "Fiber optic",
  online_security: "No",
  tech_support: "No",
};
const score = scoreCustomer(sample);
if (score.risk_band !== "high" || score.recommended_action !== "Targeted save offer") {
  throw new Error("Single-customer scoring smoke test failed.");
}

const batch = scoreBatch([sample, customers[0].features]);
if (batch.summary.customers_scored !== 2 || batch.scored.length !== 2) {
  throw new Error("Batch scoring smoke test failed.");
}

const flexibleBatch = scoreBatch([
  {
    "Account ID": "ACME-001",
    "Months Active": 7,
    MonthlyCharge: 105.2,
    LTV: 736.4,
    Sex: "Female",
    Plan: "Month-to-month",
    "Payment Method": "Electronic check",
    "Internet Type": "Fiber optic",
    Security: "No",
    Support: "No",
  },
]);
if (
  flexibleBatch.scored[0].customer_id !== "ACME-001" ||
  flexibleBatch.mapping.detected_fields.tenure !== "Months Active" ||
  flexibleBatch.mapping.detected_fields.monthly_charges !== "MonthlyCharge"
) {
  throw new Error("Flexible column mapping smoke test failed.");
}

function mockResponse() {
  return {
    statusCode: 200,
    headers: {},
    payload: null,
    setHeader(key, value) {
      this.headers[key] = value;
    },
    status(code) {
      this.statusCode = code;
      return this;
    },
    json(payload) {
      this.payload = payload;
      return this;
    },
  };
}

const statusResponse = mockResponse();
await databaseStatusApi.default({ method: "GET", headers: {} }, statusResponse);
if (statusResponse.statusCode !== 200 || !["browser", "database"].includes(statusResponse.payload.mode)) {
  throw new Error("Database status API smoke test failed.");
}

const batchesResponse = mockResponse();
await batchesApi.default({ method: "GET", headers: {} }, batchesResponse);
if (batchesResponse.statusCode !== 200 || !Array.isArray(batchesResponse.payload.batches)) {
  throw new Error("Batches API browser-fallback smoke test failed.");
}

console.log(
  `Verified web app assets, dynamic mapping, persistence APIs, single scoring, and batch scoring for ${customers.length} demo customers.`,
);
