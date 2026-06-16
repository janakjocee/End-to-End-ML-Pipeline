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
  "public/demo-pipeline-results.png",
  "public/data-drift-comparison.png",
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

console.log(`Verified web app assets for ${customers.length} scored customers.`);
