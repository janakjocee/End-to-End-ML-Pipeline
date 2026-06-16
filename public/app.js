const money = new Intl.NumberFormat("en-US", {
  style: "currency",
  currency: "USD",
  maximumFractionDigits: 0,
});

const percent = new Intl.NumberFormat("en-US", {
  style: "percent",
  maximumFractionDigits: 1,
});

const form = document.querySelector("#score-form");
const upload = document.querySelector("#csv-upload");
const uploadStatus = document.querySelector("#upload-status");
const uploadResults = document.querySelector("#upload-results");
const uploadedSummary = document.querySelector("#uploaded-summary");
const downloadButton = document.querySelector("#download-scored");
const state = {
  bundle: null,
  customers: [],
  summary: null,
  uploaded: [],
};

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
    return {
      name: "Targeted save offer",
      priority: "high",
      cost: 55,
      lift: 0.28,
      description: "Offer a tailored discount or plan credit to a high-value flexible-contract customer.",
    };
  }
  if (probability >= 0.7) {
    return {
      name: "Customer success call",
      priority: "high",
      cost: 22,
      lift: 0.18,
      description: "Route the customer to a retention specialist for plan review.",
    };
  }
  if (probability >= 0.4 && (features.tech_support === "No" || features.online_security === "No")) {
    return {
      name: "Support bundle",
      priority: "medium",
      cost: 14,
      lift: 0.12,
      description: "Bundle technical support or online security for customers lacking service add-ons.",
    };
  }
  if (probability >= 0.4) {
    return {
      name: "Education nudge",
      priority: "medium",
      cost: 3,
      lift: 0.05,
      description: "Send onboarding, product education, and billing-transparency messaging.",
    };
  }
  return {
    name: "Monitor only",
    priority: "low",
    cost: 0,
    lift: 0,
    description: "No intervention needed now; keep the account in the next scoring cycle.",
  };
}

export function scoreCustomer(bundle, features) {
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
  const action = chooseAction(features, probability);
  const annualRevenue = Number(features.monthly_charges || 0) * 12;
  const revenueAtRisk = annualRevenue * probability;
  const expectedSaved = revenueAtRisk * action.lift;
  return {
    churn_probability: Number(probability.toFixed(4)),
    risk_band: riskBand(probability),
    recommended_action: action.name,
    action_priority: action.priority,
    action_description: action.description,
    estimated_action_cost: action.cost,
    expected_save_lift: action.lift,
    annual_revenue: Number(annualRevenue.toFixed(2)),
    revenue_at_risk: Number(revenueAtRisk.toFixed(2)),
    expected_saved_revenue: Number(expectedSaved.toFixed(2)),
    expected_net_value: Number((expectedSaved - action.cost).toFixed(2)),
    top_drivers: drivers.sort((a, b) => Math.abs(b.contribution) - Math.abs(a.contribution)).slice(0, 5),
  };
}

function summarizeRows(rows) {
  const riskCounts = { high: 0, medium: 0, low: 0 };
  let revenueAtRisk = 0;
  let expectedNetValue = 0;

  rows.forEach((row) => {
    riskCounts[row.score.risk_band] += 1;
    revenueAtRisk += row.score.revenue_at_risk;
    expectedNetValue += row.score.expected_net_value;
  });

  return {
    customers_scored: rows.length,
    risk_counts: riskCounts,
    high_risk_share: rows.length ? riskCounts.high / rows.length : 0,
    revenue_at_risk: revenueAtRisk,
    expected_net_value: expectedNetValue,
  };
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let cell = "";
  let quoted = false;

  for (let index = 0; index < text.length; index += 1) {
    const char = text[index];
    const next = text[index + 1];
    if (char === '"' && quoted && next === '"') {
      cell += '"';
      index += 1;
    } else if (char === '"') {
      quoted = !quoted;
    } else if (char === "," && !quoted) {
      row.push(cell.trim());
      cell = "";
    } else if ((char === "\n" || char === "\r") && !quoted) {
      if (char === "\r" && next === "\n") index += 1;
      row.push(cell.trim());
      if (row.some((value) => value !== "")) rows.push(row);
      row = [];
      cell = "";
    } else {
      cell += char;
    }
  }
  row.push(cell.trim());
  if (row.some((value) => value !== "")) rows.push(row);

  if (rows.length < 2) {
    throw new Error("CSV needs a header row and at least one customer row.");
  }

  const headers = rows[0].map((header) => header.trim());
  return rows.slice(1).map((values) =>
    Object.fromEntries(headers.map((header, index) => [header, values[index] ?? ""])),
  );
}

function validateRecords(records) {
  const required = Object.keys(state.bundle.schema);
  const present = new Set(Object.keys(records[0] || {}));
  const missing = required.filter((name) => !present.has(name));
  if (missing.length) {
    throw new Error(`Missing required columns: ${missing.join(", ")}`);
  }

  const errors = [];
  records.forEach((record, rowIndex) => {
    required.forEach((name) => {
      const spec = state.bundle.schema[name];
      const value = record[name];
      if (spec.type === "numeric" && Number.isNaN(Number(value))) {
        errors.push(`row ${rowIndex + 2}: ${name} must be numeric`);
      }
      if (spec.type === "categorical" && !spec.values.includes(String(value))) {
        errors.push(`row ${rowIndex + 2}: ${name} must be one of ${spec.values.join(" / ")}`);
      }
    });
  });

  if (errors.length) {
    throw new Error(errors.slice(0, 5).join("; ") + (errors.length > 5 ? `; +${errors.length - 5} more` : ""));
  }
}

function scoreUploadedRecords(records) {
  return records
    .map((record, index) => {
      const features = {};
      Object.entries(state.bundle.schema).forEach(([name, spec]) => {
        features[name] = spec.type === "numeric" ? Number(record[name]) : String(record[name]);
      });
      const score = scoreCustomer(state.bundle, features);
      return {
        customer_id: record.customer_id || `uploaded-${index + 1}`,
        features,
        score,
      };
    })
    .sort((a, b) => b.score.revenue_at_risk - a.score.revenue_at_risk);
}

function renderUploadSummary(rows) {
  const summary = summarizeRows(rows);
  uploadedSummary.classList.remove("hidden");
  document.querySelector("#upload-rows").textContent = summary.customers_scored.toLocaleString();
  document.querySelector("#upload-high-risk").textContent = percent.format(summary.high_risk_share);
  document.querySelector("#upload-rar").textContent = money.format(summary.revenue_at_risk);
  document.querySelector("#upload-net").textContent = money.format(summary.expected_net_value);
}

function renderUploadedTable() {
  document.querySelector("#uploaded-table").innerHTML = state.uploaded
    .slice(0, 25)
    .map(
      (customer) => `
        <tr>
          <td>${customer.customer_id}</td>
          <td><span class="badge ${customer.score.risk_band}">${customer.score.risk_band}</span></td>
          <td>${percent.format(customer.score.churn_probability)}</td>
          <td>${money.format(customer.score.revenue_at_risk)}</td>
          <td>${money.format(customer.score.expected_net_value)}</td>
          <td>${customer.score.recommended_action}</td>
        </tr>
      `,
    )
    .join("");
}

function escapeCsv(value) {
  const text = String(value ?? "");
  return /[",\n\r]/.test(text) ? `"${text.replaceAll('"', '""')}"` : text;
}

function downloadScoredCsv() {
  const headers = [
    "customer_id",
    ...Object.keys(state.bundle.schema),
    "churn_probability",
    "risk_band",
    "revenue_at_risk",
    "expected_net_value",
    "recommended_action",
    "top_driver_1",
    "top_driver_2",
    "top_driver_3",
  ];
  const lines = [
    headers.join(","),
    ...state.uploaded.map((row) => {
      const drivers = row.score.top_drivers.slice(0, 3).map((driver) => driver.feature);
      const values = [
        row.customer_id,
        ...Object.keys(state.bundle.schema).map((name) => row.features[name]),
        row.score.churn_probability,
        row.score.risk_band,
        row.score.revenue_at_risk,
        row.score.expected_net_value,
        row.score.recommended_action,
        drivers[0] || "",
        drivers[1] || "",
        drivers[2] || "",
      ];
      return values.map(escapeCsv).join(",");
    }),
  ];
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "scored-churn-action-queue.csv";
  link.click();
  URL.revokeObjectURL(url);
}

async function handleUpload(event) {
  const file = event.target.files[0];
  if (!file) return;

  try {
    uploadStatus.className = "status-message";
    uploadStatus.textContent = `Reading ${file.name}...`;
    const records = parseCsv(await file.text());
    validateRecords(records);
    state.uploaded = scoreUploadedRecords(records);
    uploadStatus.className = "status-message success";
    uploadStatus.textContent =
      `Scored ${state.uploaded.length.toLocaleString()} rows. Showing the top 25 accounts by revenue at risk.`;
    renderUploadSummary(state.uploaded);
    renderUploadedTable();
    uploadResults.classList.remove("hidden");
  } catch (error) {
    state.uploaded = [];
    uploadedSummary.classList.add("hidden");
    uploadResults.classList.add("hidden");
    uploadStatus.className = "status-message error-text";
    uploadStatus.textContent = error.message;
  }
}

function readForm() {
  const data = new FormData(form);
  const features = {};
  Object.entries(state.bundle.schema).forEach(([name, spec]) => {
    const value = data.get(name);
    features[name] = spec.type === "numeric" ? Number(value) : value;
  });
  return features;
}

function renderForm() {
  form.innerHTML = "";
  Object.entries(state.bundle.schema).forEach(([name, spec]) => {
    const label = document.createElement("label");
    label.textContent = name.replaceAll("_", " ");
    let control;
    if (spec.type === "numeric") {
      control = document.createElement("input");
      control.type = "number";
      control.name = name;
      control.value = spec.default;
      control.min = spec.min;
      control.max = spec.max;
      control.step = name.includes("charges") ? "0.01" : "1";
    } else {
      control = document.createElement("select");
      control.name = name;
      spec.values.forEach((value) => {
        const option = document.createElement("option");
        option.value = value;
        option.textContent = value;
        option.selected = value === spec.default;
        control.append(option);
      });
    }
    label.append(control);
    form.append(label);
  });
  form.addEventListener("input", updateScore);
}

function updateScore() {
  const result = scoreCustomer(state.bundle, readForm());
  const band = document.querySelector("#risk-band");
  band.textContent = `${result.risk_band} risk`;
  band.className = `badge ${result.risk_band}`;
  document.querySelector("#score-probability").textContent =
    `${percent.format(result.churn_probability)} churn probability`;
  document.querySelector("#score-action").textContent =
    `${result.recommended_action}: ${result.action_description}`;
  document.querySelector("#score-rar").textContent = money.format(result.revenue_at_risk);
  document.querySelector("#score-net").textContent = money.format(result.expected_net_value);
  document.querySelector("#driver-list").innerHTML = result.top_drivers
    .map(
      (driver) =>
        `<li><strong>${driver.feature.replaceAll("_", " ")}</strong> = ${driver.value} (${driver.direction}, ${driver.contribution.toFixed(3)})</li>`,
    )
    .join("");
}

function renderSummary() {
  const { summary } = state;
  document.querySelector("#hero-auc").textContent = state.bundle.metrics.test.roc_auc.toFixed(3);
  document.querySelector("#customers-scored").textContent = summary.customers_scored.toLocaleString();
  document.querySelector("#high-risk-share").textContent = percent.format(summary.high_risk_share);
  document.querySelector("#revenue-at-risk").textContent = money.format(summary.revenue_at_risk);
  document.querySelector("#expected-net-value").textContent = money.format(summary.expected_net_value);
}

function renderSchemaHelp() {
  const required = Object.entries(state.bundle.schema).map(([name, spec]) => {
    if (spec.type === "numeric") return `${name} (number)`;
    return `${name} (${spec.values.join(" | ")})`;
  });
  document.querySelector("#schema-columns").textContent = `customer_id optional, then ${required.join(", ")}`;
}

function renderTable() {
  document.querySelector("#customer-table").innerHTML = state.customers
    .slice(0, 12)
    .map(
      (customer) => `
        <tr>
          <td>${customer.customer_id}</td>
          <td><span class="badge ${customer.score.risk_band}">${customer.score.risk_band}</span></td>
          <td>${percent.format(customer.score.churn_probability)}</td>
          <td>${money.format(customer.score.revenue_at_risk)}</td>
          <td>${customer.score.recommended_action}</td>
        </tr>
      `,
    )
    .join("");
}

async function load() {
  const [bundle, customers, summary] = await Promise.all([
    fetch("/model_bundle.json").then((response) => response.json()),
    fetch("/customer_scores.json").then((response) => response.json()),
    fetch("/business_summary.json").then((response) => response.json()),
  ]);
  state.bundle = bundle;
  state.customers = customers;
  state.summary = summary;
  renderSummary();
  renderSchemaHelp();
  renderForm();
  renderTable();
  updateScore();
  upload.addEventListener("change", handleUpload);
  downloadButton.addEventListener("click", downloadScoredCsv);
}

load().catch((error) => {
  document.body.insertAdjacentHTML(
    "afterbegin",
    `<div class="error">Could not load generated artifacts: ${error.message}</div>`,
  );
});
