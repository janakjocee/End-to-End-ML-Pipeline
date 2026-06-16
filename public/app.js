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
const downloadCrmButton = document.querySelector("#download-crm");
const mappingReport = document.querySelector("#mapping-report");
const monitoringReport = document.querySelector("#monitoring-report");
const historyTable = document.querySelector("#history-table");
const clearHistoryButton = document.querySelector("#clear-history");
const state = {
  bundle: null,
  customers: [],
  summary: null,
  uploaded: [],
  rawRecords: [],
  inference: null,
  history: [],
};

const historyKey = "churn-command-center:batches";

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
  multiple_lines: ["multiplelines", "multilines", "additionalLines", "lines"],
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

function inferColumnMapping(headers) {
  const normalizedHeaders = new Map(headers.map((header) => [canonicalize(header), header]));
  const mapping = {};
  const defaults = [];

  Object.keys(state.bundle.schema).forEach((feature) => {
    const candidates = [feature, ...(columnAliases[feature] || [])].map(canonicalize);
    const matched = candidates.map((candidate) => normalizedHeaders.get(candidate)).find(Boolean);
    if (matched) {
      mapping[feature] = matched;
    } else {
      defaults.push(feature);
    }
  });

  const idCandidates = (columnAliases.customer_id || []).map(canonicalize);
  const customerId = ["customer_id", ...idCandidates].map(canonicalize).map((candidate) => normalizedHeaders.get(candidate)).find(Boolean);
  return { mapping, defaults, customerId };
}

function coerceRecord(record, mapping) {
  const features = {};
  Object.entries(state.bundle.schema).forEach(([name, spec]) => {
    const sourceColumn = mapping[name];
    const rawValue = sourceColumn ? record[sourceColumn] : "";
    if (spec.type === "numeric") {
      const numeric = Number(rawValue);
      features[name] = Number.isFinite(numeric) ? numeric : Number(spec.default);
    } else {
      features[name] = normalizeCategory(rawValue, spec);
    }
  });
  return features;
}

function scoreUploadedRecords(records, inference) {
  return records
    .map((record, index) => {
      const features = coerceRecord(record, inference.mapping);
      const score = scoreCustomer(state.bundle, features);
      return {
        customer_id: inference.customerId ? record[inference.customerId] : `uploaded-${index + 1}`,
        features,
        score,
        action_status: "Not started",
        outcome: "Pending",
      };
    })
    .sort((a, b) => b.score.revenue_at_risk - a.score.revenue_at_risk);
}

function scoreUploadedRecordsAsync(records, inference) {
  if (!window.Worker || records.length < 100) {
    return Promise.resolve(scoreUploadedRecords(records, inference));
  }
  return new Promise((resolve, reject) => {
    const worker = new Worker("/scoring-worker.js");
    worker.onmessage = (event) => {
      worker.terminate();
      resolve(event.data.scored);
    };
    worker.onerror = (error) => {
      worker.terminate();
      reject(error);
    };
    worker.postMessage({ bundle: state.bundle, records, inference });
  });
}

function renderMappingReport(inference) {
  const headers = Object.keys(state.rawRecords[0] || {});
  const mapped = Object.entries(inference.mapping).map(([feature, column]) => `${feature} ← ${column}`);
  const defaults = inference.defaults.map((feature) => `${feature}=${state.bundle.schema[feature].default}`);
  mappingReport.classList.remove("hidden");
  mappingReport.innerHTML = `
    <div><strong>Column mapping:</strong> ${mapped.length} model fields detected automatically.</div>
    <div class="mapping-chips">${mapped.slice(0, 12).map((item) => `<span class="mapping-chip">${item}</span>`).join("")}</div>
    ${
      defaults.length
        ? `<div><strong>Defaults used:</strong> ${defaults.slice(0, 10).join(", ")}${defaults.length > 10 ? `, +${defaults.length - 10} more` : ""}</div>`
        : "<div><strong>Defaults used:</strong> none</div>"
    }
    <details>
      <summary>Edit mapping manually</summary>
      <div class="mapping-editor">
        ${Object.keys(state.bundle.schema)
          .map(
            (feature) => `
              <label>${feature.replaceAll("_", " ")}
                <select data-map-feature="${feature}">
                  <option value="">Use default (${state.bundle.schema[feature].default})</option>
                  ${headers
                    .map(
                      (header) =>
                        `<option value="${header}" ${inference.mapping[feature] === header ? "selected" : ""}>${header}</option>`,
                    )
                    .join("")}
                </select>
              </label>
            `,
          )
          .join("")}
      </div>
      <button id="apply-mapping" class="button secondary" type="button">Apply mapping</button>
    </details>
  `;
  document.querySelector("#apply-mapping")?.addEventListener("click", applyManualMapping);
}

async function applyManualMapping() {
  const mapping = {};
  mappingReport.querySelectorAll("[data-map-feature]").forEach((select) => {
    if (select.value) mapping[select.dataset.mapFeature] = select.value;
  });
  state.inference = {
    ...state.inference,
    mapping,
    defaults: Object.keys(state.bundle.schema).filter((feature) => !mapping[feature]),
  };
  uploadStatus.className = "status-message";
  uploadStatus.textContent = "Re-scoring with manual mapping...";
  state.uploaded = await scoreUploadedRecordsAsync(state.rawRecords, state.inference);
  uploadStatus.className = "status-message success";
  uploadStatus.textContent = `Re-scored ${state.uploaded.length.toLocaleString()} rows with your mapping.`;
  renderMappingReport(state.inference);
  renderMonitoringReport();
  renderUploadSummary(state.uploaded);
  renderUploadedTable();
  saveCurrentBatch("Manual mapping update");
}

function renderMonitoringReport() {
  if (!state.uploaded.length) return;
  const alerts = [];
  Object.entries(state.bundle.schema)
    .filter(([, spec]) => spec.type === "numeric")
    .forEach(([feature, spec]) => {
      const values = state.uploaded.map((row) => Number(row.features[feature])).filter(Number.isFinite);
      const mean = values.reduce((sum, value) => sum + value, 0) / Math.max(values.length, 1);
      const z = spec.scale ? Math.abs((mean - spec.mean) / spec.scale) : 0;
      if (z >= 0.75) alerts.push(`${feature} shifted ${z.toFixed(2)} standard deviations`);
    });
  const summary = summarizeRows(state.uploaded);
  if (Math.abs(summary.high_risk_share - state.summary.high_risk_share) >= 0.2) {
    alerts.push(`high-risk share changed from ${percent.format(state.summary.high_risk_share)} to ${percent.format(summary.high_risk_share)}`);
  }
  monitoringReport.classList.remove("hidden");
  monitoringReport.innerHTML = `
    <div><strong>Monitoring:</strong> ${alerts.length ? "review recommended before acting" : "uploaded file looks close to training profile"}</div>
    <div class="mapping-chips">
      <span class="mapping-chip">Prediction mix: ${summary.risk_counts.high} high / ${summary.risk_counts.medium} medium / ${summary.risk_counts.low} low</span>
      ${alerts.slice(0, 4).map((alert) => `<span class="mapping-chip">${alert}</span>`).join("")}
    </div>
  `;
}

function renderUploadSummary(rows) {
  const summary = summarizeRows(rows);
  uploadedSummary.classList.remove("hidden");
  document.querySelector("#upload-rows").textContent = summary.customers_scored.toLocaleString();
  document.querySelector("#upload-high-risk").textContent = percent.format(summary.high_risk_share);
  document.querySelector("#upload-rar").textContent = money.format(summary.revenue_at_risk);
  document.querySelector("#upload-net").textContent = money.format(summary.expected_net_value);
}

function loadHistory() {
  state.history = JSON.parse(localStorage.getItem(historyKey) || "[]");
}

function persistHistory() {
  localStorage.setItem(historyKey, JSON.stringify(state.history.slice(0, 15)));
}

function saveCurrentBatch(label) {
  if (!state.uploaded.length) return;
  const summary = summarizeRows(state.uploaded);
  const batch = {
    id: crypto.randomUUID ? crypto.randomUUID() : String(Date.now()),
    label,
    created_at: new Date().toISOString(),
    summary,
    rows: state.uploaded.slice(0, 100),
    outcome: "Pending review",
  };
  state.history = [batch, ...state.history].slice(0, 15);
  persistHistory();
  renderHistory();
}

function renderHistory() {
  historyTable.innerHTML = state.history
    .map(
      (batch) => `
        <tr>
          <td>${batch.label}</td>
          <td>${batch.summary.customers_scored.toLocaleString()}</td>
          <td>${percent.format(batch.summary.high_risk_share)}</td>
          <td>${money.format(batch.summary.revenue_at_risk)}</td>
          <td>${new Date(batch.created_at).toLocaleString()}</td>
          <td>
            <select data-history-outcome="${batch.id}">
              ${["Pending review", "Campaign launched", "Customers contacted", "Churn reduced", "No improvement"]
                .map((value) => `<option value="${value}" ${batch.outcome === value ? "selected" : ""}>${value}</option>`)
                .join("")}
            </select>
          </td>
        </tr>
      `,
    )
    .join("");
  historyTable.querySelectorAll("[data-history-outcome]").forEach((select) => {
    select.addEventListener("change", () => {
      const batch = state.history.find((item) => item.id === select.dataset.historyOutcome);
      if (batch) batch.outcome = select.value;
      persistHistory();
    });
  });
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
          <td>
            <strong>${customer.score.recommended_action}</strong>
            <select data-status="${customer.customer_id}">
              ${["Not started", "Queued", "Contacted", "Offer sent", "Retained", "Lost"]
                .map((status) => `<option value="${status}" ${customer.action_status === status ? "selected" : ""}>${status}</option>`)
                .join("")}
            </select>
          </td>
        </tr>
      `,
    )
    .join("");
  document.querySelector("#uploaded-table").querySelectorAll("[data-status]").forEach((select) => {
    select.addEventListener("change", () => {
      const row = state.uploaded.find((item) => item.customer_id === select.dataset.status);
      if (row) row.action_status = select.value;
      saveCurrentBatch("Action status update");
    });
  });
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

function downloadCrmCsv() {
  const headers = ["task_name", "customer_id", "priority", "action", "revenue_at_risk", "status", "notes"];
  const lines = [
    headers.join(","),
    ...state.uploaded.map((row) =>
      [
        `Call ${row.customer_id} about churn risk`,
        row.customer_id,
        row.score.action_priority,
        row.score.recommended_action,
        row.score.revenue_at_risk,
        row.action_status,
        `Top drivers: ${row.score.top_drivers.map((driver) => driver.feature).join(" | ")}`,
      ]
        .map(escapeCsv)
        .join(","),
    ),
  ];
  const blob = new Blob([lines.join("\n")], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const link = document.createElement("a");
  link.href = url;
  link.download = "crm-retention-tasks.csv";
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
    state.rawRecords = records;
    const inference = inferColumnMapping(Object.keys(records[0]));
    state.inference = inference;
    uploadStatus.textContent = `Scoring ${records.length.toLocaleString()} rows...`;
    state.uploaded = await scoreUploadedRecordsAsync(records, inference);
    uploadStatus.className = "status-message success";
    uploadStatus.textContent =
      `Scored ${state.uploaded.length.toLocaleString()} rows with ${Object.keys(inference.mapping).length} detected fields and ${inference.defaults.length} defaults. Showing the top 25 accounts by revenue at risk.`;
    renderMappingReport(inference);
    renderMonitoringReport();
    renderUploadSummary(state.uploaded);
    renderUploadedTable();
    uploadResults.classList.remove("hidden");
    saveCurrentBatch(file.name);
  } catch (error) {
    state.uploaded = [];
    uploadedSummary.classList.add("hidden");
    uploadResults.classList.add("hidden");
    mappingReport.classList.add("hidden");
    monitoringReport.classList.add("hidden");
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
  document.querySelector("#schema-columns").textContent =
    `Best results include: ${required.join(", ")}. Missing fields are filled with model defaults; similar column names are auto-detected and editable.`;
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
  loadHistory();
  renderHistory();
  updateScore();
  upload.addEventListener("change", handleUpload);
  downloadButton.addEventListener("click", downloadScoredCsv);
  downloadCrmButton.addEventListener("click", downloadCrmCsv);
  clearHistoryButton.addEventListener("click", () => {
    state.history = [];
    persistHistory();
    renderHistory();
  });
}

load().catch((error) => {
  document.body.insertAdjacentHTML(
    "afterbegin",
    `<div class="error">Could not load generated artifacts: ${error.message}</div>`,
  );
});
