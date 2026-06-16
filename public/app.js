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
const state = {
  bundle: null,
  customers: [],
  summary: null,
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
  renderForm();
  renderTable();
  updateScore();
}

load().catch((error) => {
  document.body.insertAdjacentHTML(
    "afterbegin",
    `<div class="error">Could not load generated artifacts: ${error.message}</div>`,
  );
});
