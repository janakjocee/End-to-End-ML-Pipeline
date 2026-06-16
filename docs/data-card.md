# Data Card: Synthetic Customer Churn Dataset

## Dataset Summary

The repository generates a deterministic synthetic customer churn dataset for repeatable development, testing, and demonstration. It simulates subscription customers with demographics, service usage, billing, contracts, charges, and churn labels.

## Generation

Run:

```bash
.venv/bin/python -m scripts.run_demo_pipeline
```

The generator creates:

- normal reference data
- a drift scenario with higher monthly charges, more month-to-month contracts, and higher churn rate
- holdout data for evaluation
- scored customer portfolio artifacts for the web dashboard

## Key Fields

| Field group | Examples |
|---|---|
| Identity | `customer_id` |
| Demographic | `gender`, `senior_citizen`, `partner`, `dependents` |
| Account | `tenure`, `contract`, `paperless_billing`, `payment_method` |
| Services | `internet_service`, `online_security`, `tech_support`, `streaming_tv` |
| Commercial | `monthly_charges`, `total_charges` |
| Label | `churn` |

## Drift Scenario

The drift data intentionally shifts:

- monthly charges upward
- contract mix toward month-to-month
- churn rate upward

This gives the monitoring layer something realistic to detect and explain.

## Limitations

- Synthetic data cannot prove production performance.
- Labels are generated from hand-authored probabilities, not observed customer behavior.
- Sensitive attributes are present only for demonstration and should be handled carefully in real deployments.

## Production Data Requirements

Before replacing synthetic data with real customer data, add:

- data ownership and consent review
- PII minimization and masking
- train/validation/test split by time
- feature freshness checks
- leakage checks
- segment-level performance review
- outcome feedback loop for retention actions
