# Model Card: Customer Churn Command Center

## Intended Use

The model estimates churn risk for subscription-style customers and helps customer-success teams prioritize outreach. It is intended for decision support, not fully automated customer treatment.

## Model Details

| Item | Value |
|---|---|
| Model ID | `churn-command-center-logreg-v1` |
| Algorithm | Class-balanced logistic regression |
| Features | Customer tenure, charges, contract, payment method, internet service, support services, and related account attributes |
| Output | Churn probability, risk band, top drivers, revenue at risk, and recommended retention action |
| Deployment bundle | `web/public/model_bundle.json` |

## Verified Metrics

| Metric | Holdout value |
|---|---:|
| Accuracy | 0.672 |
| Precision | 0.322 |
| Recall | 0.716 |
| F1 | 0.444 |
| ROC-AUC | 0.746 |

The model is tuned toward recall because missing a likely churner can be more expensive than sending a low-cost retention nudge.

## Decision Policy

| Risk band | Probability range | Default action |
|---|---:|---|
| Low | `< 0.40` | Monitor only |
| Medium | `0.40 - 0.69` | Education nudge or support bundle |
| High | `>= 0.70` | Customer success call or targeted save offer |

## Limitations

- The included data is synthetic and should not be treated as real telecom data.
- Intervention lift and cost assumptions are illustrative.
- The model should be recalibrated on real labeled outcomes before production use.
- Fairness, consent, and customer-treatment policies must be reviewed before real deployment.

## Monitoring Recommendations

- Track recall, precision, and calibration once ground truth arrives.
- Monitor input drift for monthly charges, contract type, tenure, and service add-ons.
- Compare action assignment volume and realized save rates by segment.
- Trigger retraining only after validating both data drift and business outcome movement.
