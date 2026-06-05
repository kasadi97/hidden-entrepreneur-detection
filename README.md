# Detecting Hidden Commercial Activity in Retail Transaction Data

Mastercard Data Quest, Case Championship — May 2026

An explainable ML pipeline that identifies "hidden entrepreneurs" — sole traders and micro-businesses running commercial activity through ordinary consumer cards — and turns transaction behaviour into ranked B2B sales leads.

---

## Problem

Banks hold transaction data on millions of cardholders. A subset of those "consumers" are actually self-employed business owners paying for cloud services, stock, logistics, and advertising on personal cards — invisible to the B2B sales team.

The goal: identify these hidden entrepreneurs from transaction behaviour alone, score them, and recommend a sales action for each.

---

## Dataset

| File | Size | Description |
| :---- | :---- | :---- |
| `business_cards.parquet` | 25K cards, \~3M transactions | Labelled business cardholders (target \= 1\) |
| `consumer_cards.parquet` | 80K cards, \~9.8M transactions | Labelled consumer cardholders (target \= 0\) |
| `merchants_reference.parquet` | \~2K merchants | MCC codes, merchant country, recurring capability |

Period: 1 October 2025 to 31 March 2026 (6 months). Currency: KZT.

Data is fully synthetic and provided for educational purposes only.

---

## Pipeline

Raw transactions (12.8M rows)

        

1\. Load & Label       merge business \+ consumer, tag target


2\. Clean              fill merchant nulls, drop amount outliers (1st-99th pct, \~2%)


3\. Time Features      parse timestamp to hour, derive weekday/weekend flag


4\. Aggregate          collapse to one row per card (105K cards total)


5\. Feature Engineering   14 behavioural features


6\. Train & Compare    Logistic Regression, Random Forest, Gradient Boosting


7\. Score & Segment    probability 0-1 mapped to HIGH / MEDIUM / LOW tiers

Single random seed (42), stratified 80/20 split, fixed library versions. Fully reproducible end-to-end.

---

## Feature Engineering

Each of the 14 features answers a specific business hypothesis.

| Feature | What it measures | Business hypothesis |
| :---- | :---- | :---- |
| `recurring_ratio` | Share of merchant-initiated recurring charges | Cloud, ads, logistics subscriptions signal B2B activity |
| `tx_mean` / `tx_sum` | Average ticket and total turnover | Businesses buy in bulk, larger spend overall |
| `unique_mcc` / `mcc_hhi` | Category count and Herfindahl concentration | A business focuses on a few categories, not scattered retail |
| `unique_merchants` | Distinct merchants used | Recurring supplier pool rather than random retail |
| `offline_ratio` | Share of in-person transactions | Trade acquiring is offline-heavy |
| `intl_ratio` / `unique_countries` | Foreign transaction share and geography | Importers and cross-border sellers |
| `business_hour_ratio` / `weekend_ratio` | When spending happens | Activity in work hours, quiet on weekends |
| `tx_count` / `active_days` / `tx_std` | Volume, regularity, spread | Operational rhythm vs household noise |

All features are aggregated to card level. The model judges a cardholder's full 6-month behaviour, not individual transactions.

---

## EDA — Key Findings

Business and consumer cards behave very differently:

| Signal | Business (median) | Consumer (median) |
| :---- | :---- | :---- |
| Weekend share | 0.12 | 0.35 |
| Unique categories | 15 | 32 |
| Spend concentration (HHI) | 0.19 | 0.05 |
| Recurring share | 0.13 | \~0 |
| Total turnover | 15.0M KZT | 2.9M KZT |
| Average ticket | 137K KZT | 26K KZT |

The average ticket gap alone is 5x. Combined with weekend rhythm and recurring charges, the separation between the two groups is sharp and consistent.

---

## Modelling

| Model | Test F1 | Selected |
| :---- | :---- | :---- |
| Logistic Regression | 0.989 |  |
| Random Forest (tuned) | 0.997 | Yes |
| Gradient Boosting | 0.997 |  |

Random Forest was selected for its F1, native feature importance, and SHAP compatibility. Tuned with 5-fold GridSearchCV.

| Metric | Score |
| :---- | :---- |
| F1 | 0.997 |
| ROC-AUC | 1.000 |
| PR-AUC | 0.9997 |
| CV F1 (5-fold) | 0.996 |

### Threshold analysis

| Threshold | Precision | Recall | F1 | Use case |
| :---- | :---- | :---- | :---- | :---- |
| 0.30 | 0.994 | 0.999 | 0.997 | Mass-market outreach, maximum coverage |
| 0.50 | 0.996 | 0.997 | 0.997 | Sales team prioritisation (default) |
| 0.70 | 0.998 | 0.994 | 0.996 | Premium cross-sell, highest confidence only |

---

## Explainability

Feature importances were cross-checked with SHAP values. Direction and ranking agree, making the model's logic fully auditable.

Top 3 drivers:

1. `weekend_ratio` (importance 0.285) — the strongest signal. Businesses spend on a work-week rhythm; consumers peak on weekends.  
2. `mcc_hhi` (0.178) — spend concentration. A business funnels money into a few merchant categories; consumers scatter widely.  
3. `recurring_ratio` (0.150) — subscriptions and auto-billing that come with running a business are near-zero for consumers.

---

## Scoring and Segmentation

| Tier | Score | Action | Cards (test set) | Accuracy |
| :---- | :---- | :---- | :---- | :---- |
| HIGH | \>= 0.75 | Personal call from relationship manager, offer business card \+ acquiring | 4,965 | 99.9% |
| MEDIUM | 0.50-0.75 | Targeted email with B2B product offer | 40 | 67.5% |
| LOW | 0.25-0.50 | Monitor, re-score after next 30-day cycle | 31 | 70.9% |
| NONE | \< 0.25 | No action, continue standard consumer journey | 15,964 | 99.97% |

---

## Limitations

- **Cold start** — cards with less than 2-4 weeks of history score unreliably.  
- **Synthetic data** — near-perfect metrics reflect highly separable generated data; expect lower numbers on real-world data.  
- **Validation gap** — the model is trained on known business vs consumer cards; applying it to detect hidden businesses inside the consumer base is an extrapolation that needs A/B validation.  
- **Pattern adaptation** — entrepreneurs may split spend across multiple cards over time, eroding accuracy.  
- **No inbound flows** — incoming transfers from many counterparties would be a strong signal but are not available in this dataset.

Recommended next steps: quarterly retraining, A/B test on HIGH-tier targets vs control group, PSI monitoring on key features.

---

## How to Run

pip install pandas numpy scikit-learn shap matplotlib seaborn pyarrow

\# Place data files in the same directory as the notebook:

\# business\_cards.parquet

\# consumer\_cards.parquet

\# merchants\_reference.parquet

jupyter notebook final-solutionFinal.ipynb

The notebook auto-detects Kaggle vs local environments. Random seed 42 throughout — results are fully reproducible.

---

## Stack

Python, pandas, scikit-learn, SHAP, matplotlib, seaborn, pyarrow

---

