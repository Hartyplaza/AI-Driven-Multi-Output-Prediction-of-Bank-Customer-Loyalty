# AI-Driven Multi-Output Prediction of Bank Customer Loyalty
## A Comparative Study with PLS-SEM

<div align="center">

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square&logo=python)
![XGBoost](https://img.shields.io/badge/XGBoost-3.1-orange?style=flat-square)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.6-blue?style=flat-square)
![Macro F1](https://img.shields.io/badge/Macro%20F1-0.555-brightgreen?style=flat-square)
![R²](https://img.shields.io/badge/SEM%20R²-0.67-brightgreen?style=flat-square)

**Predicting multidimensional customer loyalty using ML and structural equation modelling**

[LinkedIn](https://linkedin.com/in/hart-ofigwe) · [Portfolio](https://hartyplaza.github.io) · [GitHub](https://github.com/Hartyplaza)

</div>

---

## Overview

This study predicts three dimensions of bank customer loyalty simultaneously — intention to remain (LOY_1), recommendation intent (LOY_2), and repurchase likelihood (LOY_3) — from a German bank customer survey (n=675, 41 variables).

Four ML algorithms are benchmarked with and without SMOTE oversampling. Results are then compared against a PLS-SEM structural path model to identify where data-driven and theory-driven approaches converge and diverge on what drives customer loyalty.

---

## Dataset

| Property | Value |
|----------|-------|
| Source | German retail bank customer survey |
| Sample size | 675 valid responses |
| Variables | 41 (Likert 1-7 scale) |
| Constructs | Quality, Performance, CSR, Attractiveness, Likeability, Competence, Satisfaction, Trust, Loyalty |
| Target | LOY_1, LOY_2, LOY_3 binned → Low / Neutral / High |
| Class imbalance | 10.9:1 (High:Low) |
| Missing values | 3 (CSOR_5, 0.44%) — median imputed |

---

## Results

### ML Performance — Aggregated Loyalty (SMOTE)

| Model | Accuracy | Macro F1 | Weighted F1 |
|-------|----------|----------|-------------|
| Baseline (Dummy) | 0.770 | 0.290 | 0.681 |
| Random Forest + SMOTE | 0.756 | 0.532 | 0.750 |
| **XGBoost + SMOTE** | **0.756** | **0.555** | **0.752** |
| MLP + SMOTE | 0.667 | 0.459 | 0.670 |

> XGBoost + SMOTE is the best model — highest Macro F1 and best balance across all loyalty classes.

### ML Performance — Multi-Label (SMOTE vs No SMOTE)

| Model | Macro F1 (No SMOTE) | Macro F1 (SMOTE) | Δ |
|-------|---------------------|------------------|---|
| Random Forest | 0.504 | 0.513 | +0.009 |
| XGBoost | 0.533 | 0.539 | +0.006 |
| MLP | 0.442 | 0.478 | +0.036 |

### PLS-SEM Structural Paths

| Path | β | R² |
|------|---|----|
| Quality → Trust | 0.373 | — |
| Trust → Satisfaction | 0.835 | 0.65 |
| Satisfaction → Loyalty | 0.682 | 0.67 |
| Trust → Loyalty (indirect via Satisfaction) | 0.570 | — |

---

## Key Findings

**1 — Accuracy is misleading with imbalanced data**
The dummy baseline scores 0.770 accuracy by always predicting High loyalty — above every real model. Its Macro F1 is only 0.290. XGBoost's Macro F1 of 0.555 is nearly double the baseline, representing genuine discriminative ability.

**2 — SMOTE consistently improves Macro F1**
All three models improved on Macro F1 with SMOTE. MLP benefited most (+0.036) because neural networks are more sensitive to class distribution during gradient descent.

**3 — Trust and Satisfaction are the dominant drivers (convergent finding)**
Both SHAP analysis (ML) and path coefficients (SEM) independently identified Trust and Satisfaction as the top loyalty drivers — confirming convergent validity across two fundamentally different methodologies.

**4 — High R² does not imply good classification accuracy**
PLS-SEM achieved R² = 0.67 for Loyalty. When its predictions are binned into Low/Neutral/High, classification accuracy falls below the dummy baseline. SEM explains causal structure — it is not designed to classify observations.

**5 — Multidimensional loyalty dimensions behave differently**
LOY_1 (intention to remain) is easiest to predict (mean=5.94, skewed High). LOY_3 (repurchase likelihood) is hardest (mean=4.13, more balanced). Multi-label subset accuracy (~0.31) is lower than average accuracy (~0.67) because all three outputs must be correct simultaneously.

---

## Convergence vs Divergence

| Construct | ML (SHAP) | SEM (β) | Verdict |
|-----------|-----------|---------|---------|
| Trust | High importance | β=0.835 | **Convergence** |
| Satisfaction | High importance | β=0.682 | **Convergence** |
| Quality | High importance | β=0.373 | **Convergence** |
| CSR | Low importance | Weak path | **Convergence** |
| Likeability | Low-moderate | Moderate path | Slight divergence |

---

## Pipeline

```
Data Loading & Inspection
        │
        ▼
Preprocessing
  Impute CSOR_5 · Bin loyalty → Low/Neutral/High · Train/test split 80/20
        │
        ▼
Exploratory Data Analysis
  Loyalty distributions · Class imbalance · Construct correlations · Demographics
        │
        ▼
Multi-Label Classification (Section 5)
  RF · XGBoost · MLP — each tested with and without SMOTE
        │
        ▼
Aggregated Classification (Section 6)
  RF · XGBoost · MLP — SMOTE only
  Confusion matrix + feature importance per model
  SHAP on XGBoost (best model)
        │
        ▼
PLS-SEM Path Analysis (Section 7)
  Trust · Satisfaction · Loyalty models · Mediation · Path diagram
        │
        ▼
Comparison (Section 8)
  SHAP vs path coefficients · Convergence analysis
        │
        ▼
Conclusions & Findings (Section 9)
```

---

## Quickstart

```bash
git clone https://github.com/Hartyplaza/AI-Driven-Multi-Output-Prediction-of-Bank-Customer-Loyalty
cd AI-Driven-Multi-Output-Prediction-of-Bank-Customer-Loyalty
pip install pandas numpy matplotlib seaborn scikit-learn xgboost imbalanced-learn shap
jupyter notebook AI-Driven-Multi-Output-Prediction-of-Bank-Customer-Loyalty.ipynb
```

**On Google Colab:** Upload `DataInBrief_Bankdata.csv` to `/content/` then run all cells.

---

## Stack

| Layer | Technology |
|-------|-----------|
| **Data** | pandas, numpy |
| **ML** | scikit-learn, XGBoost |
| **Imbalance** | imbalanced-learn (SMOTE) |
| **Explainability** | SHAP (TreeExplainer) |
| **SEM** | OLS path regression |
| **Visualisation** | matplotlib, seaborn |

---

## Limitations

- German bank sample only — findings may not generalise to other markets
- Cross-sectional data — cannot establish true causality
- 10.9:1 class imbalance — Low loyalty remains hard to predict
- No hyperparameter tuning — grid search could improve results further

---

## Author

**Ofigwe Hart** — Data Scientist / ML Engineer

[![Portfolio](https://img.shields.io/badge/Portfolio-hartyplaza.github.io-blue?style=flat-square)](https://hartyplaza.github.io)
[![GitHub](https://img.shields.io/badge/GitHub-Hartyplaza-181717?style=flat-square&logo=github)](https://github.com/Hartyplaza)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-hart--ofigwe-0077B5?style=flat-square&logo=linkedin)](https://linkedin.com/in/hart-ofigwe)
