# Income Level Classification – Census Data Project

## Objective

This project investigates the factors associated with whether a person earns more or less than $50,000 per year, using U.S. Census data. The aim is to develop interpretable and predictive models, focusing on:

- Identifying key demographic, employment, and financial characteristics
- Balancing predictive performance with explainability
- Supporting fairness by excluding sensitive attributes like race and sex

---

## Dataset

- Source: [U.S. Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/census+income)
- Target variable: `income_level` (`<=50K` vs `>50K`)
- Number of observations: ~200,000
- Features: mix of categorical and numerical variables

---

## Methodology

### 1. Data Preparation
- Imputation and recoding of unknown/missing values
- Feature engineering (log transforms, age binning, etc.)
- Rare category grouping (e.g., `state`, `industry`)
- One-hot encoding for categorical variables
- Separation of signal strength (e.g., `log_dividends`) and signal presence (e.g., `has_dividends`)

### 2. Modeling
- **Logistic Regression** with `class_weight='balanced'` for interpretability
- **Random Forest** for validation and feature interaction capture
- **Threshold tuning** to optimize precision-recall tradeoff
- Evaluation with classification metrics and ROC/PR-AUC

### 3. Explainability
- SHAP values for feature impact and direction
- Odds ratio interpretation of logistic regression coefficients
- Multicollinearity check using VIF

---

## Key Insights

- **Education**, **weeks worked**, and **capital income** are top predictors.
- Age groups show distinct impacts: `<25` is negatively associated, `45–64` and `65+` have positive effects.
- **Executive/professional roles** and **owning a business** significantly increase the odds of higher income.
- **Working in education/social services** is not predictive unless paired with business ownership.

---

## Results

|Metric            |Logistic Regression  |Random Forest |
|------------------|---------------------|----------------|
| Accuracy         | 94%                 | 93%            |
| F1 Score (>50K)  | 0.55 (tuned)        | 0.54 (tuned)   |
| ROC-AUC          | 0.94                | 0.92           |
| PR-AUC           | 0.57                | 0.53           |

---
