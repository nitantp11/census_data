# Income Level Classification – Census Data Project

This project analyzes U.S. Census data to identify characteristics associated with earning more than $50K per year.
The goal is to build an explainable, actionable, and business-aligned model that predicts high-income individuals while balancing precision and recall.

Key highlights:
	•	Primary model: Logistic Regression for transparency and explainability
	•	Secondary validation: Calibrated Random Forest
	•	Exploration of Hybrid Business Rules to improve ROI and model performance
	•	Additional resampling and balancing techniques explored (SMOTE, Oversampling, Class Weighting)

---

## Dataset

- Source: [U.S. Census Income Dataset](https://archive.ics.uci.edu/ml/datasets/census+income)
- Target variable: `income_level` (`<=50K` vs `>50K`)
- Number of observations: ~200,000
- Features: mix of categorical and numerical variables

---

## Key Steps

1.	Data Cleaning & Feature Engineering
	•	Handle missing values and “?” entries
	•	Encode categorical variables (One-Hot and ordinal encodings)
	•	Create aggregated features (e.g., is_filer, owns_business, age_group)
2.	Exploratory Data Analysis
	•	Income distribution visualization
	•	Feature correlation and segmentation (e.g., education, occupation, industry)
3.	Modeling
	•	Logistic Regression (Primary Model): Explainable and interpretable
	•	Random Forest (Validation): Provides feature importance and probabilistic comparison
	•	Threshold Tuning: Maximized F1-score for the imbalanced target
4.	Hybrid Rule Integration
	•	Combines model probability with business rules:
	•	Example:
Model Prob > 0.7 AND Age 35-65 AND Executive/Admin
Predicted >50K AND Owns Business OR Files Taxes
	•	Improved recall in ROI-driven scenarios
	5.	Evaluation Metrics
	•	Accuracy, Precision, Recall, F1-score
	•	ROC-AUC & PR-AUC curves
	•	Confusion matrices for baseline, hybrid, and RF models

## Results Summary
| Model                      | Precision | Recall | F1-score | Accuracy | ROC-AUC | PR-AUC |
|----------------------------|----------|--------|---------|---------|--------|-------|
| Logistic Regression        | 0.49     | 0.62   | 0.55    | 0.94    | 0.938  | 0.564 |
| Hybrid Rules v1            | 0.28     | 0.73   | 0.41    | 0.87    | –      | –     |
| Hybrid Rules v2            | 0.47     | 0.22   | 0.30    | 0.94    | –      | –     |
| Calibrated Random Forest   | 0.74     | 0.30   | 0.43    | 0.95    | 0.931  | 0.578 |

## Business Takeaways
•	Logistic Regression is recommended as the primary model for transparency.
•	Random Forest confirms key income drivers and provides a second opinion.
•	Hybrid Rules allow businesses to tune precision vs. recall based on ROI.
•	Owning a business and filing taxes are the strongest income indicators.

## Requirements
•	Python 3.9+
•	pandas, numpy, scikit-learn, matplotlib, seaborn, shap
•	Install dependencies via:
``` pip install -r requirements.txt```

## Next Steps
•	Explore XGBoost or LightGBM for improved recall
•	Deploy as a scoring API or batch prediction pipeline

