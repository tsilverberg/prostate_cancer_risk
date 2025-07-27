# 📊 Assignment 4: Model Comparison – Regression vs Classification

## ✅ Problem Definition

We aimed to predict the **risk level of prostate cancer** (`High`, `Medium`, `Low`) based on lifestyle and clinical indicators using both **regression** and **classification** approaches.

**Dataset**: Prostate Cancer Risk and Lifestyle Synthetic Dataset  
**Target Variable**: `risk_level` (ordinal categorical)  
**Features**: Age, BMI, smoker status, alcohol consumption, diet type, stress level, physical activity, etc.

---

## 🔁 Regression Models

We applied:
- Linear Regression
- Decision Tree Regressor
- XGBoost Regressor

| Model              | RMSE   | R²     |
|--------------------|--------|--------|
| Linear Regression  | 0.5511 | 0.0450 |
| Decision Tree      | 0.6682 | -0.4041 |
| XGBoost            | 0.5618 | 0.0075 |

**Insight**: Regression models performed poorly with low or negative R² values, indicating weak explanatory power over the ordinal target. This suggests regression is not well-suited for this problem.

---

## 🎯 Classification Models

We re-framed the problem as a **classification task**, using:
- Logistic Regression
- Decision Tree Classifier
- XGBoost Classifier

| Model              | Accuracy | F1 Score |
|--------------------|----------|----------|
| Logistic Regression| 0.7600   | 0.7485   |
| Decision Tree      | 0.7350   | 0.7329   |
| **XGBoost**        | **0.8350** | **0.8150** |

**Insight**: Classification results were far superior. XGBoost achieved the highest accuracy and F1. However, class imbalance caused it to miss predictions for the `"High"` risk class entirely, which may require rebalancing techniques.

---

## 🧠 Final Recommendation

- **Use classification over regression** for ordinal categorical targets like `risk_level`.
- **XGBoost** is the best overall performer, but further tuning or class balancing (e.g., SMOTE or class_weight) is recommended to improve recall for rare classes.

---

## 📁 Deliverables

- `evaluation_metrics.csv` – Regression metrics  
- `evaluation_metrics_classification.csv` – Classification metrics  
- RMSE and Accuracy plots in `outputs/plots/`