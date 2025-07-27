# 🧪 Assignment 4 – Regression vs Classification Model Comparison

This project explores both regression and classification models to predict prostate cancer risk levels using synthetic patient lifestyle and health data.

---

## 📁 Project Structure

```
assignment_4_regression_comparison/
├── data/                           # Auto-downloaded prostate_cancer.csv
├── scripts/                        # Preprocessing and training scripts
│   ├── preprocess.py
│   ├── train_regression.py
│   ├── train_classification.py
├── outputs/                        # Results and plots
│   ├── evaluation_metrics.csv
│   ├── evaluation_metrics_classification.csv
│   └── plots/
│       ├── model_performance.png
│       └── classification_performance.png
├── report/
│   └── summary.md                  # Final report write-up
└── main.py                         # Runs preprocessing and both pipelines
```

---

## 📈 Models Compared

### 🔁 Regression
- Linear Regression
- Decision Tree Regressor
- XGBoost Regressor

### 🎯 Classification
- Logistic Regression
- Decision Tree Classifier
- XGBoost Classifier

---

## 📊 Key Metrics

- **Regression**: RMSE, R²
- **Classification**: Accuracy, F1 Score, Classification Report

---

## ✅ How to Run

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the pipeline:
   ```bash
   python main.py
   ```

3. Outputs will be saved in the `outputs/` folder.

---

## 🧠 Summary

- Regression models underperformed due to the ordinal categorical nature of the target.
- Classification, especially **XGBoost**, achieved the best results (Accuracy: 83.5%, F1: 81.5%).
- Future improvements could include class balancing for rare classes like `"High"` risk.