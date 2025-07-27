import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, f1_score, classification_report
import numpy as np

def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average='weighted')
    
    print(f"📋 {name} Classification Report:")
    print(classification_report(y_test, preds, target_names=['High', 'Low', 'Medium']))
    
    return {'Model': name, 'Accuracy': acc, 'F1': f1}

def run_classification_models(X_train, X_test, y_train, y_test):
    results = []

    print("\n🚀 Training and evaluating classification models...\n")

    # Logistic Regression
    lr = LogisticRegression(max_iter=1000)
    lr.fit(X_train, y_train)
    lr_result = evaluate_model('Logistic Regression', lr, X_test, y_test)
    results.append(lr_result)
    print(f"📈 Logistic Regression → Accuracy: {lr_result['Accuracy']:.4f}, F1: {lr_result['F1']:.4f}")

    # Decision Tree Classifier
    dt = DecisionTreeClassifier(max_depth=8, random_state=42)
    dt.fit(X_train, y_train)
    dt_result = evaluate_model('Decision Tree', dt, X_test, y_test)
    results.append(dt_result)
    print(f"🌳 Decision Tree        → Accuracy: {dt_result['Accuracy']:.4f}, F1: {dt_result['F1']:.4f}")

    # XGBoost Classifier
    xgb = XGBClassifier(n_estimators=100, max_depth=6, learning_rate=0.1, use_label_encoder=False, eval_metric='mlogloss', random_state=42)
    xgb.fit(X_train, y_train)
    xgb_result = evaluate_model('XGBoost', xgb, X_test, y_test)
    results.append(xgb_result)
    print(f"⚡ XGBoost              → Accuracy: {xgb_result['Accuracy']:.4f}, F1: {xgb_result['F1']:.4f}")

    # Save results
    os.makedirs("outputs", exist_ok=True)
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv("outputs/evaluation_metrics_classification.csv", index=False)

    # Plot results
    fig, ax = plt.subplots()
    ax.bar(metrics_df['Model'], metrics_df['Accuracy'], color='mediumseagreen')
    ax.set_title("Classification Accuracy by Model")
    ax.set_ylabel("Accuracy")
    plt.tight_layout()
    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig("outputs/plots/classification_performance.png")
    plt.close()

    print("\n✅ Classification results saved to outputs/evaluation_metrics_classification.csv and plotted.")
