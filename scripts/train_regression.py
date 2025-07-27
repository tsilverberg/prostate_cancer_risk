import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

def evaluate_model(name, model, X_test, y_test):
    preds = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    return {'Model': name, 'RMSE': rmse, 'R2': r2}

def run_regression_models(X_train, X_test, y_train, y_test):
    results = []

    print("\n🚀 Training and evaluating models...\n")

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    lr_result = evaluate_model('Linear Regression', lr, X_test, y_test)
    results.append(lr_result)
    print(f"📈 Linear Regression → RMSE: {lr_result['RMSE']:.4f}, R²: {lr_result['R2']:.4f}")

    # Decision Tree
    dt = DecisionTreeRegressor(max_depth=8, random_state=42)
    dt.fit(X_train, y_train)
    dt_result = evaluate_model('Decision Tree', dt, X_test, y_test)
    results.append(dt_result)
    print(f"🌳 Decision Tree     → RMSE: {dt_result['RMSE']:.4f}, R²: {dt_result['R2']:.4f}")

    # XGBoost
    xgb = XGBRegressor(n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42)
    xgb.fit(X_train, y_train)
    xgb_result = evaluate_model('XGBoost', xgb, X_test, y_test)
    results.append(xgb_result)
    print(f"⚡ XGBoost           → RMSE: {xgb_result['RMSE']:.4f}, R²: {xgb_result['R2']:.4f}")

    # Save results
    os.makedirs("outputs", exist_ok=True)
    metrics_df = pd.DataFrame(results)
    metrics_df.to_csv("outputs/evaluation_metrics.csv", index=False)

    # Plot results
    fig, ax = plt.subplots()
    ax.bar(metrics_df['Model'], metrics_df['RMSE'], color='steelblue')
    ax.set_title("RMSE for Prostate Cancer Risk Prediction Models")
    ax.set_ylabel("RMSE")
    plt.tight_layout()
    os.makedirs("outputs/plots", exist_ok=True)
    plt.savefig("outputs/plots/model_performance.png")
    plt.close()

    print("\n✅ Results saved to outputs/evaluation_metrics.csv and plotted.")
