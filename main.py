import os

os.environ["KAGGLE_CONFIG_DIR"] = os.path.join(os.getcwd(), ".kaggle")

import kagglehub
import shutil
from scripts.preprocess import load_and_preprocess
from scripts.train_regression import run_regression_models
from scripts.train_classification import run_classification_models

def main():
    # Download from Kaggle
    dataset_path = kagglehub.dataset_download("miadul/prostate-cancer-risk-and-lifestyle-synthetic-dataset")

    # Find CSV file
    csv_file = None
    for fname in os.listdir(dataset_path):
        if fname.endswith(".csv"):
            csv_file = fname
            break

    if not csv_file:
        raise FileNotFoundError("No CSV file found in downloaded dataset.")

    full_source_path = os.path.join(dataset_path, csv_file)
    destination_folder = "data"
    destination_path = os.path.join(destination_folder, "prostate_cancer.csv")

    os.makedirs(destination_folder, exist_ok=True)
    shutil.copy(full_source_path, destination_path)

    print("✅ Dataset copied to:", destination_path)

    # Run full pipeline
    X_train, X_test, y_train, y_test = load_and_preprocess(destination_path)
    
    print("\n🔁 Running regression models...")
    run_regression_models(X_train, X_test, y_train, y_test)
    
    print("\n🔁 Running classification models...")
    run_classification_models(X_train, X_test, y_train, y_test)

if __name__ == "__main__":
    main()

