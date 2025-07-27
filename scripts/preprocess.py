import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder

def load_and_preprocess(filepath):
    print(f"\n📥 Loading dataset from: {filepath}")
    df = pd.read_csv(filepath)

    print("\n📊 Initial DataFrame shape:", df.shape)
    print("🧾 Columns:\n", df.columns.tolist())

    # Drop ID column
    if 'id' in df.columns:
        df = df.drop(columns=['id'])
        print("🚫 Dropped column: 'id'")

    # Show dtypes before encoding
    print("\n🔎 Data types before encoding:")
    print(df.dtypes)

    # Print sample values from each column
    print("\n📌 Sample values:")
    for col in df.columns:
        print(f"- {col}: {df[col].unique()[:5]}")

    # Handle target variable: risk_level
    if 'risk_level' not in df.columns:
        raise ValueError("❌ Column 'risk_level' not found in the dataset")

    print("\n🎯 Unique risk_level values (pre-encoding):", df['risk_level'].unique())
    le_target = LabelEncoder()
    df['risk_level'] = le_target.fit_transform(df['risk_level'].astype(str))
    print("✅ Encoded 'risk_level' →", list(le_target.classes_))

    # Define and encode categorical feature columns
    categorical_cols = [
        'smoker', 'alcohol_consumption', 'diet_type',
        'physical_activity_level', 'family_history',
        'mental_stress_level', 'regular_health_checkup',
        'prostate_exam_done'
    ]

    print("\n🔤 Encoding categorical features:")
    for col in categorical_cols:
        print(f" - {col}: {df[col].unique().tolist()}")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # Check data types after encoding
    print("\n🔍 Data types after encoding:")
    print(df.dtypes)

    # Split features and target
    X = df.drop('risk_level', axis=1)
    y = df['risk_level']

    # Scale numerical features
    print("\n📏 Scaling numeric features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Train-test split
    print("✂️ Splitting data (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42
    )

    print("✅ Preprocessing complete. Shapes:")
    print("   X_train:", X_train.shape)
    print("   X_test: ", X_test.shape)
    print("   y_train:", y_train.shape)
    print("   y_test: ", y_test.shape)

    return X_train, X_test, y_train, y_test
