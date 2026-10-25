import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report
)

# 1. Load dataset
data = pd.read_csv("diabetes.csv")

print("First five rows:")
print(data.head())

print("\nDataset shape:", data.shape)

# 2. Remove duplicate rows
data = data.drop_duplicates()

# 3. Handle invalid zero measurements
columns_to_clean = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI"
]

for col in columns_to_clean:
    data[col] = data[col].replace(0, float("nan"))

# 4. Select seven features
# Pregnancies is intentionally excluded
features = [
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age"
]

X = data[features]
y = data["Outcome"]

# 5. Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 6. Define both machine-learning models
models = {
    "Random Forest": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", RandomForestClassifier(
            n_estimators=100,
            random_state=42,
            class_weight="balanced"
        ))
    ]),

    "Neural Network": Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("classifier", MLPClassifier(
            hidden_layer_sizes=(32, 16),
            max_iter=1000,
            early_stopping=True,
            random_state=42
        ))
    ])
}

# 7. Train and evaluate each model
results = {}

for name, model in models.items():
    print("\n" + "=" * 45)
    print(name)
    print("=" * 45)

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)
    accuracy = accuracy_score(y_test, predictions) * 100

    results[name] = round(accuracy, 2)

    print("Accuracy:", round(accuracy, 2), "%")
    print("\nConfusion Matrix:")
    print(confusion_matrix(y_test, predictions))
    print("\nClassification Report:")
    print(classification_report(
        y_test, predictions, zero_division=0
    ))

# 8. Compare model accuracy
print("\nMODEL ACCURACY COMPARISON")

for name, accuracy in results.items():
    print(f"{name}: {accuracy}%")

# 9. Save both trained models and feature names
joblib.dump(
    {"models": models, "features": features},
    "diabetes_models.pkl"
)

print("\nModels saved successfully!")