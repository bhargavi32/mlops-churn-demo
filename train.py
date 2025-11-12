# train.py — Day 2: Build ML Pipeline for Churn Prediction

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import joblib
import os

# 1️⃣ Load Dataset
csv_path = "Telco-Customer-Churn.csv"
data = pd.read_csv(csv_path)

print("✅ Dataset loaded successfully!")
print("Shape:", data.shape)

# 2️⃣ Preprocessing
# Remove customerID (not useful for prediction)
data.drop(columns=["customerID"], inplace=True)

# Drop rows with missing values
data.dropna(inplace=True)

# Convert Churn to 0/1
data["Churn"] = data["Churn"].apply(lambda x: 1 if x == "Yes" else 0)

# Encode categorical columns
le = LabelEncoder()
for col in data.select_dtypes(include="object").columns:
    data[col] = le.fit_transform(data[col])

# Split into features and label
X = data.drop("Churn", axis=1)
y = data["Churn"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3️⃣ Train random forest classifier
rf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
rf.fit(X_train, y_train)

# 4️⃣ Evaluate
y_pred = rf.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print("\n📊 Model Evaluation Results:")
print(f"Accuracy:  {accuracy:.3f}")
print(f"Precision: {precision:.3f}")
print(f"Recall:    {recall:.3f}")
print(f"F1 Score:  {f1:.3f}")

# 5️⃣ MLflow Logging
mlflow.set_experiment("churn_prediction")

with mlflow.start_run():
    mlflow.log_param("n_estimators", 100)
    mlflow.log_param("max_depth", 6)
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)
    mlflow.sklearn.log_model(rf, "model")

# 6️⃣ Save model + scaler locally
os.makedirs("models", exist_ok=True)

mlflow.sklearn.save_model(rf, "models/random_forest_model")
joblib.dump(scaler, "models/scaler.pkl")
joblib.dump(list(X.columns), "models/feature_order.pkl")

print("\n✅ Training complete!")
print("📦 Model + scaler saved inside /models/")
