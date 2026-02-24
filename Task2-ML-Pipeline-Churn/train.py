# ============================================
# Task 2: End-to-End ML Pipeline
# Customer Churn Prediction
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)

# ── 1. Load Dataset ──────────────────────────
print("Loading dataset...")
url = "https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv"
df = pd.read_csv(url)
print(f"Dataset shape: {df.shape}")

# ── 2. Preprocessing ─────────────────────────
print("Preprocessing data...")
df = df.drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

X = df.drop(columns=["Churn"])
y = df["Churn"]

cat_cols = X.select_dtypes(include="object").columns.tolist()
num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 3. Build Pipeline ────────────────────────
print("Building pipeline...")
preprocessor = ColumnTransformer(transformers=[
    ("num", StandardScaler(), num_cols),
    ("cat", OneHotEncoder(handle_unknown="ignore"), cat_cols)
])

lr_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", LogisticRegression(max_iter=1000))
])

rf_pipeline = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("classifier", RandomForestClassifier(random_state=42))
])

# ── 4. Train Models ──────────────────────────
print("Training Logistic Regression...")
lr_pipeline.fit(X_train, y_train)
lr_preds = lr_pipeline.predict(X_test)

print("Training Random Forest...")
rf_pipeline.fit(X_train, y_train)
rf_preds = rf_pipeline.predict(X_test)

print("=" * 45)
print("LOGISTIC REGRESSION RESULTS")
print(f"  Accuracy : {accuracy_score(y_test, lr_preds)*100:.2f}%")
print(f"  F1 Score : {f1_score(y_test, lr_preds):.4f}")
print("=" * 45)
print("RANDOM FOREST RESULTS")
print(f"  Accuracy : {accuracy_score(y_test, rf_preds)*100:.2f}%")
print(f"  F1 Score : {f1_score(y_test, rf_preds):.4f}")

# ── 5. Hyperparameter Tuning ─────────────────
print("Running GridSearchCV (this takes 3-5 mins)...")
param_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [5, 10, None],
    "classifier__min_samples_split": [2, 5]
}

grid_search = GridSearchCV(
    rf_pipeline,
    param_grid,
    cv=5,
    scoring="f1",
    n_jobs=-1,
    verbose=1
)
grid_search.fit(X_train, y_train)
best_preds = grid_search.predict(X_test)

print("=" * 45)
print("BEST MODEL AFTER GRIDSEARCH")
print(f"  Best Params : {grid_search.best_params_}")
print(f"  Accuracy    : {accuracy_score(y_test, best_preds)*100:.2f}%")
print(f"  F1 Score    : {f1_score(y_test, best_preds):.4f}")
print("=" * 45)
print(classification_report(y_test, best_preds, target_names=["No Churn", "Churn"]))

# ── 6. Confusion Matrix ──────────────────────
cm = confusion_matrix(y_test, best_preds)
plt.figure(figsize=(6, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Churn", "Churn"],
            yticklabels=["No Churn", "Churn"])
plt.title("Confusion Matrix — Best Model")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("confusion_matrix.png")
print("Confusion matrix saved ✅")

# ── 7. Export Pipeline ───────────────────────
joblib.dump(grid_search.best_estimator_, "churn_pipeline.pkl")
print("Pipeline exported as churn_pipeline.pkl ✅")
