import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import pickle
import os

print("🚀 Training Final Model on crypto_5000_dataset.csv...")

# Load data
df = pd.read_csv("data/crypto_5000_dataset.csv")

# Features and Target
target = 'label'
X = df.drop(columns=[target])
y = df[target]

# Features list
features = X.columns.tolist()

# Handle missing values
X = X.fillna(X.median())

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

# Training (Multi-class)
print(f"Training on {X_train.shape[0]} rows with {X_train.shape[1]} features...")
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    objective='multi:softprob',
    num_class=4,
    eval_metric='mlogloss',
    random_state=42
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=50,
    verbose=False
)

# Save artifacts
os.makedirs("model", exist_ok=True)
with open("model/fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open("model/features.pkl", "wb") as f:
    pickle.dump(features, f)

# We don't need label_encoders for this dataset
if os.path.exists("model/label_encoders.pkl"):
    os.remove("model/label_encoders.pkl")

# Save numerical columns (all of them are numerical)
with open("model/num_cols.pkl", "wb") as f:
    pickle.dump(features, f)

print("✅ Final Model, Scaler, and Features saved!")
print("\nModel Evaluation:")
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
