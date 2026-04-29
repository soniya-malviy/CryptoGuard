import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report, 
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)
import pickle
import os

print("Loading dataset...")
df = pd.read_csv("data/transaction_dataset.csv")

# Clean column names
df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")

# Drop irrelevant columns
drop_cols = ['index', 'address', 'unnamed:_0']
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

# Handle missing values
df = df.fillna(df.median(numeric_only=True))

# Features to use
features = [
    'avg_min_between_sent_tnx',
    'avg_min_between_received_tnx',
    'time_diff_between_first_and_last_(mins)',
    'sent_tnx',
    'received_tnx',
    'number_of_created_contracts',
    'max_value_received',
    'avg_val_received',
    'avg_val_sent',
    'total_ether_sent',
    'total_ether_balance',
    'erc20_total_ether_received',
    'erc20_total_ether_sent',
    'erc20_uniq_sent_addr',
    'erc20_uniq_rec_addr',
    'erc20_avg_time_between_sent_tnx',
    'erc20_avg_time_between_rec_tnx',
]

# Keep only features that exist in dataset
features = [f for f in features if f in df.columns]
target = 'flag'

X = df[features]
y = df[target]

print(f"Dataset shape: {X.shape}")
print(f"Fraud cases: {y.sum()} / {len(y)} ({y.mean():.1%})")

# Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, 
    test_size=0.2, 
    random_state=42,
    stratify=y
)

print("\nTraining XGBoost model...")
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=5,  # Increased to prioritize fraud detection
    use_label_encoder=False,
    eval_metric='logloss',
    random_state=42,
    n_jobs=-1
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    early_stopping_rounds=50,  # Prevent overfitting
    verbose=50
)

# Evaluate
y_pred = model.predict(X_test)
y_proba = model.predict_proba(X_test)[:, 1]

print("\n===== MODEL RESULTS =====")
print(f"Precision: {precision_score(y_test, y_pred):.3f}")
print(f"Recall:    {recall_score(y_test, y_pred):.3f}")
print(f"F1 Score:  {f1_score(y_test, y_pred):.3f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model and scaler as .pkl files
os.makedirs("model", exist_ok=True)

with open("model/fraud_model.pkl", "wb") as f:
    pickle.dump(model, f)

with open("model/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

# Save feature list so app knows exact order
with open("model/features.pkl", "wb") as f:
    pickle.dump(features, f)

print("\n✅ Model saved to model/fraud_model.pkl")
print("✅ Scaler saved to model/scaler.pkl")
print("✅ Features saved to model/features.pkl")
print("\nNow run: streamlit run app/main.py")
