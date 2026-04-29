import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import pickle
import os

print("🚀 Training model with IEEE-CIS Fraud Detection dataset structure...")

# Load data
df = pd.read_csv("data/ieee_fraud_sample.csv")

# Target
target = 'isFraud'

# Drop ID columns
X = df.drop(columns=[target, 'TransactionID', 'TransactionDT'])
y = df[target]

# Identify categorical and numerical columns
cat_cols = X.select_dtypes(include=['object']).columns.tolist()
num_cols = X.select_dtypes(exclude=['object']).columns.tolist()

# Label encoding for categorical columns
label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    X[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

# Handle missing values
X = X.fillna(-999)

# Scaling
scaler = StandardScaler()
X[num_cols] = scaler.fit_transform(X[num_cols])

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# Training
print(f"Training on {X_train.shape[0]} rows with {X_train.shape[1]} features...")
model = xgb.XGBClassifier(
    n_estimators=500,
    max_depth=8,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    scale_pos_weight=10, # IEEE is very imbalanced
    eval_metric='logloss',
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

with open("model/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

with open("model/features.pkl", "wb") as f:
    pickle.dump(X.columns.tolist(), f)

with open("model/num_cols.pkl", "wb") as f:
    pickle.dump(num_cols, f)

print("✅ Model, Scaler, Encoders, and Num_Cols saved!")
print(f"📊 Accuracy: {model.score(X_test, y_test):.2%}")
