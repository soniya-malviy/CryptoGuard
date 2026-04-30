import pickle
import numpy as np
import pandas as pd
import streamlit as st
import os

@st.cache_resource
def load_model():
    base_path = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_path = os.path.join(base_path, "model", "fraud_model.pkl")
    scaler_path = os.path.join(base_path, "model", "scaler.pkl")
    features_path = os.path.join(base_path, "model", "features.pkl")
    
    with open(model_path, "rb") as f:
        model = pickle.load(f)
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    with open(features_path, "rb") as f:
        features = pickle.load(f)
    
    return model, scaler, features

def predict_fraud(transaction_data: dict) -> dict:
    model, scaler, features = load_model()
    
    # Preprocess input data
    input_df = pd.DataFrame([transaction_data])
    
    # Ensure all features exist
    for feat in features:
        if feat not in input_df.columns:
            input_df[feat] = 0
            
    # Reorder columns to match training
    X = input_df[features]
    
    # Scale numerical columns
    X_scaled = scaler.transform(X)
    
    # Predict probabilities for all classes
    probs = model.predict_proba(X_scaled)[0]
    
    # Classes: 0: Legit, 1: Phishing, 2: Scam, 3: Hack
    # Fraud probability is sum of probs[1], probs[2], probs[3]
    fraud_prob = float(np.sum(probs[1:]))
    prediction = int(np.argmax(probs))
    
    class_names = {0: "Legit", 1: "Phishing", 2: "Scam", 3: "Hack"}
    verdict = class_names.get(prediction, "Unknown")
    
    # Feature impact (Local Contribution)
    # Multiplying scaled values by importance gives a proxy for which feature drove THIS specific score
    importances = model.feature_importances_
    local_impact = X_scaled[0] * importances
    top_features = sorted(zip(features, local_impact), key=lambda x: abs(x[1]), reverse=True)[:5]
    
    return {
        "risk_score": fraud_prob,
        "is_fraud": prediction != 0,
        "risk_level": ("HIGH" if fraud_prob > 0.6 else "MEDIUM" if fraud_prob > 0.3 else "LOW"),
        "verdict": verdict,
        "class_probs": {class_names[i]: float(p) for i, p in enumerate(probs)},
        "top_features": top_features,
        "transaction_data": transaction_data
    }
