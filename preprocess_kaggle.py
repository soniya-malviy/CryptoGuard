import pandas as pd
import os

print("Loading Kaggle dataset...")
df = pd.read_csv("data/ethereum_fraud_dataset.csv")

# Map to snake_case
col_mapping = {
    "FLAG": "label",
    "Sent tnx": "sent_tnx",
    "Received Tnx": "received_tnx",
    "Avg min between sent tnx": "avg_min_between_sent_tnx",
    "Avg min between received tnx": "avg_min_between_received_tnx",
    "Time Diff between first and last (Mins)": "time_diff_between_first_and_last_mins",
    "Number of Created Contracts": "number_of_created_contracts",
    "total Ether sent": "total_ether_sent",
    "total ether balance": "total_ether_balance",
    "avg val sent": "avg_val_sent",
    "avg val received": "avg_val_received",
    "max value received ": "max_value_received",
}

# Select only the reliable ETH columns
df = df.rename(columns=col_mapping)[list(col_mapping.values())]

# Drop missing values
df = df.dropna()

print(f"Dataset cleaned. {len(df)} rows. Features: {len(df.columns)-1}")
df.to_csv("data/crypto_real_dataset.csv", index=False)
print("Saved to data/crypto_real_dataset.csv")
