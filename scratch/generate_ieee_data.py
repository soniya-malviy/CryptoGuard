import pandas as pd
import numpy as np
import os

def generate_synthetic_ieee(n_rows=5000):
    np.random.seed(42)
    
    data = {
        'TransactionID': np.arange(10000, 10000 + n_rows),
        'isFraud': np.random.choice([0, 1], size=n_rows, p=[0.96, 0.04]),
        'TransactionDT': np.random.randint(100000, 1000000, size=n_rows),
        'TransactionAmt': np.random.exponential(scale=100, size=n_rows),
        'ProductCD': np.random.choice(['W', 'H', 'C', 'S', 'R'], size=n_rows),
        'card1': np.random.randint(1000, 20000, size=n_rows),
        'card2': np.random.randint(100, 600, size=n_rows),
        'card3': np.random.choice([150, 185, 102], size=n_rows),
        'card4': np.random.choice(['visa', 'mastercard', 'american express', 'discover'], size=n_rows),
        'card5': np.random.randint(100, 250, size=n_rows),
        'card6': np.random.choice(['debit', 'credit'], size=n_rows),
        'addr1': np.random.randint(100, 500, size=n_rows),
        'addr2': np.random.choice([87, 60, 96], size=n_rows),
        'dist1': np.random.randint(0, 1000, size=n_rows),
        'P_emaildomain': np.random.choice(['gmail.com', 'yahoo.com', 'anonymous.com', 'outlook.com'], size=n_rows),
    }
    
    # Add some C, D, and V features
    for i in range(1, 15):
        data[f'C{i}'] = np.random.randint(0, 10, size=n_rows)
    
    for i in range(1, 16):
        data[f'D{i}'] = np.random.randint(0, 500, size=n_rows)
        
    for i in range(1, 51): # Just 50 V features for simplicity
        data[f'V{i}'] = np.random.normal(0, 1, size=n_rows)
        
    df = pd.DataFrame(data)
    
    # Inject some fraud patterns
    df.loc[df['isFraud'] == 1, 'TransactionAmt'] *= 2.5
    df.loc[df['isFraud'] == 1, 'card6'] = 'credit'
    df.loc[df['isFraud'] == 1, 'V1'] += 3.0
    
    os.makedirs('data', exist_ok=True)
    df.to_csv('data/ieee_fraud_sample.csv', index=False)
    print(f"✅ Generated synthetic IEEE-CIS dataset: data/ieee_fraud_sample.csv ({n_rows} rows)")

if __name__ == "__main__":
    generate_synthetic_ieee()
