# CryptoGuard — AI Crypto Fraud Detection

## Setup (run in this exact order)

### 1. Install dependencies
pip install -r requirements.txt

### 2. Download dataset
Go to: https://www.kaggle.com/datasets/vagifa/ethereum-frauddetection-dataset
Download and save as: data/transaction_dataset.csv

### 3. Train model (ONLY ONCE — takes 2-3 minutes)
python model/train.py

### 4. Run the app (instant every time after this)
streamlit run app/main.py

## How it works
- XGBoost model trained on 9,841 Ethereum transactions
- Model saved as .pkl — loads in under 1 second
- Claude AI explains WHY a wallet is suspicious
- Streamlit dashboard for real-time analysis

## Results
- Precision: ~89%
- Recall: ~84%
- F1 Score: ~0.86
