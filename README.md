# 🛡️ CryptoGuard: Autonomous Forensic Pipeline

CryptoGuard is an enterprise-grade blockchain investigation platform. It combines a highly accurate XGBoost Machine Learning model with a 6-Agent AI Swarm to autonomously audit Ethereum wallets, detect criminal behavior, and generate actionable compliance reports.

## 🌟 Key Features

*   **Etherscan V2 Integration:** Automatically fetches live transaction history, ERC20 tokens, and contract data for any Ethereum wallet.
*   **XGBoost Risk Scoring:** Uses a binary Logistic XGBoost Classifier (trained on a real-world dataset of 9,841 verified wallets) to predict fraud probability with **94% accuracy**.
*   **6-Agent Swarm Intelligence:** Powered by LLMs (e.g., Groq / Llama-3.1), the system uses a sequential agent pipeline to analyze patterns, build evidence, and decide compliance actions.
*   **Hallucination Prevention:** Built with "Chain-of-Thought" (CoT) structures and explicit "I Don't Know" constraints to ensure the AI never guesses or falsely flags innocent wallets.
*   **Enterprise Dashboard:** A beautiful, non-technical Streamlit UI that clearly displays Risk Actions (FREEZE, WATCHLIST, CLEAR), Live Telemetry, and downloadable PDF/TXT reports.

## 🤖 The 6-Agent Pipeline

When an address is submitted, the autonomous pipeline triggers:

1.  **Agent 0 (Data Fetcher):** Queries the Etherscan V2 API, cleans the data, and constructs the 11 core transaction features needed by the ML model.
2.  **Agent 1 (Risk Scorer):** Injects the data into the XGBoost model to get a strict mathematical fraud probability (0% to 100%).
3.  **Agent 2 (Pattern Classifier):** Uses deterministic AI reasoning to categorize the exact *type* of fraud (e.g., Wash Trading, Phishing, Scam, or None).
4.  **Agent 3 (Evidence Collector):** Scans the raw data for specific anomalies (e.g., "Sent Transactions are 4.1x above normal") to build a criminal case file.
5.  **Agent 4 (Report Writer):** Generates a comprehensive, human-readable forensic document summarizing the investigation.
6.  **Agent 5 (Action Decider):** Acts as the final Maker-Checker. Reviews all prior agent logic and issues the final compliance action: `FREEZE`, `ESCALATE`, or `CLEAR`.

## 🛠️ Installation & Setup

### Prerequisites
*   Python 3.11+
*   An [Etherscan API Key](https://etherscan.io/apis)
*   A [Groq API Key](https://console.groq.com/keys) (for the LLM agents)

### 1. Clone the repository
```bash
git clone https://github.com/yourusername/cryptoguard.git
cd cryptoguard
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory:
```env
GROQ_API_KEY="your_groq_api_key_here"
ETHERSCAN_API_KEY="your_etherscan_api_key_here"
```

### 4. Run the Platform
Launch the Streamlit dashboard:
```bash
python -m streamlit run app/main.py
```

## 🧠 Model Training

The `fraud_model.pkl` is trained on a cleaned subset of the Kaggle *Ethereum Fraud Detection Dataset*. It relies entirely on standard Ethereum features (like transaction velocity and ETH balances) and intentionally ignores ERC20 token values to prevent feature-pipeline mismatches caused by varying token decimals.

To retrain the model locally:
1. Ensure the Kaggle dataset is preprocessed to `data/crypto_real_dataset.csv`.
2. Run `python model/train_final.py`.

## 📄 License
This project is for educational and research purposes. It is not financial advice.
