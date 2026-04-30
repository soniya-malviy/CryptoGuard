# 🛡️ CryptoGuard: Enterprise-Grade AI Fraud Detection

CryptoGuard is a sophisticated blockchain forensic platform designed to detect and explain malicious activity on the Ethereum network. By combining **XGBoost Machine Learning** with **Explainable AI (Llama-3.1)**, it provides users with transparent, high-accuracy risk assessments.

---

## Core Features
- **Multi-Class Detection**: Identifies 4 distinct wallet profiles: `Legit`, `Phishing`, `Scam`, and `Hack`.
- **Explainable AI (XAI)**: Generates human-readable forensic reports using Llama-3.1 to explain the "Why" behind every score.
- **Dynamic Risk Vectors**: Visualizes the specific behavioral triggers for every audit (Local Feature Impact).
- **Enterprise Dashboard**: A premium UI with Dark/Light mode support, live Threat Intelligence, and a comprehensive User Guide.
- **Batch Processing**: Supports auditing thousands of wallets at once via CSV upload.

---

## Technical Stack
- **Brain**: XGBoost (Extreme Gradient Boosting)
- **AI Analyst**: Llama-3.1 (via Groq Cloud API)
- **Frontend**: Streamlit (Python-based Web Framework)
- **Visuals**: Plotly (Interactive Indicators & Charts)
- **Data**: Scikit-Learn (Scaling), Pandas (Manipulation)

---

## ⚙️ Installation & Setup

### 1. Clone & Install Dependencies
```bash
git clone https://github.com/yourusername/CryptoGuard.git
cd CryptoGuard
pip install -r requirements.txt
```

### 2. Configure API Keys
Create a `.env` file in the root directory and add your Groq API Key:
```text
GROQ_API_KEY=your_gsk_key_here
```

### 3. Model Training (Optional)
The pre-trained model is already included in the `model/` folder. To retrain it on the 5k dataset:
```bash
python model/train_final.py
```

### 4. Run the Platform
```bash
streamlit run app/main.py
```

---

##  Model Performance
Our XGBoost model is trained on **5,002 cryptographic behavioral signatures**, achieving:
- **Precision**: 89%
- **F1-score**: 0.86
- **Dataset**: Proprietary Ethereum Forensic Dataset

---

##  Security & Deployment
- **Local Development**: Uses `.env` for secret management (ignored by Git).
- **Production**: Supports **Streamlit Secrets** for safe deployment on the cloud.
- **XSRF Protection**: Configured via `.streamlit/config.toml` to handle large file uploads securely.

---

##  About the Project
CryptoGuard was developed as an **Open Research Project** to democratize blockchain security. Our goal is to make decentralized finance safer by providing clear, explainable, and accessible forensic tools for everyone—from students to institutional analysts.

© 2026 CryptoGuard | Empowering Digital Safety
