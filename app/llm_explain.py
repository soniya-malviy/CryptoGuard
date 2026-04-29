import groq
import streamlit as st
import os
from dotenv import load_dotenv

# Load env variables
load_dotenv()

@st.cache_data(ttl=3600)  # cache same transaction explanation for 1 hour
def get_fraud_explanation(
    risk_score: float,
    risk_level: str,
    verdict: str,
    transaction_data: dict
) -> str:
    
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_key_here":
        return "Groq API Key not set. Please update the .env file to get AI explanations."
        
    client = groq.Groq(api_key=api_key)
    
    prompt = f"""
You are a compliance analyst for a crypto risk system.

RULES:
- Do NOT contradict the risk score.
- If risk is LOW (<20%), NEVER mention fraud, wash trading, or suspicion.
- Only describe behavior factually.
- Keep explanation neutral and consistent with prediction.
- Do NOT exaggerate risk.

INPUT DATA:
Risk Score: {risk_score:.1%}
Risk Level: {risk_level}
Verdict: {verdict}

Transaction Summary:
{transaction_data}

OUTPUT FORMAT:
Write exactly 3 sentences:
1. Behavioral summary
2. Neutral interpretation
3. Final risk confirmation
"""
    
    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=300
        )
        return response.choices[0].message.content
    except Exception as e:
        return f"Error calling Groq API: {str(e)}"
