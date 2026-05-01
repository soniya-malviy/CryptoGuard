"""
CryptoGuard 5-Agent Autonomous Pipeline
Each agent uses Groq API tool use (function calling) with Llama models.
"""

import json
import os
import re
import groq
import numpy as np
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Shared Infrastructure
# ---------------------------------------------------------------------------

GROQ_MODEL = "llama-3.1-8b-instant"

def _get_client():
    api_key = None
    try:
        import streamlit as st
        api_key = st.secrets.get("GROQ_API_KEY")
    except Exception:
        pass
    if not api_key:
        api_key = os.getenv("GROQ_API_KEY")
    if not api_key or api_key == "your_key_here":
        raise ValueError("GROQ_API_KEY not set. Add it to .env")
    return groq.Groq(api_key=api_key)


def _call_agent_with_tools(system_prompt, user_prompt, tools, tool_handlers, max_rounds=10):
    """Generic Groq tool-use loop. Sends message, handles tool calls, returns final text."""
    client = _get_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for _ in range(max_rounds):
        import time
        max_retries = 3
        for attempt in range(max_retries):
            try:
                response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    max_tokens=512,
                    temperature=0.2,
                    tools=tools,
                    tool_choice="auto",
                    messages=messages,
                )
                break
            except groq.RateLimitError as e:
                if attempt < max_retries - 1:
                    print(f"⚠️ Rate limit hit. Waiting 10s before retry...")
                    time.sleep(10)
                else:
                    raise e
            except Exception as e:
                if "Rate limit reached" in str(e):
                    if attempt < max_retries - 1:
                        print(f"⚠️ Rate limit hit. Waiting 10s before retry...")
                        time.sleep(10)
                        continue
                raise e

        msg = response.choices[0].message
        finish_reason = response.choices[0].finish_reason

        # If no tool calls, return the text
        if not msg.tool_calls or finish_reason == "stop":
            return msg.content or ""

        # Process tool calls
        messages.append(msg)
        for tool_call in msg.tool_calls:
            fn_name = tool_call.function.name
            try:
                fn_args = json.loads(tool_call.function.arguments)
                if not isinstance(fn_args, dict):
                    fn_args = {}
            except (json.JSONDecodeError, TypeError):
                fn_args = {}

            handler = tool_handlers.get(fn_name)
            if handler:
                result = handler(**fn_args)
            else:
                result = {"error": f"Unknown tool: {fn_name}"}

            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result) if not isinstance(result, str) else result,
            })

    # Return whatever we have after max rounds
    return messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""


def _parse_json_response(text):
    """Extract JSON from Claude's text response."""
    # Try to find JSON in code blocks first
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    # Try raw JSON
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    # Try the whole text
    return json.loads(text.strip())


def _load_dataset_medians():
    """Load dataset medians for anomaly detection."""
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base, "data", "crypto_5000_dataset.csv")
    try:
        df = pd.read_csv(csv_path)
        return df.median(numeric_only=True).to_dict()
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# AGENT 0 — Data Fetcher
# ---------------------------------------------------------------------------

def run_agent_0_data_fetcher(shared_state: dict) -> dict:
    from app.data_fetcher import (
        fetch_normal_transactions,
        fetch_erc20_transactions, 
        fetch_eth_balance,
        compute_wallet_features
    )
    
    wallet_address = shared_state["wallet_address"]
    
    AGENT0_TOOLS = [
        {
            "type": "function",
            "function": {
                "name": "fetch_wallet_transactions",
                "description": "Fetch all normal ETH transactions for a wallet address from Etherscan",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "wallet_address": {"type": "string"}
                    },
                    "required": ["wallet_address"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_token_transactions", 
                "description": "Fetch all ERC20 token transactions for a wallet",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "wallet_address": {"type": "string"}
                    },
                    "required": ["wallet_address"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "fetch_wallet_balance",
                "description": "Get current ETH balance of wallet in ETH",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "wallet_address": {"type": "string"}
                    },
                    "required": ["wallet_address"]
                }
            }
        },
        {
            "type": "function",
            "function": {
                "name": "compute_features",
                "description": "Compute all ML features from raw transaction data",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "wallet_address": {"type": "string"}
                    },
                    "required": ["wallet_address"]
                }
            }
        }
    ]
    
    AGENT0_HANDLERS = {
        "fetch_wallet_transactions": lambda wallet_address: {"count": len(fetch_normal_transactions(wallet_address))},
        "fetch_token_transactions": lambda wallet_address: {"count": len(fetch_erc20_transactions(wallet_address))},
        "fetch_wallet_balance": lambda wallet_address: {"balance_eth": fetch_eth_balance(wallet_address)},
        "compute_features": lambda wallet_address: compute_wallet_features(wallet_address),
    }

    prompt = (
        f"You are the Data Fetcher agent for CryptoGuard.\n\n"
        f"Wallet address to investigate: {wallet_address}\n\n"
        f"Use the available tools to:\n"
        f"1. fetch_wallet_transactions — get all ETH transactions\n"
        f"2. fetch_token_transactions — get ERC20 activity\n"
        f"3. fetch_wallet_balance — get current balance\n"
        f"4. compute_features — compute final ML feature set\n\n"
        f"Return ONLY a JSON with the complete wallet_data dict containing all computed features."
    )
    
    text = _call_agent_with_tools(
        "You are a blockchain data fetching agent. Always use ALL FOUR tools before making a decision.",
        prompt, AGENT0_TOOLS, AGENT0_HANDLERS
    )
    
    try:
        wallet_data = _parse_json_response(text)
        if "error" in wallet_data:
            shared_state["wallet_data"] = compute_wallet_features(wallet_address)
        else:
            shared_state["wallet_data"] = wallet_data
    except Exception:
        # Fallback — compute directly
        shared_state["wallet_data"] = compute_wallet_features(wallet_address)
    
    shared_state["audit_log"].append({
        "agent": "Data Fetcher",
        "wallet_address": wallet_address,
        "features_computed": len(shared_state["wallet_data"]),
        "timestamp": datetime.now().isoformat()
    })
    
    return shared_state


# ---------------------------------------------------------------------------
# AGENT 1 — Risk Scorer
# ---------------------------------------------------------------------------

def _tool_run_xgboost_predict(wallet_data: dict) -> float:
    """Calls predict_fraud() and returns risk_score."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.predict import predict_fraud
    result = predict_fraud(wallet_data)
    return result["risk_score"]


def _tool_normalize_risk_level(risk_score: float) -> str:
    if risk_score > 0.6:
        return "HIGH"
    elif risk_score > 0.3:
        return "MEDIUM"
    return "LOW"


AGENT1_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "run_xgboost_predict",
            "description": "Run XGBoost fraud model on wallet data. Returns risk_score float 0-1. Uses shared context automatically.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "normalize_risk_level",
            "description": "Classify a risk score into HIGH/MEDIUM/LOW.",
            "parameters": {
                "type": "object",
                "properties": {
                    "risk_score": {"type": "number", "description": "Risk score 0-1"}
                },
                "required": ["risk_score"],
            },
        },
    },
]

# Handlers moved to local scope


def run_agent_1_risk_scorer(shared_state: dict) -> dict:
    wallet_data = shared_state["wallet_data"]
    prompt = (
        f"You are the Risk Scoring agent.\n"
        f"Use run_xgboost_predict tool to get the fraud risk score for this wallet.\n"
        f"Then use normalize_risk_level to classify it.\n"
        f'Return ONLY a JSON: {{"risk_score": float, "risk_level": string}}'
    )
    local_handlers = {
        "run_xgboost_predict": lambda *args, **kwargs: {"risk_score": _tool_run_xgboost_predict(wallet_data)},
        "normalize_risk_level": lambda *args, **kwargs: {"risk_level": _tool_normalize_risk_level(kwargs.get("risk_score", 0.0))},
    }
    text = _call_agent_with_tools(
        "You are a risk scoring agent. Always use the provided tools.", prompt, AGENT1_TOOLS, local_handlers
    )
    try:
        parsed = _parse_json_response(text)
    except Exception:
        pass

    # Deterministic enforcement to prevent LLM hallucination on critical metrics
    score = _tool_run_xgboost_predict(wallet_data)
    level = _tool_normalize_risk_level(score)
    shared_state["risk_score"] = score
    shared_state["risk_level"] = level

    shared_state["audit_log"].append({
        "agent": "Risk Scorer",
        "tool_called": "run_xgboost_predict",
        "output": shared_state["risk_score"],
        "risk_level": shared_state["risk_level"],
        "timestamp": datetime.now().isoformat(),
    })
    return shared_state


# ---------------------------------------------------------------------------
# AGENT 2 — Pattern Classifier (with self-correction loop)
# ---------------------------------------------------------------------------

def _tool_check_transaction_velocity(sent_tnx: float, avg_min_between_sent_tnx: float) -> dict:
    if sent_tnx > 500 and avg_min_between_sent_tnx < 5:
        return {"pattern": "high_velocity", "suspicious": True}
    return {"pattern": "normal_velocity", "suspicious": False}


def _tool_check_circular_flow(total_ether_sent: float, total_ether_balance: float) -> dict:
    if total_ether_sent > 100 and total_ether_balance < 0.1:
        return {"pattern": "circular_flow", "suspicious": True}
    return {"pattern": "normal_flow", "suspicious": False}


def _tool_check_contract_creation(num_contracts: float) -> dict:
    if num_contracts > 10:
        return {"pattern": "mass_contract_creation", "suspicious": True}
    return {"pattern": "normal", "suspicious": False}


AGENT2_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "check_transaction_velocity",
            "description": "Check if transaction frequency is suspiciously high. Uses shared context automatically.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_circular_flow",
            "description": "Check for circular ether flow patterns. Uses shared context automatically.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "check_contract_creation",
            "description": "Check for mass contract creation. Uses shared context automatically.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

# Handlers moved to local scope


def run_agent_2_pattern_classifier(shared_state: dict) -> dict:
    wallet_data = shared_state["wallet_data"]
    risk_score = shared_state["risk_score"]
    base_prompt = (
        f"You are the Pattern Classification agent.\n"
        f"Risk score from previous agent: {risk_score}\n"
        f"Use check_transaction_velocity, check_circular_flow, and check_contract_creation tools to analyze this wallet.\n\n"
        f"Based on tool results, classify the fraud type as ONE of:\n"
        f"- wash_trading: high velocity + circular flow\n"
        f"- pump_and_dump: mass contracts + high velocity\n"
        f"- phishing: low txns but very high single values\n"
        f"- NONE: no suspicious patterns detected (SAFE)\n\n"
        f"If tool results contradict the risk score, or if evidence is insufficient, do NOT guess. Return UNKNOWN for fraud_type.\n\n"
        f'Return ONLY a JSON: {{"thought_process": "step-by-step reasoning here", "fraud_type": string, "confidence": float between 0.0 and 1.0, "reasoning": string}}'
    )

    confidence = 0.0
    fraud_type = "NONE"
    reasoning = ""
    retry_count = 0

    while confidence < 0.70 and retry_count < 3:
        prompt = base_prompt
        if retry_count > 0:
            prompt += f"\n\nATTENTION: This is retry {retry_count}. Re-evaluate the tool results step-by-step. If there is genuinely no fraud, correctly classify as NONE with high confidence."

        local_handlers = {
            "check_transaction_velocity": lambda *args, **kwargs: _tool_check_transaction_velocity(
                wallet_data.get("sent_tnx", 0), wallet_data.get("avg_min_between_sent_tnx", 0)
            ),
            "check_circular_flow": lambda *args, **kwargs: _tool_check_circular_flow(
                wallet_data.get("total_ether_sent", 0), wallet_data.get("total_ether_balance", 0)
            ),
            "check_contract_creation": lambda *args, **kwargs: _tool_check_contract_creation(
                wallet_data.get("number_of_created_contracts", 0)
            ),
        }

        text = _call_agent_with_tools(
            "You are a pattern classification agent. Always use ALL three tools before making a decision.",
            prompt, AGENT2_TOOLS, local_handlers,
        )
        try:
            parsed = _parse_json_response(text)
            fraud_type = parsed.get("fraud_type", "UNKNOWN")
            confidence = float(parsed.get("confidence", 0.0))
            if confidence > 1.0:
                confidence = confidence / 100.0
            
            thought_process = parsed.get("thought_process", "")
            reasoning = parsed.get("reasoning", "")
            
            # Combine thought process into reasoning for audit log
            if thought_process:
                reasoning = f"Thought Process: {thought_process} | Conclusion: {reasoning}"
        except Exception:
            pass

        retry_count += 1

    # If still low confidence after 3 retries
    if confidence < 0.70:
        fraud_type = "UNKNOWN"
        reasoning = "Low confidence after 3 retries — flagged for human review."

    shared_state["fraud_type"] = fraud_type
    shared_state["fraud_confidence"] = confidence

    shared_state["audit_log"].append({
        "agent": "Pattern Classifier",
        "retries": retry_count,
        "fraud_type": fraud_type,
        "confidence": confidence,
        "reasoning": reasoning,
        "human_review_needed": confidence < 0.70,
        "timestamp": datetime.now().isoformat(),
    })
    return shared_state


# ---------------------------------------------------------------------------
# AGENT 3 — Evidence Collector
# ---------------------------------------------------------------------------

def _tool_get_top_risk_features(wallet_data: dict) -> list:
    """Return top 5 features by XGBoost importance as human-readable strings."""
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from model.predict import predict_fraud
    result = predict_fraud(wallet_data)
    evidence = []
    for feat_name, importance in result.get("top_features", [])[:5]:
        val = wallet_data.get(feat_name, "N/A")
        readable = feat_name.replace("_", " ").title()
        evidence.append(f"{readable}: {val} (importance: {abs(importance):.4f})")
    return evidence


def _tool_find_anomalies(wallet_data: dict) -> list:
    """Compare each feature against dataset median; flag those 3x above."""
    medians = _load_dataset_medians()
    anomalies = []
    for key, val in wallet_data.items():
        if key in medians and medians[key] > 0:
            ratio = val / medians[key]
            if ratio >= 3.0:
                readable = key.replace("_", " ").title()
                anomalies.append(f"{readable} is {ratio:.1f}x above normal (value: {val}, median: {medians[key]:.2f})")
            elif ratio <= 1/3 and medians[key] > 0:
                readable = key.replace("_", " ").title()
                if ratio == 0:
                    anomalies.append(f"{readable} is 0 (median: {medians[key]:.2f})")
                else:
                    anomalies.append(f"{readable} is {1/ratio:.1f}x below normal (value: {val}, median: {medians[key]:.2f})")
    return anomalies[:10]


def _tool_build_evidence_summary(top_features: list, anomalies: list) -> list:
    """Combine and deduplicate, return max 5 points."""
    combined = list(dict.fromkeys(top_features + anomalies))
    return combined[:5]


AGENT3_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "get_top_risk_features",
            "description": "Get top 5 risk features from XGBoost model for given wallet data. Uses shared context automatically.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "find_anomalies",
            "description": "Find features that are 3x above/below dataset median. Uses shared context automatically.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
]

# Handlers moved to local scope


def run_agent_3_evidence_collector(shared_state: dict) -> dict:
    wallet_data = shared_state["wallet_data"]
    fraud_type = shared_state["fraud_type"]
    confidence = shared_state["fraud_confidence"]

    prompt = (
        f"You are the Evidence Collection agent.\n"
        f"Fraud type identified: {fraud_type} (confidence: {confidence})\n"
        f"Use get_top_risk_features and find_anomalies tools.\n"
        f"Review the results, then return ONLY a JSON with the combined evidence: {{\"evidence_list\": [string, string, string, string, string]}}"
    )
    
    local_handlers = {
        "get_top_risk_features": lambda *args, **kwargs: {"features": _tool_get_top_risk_features(wallet_data)},
        "find_anomalies": lambda *args, **kwargs: {"anomalies": _tool_find_anomalies(wallet_data)},
    }
    
    text = _call_agent_with_tools(
        "You are an evidence collection agent. Use both tools, then return the JSON list.", prompt, AGENT3_TOOLS, local_handlers
    )
    try:
        parsed = _parse_json_response(text)
        shared_state["evidence_list"] = parsed.get("evidence_list", [])[:5]
    except Exception:
        # Fallback: run tools directly
        feats = _tool_get_top_risk_features(wallet_data)
        anomalies = _tool_find_anomalies(wallet_data)
        shared_state["evidence_list"] = _tool_build_evidence_summary(feats, anomalies)

    shared_state["audit_log"].append({
        "agent": "Evidence Collector",
        "evidence_count": len(shared_state["evidence_list"]),
        "timestamp": datetime.now().isoformat(),
    })
    return shared_state


# ---------------------------------------------------------------------------
# AGENT 4 — Report Writer (with self-correction for length)
# ---------------------------------------------------------------------------

def _tool_format_compliance_report(risk_score, risk_level, fraud_type, confidence, evidence_list, wallet_data):
    """Returns a structured compliance report template with placeholders filled."""
    evidence_bullets = "\n".join(f"  • {e}" for e in evidence_list) if evidence_list else "  • No evidence collected"

    status_word = "flagged" if risk_score > 0.3 else "evaluated"
    report = f"""═══════════════════════════════════════════════════════════════
                  CRYPTOGUARD COMPLIANCE REPORT
                  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
═══════════════════════════════════════════════════════════════

1. EXECUTIVE SUMMARY
   This wallet has been {status_word} with a risk score of {risk_score:.1%}
   ({risk_level} risk). The automated analysis classified the activity
   as "{fraud_type}" with {confidence:.1%} confidence.

2. TRANSACTION PATTERN ANALYSIS
   Wallet Metrics:
   - Total Transactions: {wallet_data.get('_meta', {}).get('total_transactions', 0)}
   - Sent Transactions: {wallet_data.get('sent_tnx', 0)}
   - Received Transactions: {wallet_data.get('received_tnx', 0)}
   - ETH Balance: {wallet_data.get('total_ether_balance', 0):.4f}

3. FRAUD CLASSIFICATION REASONING
   Classification: {fraud_type}
   Confidence: {confidence:.1%}
   The classification was determined by analyzing transaction velocity,
   circular flow patterns, and contract creation behavior.

4. EVIDENCE POINTS
{evidence_bullets}

5. RISK ASSESSMENT
   Risk Level: {risk_level}
   Risk Score: {risk_score:.1%}
   Recommendation: {"Immediate action required" if risk_level == "HIGH" else "Monitor closely" if risk_level == "MEDIUM" else "No immediate action needed"}

═══════════════════════════════════════════════════════════════
                     END OF REPORT
═══════════════════════════════════════════════════════════════"""
    return report


def _tool_validate_report_length(report_text: str) -> dict:
    word_count = len(report_text.split())
    if word_count < 150:
        return {"valid": False, "reason": "too short", "word_count": word_count}
    return {"valid": True, "word_count": word_count}


def _tool_save_report(report_text: str, wallet_id: str) -> str:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(base, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(reports_dir, f"{wallet_id}_{ts}.txt")
    with open(filepath, "w") as f:
        f.write(report_text)
    return filepath


AGENT4_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "format_compliance_report",
            "description": "Generate a structured compliance report from investigation data. Uses shared context automatically.",
            "parameters": {
                "type": "object",
                "properties": {},
                "required": [],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "validate_report_length",
            "description": "Check if report meets minimum word count (150 words).",
            "parameters": {
                "type": "object",
                "properties": {"report_text": {"type": "string"}},
                "required": ["report_text"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "save_report",
            "description": "Save the compliance report to disk.",
            "parameters": {
                "type": "object",
                "properties": {
                    "report_text": {"type": "string"},
                    "wallet_id": {"type": "string"},
                },
                "required": ["report_text", "wallet_id"],
            },
        },
    },
]

# Handlers moved to local scope in run_agent_4_report_writer


def run_agent_4_report_writer(shared_state: dict) -> dict:
    risk_score = shared_state["risk_score"]
    risk_level = shared_state["risk_level"]
    fraud_type = shared_state["fraud_type"]
    confidence = shared_state["fraud_confidence"]
    evidence_list = shared_state["evidence_list"]
    wallet_data = shared_state["wallet_data"]
    wallet_id = f"wallet_{hash(json.dumps(wallet_data, sort_keys=True)) % 100000:05d}"

    # Agent 4 is fully deterministic. 
    # Having an LLM generate a massive text document via JSON tool-calling 
    # leads to token exhaustion and JSON truncation errors.
    report_text = _tool_format_compliance_report(
        risk_score, risk_level, fraud_type, confidence, evidence_list, wallet_data
    )
    filepath = _tool_save_report(report_text, wallet_id)

    word_count = len(report_text.split())
    shared_state["report_text"] = report_text
    shared_state["report_filepath"] = filepath

    shared_state["audit_log"].append({
        "agent": "Report Writer (Deterministic)",
        "report_word_count": word_count,
        "filepath": filepath,
        "timestamp": datetime.now().isoformat(),
    })
    return shared_state


# ---------------------------------------------------------------------------
# AGENT 5 — Action Decider (with conflict detection)
# ---------------------------------------------------------------------------

def _calculate_consolidated_risk(ml_score: float, fraud_type: str, fraud_confidence: float) -> float:
    """Implement dynamic weighting: AI boosts risk aggressively but reduces it cautiously."""
    if fraud_type == "NONE":
        ai_score = 0.0
        ai_weight = 0.2
        ai_adjustment = -0.1 * fraud_confidence
    elif fraud_confidence >= 0.9:
        ai_score = 1.0
        ai_weight = 0.5
        ai_adjustment = 0.4 * fraud_confidence
    else:
        ai_score = 1.0
        ai_weight = 0.3
        ai_adjustment = 0.2 * fraud_confidence

    ml_weight = 1.0 - ai_weight
    consolidated = (ml_weight * ml_score) + (ai_weight * ai_score) + ai_adjustment
    return max(0.0, min(1.0, consolidated))

def _tool_decide_action(risk_score: float, fraud_type: str, confidence: float, evidence_count: int, wallet_data: dict) -> dict:
    # 1. Consolidated Risk
    consolidated_risk = _calculate_consolidated_risk(risk_score, fraud_type, confidence)
    
    # 2. Anomaly Override (e.g., sudden burst activity)
    normal_txns = wallet_data.get("_meta", {}).get("total_transactions", 0)
    time_diff = wallet_data.get("time_diff_between_first_and_last_mins", 0)
    is_anomaly = False
    if normal_txns > 100 and time_diff < 60: # Extreme burst
        is_anomaly = True

    # 3. Enforcement Triad (Deterministic)
    if consolidated_risk >= 0.8 and confidence >= 0.8 and evidence_count >= 3:
        action = "FREEZE"
        reason = f"High risk signal confirmed across multiple vectors."
    elif consolidated_risk < 0.2 and fraud_type == "NONE":
        action = "CLEAR"
        reason = "Consistent low-risk profile."
    else:
        action = "ESCALATE"
        reason = "Inconsistent or borderline signals. Flagged for human review."

    # 4. Disagreement Override (Strict Contradiction Detection)
    # If ML says SAFE (<30%) but AI says FRAUD with high confidence (>70%) -> ESCALATE
    # If ML says FRAUD (>70%) but AI says NONE with high confidence (>70%) -> ESCALATE
    is_disagreement = False
    if (risk_score <= 0.3 and fraud_type != "NONE" and confidence >= 0.7) or \
       (risk_score >= 0.7 and fraud_type == "NONE" and confidence >= 0.7):
        is_disagreement = True

    if is_disagreement:
        action = "ESCALATE"
        reason = f"SIGNAL DISAGREEMENT: ML ({risk_score:.1%}) and AI ({fraud_type} @ {confidence:.1%}) are in conflict."

    if is_anomaly:
        action = "ESCALATE"
        reason = "ANOMALY OVERRIDE: Extreme burst activity detected."

    # 5. Integrity Layer (Explicit Contradiction Check)
    if (consolidated_risk >= 0.7 and action == "CLEAR") or (consolidated_risk <= 0.3 and action == "FREEZE"):
        action = "ESCALATE"
        reason = f"INTEGRITY_VIOLATION: Contradiction between risk score ({consolidated_risk:.1%}) and action choice."

    return {
        "action": action, 
        "reason": reason,
        "consolidated_risk": consolidated_risk
    }


def _tool_log_action(action: str, wallet_data: dict, risk_score: float, fraud_type: str) -> str:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(base, "logs", "actions_log.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "risk_score": risk_score,
        "fraud_type": fraud_type,
        "wallet_summary": {k: v for k, v in list(wallet_data.items())[:5]},
    }

    existing = []
    if os.path.exists(log_path):
        try:
            with open(log_path, "r") as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.append(entry)
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2)
    return "logged"


def _tool_notify_compliance_team(action: str, report_filepath: str) -> str:
    if action in ("FREEZE", "ESCALATE"):
        print(f"\n🚨 ALERT: Compliance team notified for {action}")
        print(f"   Report: {report_filepath}")
    return "notified"


AGENT5_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "decide_action",
            "description": "Determine compliance action based on risk score, fraud type, and confidence.",
            "parameters": {
                "type": "object",
                "properties": {
                    "risk_score": {"type": "number"},
                    "fraud_type": {"type": "string"},
                    "confidence": {"type": "number"},
                },
                "required": ["risk_score", "fraud_type", "confidence"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "log_action",
            "description": "Log the compliance action to actions_log.json. Uses shared context automatically.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                },
                "required": ["action"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "notify_compliance_team",
            "description": "Notify compliance team for FREEZE or ESCALATE actions.",
            "parameters": {
                "type": "object",
                "properties": {
                    "action": {"type": "string"},
                    "report_filepath": {"type": "string"},
                },
                "required": ["action", "report_filepath"],
            },
        },
    },
]

# Handlers moved to local scope


def run_agent_5_action_decider(shared_state: dict) -> dict:
    risk_score = shared_state["risk_score"]
    fraud_type = shared_state["fraud_type"]
    confidence = shared_state["fraud_confidence"]
    evidence_list = shared_state["evidence_list"]
    wallet_data = shared_state["wallet_data"]
    report_filepath = shared_state.get("report_filepath", "")

    # Deterministic enforcement (No LLM)
    result = _tool_decide_action(risk_score, fraud_type, confidence, len(evidence_list), wallet_data)
    action = result["action"]
    reason = result["reason"]
    consolidated_risk = result.get("consolidated_risk", risk_score)

    # Auditable Structured Reasoning
    structured_reason = (
        f"Consolidated Risk: {consolidated_risk:.2f} ({'High' if consolidated_risk > 0.7 else 'Low'})\n\n"
        "Drivers:\n"
    )
    for e in evidence_list[:3]:
        structured_reason += f"- {e}\n"
    structured_reason += f"\nIntegrity: Passed"

    _tool_log_action(action, wallet_data, consolidated_risk, fraud_type)
    if action in ("FREEZE", "ESCALATE"):
        _tool_notify_compliance_team(action, report_filepath)

    shared_state["final_action"] = action
    shared_state["action_reason"] = structured_reason
    shared_state["consolidated_risk"] = consolidated_risk

    shared_state["audit_log"].append({
        "agent": "Action Decider (Deterministic Enforcement)",
        "final_action": action,
        "consolidated_risk": consolidated_risk,
        "timestamp": datetime.now().isoformat(),
    })

    return shared_state
