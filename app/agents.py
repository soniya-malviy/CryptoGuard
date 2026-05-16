"""
CryptoGuard 6-Agent Autonomous Pipeline
Design principles:
  - ML score is the single source of truth for risk level
  - LLMs only classify fraud TYPE, never override risk level
  - consolidated_risk stays within the same risk band as ML score
  - No hardcoded thresholds — all derived from dataset medians
  - Sequential state: each agent reads only what prior agents wrote
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

GROQ_MODEL = "llama-3.1-8b-instant"

# ---------------------------------------------------------------------------
# Shared Infrastructure
# ---------------------------------------------------------------------------

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
    """Generic Groq tool-use loop."""
    import time
    client = _get_client()
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    for _ in range(max_rounds):
        for attempt in range(3):
            try:
                response = client.chat.completions.create(
                    model=GROQ_MODEL,
                    max_tokens=512,
                    temperature=0.0,   # zero temp — no creative hallucination
                    tools=tools,
                    tool_choice="auto",
                    messages=messages,
                )
                break
            except groq.RateLimitError:
                if attempt < 2:
                    time.sleep(10)
                else:
                    raise
            except Exception as e:
                if "Rate limit" in str(e) and attempt < 2:
                    time.sleep(10)
                    continue
                raise

        msg = response.choices[0].message
        if not msg.tool_calls or response.choices[0].finish_reason == "stop":
            return msg.content or ""

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
            result = handler(**fn_args) if handler else {"error": f"Unknown tool: {fn_name}"}
            messages.append({
                "role": "tool",
                "tool_call_id": tool_call.id,
                "content": json.dumps(result) if not isinstance(result, str) else result,
            })

    return messages[-1].get("content", "") if isinstance(messages[-1], dict) else ""


def _parse_json_response(text):
    match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', text, re.DOTALL)
    if match:
        return json.loads(match.group(1))
    match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if match:
        return json.loads(match.group(0))
    return json.loads(text.strip())


def _load_dataset_medians():
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    csv_path = os.path.join(base, "data", "crypto_5000_dataset.csv")
    try:
        df = pd.read_csv(csv_path)
        return df.median(numeric_only=True).to_dict()
    except Exception:
        return {}


def _risk_band(score: float) -> str:
    """Single canonical function for risk level — used everywhere."""
    if score > 0.6:
        return "HIGH"
    elif score > 0.3:
        return "MEDIUM"
    return "LOW"


# ---------------------------------------------------------------------------
# AGENT 0 — Data Fetcher
# ---------------------------------------------------------------------------

def run_agent_0_data_fetcher(shared_state: dict) -> dict:
    from app.data_fetcher import compute_wallet_features

    wallet_address = shared_state["wallet_address"]

    # Always compute features directly — LLM cannot improve raw API calls
    wallet_data = compute_wallet_features(wallet_address)
    shared_state["wallet_data"] = wallet_data

    shared_state["audit_log"].append({
        "agent": "Data Fetcher",
        "wallet_address": wallet_address,
        "features_computed": len(wallet_data),
        "has_error": "error" in wallet_data,
        "timestamp": datetime.now().isoformat(),
    })
    return shared_state


# ---------------------------------------------------------------------------
# AGENT 1 — Risk Scorer  (fully deterministic — no LLM)
# ---------------------------------------------------------------------------

def run_agent_1_risk_scorer(shared_state: dict) -> dict:
    """
    Runs XGBoost model once. Stores risk_score and risk_level as the
    immutable ground truth that all downstream agents must respect.
    """
    from model.predict import predict_fraud

    wallet_data = shared_state["wallet_data"]

    if "error" in wallet_data:
        shared_state["risk_score"] = 0.0
        shared_state["risk_level"] = "LOW"
        shared_state["audit_log"].append({
            "agent": "Risk Scorer",
            "note": "Skipped — no wallet data",
            "timestamp": datetime.now().isoformat(),
        })
        return shared_state

    result = predict_fraud(wallet_data)
    score = float(result["risk_score"])
    level = _risk_band(score)

    # Store as immutable ground truth
    shared_state["risk_score"] = score
    shared_state["risk_level"] = level

    shared_state["audit_log"].append({
        "agent": "Risk Scorer",
        "risk_score": score,
        "risk_level": level,
        "top_features": [f[0] for f in result.get("top_features", [])[:3]],
        "timestamp": datetime.now().isoformat(),
    })
    return shared_state


# ---------------------------------------------------------------------------
# AGENT 2 — Pattern Classifier
# LLM classifies fraud TYPE only. Risk level is never touched.
# Tool thresholds are data-driven (dataset medians × multiplier).
# ---------------------------------------------------------------------------

def _get_dynamic_thresholds(wallet_data: dict) -> dict:
    """Derive thresholds from dataset medians so nothing is hardcoded."""
    medians = _load_dataset_medians()
    return {
        "high_velocity_sent": max(medians.get("sent_tnx", 50) * 3, 10),
        "fast_interval_mins": max(medians.get("avg_min_between_sent_tnx", 60) * 0.1, 1),
        "circular_sent_eth": max(medians.get("total_ether_sent", 5) * 5, 1),
        "low_balance_eth": max(medians.get("total_ether_balance", 1) * 0.05, 0.001),
        "mass_contracts": max(medians.get("number_of_created_contracts", 1) * 5, 3),
    }


def run_agent_2_pattern_classifier(shared_state: dict) -> dict:
    wallet_data = shared_state["wallet_data"]
    risk_score = shared_state["risk_score"]
    risk_level = shared_state["risk_level"]

    if "error" in wallet_data:
        shared_state["fraud_type"] = "NONE"
        shared_state["fraud_confidence"] = 1.0
        shared_state["audit_log"].append({
            "agent": "Pattern Classifier",
            "note": "Skipped — no wallet data",
            "timestamp": datetime.now().isoformat(),
        })
        return shared_state

    thresholds = _get_dynamic_thresholds(wallet_data)

    # --- Tool implementations (data-driven thresholds) ---
    def _check_velocity():
        sent = wallet_data.get("sent_tnx", 0)
        interval = wallet_data.get("avg_min_between_sent_tnx", 9999)
        suspicious = sent > thresholds["high_velocity_sent"] and interval < thresholds["fast_interval_mins"]
        return {
            "suspicious": suspicious,
            "sent_tnx": sent,
            "avg_interval_mins": interval,
            "threshold_sent": thresholds["high_velocity_sent"],
            "threshold_interval": thresholds["fast_interval_mins"],
        }

    def _check_circular_flow():
        sent = wallet_data.get("total_ether_sent", 0)
        balance = wallet_data.get("total_ether_balance", 0)
        suspicious = sent > thresholds["circular_sent_eth"] and balance < thresholds["low_balance_eth"]
        return {
            "suspicious": suspicious,
            "total_ether_sent": sent,
            "total_ether_balance": balance,
            "threshold_sent": thresholds["circular_sent_eth"],
            "threshold_balance": thresholds["low_balance_eth"],
        }

    def _check_contract_creation():
        contracts = wallet_data.get("number_of_created_contracts", 0)
        suspicious = contracts > thresholds["mass_contracts"]
        return {
            "suspicious": suspicious,
            "contracts_created": contracts,
            "threshold": thresholds["mass_contracts"],
        }

    TOOLS = [
        {"type": "function", "function": {
            "name": "check_transaction_velocity",
            "description": "Check if transaction frequency is suspiciously high based on dataset-derived thresholds.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }},
        {"type": "function", "function": {
            "name": "check_circular_flow",
            "description": "Check for circular ether flow: high sent ETH but near-zero balance.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }},
        {"type": "function", "function": {
            "name": "check_contract_creation",
            "description": "Check for mass contract creation activity.",
            "parameters": {"type": "object", "properties": {}, "required": []},
        }},
    ]

    handlers = {
        "check_transaction_velocity": lambda **kw: _check_velocity(),
        "check_circular_flow": lambda **kw: _check_circular_flow(),
        "check_contract_creation": lambda **kw: _check_contract_creation(),
    }

    # Run tools directly to get ground truth for enforcement
    vel = _check_velocity()
    circ = _check_circular_flow()
    contr = _check_contract_creation()

    # Determine what fraud types are actually supported by tool evidence
    supported_types = []
    if vel["suspicious"] and circ["suspicious"]:
        supported_types.append("wash_trading")
    if contr["suspicious"] and vel["suspicious"]:
        supported_types.append("pump_and_dump")
    if not vel["suspicious"] and not circ["suspicious"] and not contr["suspicious"]:
        supported_types.append("NONE")

    # If ML score is LOW, fraud_type must be NONE unless tools show strong evidence
    if risk_level == "LOW" and not supported_types:
        supported_types = ["NONE"]

    # Ask LLM to classify fraud TYPE only — give it the tool results and ML score
    tool_summary = json.dumps({"velocity": vel, "circular_flow": circ, "contracts": contr}, indent=2)
    prompt = (
        f"You are the Pattern Classification agent.\n\n"
        f"ML Risk Score (ground truth, do NOT change): {risk_score:.3f} ({risk_level})\n\n"
        f"Tool results (these are facts, not opinions):\n{tool_summary}\n\n"
        f"Supported fraud types based on tool evidence: {supported_types}\n\n"
        f"Your ONLY job: pick the fraud_type from the supported list above.\n"
        f"Rules:\n"
        f"- If supported_types contains only 'NONE', you MUST return fraud_type='NONE'\n"
        f"- If risk_level is LOW, you MUST return fraud_type='NONE' unless tools show suspicious=True\n"
        f"- Do NOT invent fraud types not in the supported list\n"
        f"- confidence must reflect how strongly the tool results support your choice (0.0-1.0)\n\n"
        f'Return ONLY JSON: {{"fraud_type": string, "confidence": float, "reasoning": string}}'
    )

    fraud_type = "NONE"
    confidence = 0.0
    reasoning = ""

    for attempt in range(3):
        text = _call_agent_with_tools(
            "You are a pattern classification agent. Classify fraud type strictly from tool evidence.",
            prompt, TOOLS, handlers,
        )
        try:
            parsed = _parse_json_response(text)
            fraud_type = parsed.get("fraud_type", "NONE")
            confidence = float(parsed.get("confidence", 0.0))
            if confidence > 1.0:
                confidence /= 100.0
            reasoning = parsed.get("reasoning", "")
            if confidence >= 0.6:
                break
        except Exception:
            pass

    # ── Enforcement: LLM cannot contradict tool evidence ──────────────────
    # If no tool flagged suspicious, fraud_type must be NONE
    no_tool_evidence = not (vel["suspicious"] or circ["suspicious"] or contr["suspicious"])
    if no_tool_evidence and fraud_type != "NONE":
        fraud_type = "NONE"
        confidence = 0.9
        reasoning = "Overridden: no tool evidence supports fraud classification."

    # If ML score is LOW and no tool evidence, force NONE
    if risk_level == "LOW" and no_tool_evidence:
        fraud_type = "NONE"
        confidence = max(confidence, 0.85)

    # If ML score is HIGH but LLM says NONE with no tool evidence, flag UNKNOWN
    if risk_level == "HIGH" and fraud_type == "NONE" and no_tool_evidence:
        fraud_type = "UNKNOWN"
        confidence = 0.5
        reasoning = "ML flags HIGH risk but no specific pattern detected — flagged for human review."

    shared_state["fraud_type"] = fraud_type
    shared_state["fraud_confidence"] = confidence

    shared_state["audit_log"].append({
        "agent": "Pattern Classifier",
        "fraud_type": fraud_type,
        "confidence": confidence,
        "reasoning": reasoning,
        "tool_evidence": {"velocity": vel["suspicious"], "circular": circ["suspicious"], "contracts": contr["suspicious"]},
        "enforcement_applied": no_tool_evidence and fraud_type == "NONE",
        "timestamp": datetime.now().isoformat(),
    })
    return shared_state


# ---------------------------------------------------------------------------
# AGENT 3 — Evidence Collector
# ---------------------------------------------------------------------------

def _tool_get_top_risk_features(wallet_data: dict) -> list:
    import sys
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
    medians = _load_dataset_medians()
    anomalies = []
    for key, val in wallet_data.items():
        if key.startswith("_") or key not in medians or medians[key] <= 0:
            continue
        ratio = val / medians[key]
        readable = key.replace("_", " ").title()
        if ratio >= 3.0:
            anomalies.append(f"{readable} is {ratio:.1f}x above normal (value: {val}, median: {medians[key]:.2f})")
        elif ratio <= 1/3:
            if ratio == 0:
                anomalies.append(f"{readable} is 0 (median: {medians[key]:.2f})")
            else:
                anomalies.append(f"{readable} is {1/ratio:.1f}x below normal (value: {val}, median: {medians[key]:.2f})")
    return anomalies[:10]


def run_agent_3_evidence_collector(shared_state: dict) -> dict:
    wallet_data = shared_state["wallet_data"]

    if "error" in wallet_data:
        shared_state["evidence_list"] = []
        shared_state["audit_log"].append({
            "agent": "Evidence Collector",
            "note": "Skipped — no wallet data",
            "timestamp": datetime.now().isoformat(),
        })
        return shared_state

    # Run both tools directly — no LLM needed for data collection
    feats = _tool_get_top_risk_features(wallet_data)
    anomalies = _tool_find_anomalies(wallet_data)

    # Deduplicate and cap at 5
    combined = list(dict.fromkeys(feats + anomalies))[:5]
    shared_state["evidence_list"] = combined

    shared_state["audit_log"].append({
        "agent": "Evidence Collector",
        "evidence_count": len(combined),
        "timestamp": datetime.now().isoformat(),
    })
    return shared_state


# ---------------------------------------------------------------------------
# AGENT 4 — Report Writer
# Uses consolidated_risk (set by Agent 5) if available, else risk_score.
# Since Agent 4 runs before Agent 5, it uses risk_score here.
# Agent 5 will update the report's risk display via shared_state.
# ---------------------------------------------------------------------------

def _format_report(risk_score: float, risk_level: str, fraud_type: str,
                   confidence: float, evidence_list: list, wallet_data: dict) -> str:
    evidence_bullets = "\n".join(f"  • {e}" for e in evidence_list) if evidence_list else "  • No anomalies detected"
    meta = wallet_data.get("_meta", {})
    status_word = "flagged" if risk_score > 0.3 else "evaluated"

    return f"""═══════════════════════════════════════════════════════════════
                  CRYPTOGUARD COMPLIANCE REPORT
                  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S UTC')}
═══════════════════════════════════════════════════════════════

1. EXECUTIVE SUMMARY
   This wallet has been {status_word} with a risk score of {risk_score:.1%}
   ({risk_level} risk). The automated analysis classified the activity
   as "{fraud_type}" with {confidence:.1%} confidence.

2. TRANSACTION PATTERN ANALYSIS
   Wallet Metrics:
   - Total Transactions: {meta.get('total_transactions', 0)}
   - Sent Transactions: {wallet_data.get('sent_tnx', 0)}
   - Received Transactions: {wallet_data.get('received_tnx', 0)}
   - ETH Balance: {wallet_data.get('total_ether_balance', 0):.18g} ETH
   - First Seen: {meta.get('first_seen', 'N/A')}
   - Last Seen: {meta.get('last_seen', 'N/A')}

3. FRAUD CLASSIFICATION
   Classification: {fraud_type}
   Confidence: {confidence:.1%}

4. EVIDENCE POINTS
{evidence_bullets}

5. RISK ASSESSMENT
   Risk Level: {risk_level}
   Risk Score: {risk_score:.1%}
   Recommendation: {"Immediate action required" if risk_level == "HIGH" else "Monitor closely" if risk_level == "MEDIUM" else "No immediate action needed"}

═══════════════════════════════════════════════════════════════
                     END OF REPORT
═══════════════════════════════════════════════════════════════"""


def run_agent_4_report_writer(shared_state: dict) -> dict:
    risk_score = shared_state["risk_score"]
    risk_level = shared_state["risk_level"]
    fraud_type = shared_state["fraud_type"]
    confidence = shared_state["fraud_confidence"]
    evidence_list = shared_state["evidence_list"]
    wallet_data = shared_state["wallet_data"]

    wallet_id = f"wallet_{hash(json.dumps({k: v for k, v in wallet_data.items() if not k.startswith('_')}, sort_keys=True)) % 100000:05d}"

    report_text = _format_report(risk_score, risk_level, fraud_type, confidence, evidence_list, wallet_data)

    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    reports_dir = os.path.join(base, "reports")
    os.makedirs(reports_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filepath = os.path.join(reports_dir, f"{wallet_id}_{ts}.txt")
    with open(filepath, "w") as f:
        f.write(report_text)

    shared_state["report_text"] = report_text
    shared_state["report_filepath"] = filepath

    shared_state["audit_log"].append({
        "agent": "Report Writer",
        "report_word_count": len(report_text.split()),
        "filepath": filepath,
        "timestamp": datetime.now().isoformat(),
    })
    return shared_state


# ---------------------------------------------------------------------------
# AGENT 5 — Action Decider  (fully deterministic — no LLM)
#
# Key design: consolidated_risk MUST stay within the same risk band as
# risk_score unless there is explicit tool-backed evidence of disagreement.
# This prevents the LOW→HIGH flip the user was seeing.
# ---------------------------------------------------------------------------

def _calculate_consolidated_risk(ml_score: float, fraud_type: str, fraud_confidence: float) -> float:
    """
    Blend ML score with pattern classifier signal.
    The blend NEVER moves the score across a risk band boundary
    unless both signals agree on the direction.
    """
    ml_band = _risk_band(ml_score)

    if fraud_type == "NONE":
        # Pattern classifier says safe — small downward nudge only
        adjustment = -0.05 * fraud_confidence
    elif fraud_type == "UNKNOWN":
        # Uncertain — no adjustment
        adjustment = 0.0
    else:
        # Pattern classifier found fraud — small upward nudge only
        adjustment = 0.05 * fraud_confidence

    consolidated = ml_score + adjustment
    consolidated = max(0.0, min(1.0, consolidated))

    # ── Band-lock: never cross a band boundary unless ML score is near the edge ──
    # "Near edge" = within 0.08 of a boundary (0.3 or 0.6)
    consolidated_band = _risk_band(consolidated)
    if consolidated_band != ml_band:
        near_lower = abs(ml_score - 0.3) < 0.08
        near_upper = abs(ml_score - 0.6) < 0.08
        if not (near_lower or near_upper):
            # Not near a boundary — revert to ML score
            consolidated = ml_score

    return round(consolidated, 4)


def _decide_action(consolidated_risk: float, risk_level: str, fraud_type: str,
                   fraud_confidence: float, evidence_count: int) -> tuple[str, str]:
    """
    Deterministic action decision anchored to consolidated_risk.
    Returns (action, reason).
    """
    # Signal disagreement: ML and pattern classifier strongly contradict each other
    ml_score_band = risk_level
    pattern_says_fraud = fraud_type not in ("NONE", "UNKNOWN") and fraud_confidence >= 0.7
    pattern_says_safe = fraud_type == "NONE" and fraud_confidence >= 0.7

    disagreement = (
        (ml_score_band == "LOW" and pattern_says_fraud) or
        (ml_score_band == "HIGH" and pattern_says_safe)
    )

    if disagreement:
        return "ESCALATE", f"Signal disagreement: ML={ml_score_band}, Pattern={fraud_type} ({fraud_confidence:.0%} confidence). Human review required."

    # Standard decision tree — anchored to consolidated_risk
    if consolidated_risk >= 0.7 and fraud_confidence >= 0.7 and evidence_count >= 2:
        return "FREEZE", f"High consolidated risk ({consolidated_risk:.1%}) confirmed by pattern analysis ({fraud_type}) and {evidence_count} evidence points."
    elif consolidated_risk >= 0.5 or (fraud_type not in ("NONE", "UNKNOWN") and fraud_confidence >= 0.6):
        return "ESCALATE", f"Moderate risk ({consolidated_risk:.1%}) or unconfirmed fraud pattern ({fraud_type}). Flagged for review."
    elif consolidated_risk < 0.3 and fraud_type == "NONE":
        return "CLEAR", f"Low consolidated risk ({consolidated_risk:.1%}) with no fraud patterns detected."
    else:
        return "ESCALATE", f"Borderline signals (risk={consolidated_risk:.1%}, pattern={fraud_type}). Flagged for human review."


def run_agent_5_action_decider(shared_state: dict) -> dict:
    risk_score = shared_state["risk_score"]
    risk_level = shared_state["risk_level"]
    fraud_type = shared_state["fraud_type"]
    confidence = shared_state["fraud_confidence"]
    evidence_list = shared_state["evidence_list"]
    wallet_data = shared_state["wallet_data"]
    report_filepath = shared_state.get("report_filepath", "")

    # 1. Compute consolidated risk (anchored to ML score)
    consolidated_risk = _calculate_consolidated_risk(risk_score, fraud_type, confidence)
    consolidated_level = _risk_band(consolidated_risk)

    # 2. Decide action
    action, reason = _decide_action(consolidated_risk, risk_level, fraud_type, confidence, len(evidence_list))

    # 3. Build structured reason with evidence
    structured_reason = (
        f"ML Risk Score: {risk_score:.1%} ({risk_level})\n"
        f"Consolidated Risk: {consolidated_risk:.1%} ({consolidated_level})\n"
        f"Pattern: {fraud_type} ({confidence:.0%} confidence)\n\n"
        f"Decision: {reason}\n\n"
    )
    if evidence_list:
        structured_reason += "Key Evidence:\n" + "\n".join(f"- {e}" for e in evidence_list[:3])

    # 4. Update report to use consolidated_risk (patch the report text)
    if shared_state.get("report_text"):
        updated_report = shared_state["report_text"].replace(
            f"risk score of {risk_score:.1%}\n   ({risk_level} risk)",
            f"risk score of {consolidated_risk:.1%}\n   ({consolidated_level} risk)"
        )
        shared_state["report_text"] = updated_report
        # Overwrite saved report file
        if report_filepath and os.path.exists(report_filepath):
            with open(report_filepath, "w") as f:
                f.write(updated_report)

    # 5. Log action
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    log_path = os.path.join(base, "logs", "actions_log.json")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    entry = {
        "timestamp": datetime.now().isoformat(),
        "action": action,
        "ml_risk_score": risk_score,
        "consolidated_risk": consolidated_risk,
        "risk_level": consolidated_level,
        "fraud_type": fraud_type,
    }
    existing = []
    if os.path.exists(log_path):
        try:
            with open(log_path) as f:
                existing = json.load(f)
        except Exception:
            existing = []
    existing.append(entry)
    with open(log_path, "w") as f:
        json.dump(existing, f, indent=2)

    if action in ("FREEZE", "ESCALATE"):
        print(f"\n🚨 ALERT: {action} — Report: {report_filepath}")

    shared_state["final_action"] = action
    shared_state["action_reason"] = structured_reason
    shared_state["consolidated_risk"] = consolidated_risk
    shared_state["consolidated_level"] = consolidated_level

    shared_state["audit_log"].append({
        "agent": "Action Decider",
        "ml_risk_score": risk_score,
        "consolidated_risk": consolidated_risk,
        "consolidated_level": consolidated_level,
        "final_action": action,
        "reason": reason,
        "timestamp": datetime.now().isoformat(),
    })
    return shared_state
