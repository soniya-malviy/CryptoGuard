import sys
import os
import json

# Add root directory to sys.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.agents import _tool_decide_action, _calculate_consolidated_risk

def test_logic_boundaries():
    print("🧪 Running CryptoGuard Logic Boundary Tests...")
    print("=" * 50)

    test_cases = [
        {
            "name": "Contradiction: ML High / AI None",
            "ml_score": 0.85,
            "fraud_type": "NONE",
            "confidence": 0.95,
            "evidence_count": 5,
            "expected_action": "ESCALATE",
            "desc": "ML thinks fraud, AI says NONE. Must ESCALATE."
        },
        {
            "name": "Contradiction: ML Low / AI Scam",
            "ml_score": 0.05,
            "fraud_type": "SCAM",
            "confidence": 0.95,
            "evidence_count": 5,
            "expected_action": "ESCALATE",
            "desc": "ML says SAFE, AI says SCAM. Must ESCALATE."
        },
        {
            "name": "Boundary: Just below FREEZE threshold",
            "ml_score": 0.79,
            "fraud_type": "HACK",
            "confidence": 0.79,
            "evidence_count": 3,
            "expected_action": "ESCALATE",
            "desc": "Risk/Confidence just under 80%. Must default to ESCALATE."
        },
        {
            "name": "Boundary: Just at FREEZE threshold",
            "ml_score": 0.81,
            "fraud_type": "HACK",
            "confidence": 0.95,
            "evidence_count": 3,
            "expected_action": "FREEZE",
            "desc": "Strict match for FREEZE triad."
        },
        {
            "name": "Anomaly: Extreme Burst activity",
            "ml_score": 0.10,
            "fraud_type": "NONE",
            "confidence": 0.99,
            "evidence_count": 0,
            "wallet_data": {"_meta": {"total_transactions": 500}, "time_diff_between_first_and_last_mins": 10},
            "expected_action": "ESCALATE",
            "desc": "Low risk metrics, but extreme burst activity detected. Anomaly override must fire."
        }
    ]

    passed = 0
    for case in test_cases:
        wallet_data = case.get("wallet_data", {"_meta": {"total_transactions": 1}, "time_diff_between_first_and_last_mins": 100})
        result = _tool_decide_action(
            case["ml_score"], 
            case["fraud_type"], 
            case["confidence"], 
            case["evidence_count"], 
            wallet_data
        )
        
        action = result["action"]
        consolidated = result.get("consolidated_risk", 0)
        
        if action == case["expected_action"]:
            print(f"✅ PASS: {case['name']}")
            print(f"   [Consolidated Risk: {consolidated:.2f}] -> {action}")
            passed += 1
        else:
            print(f"❌ FAIL: {case['name']}")
            print(f"   Expected: {case['expected_action']}, Got: {action}")
            print(f"   Reason: {result['reason']}")

    print("=" * 50)
    print(f"Result: {passed}/{len(test_cases)} tests passed.")

if __name__ == "__main__":
    test_logic_boundaries()
