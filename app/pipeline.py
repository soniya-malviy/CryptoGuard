"""
CryptoGuard Agent Pipeline Orchestrator
Runs all 5 agents sequentially, passing shared state between them.
"""

import json
import os
from datetime import datetime
from app.agents import (
    run_agent_0_data_fetcher,
    run_agent_1_risk_scorer,
    run_agent_2_pattern_classifier,
    run_agent_3_evidence_collector,
    run_agent_4_report_writer,
    run_agent_5_action_decider,
)


def run_full_pipeline(wallet_address: str, progress_callback=None) -> dict:
    """
    Run the complete 6-agent fraud investigation pipeline (Agent 0 to 5).
    
    Args:
        wallet_address: Ethereum wallet address string.
        progress_callback: Optional callable(agent_num, agent_name, status) for UI updates.
    
    Returns:
        shared_state dict with all investigation results.
    """
    # Ensure output directories exist
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.makedirs(os.path.join(base, "logs"), exist_ok=True)
    os.makedirs(os.path.join(base, "reports"), exist_ok=True)

    # Initialize shared state
    shared_state = {
        "wallet_address": wallet_address,
        "wallet_data": {},
        "risk_score": 0.0,
        "risk_level": "",
        "fraud_type": "",
        "fraud_confidence": 0.0,
        "evidence_list": [],
        "report_text": "",
        "report_filepath": "",
        "final_action": "",
        "action_reason": "",
        "audit_log": [],
    }

    agents = [
        (0, "Data Fetcher", run_agent_0_data_fetcher),
        (1, "Risk Scorer", run_agent_1_risk_scorer),
        (2, "Pattern Classifier", run_agent_2_pattern_classifier),
        (3, "Evidence Collector", run_agent_3_evidence_collector),
        (4, "Report Writer", run_agent_4_report_writer),
        (5, "Action Decider", run_agent_5_action_decider),
    ]

    print("\n🚀 CryptoGuard Agent Pipeline Starting...")
    print("=" * 50)

    for num, name, agent_fn in agents:
        icons = {0: "🌐", 1: "🔍", 2: "🔎", 3: "📋", 4: "📝", 5: "⚡"}
        print(f"{icons.get(num, '▶')} Agent {num}: {name} running...")

        if progress_callback:
            progress_callback(num, name, "running")

        try:
            shared_state = agent_fn(shared_state)
        except Exception as e:
            shared_state["audit_log"].append({
                "agent": name,
                "error": str(e),
                "timestamp": datetime.now().isoformat(),
            })
            print(f"   ❌ Agent {num} failed: {e}")
            if progress_callback:
                progress_callback(num, name, f"failed: {e}")
            continue

        # Print agent result summary
        if num == 0:
            print(f"   ✅ Fetched {len(shared_state['wallet_data'])} features from Etherscan")
        elif num == 1:
            print(f"   ✅ Risk Score: {shared_state['risk_score']:.1%} ({shared_state['risk_level']})")
        elif num == 2:
            print(f"   ✅ Fraud Type: {shared_state['fraud_type']} ({shared_state['fraud_confidence']:.1%} confidence)")
        elif num == 3:
            print(f"   ✅ Evidence points collected: {len(shared_state['evidence_list'])}")
        elif num == 4:
            print(f"   ✅ Report saved: {shared_state.get('report_filepath', 'N/A')}")
        elif num == 5:
            print(f"   ✅ Final Action: {shared_state['final_action']}")

        if progress_callback:
            progress_callback(num, name, "done")

    print("=" * 50)
    print(f"✅ Pipeline Complete | Audit log: {len(shared_state['audit_log'])} entries")

    # Save full audit log
    try:
        audit_path = os.path.join(
            base, "logs", f"audit_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(audit_path, "w") as f:
            json.dump(shared_state["audit_log"], f, indent=2, default=str)
        print(f"📁 Audit log saved: {audit_path}")
    except Exception as e:
        print(f"⚠️ Could not save audit log: {e}")

    return shared_state
