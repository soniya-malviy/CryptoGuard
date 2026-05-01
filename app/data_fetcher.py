import requests
import os
import time
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

ETHERSCAN_BASE = "https://api.etherscan.io/v2/api"

def _get_etherscan_key():
    key = None
    try:
        import streamlit as st
        key = st.secrets.get("ETHERSCAN_API_KEY")
    except Exception:
        pass
    if not key:
        key = os.getenv("ETHERSCAN_API_KEY")
    return key

API_KEY = _get_etherscan_key()

def fetch_normal_transactions(wallet_address: str) -> list:
    """Fetch all normal ETH transactions for wallet"""
    
    params = {
        "chainid": 1,
        "module": "account",
        "action": "txlist",
        "address": wallet_address,
        "startblock": 0,
        "endblock": 99999999,
        "sort": "asc",
        "apikey": API_KEY
    }
    
    response = requests.get(ETHERSCAN_BASE, params=params)
    data = response.json()
    
    if data["status"] == "1":
        return data["result"]
    if data.get("message") != "No transactions found":
        print(f"⚠️ Etherscan API Notice: {data.get('message')} - {data.get('result')}")
    return []

def fetch_erc20_transactions(wallet_address: str) -> list:
    """Fetch all ERC20 token transactions"""
    
    params = {
        "chainid": 1,
        "module": "account",
        "action": "tokentx",
        "address": wallet_address,
        "sort": "asc",
        "apikey": API_KEY
    }
    
    response = requests.get(ETHERSCAN_BASE, params=params)
    data = response.json()
    
    if data["status"] == "1":
        return data["result"]
    return []

def fetch_eth_balance(wallet_address: str) -> float:
    """Fetch current ETH balance"""
    
    params = {
        "chainid": 1,
        "module": "account",
        "action": "balance",
        "address": wallet_address,
        "tag": "latest",
        "apikey": API_KEY
    }
    
    response = requests.get(ETHERSCAN_BASE, params=params)
    data = response.json()
    
    if data["status"] == "1":
        # Convert Wei to ETH
        return int(data["result"]) / 1e18
    return 0.0

def fetch_created_contracts(wallet_address: str) -> list:
    """Fetch contracts created by this wallet"""
    
    params = {
        "chainid": 1,
        "module": "account",
        "action": "txlistinternal",
        "address": wallet_address,
        "sort": "asc",
        "apikey": API_KEY
    }
    
    response = requests.get(ETHERSCAN_BASE, params=params)
    data = response.json()
    
    if data["status"] == "1":
        # Filter only contract creations
        contracts = [
            tx for tx in data["result"] 
            if tx.get("type") == "create"
        ]
        return contracts
    return []

def compute_wallet_features(wallet_address: str) -> dict:
    """
    Main function — fetches everything and computes 
    all features needed by XGBoost model.
    Single wallet address in → full feature dict out.
    """
    
    print(f"Fetching data for wallet: {wallet_address}")
    
    # Fetch all data
    normal_txns = fetch_normal_transactions(wallet_address)
    erc20_txns = fetch_erc20_transactions(wallet_address)
    eth_balance = fetch_eth_balance(wallet_address)
    contracts = fetch_created_contracts(wallet_address)
    
    if not normal_txns:
        # Check if balance exists even if no normal txns (e.g. pure contract/ERC20 wallet)
        if eth_balance > 0 or erc20_txns:
            normal_txns = [] # Continue with empty txns list
        else:
            return {"error": "No transaction history or balance found for this wallet on Etherscan."}
    
    # Separate sent and received
    sent_txns = [
        tx for tx in normal_txns 
        if tx["from"].lower() == wallet_address.lower()
    ]
    received_txns = [
        tx for tx in normal_txns 
        if tx["to"].lower() == wallet_address.lower()
    ]
    
    # Convert values from Wei to ETH
    def wei_to_eth(wei_str):
        try:
            return int(wei_str) / 1e18
        except:
            return 0.0
    
    sent_values = [wei_to_eth(tx["value"]) for tx in sent_txns]
    received_values = [wei_to_eth(tx["value"]) for tx in received_txns]
    
    # Compute time differences between transactions
    def avg_time_between(txns):
        if len(txns) < 2:
            return 0.0
        timestamps = sorted([int(tx["timeStamp"]) for tx in txns])
        diffs = [
            (timestamps[i+1] - timestamps[i]) / 60  # convert to minutes
            for i in range(len(timestamps)-1)
        ]
        return sum(diffs) / len(diffs) if diffs else 0.0
    
    # First and last transaction timestamps
    all_timestamps = sorted([int(tx["timeStamp"]) for tx in normal_txns])
    time_diff_first_last = (
        (all_timestamps[-1] - all_timestamps[0]) / 60 
        if len(all_timestamps) > 1 else 0.0
    )
    
    # ERC20 computations
    erc20_sent = [
        tx for tx in erc20_txns 
        if tx["from"].lower() == wallet_address.lower()
    ]
    erc20_received = [
        tx for tx in erc20_txns 
        if tx["to"].lower() == wallet_address.lower()
    ]
    
    erc20_sent_values = [wei_to_eth(tx["value"]) for tx in erc20_sent]
    erc20_received_values = [wei_to_eth(tx["value"]) for tx in erc20_received]
    
    # Unique addresses
    unique_sent_addresses = list(set([tx["to"] for tx in erc20_sent]))
    unique_received_addresses = list(set([tx["from"] for tx in erc20_received]))
    
    # Build final feature dict matching XGBoost model features
    features = {
        # Transaction counts
        "sent_tnx": len(sent_txns),
        "received_tnx": len(received_txns),
        
        # Time features
        "avg_min_between_sent_tnx": avg_time_between(sent_txns),
        "avg_min_between_received_tnx": avg_time_between(received_txns),
        "time_diff_between_first_and_last_mins": time_diff_first_last,
        
        # Value features
        "avg_val_sent": sum(sent_values)/len(sent_values) if sent_values else 0.0,
        "avg_val_received": sum(received_values)/len(received_values) if received_values else 0.0,
        "max_value_received": max(received_values) if received_values else 0.0,
        "total_ether_sent": sum(sent_values),
        "total_ether_balance": eth_balance,
        
        # Contracts
        "number_of_created_contracts": len(contracts),
        
        # ERC20 features
        "erc20_total_received": sum(erc20_received_values),
        "erc20_total_sent": sum(erc20_sent_values),
        "erc20_uniq_sent_addr": len(unique_sent_addresses),
        "erc20_uniq_rec_addr": len(unique_received_addresses),
        "erc20_avg_time_sent": avg_time_between(erc20_sent),
        "erc20_avg_time_rec": avg_time_between(erc20_received),
        
        # Extra context for agents (not used by XGBoost but useful for LLM)
        "_meta": {
            "wallet_address": wallet_address,
            "total_transactions": len(normal_txns),
            "eth_balance": eth_balance,
            "first_seen": datetime.fromtimestamp(all_timestamps[0]).strftime("%Y-%m-%d"),
            "last_seen": datetime.fromtimestamp(all_timestamps[-1]).strftime("%Y-%m-%d"),
            "connected_wallets": list(set(
                [tx["to"] for tx in sent_txns[:20]] + 
                [tx["from"] for tx in received_txns[:20]]
            ))[:10]  # top 10 connected wallets
        }
    }
    
    return features
