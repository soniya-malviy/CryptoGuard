
import sys, os
sys.path.insert(0, '.')
from app.pipeline import run_full_pipeline
import json
data = {
    'sent_tnx': 71, 'received_tnx': 81, 'avg_min_between_sent_tnx': 25.5, 'avg_min_between_received_tnx': 98.2,
    'time_diff_between_first_and_last_mins': 22333.3, 'number_of_created_contracts': 0, 'total_ether_sent': 62.6,
    'total_ether_balance': 13.8, 'avg_val_sent': 1.9, 'avg_val_received': 2.8, 'max_value_received': 9.5,
    'erc20_total_received': 9.9, 'erc20_total_sent': 6.2, 'erc20_uniq_sent_addr': 10, 'erc20_uniq_rec_addr': 11,
    'erc20_avg_time_sent': 41.1, 'erc20_avg_time_rec': 56.0
}
try:
    res = run_full_pipeline(data)
    print(json.dumps(res['audit_log'], indent=2))
except Exception as e:
    import traceback
    traceback.print_exc()

