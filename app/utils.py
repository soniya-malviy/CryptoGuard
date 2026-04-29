def format_eth(value):
    return f"{value:.4f} ETH"

def format_timestamp(ts):
    import datetime
    return datetime.datetime.fromtimestamp(ts).strftime('%Y-%m-%d %H:%M:%S')
