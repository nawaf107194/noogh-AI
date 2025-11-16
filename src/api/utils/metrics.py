import time

start_time = time.time()

def get_uptime_seconds():
    return time.time() - start_time

def get_all_metrics():
    return {
        "uptime": get_uptime_seconds()
    }
