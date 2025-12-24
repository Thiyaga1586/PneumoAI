import time
from collections import deque

latencies = deque(maxlen=500)
errors = deque(maxlen=500)

def record_latency(ms: float):
    latencies.append(ms)

def record_error():
    errors.append(time.time())

def p95(vals):
    if not vals:
        return 0.0
    s = sorted(vals)
    idx = int(0.95 * (len(s) - 1))
    return float(s[idx])
