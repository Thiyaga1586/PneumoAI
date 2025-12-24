import time
from ..mlops.drift import drift_check_and_maybe_rollback
from ..mlops.rollback import get_current_version

if __name__ == "__main__":
    while True:
        try:
            v = get_current_version()
            score = drift_check_and_maybe_rollback(version=v, window=200, threshold=0.25)
            print(f"[drift] version={v} psi={score:.4f}")
        except Exception as e:
            print("[drift] error:", e)
        time.sleep(60)
