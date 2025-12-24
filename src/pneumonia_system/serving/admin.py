from typing import Callable, Dict, Any, Optional

import numpy as np
from fastapi import APIRouter

from ..mlops.drift import drift_check_and_maybe_rollback
from ..mlops.rollback import get_current_version, set_current_version
from ..observability.store import latest_rows



def create_admin_router(
    reload_model: Callable[[], None],
    get_served_version: Callable[[], str],
) -> APIRouter:
    """
    Admin router factory to avoid circular imports.
    - reload_model(): reloads in-memory model used by API
    - get_served_version(): returns in-memory served model version
    """
    router = APIRouter()

    @router.get("/status")
    def status() -> Dict[str, Any]:
        return {
            "registry_current": get_current_version(),
            "served_in_memory": get_served_version(),
        }

    @router.post("/reload")
    def reload_now() -> Dict[str, Any]:
        reload_model()
        return {
            "reloaded": True,
            "served_in_memory": get_served_version(),
            "registry_current": get_current_version(),
        }

    @router.get("/drift")
    def drift(window: int = 50, threshold: float = 0.25) -> Dict[str, Any]:
        """
        Computes PSI drift for registry current version.
        If drift >= threshold, rollback() is triggered (registry flips).
        Then we reload the in-memory model so serving matches registry immediately.
        """
        before = get_current_version()
        psi_score = drift_check_and_maybe_rollback(version=before, window=window, threshold=threshold)
        after = get_current_version()

        rolled_back = (after != before)
        if rolled_back:
            reload_model()

        return {
            "before_version": before,
            "after_version": after,
            "rolled_back": rolled_back,
            "window": window,
            "threshold": threshold,
            "psi": round(float(psi_score), 6),
            "served_in_memory": get_served_version(),
        }

    @router.post("/promote/{version}")
    def promote(version: str) -> Dict[str, Any]:
        """
        Promote a version (update registry) + reload in-memory model.
        """
        set_current_version(version)
        reload_model()
        return {
            "promoted_to": version,
            "registry_current": get_current_version(),
            "served_in_memory": get_served_version(),
        }
    
    @router.get("/eval_latest")
    def eval_latest(window: int = 200, model_version: Optional[str] = None) -> Dict[str, Any]:
        """
        Rolling evaluation over last N logged requests from SQLite.
        - Always returns ops metrics (count, error_rate, latency p50/p95/mean)
        - Returns classification metrics only if true_label exists in rows.
        """
        rows = latest_rows(limit=window, model_version=model_version)

        n = len(rows)
        if n == 0:
            return {"window": window, "count": 0, "message": "No requests logged yet."}

        # ops metrics
        ok_rows = [r for r in rows if r["error"] is None]
        err_count = n - len(ok_rows)
        error_rate = err_count / n

        out: Dict[str, Any] = {
            "window": window,
            "count": n,
            "model_version_filter": model_version,
            "error_count": err_count,
            "error_rate": round(float(error_rate), 6),
        }

        if ok_rows:
            lat = np.array([r["latency_ms"] for r in ok_rows], dtype=float)
            out["latency_ms"] = {
                "p50": round(float(np.percentile(lat, 50)), 3),
                "p95": round(float(np.percentile(lat, 95)), 3),
                "mean": round(float(lat.mean()), 3),
            }

        # classification metrics only if true_label is present
        labeled = [r for r in ok_rows if r.get("true_label")]
        out["labeled_count"] = len(labeled)

        if len(labeled) == 0:
            out["message"] = "No true_label present. Send true_label in /predict to compute confusion matrix."
            return out

        def norm(lbl: str) -> str:
            x = lbl.strip().upper()
            if x in ("0", "NORMAL"):
                return "NORMAL"
            if x in ("1", "PNEUMONIA"):
                return "PNEUMONIA"
            return x

        y_true = np.array([1 if norm(r["true_label"]) == "PNEUMONIA" else 0 for r in labeled], dtype=int)
        y_pred = np.array([1 if norm(r["pred_label"]) == "PNEUMONIA" else 0 for r in labeled], dtype=int)

        tn = int(((y_pred == 0) & (y_true == 0)).sum())
        fp = int(((y_pred == 1) & (y_true == 0)).sum())
        fn = int(((y_pred == 0) & (y_true == 1)).sum())
        tp = int(((y_pred == 1) & (y_true == 1)).sum())

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
        acc = float((y_pred == y_true).mean())

        out["classification"] = {
            "accuracy": round(acc, 6),
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1": round(float(f1), 6),
            "confusion_matrix": {"tn": tn, "fp": fp, "fn": fn, "tp": tp},
        }
        return out
    
    return router
