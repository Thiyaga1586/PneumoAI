from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST

PREDICTION_REQUESTS_TOTAL = Counter(
    "pneumoai_prediction_requests_total",
    "Total number of prediction requests",
)

PREDICTION_ERRORS_TOTAL = Counter(
    "pneumoai_prediction_errors_total",
    "Total number of prediction errors",
)

PREDICTION_LATENCY_MS = Histogram(
    "pneumoai_prediction_latency_ms",
    "Prediction latency in milliseconds",
    buckets=(10, 25, 50, 100, 250, 500, 1000, 2000, 5000),
)

ASYNC_REQUESTS_TOTAL = Counter(
    "pneumoai_async_requests_total",
    "Total number of async prediction requests",
)

ASYNC_COMPLETIONS_TOTAL = Counter(
    "pneumoai_async_completions_total",
    "Total number of completed async prediction jobs",
)

ADMIN_ACTIONS_TOTAL = Counter(
    "pneumoai_admin_actions_total",
    "Total number of admin actions",
    ["action"],
)

DRIFT_CHECKS_TOTAL = Counter(
    "pneumoai_drift_checks_total",
    "Total number of drift checks",
)

DRIFT_SCORE = Histogram(
    "pneumoai_drift_js_divergence",
    "Jensen-Shannon divergence score for drift detection",
    buckets=(0.01, 0.03, 0.05, 0.08, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0),
)


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST