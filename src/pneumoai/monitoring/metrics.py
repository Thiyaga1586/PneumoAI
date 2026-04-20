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


def render_metrics() -> tuple[bytes, str]:
    return generate_latest(), CONTENT_TYPE_LATEST