from pneumoai.mlops.promotion_gate import should_promote


def test_promotion_gate_approves_better_candidate():
    candidate = {
        "version": "v2",
        "f1": 0.946,
        "recall": 0.958,
        "latency_p95_ms": 1000.0,
    }
    champion = {
        "version": "v1",
        "f1": 0.941,
        "recall": 0.955,
        "latency_p95_ms": 950.0,
    }

    approved, reasons = should_promote(candidate, champion)

    assert approved is True
    assert reasons == []


def test_promotion_gate_rejects_lower_f1():
    candidate = {
        "version": "v2",
        "f1": 0.900,
        "recall": 0.958,
        "latency_p95_ms": 900.0,
    }
    champion = {
        "version": "v1",
        "f1": 0.941,
        "recall": 0.955,
        "latency_p95_ms": 950.0,
    }

    approved, reasons = should_promote(candidate, champion)

    assert approved is False
    assert any("f1" in reason for reason in reasons)


def test_promotion_gate_rejects_lower_recall():
    candidate = {
        "version": "v2",
        "f1": 0.950,
        "recall": 0.900,
        "latency_p95_ms": 900.0,
    }
    champion = {
        "version": "v1",
        "f1": 0.941,
        "recall": 0.955,
        "latency_p95_ms": 950.0,
    }

    approved, reasons = should_promote(candidate, champion)

    assert approved is False
    assert any("recall" in reason for reason in reasons)


def test_promotion_gate_rejects_large_latency_regression():
    candidate = {
        "version": "v2",
        "f1": 0.950,
        "recall": 0.960,
        "latency_p95_ms": 1200.0,
    }
    champion = {
        "version": "v1",
        "f1": 0.941,
        "recall": 0.955,
        "latency_p95_ms": 950.0,
    }

    approved, reasons = should_promote(candidate, champion)

    assert approved is False
    assert any("latency" in reason for reason in reasons)