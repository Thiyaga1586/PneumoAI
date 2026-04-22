from __future__ import annotations


def should_promote(candidate: dict, champion: dict) -> tuple[bool, list[str]]:
    reasons: list[str] = []

    candidate_f1 = float(candidate.get("f1", 0.0))
    champion_f1 = float(champion.get("f1", 0.0))

    candidate_recall = float(candidate.get("recall", 0.0))
    champion_recall = float(champion.get("recall", 0.0))

    candidate_latency = float(candidate.get("latency_p95_ms", 0.0))
    champion_latency = float(champion.get("latency_p95_ms", 0.0))

    if candidate_f1 < champion_f1:
        reasons.append(
            f"candidate f1 {candidate_f1:.4f} is lower than champion f1 {champion_f1:.4f}"
        )

    if candidate_recall < champion_recall:
        reasons.append(
            f"candidate recall {candidate_recall:.4f} is lower than champion recall {champion_recall:.4f}"
        )

    if champion_latency > 0 and candidate_latency > champion_latency * 1.10:
        reasons.append(
            f"candidate latency_p95_ms {candidate_latency:.2f} exceeds allowed limit based on champion {champion_latency:.2f}"
        )

    return len(reasons) == 0, reasons