from __future__ import annotations

import json
from pathlib import Path

from pneumoai.mlops.promotion_gate import should_promote


def main() -> None:
    candidate_path = Path("artifacts") / "candidate_metrics.json"
    champion_path = Path("artifacts") / "champion_metrics.json"

    if not candidate_path.exists():
        raise FileNotFoundError(f"Missing {candidate_path}")
    if not champion_path.exists():
        raise FileNotFoundError(f"Missing {champion_path}")

    candidate = json.loads(candidate_path.read_text(encoding="utf-8"))
    champion = json.loads(champion_path.read_text(encoding="utf-8"))

    approved, reasons = should_promote(candidate, champion)

    result = {
        "approved": approved,
        "candidate": candidate,
        "champion": champion,
        "reasons": reasons,
    }

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()