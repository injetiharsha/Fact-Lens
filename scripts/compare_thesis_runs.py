"""Compare two thesis run summaries and emit delta report.

Usage:
  .\\.venv-gpu\\Scripts\\python.exe scripts\\compare_thesis_runs.py \
    --a tests/benchmarks/thesis_runs/<runA>/run_summary.json \
    --b tests/benchmarks/thesis_runs/<runB>/run_summary.json
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


KEYS = [
    "accuracy",
    "neutral_rate",
    "false_positive_rate",
    "false_negative_rate",
    "latency_avg_sec",
    "latency_median_sec",
    "source_diversity_avg",
    "trusted_source_ratio_avg",
    "provider_calls_total",
    "structured_calls_total",
    "estimated_cost_usd_total",
    "estimated_cost_usd_per_claim",
    "evidence_recall_at_k_avg",
]


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric(summary: Dict[str, Any], split: str, key: str) -> float | None:
    out = (summary.get("outputs") or {}).get(split) or {}
    metrics = out.get("metrics") or {}
    value = metrics.get(key)
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _diff(a: float | None, b: float | None) -> float | None:
    if a is None or b is None:
        return None
    return b - a


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="Path to run_summary.json for baseline")
    parser.add_argument("--b", required=True, help="Path to run_summary.json for candidate")
    parser.add_argument("--out", default="", help="Optional output JSON path")
    args = parser.parse_args()

    a_path = Path(args.a)
    b_path = Path(args.b)
    a = _load(a_path)
    b = _load(b_path)

    splits = sorted(set((a.get("outputs") or {}).keys()) | set((b.get("outputs") or {}).keys()))
    delta: Dict[str, Any] = {}
    for split in splits:
        split_delta: Dict[str, Any] = {}
        for key in KEYS:
            av = _metric(a, split, key)
            bv = _metric(b, split, key)
            split_delta[key] = {
                "a": av,
                "b": bv,
                "delta_b_minus_a": _diff(av, bv),
            }
        delta[split] = split_delta

    report = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_summary": str(a_path.resolve()),
        "candidate_summary": str(b_path.resolve()),
        "baseline_profile": a.get("profile_name"),
        "candidate_profile": b.get("profile_name"),
        "delta": delta,
    }

    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
