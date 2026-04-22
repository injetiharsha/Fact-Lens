"""Compare two mix10 run JSON files and emit exact diffs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def _load(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _claim_map(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {str(r.get("id")): r for r in rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--current", required=True)
    parser.add_argument("--old", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    current = _load(Path(args.current))
    old = _load(Path(args.old))

    m_cur = dict(current.get("benchmark_metrics", {}) or {})
    m_old = dict(old.get("benchmark_metrics", {}) or {})
    c_cur = _claim_map(list(current.get("claims", []) or []))
    c_old = _claim_map(list(old.get("claims", []) or []))
    ids = sorted(set(c_cur).union(c_old), key=lambda x: int(x))

    changed: List[Dict[str, Any]] = []
    regressions: List[Dict[str, Any]] = []
    improvements: List[Dict[str, Any]] = []

    for cid in ids:
        a = c_cur.get(cid)
        b = c_old.get(cid)
        if not a or not b:
            continue
        if a.get("predicted_verdict") != b.get("predicted_verdict") or bool(a.get("correct")) != bool(b.get("correct")):
            row = {
                "id": cid,
                "language": a.get("language"),
                "expected": a.get("expected_verdict"),
                "current_pred": a.get("predicted_verdict"),
                "old_pred": b.get("predicted_verdict"),
                "current_correct": bool(a.get("correct")),
                "old_correct": bool(b.get("correct")),
            }
            changed.append(row)
            if bool(a.get("correct")) and (not bool(b.get("correct"))):
                regressions.append(row)
            if (not bool(a.get("correct"))) and bool(b.get("correct")):
                improvements.append(row)

    out = {
        "current_file": args.current,
        "old_file": args.old,
        "summary": {
            "accuracy_current": m_cur.get("accuracy"),
            "accuracy_old": m_old.get("accuracy"),
            "accuracy_delta_old_minus_current": round(float(m_old.get("accuracy", 0.0)) - float(m_cur.get("accuracy", 0.0)), 3),
            "neutral_rate_current": m_cur.get("neutral_rate"),
            "neutral_rate_old": m_old.get("neutral_rate"),
            "neutral_delta_old_minus_current": round(float(m_old.get("neutral_rate", 0.0)) - float(m_cur.get("neutral_rate", 0.0)), 3),
            "time_current_sec": current.get("total_time_seconds"),
            "time_old_sec": old.get("total_time_seconds"),
            "time_delta_old_minus_current_sec": round(float(old.get("total_time_seconds", 0.0)) - float(current.get("total_time_seconds", 0.0)), 3),
            "changed_claims": len(changed),
            "regressions": len(regressions),
            "improvements": len(improvements),
        },
        "changed_claims": changed,
        "regressions": regressions,
        "improvements": improvements,
    }

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(out["summary"], indent=2, ensure_ascii=False))
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()

