"""Run 30 EN benchmark claims through full pipeline (all enabled sources)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.routes.claim import _pipeline_config  # noqa: E402
from pipeline.claim_pipeline import ClaimPipeline  # noqa: E402

LABELS = ["support", "refute", "neutral"]


def _load_cases(path: Path, n: int) -> List[Dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    en_rows = [r for r in rows if str(r.get("lang_bucket", "")).upper() == "EN"]
    return en_rows[:n]


def _metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    y_true = [r["expected"] for r in rows]
    y_pred = [r["predicted"] for r in rows]
    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    neutral_pred = sum(1 for p in y_pred if p == "neutral")
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist() if total else [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    return {
        "total_claims": total,
        "correct_predictions": correct,
        "incorrect_predictions": total - correct,
        "accuracy": float(accuracy_score(y_true, y_pred)) if total else 0.0,
        "neutral_predictions": neutral_pred,
        "neutral_rate": float(neutral_pred / total) if total else 0.0,
        "expected_distribution": dict(Counter(y_true)),
        "predicted_distribution": dict(Counter(y_pred)),
        "confusion_matrix": {"labels": LABELS, "matrix": cm},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default="tests/benchmarks/rfcs_benchmark_en/benchmark_cases_en.json",
    )
    parser.add_argument("--num-claims", type=int, default=30)
    parser.add_argument(
        "--out-dir",
        default="tests/benchmarks/rfcs_benchmark_en_full_pipeline",
    )
    parser.add_argument("--llm-mode", choices=["env", "on", "off"], default="env")
    args = parser.parse_args()

    load_dotenv(override=True)
    if args.llm_mode == "on":
        os.environ["ENABLE_LLM_VERIFIER"] = "1"
    elif args.llm_mode == "off":
        os.environ["ENABLE_LLM_VERIFIER"] = "0"

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_cases(ROOT / args.cases, args.num_claims)
    print("Initializing EN pipeline (loading models/checkpoints)...", flush=True)
    t0 = datetime.now(timezone.utc)
    pipeline = ClaimPipeline(_pipeline_config("en"))
    t1 = datetime.now(timezone.utc)
    print(
        f"Pipeline ready in {(t1 - t0).total_seconds():.2f}s. Starting claims...",
        flush=True,
    )

    predictions: List[Dict[str, Any]] = []
    for idx, item in enumerate(rows, start=1):
        claim = str(item["claim"])
        expected = str(item["expected"]).lower()
        started = datetime.now(timezone.utc)
        err = ""
        verdict = "neutral"
        confidence = 0.0
        evidence_count = 0
        details: Dict[str, Any] = {}
        try:
            result = pipeline.analyze(claim=claim, language="en")
            verdict = str(getattr(result, "verdict", "neutral") or "neutral").lower()
            if verdict not in LABELS:
                verdict = "neutral"
            confidence = float(getattr(result, "confidence", 0.0) or 0.0)
            evidence_count = len(getattr(result, "evidence", []) or [])
            details = getattr(result, "details", {}) or {}
        except Exception as exc:
            err = str(exc)
        ended = datetime.now(timezone.utc)

        row = {
            "index": idx,
            "id": item.get("id", idx),
            "claim": claim,
            "expected": expected,
            "predicted": verdict,
            "correct": verdict == expected,
            "confidence": confidence,
            "evidence_count": evidence_count,
            "error": err,
            "details": details,
            "started_utc": started.isoformat(),
            "ended_utc": ended.isoformat(),
            "duration_sec": (ended - started).total_seconds(),
            "mode": "full_pipeline_all_sources",
        }
        predictions.append(row)
        print(
            f"[{idx}/{len(rows)}] id={row['id']} expected={expected} "
            f"predicted={verdict} ok={row['correct']} ev={evidence_count}"
        )

    metrics = _metrics(predictions)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "full_pipeline_all_sources",
        "claims": len(rows),
        "metrics": metrics,
        "outputs": {
            "predictions": str((out_dir / "predictions_en_30_full_pipeline.json").resolve()),
            "metrics": str((out_dir / "metrics_en_30_full_pipeline.json").resolve()),
        },
    }

    (out_dir / "predictions_en_30_full_pipeline.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "metrics_en_30_full_pipeline.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
