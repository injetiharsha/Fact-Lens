"""Direct benchmark runner (EN/MULTI) with one command.

Usage:
  .\.venv-gpu\Scripts\python.exe scripts\run_benchmark_direct.py --split en --num-claims 30
  .\.venv-gpu\Scripts\python.exe scripts\run_benchmark_direct.py --split multi --num-claims 30
  .\.venv-gpu\Scripts\python.exe scripts\run_benchmark_direct.py --split both --num-claims 30
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.routes.claim import _pipeline_config  # noqa: E402
from pipeline.claim_pipeline import ClaimPipeline  # noqa: E402

LABELS = {"support", "refute", "neutral"}


def _load_cases(path: Path, bucket: str, n: int) -> List[Dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = [r for r in rows if str(r.get("lang_bucket", "")).upper() == bucket.upper()]
    return out[:n]


def _binary_counts(rows: List[Dict[str, Any]]) -> Tuple[int, int, int, int]:
    # positive class = support, negative class = refute. ignore expected neutral.
    filtered = [r for r in rows if r.get("expected") in {"support", "refute"}]
    tp = sum(1 for r in filtered if r["expected"] == "support" and r["predicted"] == "support")
    tn = sum(1 for r in filtered if r["expected"] == "refute" and r["predicted"] == "refute")
    fp = sum(1 for r in filtered if r["expected"] == "refute" and r["predicted"] == "support")
    fn = sum(1 for r in filtered if r["expected"] == "support" and r["predicted"] != "support")
    return tp, tn, fp, fn


def _metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    y_true = [r["expected"] for r in rows]
    y_pred = [r["predicted"] for r in rows]
    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    neutral_pred = sum(1 for p in y_pred if p == "neutral")
    tp, tn, fp, fn = _binary_counts(rows)
    binary_total = sum(1 for r in rows if r.get("expected") in {"support", "refute"})

    return {
        "total_claims": total,
        "correct_predictions": correct,
        "incorrect_predictions": total - correct,
        "accuracy": float(correct / total) if total else 0.0,
        "neutral_predictions": neutral_pred,
        "neutral_rate": float(neutral_pred / total) if total else 0.0,
        "expected_distribution": dict(Counter(y_true)),
        "predicted_distribution": dict(Counter(y_pred)),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "false_positive_rate": float(fp / binary_total) if binary_total else 0.0,
        "false_negative_rate": float(fn / binary_total) if binary_total else 0.0,
    }


def _run_en(num_claims: int, cases_path: Path) -> List[Dict[str, Any]]:
    rows = _load_cases(cases_path, "EN", num_claims)
    pipeline = ClaimPipeline(_pipeline_config("en"))
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
            "language": "en",
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
        }
        predictions.append(row)
        print(
            f"[EN {idx}/{len(rows)}] id={row['id']} expected={expected} "
            f"predicted={verdict} ok={row['correct']} ev={evidence_count}"
        )
    return predictions


def _run_multi(num_claims: int, cases_path: Path) -> List[Dict[str, Any]]:
    rows = _load_cases(cases_path, "MULTI", num_claims)
    pipelines: Dict[str, ClaimPipeline] = {}
    predictions: List[Dict[str, Any]] = []

    for idx, item in enumerate(rows, start=1):
        claim = str(item["claim"])
        expected = str(item["expected"]).lower()
        language = str(item.get("language") or "hi")
        started = datetime.now(timezone.utc)
        err = ""
        verdict = "neutral"
        confidence = 0.0
        evidence_count = 0
        details: Dict[str, Any] = {}
        try:
            if language not in pipelines:
                pipelines[language] = ClaimPipeline(_pipeline_config(language))
            result = pipelines[language].analyze(claim=claim, language=language)
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
            "language": language,
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
        }
        predictions.append(row)
        print(
            f"[MULTI {idx}/{len(rows)}] id={row['id']} lang={language} expected={expected} "
            f"predicted={verdict} ok={row['correct']} ev={evidence_count}"
        )
    return predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", choices=["en", "multi", "both"], default="both")
    parser.add_argument("--num-claims", type=int, default=30)
    parser.add_argument("--llm-mode", choices=["env", "on", "off"], default="env")
    parser.add_argument("--cases", default="tests/benchmarks/rfcs_benchmark_multi/benchmark_cases_multi.json")
    parser.add_argument("--out-dir", default="tests/benchmarks/direct_benchmark_run")
    args = parser.parse_args()

    load_dotenv(override=True)
    if args.llm_mode == "on":
        os.environ["ENABLE_LLM_VERIFIER"] = "1"
    elif args.llm_mode == "off":
        os.environ["ENABLE_LLM_VERIFIER"] = "0"

    cases_path = ROOT / args.cases
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    outputs: Dict[str, Any] = {}
    if args.split in {"en", "both"}:
        en_rows = _run_en(args.num_claims, cases_path)
        en_metrics = _metrics(en_rows)
        (out_dir / "predictions_en.json").write_text(json.dumps(en_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        (out_dir / "metrics_en.json").write_text(json.dumps(en_metrics, ensure_ascii=False, indent=2), encoding="utf-8")
        outputs["en"] = {"claims": len(en_rows), "metrics": en_metrics}

    if args.split in {"multi", "both"}:
        multi_rows = _run_multi(args.num_claims, cases_path)
        multi_metrics = _metrics(multi_rows)
        (out_dir / "predictions_multi.json").write_text(
            json.dumps(multi_rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / "metrics_multi.json").write_text(
            json.dumps(multi_metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        outputs["multi"] = {"claims": len(multi_rows), "metrics": multi_metrics}

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "split": args.split,
        "num_claims_requested": args.num_claims,
        "out_dir": str(out_dir.resolve()),
        "outputs": outputs,
    }
    (out_dir / "run_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
