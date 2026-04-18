"""EN benchmark runner (parallel_test style, no required args)."""

from __future__ import annotations

import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.routes.claim import _pipeline_config
from pipeline.claim_pipeline import ClaimPipeline

DEFAULT_CASES = ROOT / "tests/benchmarks/rfcs_benchmark_en/benchmark_cases_en.json"
DEFAULT_OUTPUT = ROOT / "tests/benchmarks/rfcs_benchmark_en_full_pipeline/parallel_like_results_en.json"
NUM_CLAIMS = 30


def load_claim_batch(path: Path, n: int = NUM_CLAIMS) -> List[Dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in rows if str(r.get("lang_bucket", "")).upper() == "EN"]
    return rows[:n]


def process_claim(i: int, row: Dict[str, Any], total: int, pipeline: ClaimPipeline) -> Dict[str, Any]:
    claim = str(row["claim"])
    expected = str(row["expected"]).lower()
    print("\n==============================")
    print(f"Processing claim {i + 1}/{total}")
    print("Claim:", claim)

    start = time.time()
    error_text = None
    verdict = "neutral"
    confidence = 0.0
    evidence = []
    details = {}

    try:
        result = pipeline.analyze(claim=claim, language="en")
        verdict = str(getattr(result, "verdict", "neutral") or "neutral").lower()
        confidence = float(getattr(result, "confidence", 0.0) or 0.0)
        evidence = list(getattr(result, "evidence", []) or [])
        details = dict(getattr(result, "details", {}) or {})
    except Exception as exc:
        error_text = str(exc)
        print("Claim processing error:", error_text)

    elapsed = round(time.time() - start, 3)
    print("Verdict:", verdict)
    print("Time:", elapsed, "sec")
    timings = details.get("timings") if isinstance(details, dict) else None
    if isinstance(timings, dict) and timings:
        timings_ms = {}
        for key, value in timings.items():
            try:
                timings_ms[key] = round(float(value) * 1000.0, 2)
            except Exception:
                continue
        print("Stage timings (ms):", json.dumps(timings_ms, ensure_ascii=False))

    return {
        "id": row.get("id", i + 1),
        "claim": claim,
        "expected_verdict": expected,
        "predicted_verdict": verdict,
        "correct": verdict == expected,
        "confidence": confidence,
        "evidence_count": len(evidence),
        "time_seconds": elapsed,
        "details": details,
        "error": error_text,
    }


def run_benchmark(rows: List[Dict[str, Any]]) -> tuple[list[Dict[str, Any]], list[float], float]:
    start = time.time()
    results: List[Dict[str, Any]] = []
    claim_times: List[float] = []
    print("Initializing EN pipeline (loading models/checkpoints)...", flush=True)
    init_start = time.time()
    pipeline = ClaimPipeline(_pipeline_config("en"))
    print(f"Pipeline ready in {round(time.time() - init_start, 3)} sec", flush=True)

    # Old-style benchmark flow: claim-level sequential processing.
    # Inner pipeline stages retain their own async/parallel behavior.
    for i, row in enumerate(rows):
        out = process_claim(i, row, len(rows), pipeline)
        results.append(out)
        claim_times.append(float(out["time_seconds"]))

    total_time = round(time.time() - start, 3)
    print("\n==============================")
    print("Total benchmark time:", total_time, "sec")
    return results, claim_times, total_time


def evaluate(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    correct = 0
    neutral = 0
    tp = tn = fp = fn = 0
    failed_by_expected = Counter()
    failed_by_predicted = Counter()
    failed_claims = []

    for r in results:
        pred = r["predicted_verdict"]
        truth = r["expected_verdict"]
        if pred == truth:
            correct += 1
        if pred == "neutral":
            neutral += 1

        if truth in {"support", "refute"}:
            if pred == "support" and truth == "support":
                tp += 1
            elif pred == "support" and truth == "refute":
                fp += 1
            elif pred != "support" and truth == "support":
                fn += 1
            elif pred != "support" and truth == "refute":
                tn += 1

        if pred != truth:
            failed_by_expected[truth] += 1
            failed_by_predicted[pred] += 1
            failed_claims.append(
                {
                    "id": r.get("id"),
                    "claim": r["claim"],
                    "expected_verdict": truth,
                    "predicted_verdict": pred,
                    "evidence_count": r.get("evidence_count"),
                    "time_seconds": r.get("time_seconds"),
                }
            )

    total = len(results)
    binary_total = sum(1 for r in results if r["expected_verdict"] in {"support", "refute"})
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0

    return {
        "total_claims": total,
        "correct_predictions": correct,
        "accuracy": round(correct / total, 3) if total else 0.0,
        "neutral_rate": round(neutral / total, 3) if total else 0.0,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "false_positive_rate": round(fp / binary_total, 3) if binary_total else 0.0,
        "false_negative_rate": round(fn / binary_total, 3) if binary_total else 0.0,
        "precision_true_class": round(precision, 3),
        "recall_true_class": round(recall, 3),
        "f1_true_class": round(f1, 3),
        "failed_by_expected_verdict": dict(failed_by_expected),
        "failed_by_predicted_verdict": dict(failed_by_predicted),
        "failed_claims": failed_claims,
    }


def summarize_stage_timings(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    totals = Counter()
    totals_ms = Counter()
    per_claim = []
    for row in results:
        timings = {}
        details = row.get("details", {})
        if isinstance(details, dict):
            timings = details.get("timings", {}) or {}

        clean_timings: Dict[str, float] = {}
        for key, value in timings.items():
            try:
                numeric = float(value)
            except Exception:
                continue
            totals[key] += numeric
            totals_ms[key] += (numeric * 1000.0)
            clean_timings[key] = round(numeric, 4)

        if clean_timings:
            per_claim.append(
                {
                    "id": row.get("id"),
                    "claim": row.get("claim"),
                    "time_seconds": row.get("time_seconds"),
                    "stage_timings_seconds": clean_timings,
                    "stage_timings_ms": {k: round(v * 1000.0, 2) for k, v in clean_timings.items()},
                }
            )

    rounded_totals = {key: round(value, 4) for key, value in totals.items()}
    rounded_totals_ms = {key: round(value, 2) for key, value in totals_ms.items()}
    dominant_stage = None
    dominant_value = 0.0
    for key, value in rounded_totals.items():
        if key == "total_pipeline":
            continue
        if value > dominant_value:
            dominant_stage = key
            dominant_value = value

    model_locked_total = round(
        rounded_totals.get("stage6_relevance", 0.0)
        + rounded_totals.get("stage7_stance", 0.0)
        + rounded_totals.get("stage10_llm_verify", 0.0),
        3,
    )

    return {
            "stage_timing_totals_seconds": rounded_totals,
            "stage_timing_totals_ms": rounded_totals_ms,
            "dominant_stage": dominant_stage,
            "dominant_stage_seconds": dominant_value,
            "model_locked_total_seconds": model_locked_total,
            "per_claim_stage_timings": per_claim,
    }


def save_results(results: List[Dict[str, Any]], metrics: Dict[str, Any], claim_times: List[float], total_time: float, output_path: Path) -> None:
    stage_summary = summarize_stage_timings(results)
    output = {
        "benchmark_metrics": metrics,
        "total_time_seconds": total_time,
        "average_claim_time": round(sum(claim_times) / len(claim_times), 3) if claim_times else 0.0,
        "stage_timing_summary": stage_summary,
        "claims": results,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nResults saved to {output_path}")


def main() -> None:
    load_dotenv(override=True)
    rows = load_claim_batch(DEFAULT_CASES, NUM_CLAIMS)
    results, claim_times, total_time = run_benchmark(rows)
    metrics = evaluate(results)

    print("\n==============================")
    print("BENCHMARK METRICS")
    print("==============================")
    for key in [
        "total_claims",
        "correct_predictions",
        "accuracy",
        "neutral_rate",
        "tp",
        "tn",
        "fp",
        "fn",
        "false_positive_rate",
        "false_negative_rate",
        "precision_true_class",
        "recall_true_class",
        "f1_true_class",
    ]:
        print(f"{key} : {metrics.get(key)}")

    print("failed_by_expected_verdict :", metrics.get("failed_by_expected_verdict"))
    print("failed_by_predicted_verdict :", metrics.get("failed_by_predicted_verdict"))
    stage_summary = summarize_stage_timings(results)
    print("dominant_stage :", stage_summary.get("dominant_stage"))
    print("dominant_stage_seconds :", stage_summary.get("dominant_stage_seconds"))
    print("model_locked_total_seconds :", stage_summary.get("model_locked_total_seconds"))

    save_results(results, metrics, claim_times, total_time, DEFAULT_OUTPUT)


if __name__ == "__main__":
    main()
