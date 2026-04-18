"""Rerun 15-claim benchmark with LLM OFF/ON + full trace.

Output:
- tests/llm_toggle/venv_gpu_trace_rerun/trace_run_results.json

Contains:
- per-claim full trace (search -> retrieval -> relevance -> stance -> verdict -> llm)
- final benchmark summary (no macro/weighted):
  total accuracy, neutral rate, correct/incorrect counts, confusion matrix
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from sklearn.metrics import accuracy_score, confusion_matrix

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.routes.claim import _pipeline_config  # noqa: E402
from pipeline.claim_pipeline import ClaimPipeline  # noqa: E402

LABELS = ["support", "refute", "neutral"]

CLAIMS_15 = [
    {"claim": "The Earth revolves around the Sun.", "expected": "support"},
    {"claim": "At sea level, pure water boils at one hundred degrees Celsius.", "expected": "support"},
    {"claim": "The capital city of India is New Delhi.", "expected": "support"},
    {"claim": "An adult human typically has two hundred and six bones.", "expected": "support"},
    {"claim": "The Great Wall is located in China.", "expected": "support"},
    {"claim": "The Sun revolves around the Earth once every day.", "expected": "refute"},
    {"claim": "Humans use only ten percent of their brain in daily life.", "expected": "refute"},
    {"claim": "The Great Wall of China is clearly visible from the Moon without aid.", "expected": "refute"},
    {"claim": "COVID-19 vaccines cause infertility in all women.", "expected": "refute"},
    {"claim": "NASA officially confirmed active microbial life on Mars in 2025.", "expected": "refute"},
    {"claim": "This restaurant serves the best biryani in Hyderabad.", "expected": "neutral"},
    {"claim": "My neighbor's dog understands three different human languages.", "expected": "neutral"},
    {"claim": "Most residents in my city prefer tea over coffee.", "expected": "neutral"},
    {"claim": "The latest smartphone model is better than every previous model.", "expected": "neutral"},
    {"claim": "A secret project currently controls global weather patterns.", "expected": "neutral"},
]


def _safe(v: Any) -> Any:
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    if isinstance(v, dict):
        return {str(k): _safe(val) for k, val in v.items()}
    if isinstance(v, list):
        return [_safe(x) for x in v]
    if isinstance(v, tuple):
        return [_safe(x) for x in v]
    return str(v)


def _set_env_file_key(env_path: Path, key: str, value: str) -> str | None:
    lines = env_path.read_text(encoding="utf-8").splitlines()
    old = None
    changed = False
    out: List[str] = []
    for line in lines:
        if line.startswith(f"{key}="):
            old = line.split("=", 1)[1]
            out.append(f"{key}={value}")
            changed = True
        else:
            out.append(line)
    if not changed:
        out.append(f"{key}={value}")
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")
    return old


def _restore_env_file_key(env_path: Path, key: str, old: str | None) -> None:
    lines = env_path.read_text(encoding="utf-8").splitlines()
    out: List[str] = []
    found = False
    for line in lines:
        if line.startswith(f"{key}="):
            found = True
            if old is not None:
                out.append(f"{key}={old}")
        else:
            out.append(line)
    if (not found) and (old is not None):
        out.append(f"{key}={old}")
    env_path.write_text("\n".join(out) + "\n", encoding="utf-8")


def _attach_trace(pipeline: ClaimPipeline, trace: List[Dict[str, Any]]) -> None:
    def wrap_method(obj: Any, name: str):
        if not hasattr(obj, name):
            return
        original = getattr(obj, name)

        def wrapped(*args, **kwargs):
            t0 = time.time()
            event: Dict[str, Any] = {
                "event": f"{obj.__class__.__name__}.{name}",
                "args": _safe(args[:4]),
                "kwargs": _safe(kwargs),
            }
            try:
                out = original(*args, **kwargs)
                event["ok"] = True
                event["duration_ms"] = int((time.time() - t0) * 1000)
                if isinstance(out, list):
                    event["result_count"] = len(out)
                    if out and isinstance(out[0], dict):
                        event["result_sample"] = _safe(out[:2])
                elif isinstance(out, tuple):
                    event["result_tuple"] = _safe(out)
                elif isinstance(out, dict):
                    event["result_dict"] = _safe(out)
                else:
                    event["result"] = _safe(out)
                trace.append(event)
                return out
            except Exception as exc:
                event["ok"] = False
                event["duration_ms"] = int((time.time() - t0) * 1000)
                event["error"] = str(exc)
                trace.append(event)
                raise

        setattr(obj, name, wrapped)

    wrap_method(pipeline.normalizer, "normalize")
    wrap_method(pipeline.normalizer, "rephrase_for_search")
    wrap_method(pipeline.context_classifier, "classify")
    wrap_method(pipeline.domain_router, "route")
    wrap_method(pipeline.evidence_gatherer, "gather")
    wrap_method(pipeline.evidence_gatherer.web_search, "search")
    wrap_method(pipeline.evidence_gatherer.web_search, "_search_tavily")
    wrap_method(pipeline.evidence_gatherer.web_search, "_search_serpapi")
    wrap_method(pipeline.evidence_gatherer.web_search, "_search_newsapi")
    wrap_method(pipeline.evidence_gatherer.web_search, "_search_ddg")
    wrap_method(pipeline.evidence_gatherer.api_client, "query")
    wrap_method(pipeline.evidence_gatherer.scraper, "scrape")
    wrap_method(pipeline.evidence_gatherer.scraper, "scrape_urls")
    wrap_method(pipeline.relevance_scorer, "rank_evidence")
    wrap_method(pipeline.stance_detector, "detect")
    wrap_method(pipeline.evidence_scorer, "calculate_weight")
    wrap_method(pipeline.verdict_engine, "compute")
    wrap_method(pipeline.llm_verifier, "verify")


def run_mode(mode: str, claims: List[Dict[str, str]], language: str) -> Dict[str, Any]:
    cfg = _pipeline_config(language)
    pipeline = ClaimPipeline(cfg)
    mode_rows: List[Dict[str, Any]] = []

    for idx, item in enumerate(claims, start=1):
        claim = item["claim"]
        expected = item["expected"]
        trace: List[Dict[str, Any]] = []
        _attach_trace(pipeline, trace)
        started = datetime.now(timezone.utc).isoformat()
        result = pipeline.analyze(claim=claim, language=language)
        ended = datetime.now(timezone.utc).isoformat()
        predicted = str(result.verdict).strip().lower()
        row = {
            "id": idx,
            "claim": claim,
            "expected": expected,
            "mode": mode,
            "started_utc": started,
            "ended_utc": ended,
            "verdict": result.verdict,
            "predicted": predicted,
            "correct": predicted == expected,
            "confidence": float(result.confidence),
            "evidence_count": len(result.evidence),
            "details": _safe(result.details),
            "reasoning": result.reasoning,
            "evidence": _safe(result.evidence),
            "trace": trace,
        }
        mode_rows.append(row)

    return {"mode": mode, "rows": mode_rows}


def compare_rows(off_rows: List[Dict[str, Any]], on_rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for off, on in zip(off_rows, on_rows):
        out.append(
            {
                "id": off["id"],
                "claim": off["claim"],
                "off": {
                    "verdict": off["verdict"],
                    "confidence": off["confidence"],
                    "evidence_count": off["evidence_count"],
                },
                "on": {
                    "verdict": on["verdict"],
                    "confidence": on["confidence"],
                    "evidence_count": on["evidence_count"],
                },
                "diff": {
                    "verdict_changed": off["verdict"] != on["verdict"],
                    "confidence_delta": on["confidence"] - off["confidence"],
                    "evidence_count_delta": on["evidence_count"] - off["evidence_count"],
                    "llm_triggered_on": bool((on.get("details") or {}).get("llm_verifier", {}).get("triggered", False)),
                },
            }
        )
    return out


def compute_metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    labeled = [r for r in rows if r.get("expected") in LABELS]
    y_true = [r["expected"] for r in labeled]
    y_pred = [r.get("predicted", "neutral") for r in labeled]
    total = len(labeled)
    correct = sum(1 for r in labeled if r.get("correct"))
    neutral_pred = sum(1 for x in y_pred if x == "neutral")

    if total == 0:
        return {
            "total_claims": 0,
            "correct_predictions": 0,
            "incorrect_predictions": 0,
            "accuracy": 0.0,
            "neutral_predictions": 0,
            "neutral_rate": 0.0,
            "expected_distribution": {},
            "predicted_distribution": {},
            "confusion_matrix": {"labels": LABELS, "matrix": [[0, 0, 0], [0, 0, 0], [0, 0, 0]]},
        }

    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist()
    return {
        "total_claims": int(total),
        "correct_predictions": int(correct),
        "incorrect_predictions": int(total - correct),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "neutral_predictions": int(neutral_pred),
        "neutral_rate": float(neutral_pred / total),
        "expected_distribution": dict(sorted(Counter(y_true).items())),
        "predicted_distribution": dict(sorted(Counter(y_pred).items())),
        "confusion_matrix": {"labels": LABELS, "matrix": cm},
    }


def compute_comparison_metrics(compare: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(compare)
    verdict_changes = sum(1 for x in compare if x["diff"]["verdict_changed"])
    llm_triggered = sum(1 for x in compare if x["diff"]["llm_triggered_on"])
    return {
        "total_claims_compared": int(total),
        "verdict_changes_off_vs_on": int(verdict_changes),
        "verdict_change_rate": float(verdict_changes / total if total else 0.0),
        "llm_triggered_on_count": int(llm_triggered),
        "llm_triggered_on_rate": float(llm_triggered / total if total else 0.0),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--language", default="en")
    ap.add_argument("--out-dir", default="tests/llm_toggle/venv_gpu_trace_rerun")
    args = ap.parse_args()

    env_path = ROOT / ".env"
    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    old = _set_env_file_key(env_path, "ENABLE_LLM_VERIFIER", "0")
    os.environ["ENABLE_LLM_VERIFIER"] = "0"
    try:
        off_data = run_mode("llm_off", CLAIMS_15, args.language)
    finally:
        _restore_env_file_key(env_path, "ENABLE_LLM_VERIFIER", old)

    old2 = _set_env_file_key(env_path, "ENABLE_LLM_VERIFIER", "1")
    os.environ["ENABLE_LLM_VERIFIER"] = "1"
    try:
        on_data = run_mode("llm_on", CLAIMS_15, args.language)
    finally:
        _restore_env_file_key(env_path, "ENABLE_LLM_VERIFIER", old2)

    compare = compare_rows(off_data["rows"], on_data["rows"])
    off_metrics = compute_metrics(off_data["rows"])
    on_metrics = compute_metrics(on_data["rows"])
    cmp_metrics = compute_comparison_metrics(compare)

    payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "language": args.language,
        "claims_count": len(CLAIMS_15),
        "metrics": {"llm_off": off_metrics, "llm_on": on_metrics},
        "comparison_metrics": cmp_metrics,
        "off": off_data,
        "on": on_data,
        "comparison": compare,
    }
    out_file = out_dir / "trace_run_results.json"
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"Saved: {out_file}")
    print(
        json.dumps(
            {
                "claims": len(CLAIMS_15),
                "off_accuracy": off_metrics["accuracy"],
                "on_accuracy": on_metrics["accuracy"],
                "off_neutral_rate": off_metrics["neutral_rate"],
                "on_neutral_rate": on_metrics["neutral_rate"],
                "verdict_changes": cmp_metrics["verdict_changes_off_vs_on"],
                "llm_triggered_on": cmp_metrics["llm_triggered_on_count"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

