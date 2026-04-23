"""Run thesis experiments with static profiles and save comparable metrics.

Usage examples:
  .\\.venv-gpu\\Scripts\\python.exe scripts\\run_thesis_experiment.py --profile base_thesis_v1 --split both --num-claims 30
  .\\.venv-gpu\\Scripts\\python.exe scripts\\run_thesis_experiment.py --profile research_score_v1 --split multi --num-claims 120 --cases tests/benchmarks/thesis_dataset_v1.json
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
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
TRUSTED_CRED_THRESHOLD = 0.80
COST_PER_SEARCH_PROVIDER = {
    "ddg": 0.0,
    "duckduckgo": 0.0,
    "serpapi": 0.01,
    "tavily": 0.008,
    "newsapi": 0.005,
}
STRUCTURED_CALL_COST_USD = 0.001


def _load_profiles(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _apply_profile_env(profile: Dict[str, Any]) -> Dict[str, str | None]:
    env_map = dict(profile.get("env", {}) or {})
    old: Dict[str, str | None] = {}
    for key, value in env_map.items():
        old[key] = os.getenv(key)
        os.environ[key] = str(value)
    return old


def _restore_env(old: Dict[str, str | None]) -> None:
    for key, value in old.items():
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = value


def _load_cases(path: Path, bucket: str, n: int) -> List[Dict[str, Any]]:
    rows = json.loads(path.read_text(encoding="utf-8"))
    out = [r for r in rows if str(r.get("lang_bucket", "")).upper() == bucket.upper()]
    return out[:n]


def _binary_counts(rows: List[Dict[str, Any]]) -> Tuple[int, int, int, int]:
    filtered = [r for r in rows if r.get("expected") in {"support", "refute"}]
    tp = sum(1 for r in filtered if r["expected"] == "support" and r["predicted"] == "support")
    tn = sum(1 for r in filtered if r["expected"] == "refute" and r["predicted"] == "refute")
    fp = sum(1 for r in filtered if r["expected"] == "refute" and r["predicted"] == "support")
    fn = sum(1 for r in filtered if r["expected"] == "support" and r["predicted"] != "support")
    return tp, tn, fp, fn


def _host(url: str) -> str:
    value = str(url or "").strip().lower()
    if not value:
        return ""
    no_proto = value.split("://", 1)[-1]
    return no_proto.split("/", 1)[0].split(":", 1)[0].strip(".")


def _gather_row_metrics(row: Dict[str, Any]) -> Dict[str, Any]:
    details = row.get("details") or {}
    telemetry = details.get("retrieval_telemetry") or {}
    evidence = row.get("evidence") or []

    unique_hosts = {h for h in (_host(ev.get("url", "")) for ev in evidence) if h}
    trusted = sum(1 for ev in evidence if float(ev.get("credibility", 0.0) or 0.0) >= TRUSTED_CRED_THRESHOLD)
    trusted_ratio = (trusted / len(evidence)) if evidence else 0.0

    provider_counts = telemetry.get("search_provider_counts") or {}
    provider_calls = int(sum(int(v) for v in provider_counts.values())) if provider_counts else 0
    structured_counts = telemetry.get("source_type_counts") or {}
    structured_calls = int(structured_counts.get("structured_api", 0))

    est_cost = 0.0
    for provider, count in provider_counts.items():
        est_cost += float(COST_PER_SEARCH_PROVIDER.get(str(provider).lower(), 0.0)) * int(count)
    est_cost += STRUCTURED_CALL_COST_USD * structured_calls

    evidence_recall_at_k = None
    expected_urls = row.get("expected_evidence_urls") or row.get("reference_urls") or []
    if expected_urls:
        expected = {str(u).strip().lower() for u in expected_urls if str(u).strip()}
        got = {str(ev.get("url") or "").strip().lower() for ev in evidence if str(ev.get("url") or "").strip()}
        if expected:
            evidence_recall_at_k = len(expected.intersection(got)) / len(expected)

    return {
        "source_diversity": len(unique_hosts),
        "trusted_source_ratio": trusted_ratio,
        "provider_calls": provider_calls,
        "structured_calls": structured_calls,
        "estimated_cost_usd": est_cost,
        "evidence_recall_at_k": evidence_recall_at_k,
    }


def _metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    y_true = [r["expected"] for r in rows]
    y_pred = [r["predicted"] for r in rows]
    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    neutral_pred = sum(1 for p in y_pred if p == "neutral")
    tp, tn, fp, fn = _binary_counts(rows)
    binary_total = sum(1 for r in rows if r.get("expected") in {"support", "refute"})

    latencies = [float(r.get("duration_sec", 0.0) or 0.0) for r in rows]
    row_metrics = [_gather_row_metrics(r) for r in rows]

    recall_values = [m["evidence_recall_at_k"] for m in row_metrics if m["evidence_recall_at_k"] is not None]

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
        "latency_avg_sec": float(sum(latencies) / len(latencies)) if latencies else 0.0,
        "latency_median_sec": float(statistics.median(latencies)) if latencies else 0.0,
        "source_diversity_avg": float(sum(m["source_diversity"] for m in row_metrics) / len(row_metrics)) if row_metrics else 0.0,
        "trusted_source_ratio_avg": float(sum(m["trusted_source_ratio"] for m in row_metrics) / len(row_metrics)) if row_metrics else 0.0,
        "provider_calls_total": int(sum(m["provider_calls"] for m in row_metrics)),
        "structured_calls_total": int(sum(m["structured_calls"] for m in row_metrics)),
        "estimated_cost_usd_total": float(sum(m["estimated_cost_usd"] for m in row_metrics)),
        "estimated_cost_usd_per_claim": float(sum(m["estimated_cost_usd"] for m in row_metrics) / len(row_metrics)) if row_metrics else 0.0,
        "evidence_recall_at_k_avg": float(sum(recall_values) / len(recall_values)) if recall_values else None,
    }


def _analyze_rows(rows: List[Dict[str, Any]], default_language: str, pipelines: Dict[str, ClaimPipeline]) -> List[Dict[str, Any]]:
    predictions: List[Dict[str, Any]] = []
    for idx, item in enumerate(rows, start=1):
        claim = str(item["claim"])
        expected = str(item["expected"]).lower()
        language = str(item.get("language") or default_language)
        if language not in pipelines:
            pipelines[language] = ClaimPipeline(_pipeline_config(language))

        started = datetime.now(timezone.utc)
        err = ""
        verdict = "neutral"
        confidence = 0.0
        evidence: List[Dict[str, Any]] = []
        details: Dict[str, Any] = {}
        try:
            result = pipelines[language].analyze(claim=claim, language=language)
            verdict = str(getattr(result, "verdict", "neutral") or "neutral").lower()
            if verdict not in LABELS:
                verdict = "neutral"
            confidence = float(getattr(result, "confidence", 0.0) or 0.0)
            evidence = list(getattr(result, "evidence", []) or [])
            details = dict(getattr(result, "details", {}) or {})
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
            "evidence_count": len(evidence),
            "evidence": evidence,
            "details": details,
            "error": err,
            "expected_evidence_urls": item.get("expected_evidence_urls") or item.get("reference_urls") or [],
            "started_utc": started.isoformat(),
            "ended_utc": ended.isoformat(),
            "duration_sec": (ended - started).total_seconds(),
        }
        predictions.append(row)
        print(
            f"[{language} {idx}/{len(rows)}] id={row['id']} expected={expected} "
            f"predicted={verdict} ok={row['correct']} ev={row['evidence_count']}"
        )
    return predictions


def _save_json(path: Path, payload: Dict[str, Any] | List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, help="Profile name in configs/research/thesis_profiles.json")
    parser.add_argument("--profile-file", default="configs/research/thesis_profiles.json")
    parser.add_argument("--split", choices=["en", "multi", "both"], default="both")
    parser.add_argument("--num-claims", type=int, default=30)
    parser.add_argument("--cases", default="tests/benchmarks/rfcs_benchmark_multi/benchmark_cases_multi.json")
    parser.add_argument("--out-root", default="tests/benchmarks/thesis_runs")
    args = parser.parse_args()

    load_dotenv(override=True)
    profile_file = ROOT / args.profile_file
    profiles = _load_profiles(profile_file)
    if args.profile not in profiles:
        raise ValueError(f"Unknown profile '{args.profile}'. Available: {', '.join(sorted(profiles.keys()))}")

    profile = profiles[args.profile]
    old_env = _apply_profile_env(profile)
    try:
        cases_path = ROOT / args.cases
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / args.out_root / f"{timestamp}_{args.profile}_{args.split}"

        outputs: Dict[str, Any] = {}
        if args.split in {"en", "both"}:
            en_rows = _load_cases(cases_path, "EN", args.num_claims)
            en_pred = _analyze_rows(en_rows, "en", pipelines={"en": ClaimPipeline(_pipeline_config("en"))})
            en_metrics = _metrics(en_pred)
            _save_json(out_dir / "predictions_en.json", en_pred)
            _save_json(out_dir / "metrics_en.json", en_metrics)
            outputs["en"] = {"claims": len(en_pred), "metrics": en_metrics}

        if args.split in {"multi", "both"}:
            multi_rows = _load_cases(cases_path, "MULTI", args.num_claims)
            multi_pred = _analyze_rows(multi_rows, "hi", pipelines={})
            multi_metrics = _metrics(multi_pred)
            _save_json(out_dir / "predictions_multi.json", multi_pred)
            _save_json(out_dir / "metrics_multi.json", multi_metrics)
            outputs["multi"] = {"claims": len(multi_pred), "metrics": multi_metrics}

        run_summary = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "profile_name": args.profile,
            "profile_description": profile.get("description", ""),
            "profile_env": profile.get("env", {}),
            "profile_file": str(profile_file.resolve()),
            "split": args.split,
            "num_claims_requested": args.num_claims,
            "cases": str(cases_path.resolve()),
            "out_dir": str(out_dir.resolve()),
            "outputs": outputs,
        }
        _save_json(out_dir / "run_summary.json", run_summary)
        print(json.dumps(run_summary, ensure_ascii=False, indent=2))
    finally:
        _restore_env(old_env)


if __name__ == "__main__":
    main()
