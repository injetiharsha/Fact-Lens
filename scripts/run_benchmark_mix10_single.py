"""Run single 10-claim mixed benchmark profile and save JSON output.

Profiles:
- current
- old_stage56
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.routes.claim import _pipeline_config
from pipeline.claim_pipeline import ClaimPipeline

EN_CASES = ROOT / "tests/benchmarks/rfcs_benchmark_en/benchmark_cases_en.json"
MULTI_CASES = ROOT / "tests/benchmarks/rfcs_benchmark_multi/benchmark_cases_multi.json"
TARGET_MULTI_LANGS = ["te", "hi", "ta", "kn", "ml"]


def _load_mix10() -> List[Dict[str, Any]]:
    en_rows = json.loads(EN_CASES.read_text(encoding="utf-8"))
    en_rows = [r for r in en_rows if str(r.get("lang_bucket", "")).upper() == "EN"][:5]

    multi_rows = json.loads(MULTI_CASES.read_text(encoding="utf-8"))
    picks: List[Dict[str, Any]] = []
    seen = set()
    for lang in TARGET_MULTI_LANGS:
        for row in multi_rows:
            if str(row.get("language", "")).lower() == lang and row.get("id") not in seen:
                picks.append(row)
                seen.add(row.get("id"))
                break
    if len(picks) != len(TARGET_MULTI_LANGS):
        found = {str(r.get("language", "")).lower() for r in picks}
        missing = [l for l in TARGET_MULTI_LANGS if l not in found]
        raise RuntimeError(f"Missing MULTI rows for languages: {missing}")
    return en_rows + picks


@contextmanager
def _env_overrides(overrides: Dict[str, str]):
    prev: Dict[str, str | None] = {}
    for k, v in overrides.items():
        prev[k] = os.getenv(k)
        os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, old_v in prev.items():
            if old_v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old_v


def _profile_snapshot() -> Dict[str, Any]:
    keys = [
        "MULTI_PHASE2_MODE",
        "EVIDENCE_DAG_ENABLED",
        "EVIDENCE_GATHER_CACHE_ENABLE",
        "EVIDENCE_MMR_ENABLED",
        "EVIDENCE_DOMAIN_DIVERSITY_ENABLED",
        "EVIDENCE_SCRAPER_ENRICH_MAX_RESULTS",
        "SCRAPER_URL_CANDIDATE_MULT",
        "WEB_SEARCH_PROVIDER_ORDER",
        "WEB_SEARCH_MAX_QUERIES",
        "WEB_SEARCH_MIN_PROVIDERS_BEFORE_STOP",
        "WEB_SEARCH_MAX_TOTAL_RESULTS_EN",
        "WEB_SEARCH_MAX_TOTAL_RESULTS_MULTI",
    ]
    return {k: os.getenv(k) for k in keys}


def _metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    neutral = sum(1 for r in rows if r["predicted_verdict"] == "neutral")

    by_bucket: Dict[str, Dict[str, int]] = {}
    by_lang: Dict[str, Dict[str, int]] = {}
    for r in rows:
        bucket = str(r.get("lang_bucket") or "MULTI").upper()
        lang = str(r.get("language") or "en").lower()
        by_bucket.setdefault(bucket, {"total": 0, "correct": 0})
        by_bucket[bucket]["total"] += 1
        by_bucket[bucket]["correct"] += int(bool(r["correct"]))
        by_lang.setdefault(lang, {"total": 0, "correct": 0})
        by_lang[lang]["total"] += 1
        by_lang[lang]["correct"] += int(bool(r["correct"]))

    return {
        "total_claims": total,
        "correct_predictions": correct,
        "accuracy": round(correct / total, 3) if total else 0.0,
        "neutral_rate": round(neutral / total, 3) if total else 0.0,
        "by_bucket": {
            k: {
                "total": v["total"],
                "correct": v["correct"],
                "accuracy": round(v["correct"] / v["total"], 3) if v["total"] else 0.0,
            }
            for k, v in by_bucket.items()
        },
        "by_language": {
            k: {
                "total": v["total"],
                "correct": v["correct"],
                "accuracy": round(v["correct"] / v["total"], 3) if v["total"] else 0.0,
            }
            for k, v in sorted(by_lang.items())
        },
    }


def _run(rows: List[Dict[str, Any]]) -> tuple[list[Dict[str, Any]], dict[str, Any], float]:
    t0 = time.time()
    en_pipe = ClaimPipeline(_pipeline_config("en"))
    multi_pipe = ClaimPipeline(_pipeline_config("hi"))
    out: List[Dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        claim = str(row["claim"])
        expected = str(row["expected"]).lower()
        language = str(row.get("language") or "en").lower()
        bucket = str(row.get("lang_bucket") or "MULTI").upper()
        print(f"[{idx}/{len(rows)}] {bucket}/{language} id={row.get('id')}", flush=True)
        pipe = en_pipe if language == "en" else multi_pipe
        result = pipe.analyze(claim=claim, language=language)
        pred = str(getattr(result, "verdict", "neutral") or "neutral").lower()
        out.append(
            {
                "id": row.get("id"),
                "lang_bucket": bucket,
                "language": language,
                "claim": claim,
                "expected_verdict": expected,
                "predicted_verdict": pred,
                "correct": pred == expected,
                "confidence": float(getattr(result, "confidence", 0.0) or 0.0),
                "evidence_count": len(getattr(result, "evidence", []) or []),
                "details": dict(getattr(result, "details", {}) or {}),
            }
        )
    total_time = round(time.time() - t0, 3)
    return out, _metrics(out), total_time


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", choices=["current", "old_stage56"], required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    load_dotenv(override=True)
    rows = _load_mix10()

    overrides: Dict[str, str] = {}
    if args.profile == "old_stage56":
        overrides = {
            "MULTI_PHASE2_MODE": "scored",
            "EVIDENCE_DAG_ENABLED": "0",
            "EVIDENCE_GATHER_CACHE_ENABLE": "0",
            "EVIDENCE_MMR_ENABLED": "0",
            "EVIDENCE_DOMAIN_DIVERSITY_ENABLED": "1",
            "EVIDENCE_SCRAPER_ENRICH_MAX_RESULTS": "6",
            "SCRAPER_URL_CANDIDATE_MULT": "1",
        }

    with _env_overrides(overrides):
        profile = _profile_snapshot()
        claims, metrics, total_time = _run(rows)

    payload = {
        "profile_name": args.profile,
        "run_profile": profile,
        "benchmark_metrics": metrics,
        "total_time_seconds": total_time,
        "claims": claims,
    }
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved: {out}")
    print(json.dumps({"profile": args.profile, **metrics, "total_time_seconds": total_time}, ensure_ascii=False))


if __name__ == "__main__":
    main()

