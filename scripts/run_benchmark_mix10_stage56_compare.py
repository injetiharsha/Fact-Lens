"""Run 10-claim mixed benchmark and compare current vs old-style Stage 5/6 profile.

Sample:
- 5 EN claims (first 5 from EN benchmark set)
- 5 MULTI claims (first one each from te/hi/ta/kn/ml)
"""

from __future__ import annotations

import json
import os
import sys
import time
import gc
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Dict, List, Tuple

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.routes.claim import _pipeline_config
from pipeline.claim_pipeline import ClaimPipeline

EN_CASES = ROOT / "tests/benchmarks/rfcs_benchmark_en/benchmark_cases_en.json"
MULTI_CASES = ROOT / "tests/benchmarks/rfcs_benchmark_multi/benchmark_cases_multi.json"
OUT_DIR = ROOT / "tests/benchmarks/rfcs_benchmark_mix10"
OUT_CURRENT = OUT_DIR / "mix10_current.json"
OUT_OLD = OUT_DIR / "mix10_old_stage56.json"
OUT_COMPARE = OUT_DIR / "mix10_compare.json"

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
        missing = [l for l in TARGET_MULTI_LANGS if l not in {str(r.get("language", "")).lower() for r in picks}]
        raise RuntimeError(f"Missing MULTI language rows for: {missing}")

    return en_rows + picks


@contextmanager
def _env_overrides(overrides: Dict[str, str]):
    old: Dict[str, str | None] = {}
    for k, v in overrides.items():
        old[k] = os.getenv(k)
        os.environ[k] = str(v)
    try:
        yield
    finally:
        for k, old_v in old.items():
            if old_v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = old_v


def _metrics(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    total = len(rows)
    correct = sum(1 for r in rows if r["correct"])
    neutral = sum(1 for r in rows if r["predicted_verdict"] == "neutral")
    per_bucket = {"EN": {"total": 0, "correct": 0}, "MULTI": {"total": 0, "correct": 0}}
    per_lang: Dict[str, Dict[str, int]] = {}
    for r in rows:
        bucket = str(r.get("lang_bucket") or "MULTI").upper()
        lang = str(r.get("language") or "en").lower()
        per_bucket.setdefault(bucket, {"total": 0, "correct": 0})
        per_bucket[bucket]["total"] += 1
        per_bucket[bucket]["correct"] += int(bool(r["correct"]))
        if lang not in per_lang:
            per_lang[lang] = {"total": 0, "correct": 0}
        per_lang[lang]["total"] += 1
        per_lang[lang]["correct"] += int(bool(r["correct"]))

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
            for k, v in per_bucket.items()
        },
        "by_language": {
            k: {
                "total": v["total"],
                "correct": v["correct"],
                "accuracy": round(v["correct"] / v["total"], 3) if v["total"] else 0.0,
            }
            for k, v in sorted(per_lang.items())
        },
    }


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


def _run_once(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, Any], float]:
    t0 = time.time()
    en_pipe = ClaimPipeline(_pipeline_config("en"))
    multi_pipe = ClaimPipeline(_pipeline_config("hi"))
    out: List[Dict[str, Any]] = []

    for i, row in enumerate(rows, start=1):
        claim = str(row["claim"])
        expected = str(row["expected"]).lower()
        language = str(row.get("language") or "en").lower()
        bucket = str(row.get("lang_bucket") or "MULTI").upper()
        print(f"[{i}/{len(rows)}] {bucket}/{language} id={row.get('id')}")
        pipe = en_pipe if language == "en" else multi_pipe
        rs = pipe.analyze(claim=claim, language=language)
        predicted = str(getattr(rs, "verdict", "neutral") or "neutral").lower()
        details = dict(getattr(rs, "details", {}) or {})
        out.append(
            {
                "id": row.get("id"),
                "lang_bucket": bucket,
                "language": language,
                "claim": claim,
                "expected_verdict": expected,
                "predicted_verdict": predicted,
                "correct": predicted == expected,
                "confidence": float(getattr(rs, "confidence", 0.0) or 0.0),
                "evidence_count": len(getattr(rs, "evidence", []) or []),
                "details": details,
            }
        )
    total_time = round(time.time() - t0, 3)
    del en_pipe
    del multi_pipe
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass
    return out, _metrics(out), total_time


def main() -> None:
    load_dotenv(override=True)
    rows = _load_mix10()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    current_profile = _profile_snapshot()
    print("Running CURRENT profile...")
    current_rows, current_metrics, current_time = _run_once(rows)
    current_payload = {
        "run_profile": current_profile,
        "benchmark_metrics": current_metrics,
        "total_time_seconds": current_time,
        "claims": current_rows,
    }
    OUT_CURRENT.write_text(json.dumps(current_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    old_stage56_overrides = {
        # Stage-5/6 old-style behavior knobs only.
        "MULTI_PHASE2_MODE": "scored",
        "EVIDENCE_DAG_ENABLED": "0",
        "EVIDENCE_GATHER_CACHE_ENABLE": "0",
        "EVIDENCE_MMR_ENABLED": "0",
        "EVIDENCE_DOMAIN_DIVERSITY_ENABLED": "1",
        "EVIDENCE_SCRAPER_ENRICH_MAX_RESULTS": "6",
        "SCRAPER_URL_CANDIDATE_MULT": "1",
    }
    print("Running OLD Stage5/6 profile...")
    with _env_overrides(old_stage56_overrides):
        old_profile = _profile_snapshot()
        old_rows, old_metrics, old_time = _run_once(rows)
    old_payload = {
        "run_profile": old_profile,
        "benchmark_metrics": old_metrics,
        "total_time_seconds": old_time,
        "claims": old_rows,
    }
    OUT_OLD.write_text(json.dumps(old_payload, indent=2, ensure_ascii=False), encoding="utf-8")

    compare = {
        "current_file": str(OUT_CURRENT),
        "old_stage56_file": str(OUT_OLD),
        "accuracy_current": current_metrics["accuracy"],
        "accuracy_old_stage56": old_metrics["accuracy"],
        "accuracy_delta_old_minus_current": round(old_metrics["accuracy"] - current_metrics["accuracy"], 3),
        "neutral_rate_current": current_metrics["neutral_rate"],
        "neutral_rate_old_stage56": old_metrics["neutral_rate"],
        "neutral_rate_delta_old_minus_current": round(
            old_metrics["neutral_rate"] - current_metrics["neutral_rate"], 3
        ),
        "time_current_sec": current_time,
        "time_old_stage56_sec": old_time,
    }
    OUT_COMPARE.write_text(json.dumps(compare, indent=2, ensure_ascii=False), encoding="utf-8")

    print("\nMIX10 COMPARISON")
    print(json.dumps(compare, indent=2, ensure_ascii=False))
    print(f"\nSaved:\n- {OUT_CURRENT}\n- {OUT_OLD}\n- {OUT_COMPARE}")


if __name__ == "__main__":
    main()
