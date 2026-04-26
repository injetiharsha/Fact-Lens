"""Run thesis experiments with static profiles and save comparable metrics.

Usage examples:
  .\\.venv-gpu\\Scripts\\python.exe scripts\\run_thesis_experiment.py --profile base_thesis_v1 --split both --num-claims 30
  .\\.venv-gpu\\Scripts\\python.exe scripts\\run_thesis_experiment.py --profile research_score_v1 --split multi --num-claims 120 --cases tests/benchmarks/thesis_dataset_v1.json
"""

from __future__ import annotations

import argparse
import json
import os
import random
import statistics
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
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
ALL_EXPECTED_LABELS = LABELS | {"uncheckable"}
TRUSTED_CRED_THRESHOLD = 0.80
COST_PER_SEARCH_PROVIDER = {
    "ddg": 0.0,
    "duckduckgo": 0.0,
    "serpapi": 0.01,
    "tavily": 0.008,
    "newsapi": 0.005,
}
STRUCTURED_CALL_COST_USD = 0.001


def _env_flag(name: str, default: bool = False) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _is_blocked_by_checkability(details: Dict[str, Any]) -> bool:
    checkability = str((details or {}).get("checkability", "")).strip().lower()
    if not checkability:
        return False
    return checkability.startswith("uncheckable")


def _canonical_label(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().lower()
    if not text or text in {"none", "null", "nan", "n/a"}:
        return None
    aliases = {
        "true": "support",
        "false": "refute",
        "entailment": "support",
        "contradiction": "refute",
        "not enough info": "neutral",
        "nei": "neutral",
        "unverifiable": "uncheckable",
        "unchecked": "uncheckable",
    }
    return aliases.get(text, text)


def _resolve_expected_label(item: Dict[str, Any]) -> str | None:
    """Resolve dataset expected label from common field names."""
    # Prefer explicit 3-way labels when available.
    for key in (
        "expected",
        "expected_verdict",
        "expected_verdict_3way",
        "verdict",
        "ground_truth",
        "expected_label",
    ):
        text = _canonical_label(item.get(key))
        if text:
            return text
    # Fall back to 4-way expected label (includes uncheckable).
    text4 = _canonical_label(item.get("expected_verdict_4way"))
    if text4:
        return text4
    # Last resort: infer uncheckable from checkability label.
    chk = str(item.get("checkability_label", "")).strip().upper()
    if chk in {"PERSONAL_STATEMENT", "OPINION", "QUESTION_OR_REWRITE", "OTHER_UNCHECKABLE"}:
        return "uncheckable"
    return None


def _validate_row_label_consistency(item: Dict[str, Any], expected: str, idx: int) -> None:
    row_id = item.get("id", idx)
    if expected not in ALL_EXPECTED_LABELS:
        raise ValueError(
            f"Row id={row_id} has unsupported expected label '{expected}'. "
            f"Allowed: {sorted(ALL_EXPECTED_LABELS)}"
        )
    chk = str(item.get("checkability_label", "")).strip().upper()
    uncheckable_chk = {"PERSONAL_STATEMENT", "OPINION", "QUESTION_OR_REWRITE", "OTHER_UNCHECKABLE"}
    if chk in uncheckable_chk and expected != "uncheckable":
        raise ValueError(
            f"Row id={row_id} inconsistent labels: checkability_label={chk} but expected={expected}. "
            "Expected should be 'uncheckable' for this checkability label."
        )


def _load_profiles(path: Path) -> Dict[str, Any]:
    # Be tolerant to BOM-encoded JSON files from Windows editors.
    return json.loads(path.read_text(encoding="utf-8-sig"))


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


def _load_cases(
    path: Path,
    bucket: str,
    n: int,
    shuffle: bool = False,
    seed: int | None = None,
) -> List[Dict[str, Any]]:
    # Allow UTF-8 with optional BOM in dataset files.
    payload = json.loads(path.read_text(encoding="utf-8-sig"))

    if isinstance(payload, dict):
        rows_raw = payload.get("claims", [])
    elif isinstance(payload, list):
        rows_raw = payload
    else:
        raise ValueError(f"Unsupported dataset shape in {path}. Expected list or object with 'claims'.")

    rows: List[Dict[str, Any]] = [r for r in rows_raw if isinstance(r, dict)]
    if not rows:
        raise ValueError(f"No valid claim rows found in {path}.")

    target_bucket = bucket.upper()
    out: List[Dict[str, Any]] = []
    for row in rows:
        lang_bucket = str(row.get("lang_bucket", "")).upper().strip()
        if not lang_bucket:
            lang = str(row.get("language", "en")).strip().lower()
            lang_bucket = "EN" if lang == "en" else "MULTI"
        if lang_bucket == target_bucket:
            out.append(row)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(out)
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
    total = len(rows)
    blocked_rows = [r for r in rows if bool(r.get("blocked_by_checkability", False))]
    checked_rows = [r for r in rows if not bool(r.get("blocked_by_checkability", False))]

    # Primary evaluation set: checked (unblocked) claims only.
    y_true = [r["expected"] for r in checked_rows]
    y_pred = [r["predicted"] for r in checked_rows]
    checked_total = len(checked_rows)
    checked_correct = sum(1 for r in checked_rows if r["correct"])

    # Raw/all-claims metrics kept for audit.
    raw_correct = sum(1 for r in rows if r["correct"])

    total = len(rows)
    neutral_pred = sum(1 for p in y_pred if p == "neutral")
    tp, tn, fp, fn = _binary_counts(checked_rows)
    binary_total = sum(1 for r in checked_rows if r.get("expected") in {"support", "refute"})

    latencies = [float(r.get("duration_sec", 0.0) or 0.0) for r in rows]
    row_metrics = [_gather_row_metrics(r) for r in rows]
    blocked_count = len(blocked_rows)
    expected_uncheckable_rows = [r for r in rows if str(r.get("expected", "")).lower() == "uncheckable"]
    expected_uncheckable_count = len(expected_uncheckable_rows)

    checkable_expected_rows = [r for r in rows if str(r.get("expected", "")).lower() in LABELS]
    checkable_unblocked_rows = [r for r in checkable_expected_rows if not bool(r.get("blocked_by_checkability", False))]
    checkable_unblocked_total = len(checkable_unblocked_rows)
    checkable_unblocked_correct = sum(1 for r in checkable_unblocked_rows if r.get("correct"))

    blocked_expected_uncheckable = sum(
        1
        for r in blocked_rows
        if str(r.get("expected", "")).lower() == "uncheckable"
    )

    recall_values = [m["evidence_recall_at_k"] for m in row_metrics if m["evidence_recall_at_k"] is not None]

    return {
        "total_claims": total,
        "evaluated_claims": checked_total,
        "correct_predictions": checked_correct,
        "incorrect_predictions": checked_total - checked_correct,
        "accuracy": float(checked_correct / checked_total) if checked_total else 0.0,
        "neutral_predictions": neutral_pred,
        "neutral_rate": float(neutral_pred / checked_total) if checked_total else 0.0,
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
        # Raw/all-claims accuracy for auditing.
        "raw_correct_predictions": raw_correct,
        "raw_incorrect_predictions": total - raw_correct,
        "raw_accuracy": float(raw_correct / total) if total else 0.0,
        # Blocked-aware reporting (keeps raw accuracy unchanged).
        "blocked_claims": blocked_count,
        "blocked_rate": float(blocked_count / total) if total else 0.0,
        "expected_uncheckable_claims": expected_uncheckable_count,
        "checkable_claims_total": len(checkable_expected_rows),
        "checkable_unblocked_claims": checkable_unblocked_total,
        "checkable_unblocked_correct": checkable_unblocked_correct,
        "accuracy_checkable_unblocked": float(checkable_unblocked_correct / checkable_unblocked_total)
        if checkable_unblocked_total
        else 0.0,
        "blocked_expected_uncheckable": blocked_expected_uncheckable,
        # Gives credit when uncheckable rows are correctly blocked.
        "accuracy_blocked_adjusted": float((checkable_unblocked_correct + blocked_expected_uncheckable) / total) if total else 0.0,
    }


def _process_row_with_pipeline(
    idx: int,
    total: int,
    item: Dict[str, Any],
    pipeline: ClaimPipeline,
    default_language: str,
) -> Dict[str, Any]:
    claim = str(item["claim"])
    expected_raw = _resolve_expected_label(item)
    if expected_raw is None:
        raise KeyError(
            f"Row id={item.get('id', idx)} missing expected label. "
            "Provide one of: expected, expected_verdict, verdict, ground_truth, expected_label."
        )
    expected = str(expected_raw).strip().lower()
    _validate_row_label_consistency(item, expected, idx)
    language = str(item.get("language") or default_language)

    started = datetime.now(timezone.utc)
    err = ""
    verdict = "neutral"
    confidence = 0.0
    evidence: List[Dict[str, Any]] = []
    details: Dict[str, Any] = {}
    try:
        result = pipeline.analyze(claim=claim, language=language)
        verdict = _canonical_label(getattr(result, "verdict", "neutral")) or "neutral"
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
        "blocked_by_checkability": _is_blocked_by_checkability(details),
        "error": err,
        "expected_evidence_urls": item.get("expected_evidence_urls") or item.get("reference_urls") or [],
        "started_utc": started.isoformat(),
        "ended_utc": ended.isoformat(),
        "duration_sec": (ended - started).total_seconds(),
    }
    print(
        f"[{language} {idx}/{total}] id={row['id']} expected={expected} "
        f"predicted={verdict} ok={row['correct']} ev={row['evidence_count']}"
    )
    return row


def _is_retryable_rate_limit(error_text: str) -> bool:
    txt = str(error_text or "").strip().lower()
    if not txt:
        return False
    return ("429" in txt) or ("too many requests" in txt) or ("rate limit" in txt)


def _process_row_with_retry(
    idx: int,
    total: int,
    item: Dict[str, Any],
    pipeline: ClaimPipeline,
    default_language: str,
    retry_429: int,
    retry_backoff_sec: float,
) -> Dict[str, Any]:
    attempts = max(0, int(retry_429))
    for attempt in range(attempts + 1):
        row = _process_row_with_pipeline(
            idx=idx,
            total=total,
            item=item,
            pipeline=pipeline,
            default_language=default_language,
        )
        if not row.get("error") or attempt >= attempts or not _is_retryable_rate_limit(str(row.get("error"))):
            return row
        wait = max(0.0, float(retry_backoff_sec)) * (2 ** attempt)
        wait += random.uniform(0.0, 0.25)
        print(
            f"[retry {attempt + 1}/{attempts}] id={row.get('id')} "
            f"reason=rate_limit sleep={wait:.2f}s"
        )
        time.sleep(wait)
    return row


def _analyze_rows_shared_pipeline(
    rows: List[Dict[str, Any]],
    pipeline: ClaimPipeline,
    default_language: str,
    workers: int,
    max_in_flight: int,
    dispatch_delay_sec: float,
    retry_429: int,
    retry_backoff_sec: float,
) -> List[Dict[str, Any]]:
    if not rows:
        return []
    worker_count = max(1, int(workers))
    in_flight_limit = max(1, int(max_in_flight))
    dispatch_delay = max(0.0, float(dispatch_delay_sec))
    total = len(rows)
    if worker_count <= 1:
        return [
            _process_row_with_retry(
                i,
                total,
                item,
                pipeline,
                default_language,
                retry_429=retry_429,
                retry_backoff_sec=retry_backoff_sec,
            )
            for i, item in enumerate(rows, start=1)
        ]

    out: List[Dict[str, Any]] = [{} for _ in rows]
    with ThreadPoolExecutor(max_workers=worker_count) as pool:
        pending: Dict[Any, int] = {}
        for i, item in enumerate(rows, start=1):
            while len(pending) >= in_flight_limit:
                done = next(as_completed(list(pending.keys())))
                pos = pending.pop(done)
                out[pos] = done.result()
            fut = pool.submit(
                _process_row_with_retry,
                i,
                total,
                item,
                pipeline,
                default_language,
                retry_429,
                retry_backoff_sec,
            )
            pending[fut] = i - 1
            if dispatch_delay > 0 and i < total:
                time.sleep(dispatch_delay)
        for fut in as_completed(list(pending.keys())):
            pos = pending[fut]
            out[pos] = fut.result()
    return out


def _analyze_rows_shared_multi(
    rows: List[Dict[str, Any]],
    pipeline: ClaimPipeline,
    default_language: str = "hi",
    max_in_flight: int = 3,
    dispatch_delay_sec: float = 0.0,
    retry_429: int = 0,
    retry_backoff_sec: float = 4.0,
) -> List[Dict[str, Any]]:
    """
    Analyze rows with one shared pipeline instance for multi-language runs.
    Avoids loading duplicate model copies per language (critical on small GPUs).
    """
    worker_count = max(1, int(os.getenv("THESIS_CLAIM_WORKERS", "3")))
    return _analyze_rows_shared_pipeline(
        rows=rows,
        pipeline=pipeline,
        default_language=default_language,
        workers=worker_count,
        max_in_flight=max_in_flight,
        dispatch_delay_sec=dispatch_delay_sec,
        retry_429=retry_429,
        retry_backoff_sec=retry_backoff_sec,
    )


def _save_json(path: Path, payload: Dict[str, Any] | List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def _print_model_summary(label: str, cfg: Dict[str, Any]) -> None:
    llm_provider = os.getenv("LLM_VERIFIER_PROVIDER_EN" if label == "en" else "LLM_VERIFIER_PROVIDER_MULTI") or os.getenv(
        "LLM_VERIFIER_PROVIDER"
    )
    llm_model = os.getenv("LLM_VERIFIER_MODEL_EN" if label == "en" else "LLM_VERIFIER_MODEL_MULTI") or os.getenv(
        "LLM_VERIFIER_MODEL"
    )
    print(f"[models:{label}] pipeline_language={cfg.get('pipeline_language')}")
    print(f"[models:{label}] checkability={cfg.get('claim_checkability_checkpoint')}")
    print(f"[models:{label}] context={cfg.get('context_checkpoint')}")
    print(f"[models:{label}] relevance={cfg.get('relevance_checkpoint')}")
    print(f"[models:{label}] stance={cfg.get('stance_checkpoint')}")
    print(
        f"[models:{label}] llm_verifier_enabled={os.getenv('ENABLE_LLM_VERIFIER')} "
        f"provider={llm_provider} model={llm_model}"
    )
    print(
        f"[models:{label}] scoring_mode={os.getenv('SCORING_MODE', 'heuristic')} "
        f"relevance_bi_encoder={cfg.get('relevance_bi_encoder_model')}"
    )


def _print_effective_translation_web_env() -> None:
    keys = [
        "EVIDENCE_SOURCE_MODE",
        "EVIDENCE_STAGE_ORDER",
        "EVIDENCE_STAGE_MIN_RESULTS",
        "WEB_SEARCH_ENABLE_DDG",
        "WEB_SEARCH_ENABLE_TAVILY",
        "WEB_SEARCH_ENABLE_SERPAPI",
        "WEB_SEARCH_PROVIDER_ORDER",
        "MULTI_ENABLE_EN_QUERY_TRANSLATION",
        "EVIDENCE_ENABLE_TRANSLATED_QUERY_SEARCH",
        "MULTI_QUERY_TRANSLATION_LOCAL_ENABLE",
        "MULTI_QUERY_TRANSLATION_WEB_ENABLE",
        "MULTI_QUERY_TRANSLATION_PREFER_LLM",
        "TRANSLATION_LLM_PROVIDER",
    ]
    print("[env-check] effective translation/web settings:")
    for key in keys:
        print(f"[env-check] {key}={os.getenv(key, '')}")


def _validate_translation_web_config() -> None:
    translation_web_enabled = _env_flag("MULTI_QUERY_TRANSLATION_WEB_ENABLE", False) or _env_flag(
        "EVIDENCE_ENABLE_TRANSLATED_QUERY_SEARCH", False
    ) or _env_flag("MULTI_ENABLE_EN_QUERY_TRANSLATION", False)
    if not translation_web_enabled:
        return

    web_provider_enabled = _env_flag("WEB_SEARCH_ENABLE_DDG", False) or _env_flag(
        "WEB_SEARCH_ENABLE_TAVILY", False
    ) or _env_flag("WEB_SEARCH_ENABLE_SERPAPI", False)
    if not web_provider_enabled:
        raise RuntimeError(
            "Translation-to-web is enabled, but all web providers are disabled. "
            "Enable at least one of WEB_SEARCH_ENABLE_DDG/TAVILY/SERPAPI."
        )

    source_mode = str(os.getenv("EVIDENCE_SOURCE_MODE", "")).strip().lower()
    stage_order = [p.strip().lower() for p in str(os.getenv("EVIDENCE_STAGE_ORDER", "")).split(",") if p.strip()]
    if source_mode == "staged_fallback" and "web_search" not in stage_order:
        raise RuntimeError(
            "Translation-to-web is enabled, but staged retrieval does not include 'web_search' in EVIDENCE_STAGE_ORDER."
        )

    # Soft warning: web search may be skipped often when structured API appears first and threshold is low.
    if source_mode == "staged_fallback" and stage_order and stage_order[0] != "web_search":
        print(
            "[env-check] warning: web_search is not first stage; translation queries may be underused when earlier "
            "stages satisfy EVIDENCE_STAGE_MIN_RESULTS."
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", required=True, help="Profile name in configs/research/thesis_profiles.json")
    parser.add_argument("--profile-file", default="configs/research/thesis_profiles.json")
    parser.add_argument(
        "--env-only",
        action="store_true",
        help="Ignore profile env overrides and run using only current process/.env environment.",
    )
    parser.add_argument("--split", choices=["en", "multi", "both"], default="both")
    parser.add_argument("--num-claims", type=int, default=30)
    parser.add_argument("--cases", default="tests/benchmarks/rfcs_benchmark_multi/benchmark_cases_multi.json")
    parser.add_argument("--out-root", default="tests/benchmarks/thesis_runs")
    parser.add_argument("--device", choices=["auto", "cpu"], default="auto")
    parser.add_argument("--claim-workers", type=int, default=3)
    parser.add_argument("--max-in-flight", type=int, default=3, help="Max in-flight claim jobs (scheduler cap)")
    parser.add_argument("--dispatch-delay-ms", type=int, default=350, help="Delay between claim dispatches (ms)")
    parser.add_argument("--retry-429", type=int, default=1, help="Retries per claim on 429/rate-limit errors")
    parser.add_argument("--retry-backoff-sec", type=float, default=4.0, help="Base backoff seconds for 429 retries")
    parser.add_argument("--shuffle", action="store_true", default=True, help="Shuffle case order before slicing")
    parser.add_argument("--no-shuffle", dest="shuffle", action="store_false")
    parser.add_argument("--shuffle-seed", type=int, default=42, help="Seed for deterministic shuffling")
    parser.add_argument("--show-models", action="store_true", help="Print resolved model/checkpoint config at startup")
    parser.add_argument("--strict-profile-env", action="store_true", default=True)
    parser.add_argument("--no-strict-profile-env", dest="strict_profile_env", action="store_false")
    args = parser.parse_args()

    load_dotenv(override=True)
    if args.device == "cpu":
        # Must be set before model load so torch skips CUDA.
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    os.environ["THESIS_CLAIM_WORKERS"] = str(max(1, int(args.claim_workers)))
    if args.strict_profile_env:
        # Prevent api/routes/claim._env_or_file from pulling unset keys from .env.
        os.environ["IGNORE_ENV_FILE_FALLBACK"] = "1"

    profile_file = ROOT / args.profile_file
    profiles = _load_profiles(profile_file)
    if args.profile not in profiles:
        raise ValueError(f"Unknown profile '{args.profile}'. Available: {', '.join(sorted(profiles.keys()))}")

    profile = profiles[args.profile]
    old_env: Dict[str, str | None] = {}
    if not args.env_only:
        old_env = _apply_profile_env(profile)
    try:
        _print_effective_translation_web_env()
        _validate_translation_web_config()
        cases_path = ROOT / args.cases
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        out_dir = ROOT / args.out_root / f"{timestamp}_{args.profile}_{args.split}"

        outputs: Dict[str, Any] = {}
        if args.split in {"en", "both"}:
            en_rows = _load_cases(
                cases_path,
                "EN",
                args.num_claims,
                shuffle=bool(args.shuffle),
                seed=int(args.shuffle_seed),
            )
            en_cfg = _pipeline_config("en")
            if args.show_models:
                _print_model_summary("en", en_cfg)
            shared_en = ClaimPipeline(en_cfg)
            en_pred = _analyze_rows_shared_pipeline(
                rows=en_rows,
                pipeline=shared_en,
                default_language="en",
                workers=max(1, int(args.claim_workers)),
                max_in_flight=max(1, int(args.max_in_flight)),
                dispatch_delay_sec=max(0.0, float(args.dispatch_delay_ms) / 1000.0),
                retry_429=max(0, int(args.retry_429)),
                retry_backoff_sec=max(0.0, float(args.retry_backoff_sec)),
            )
            en_metrics = _metrics(en_pred)
            _save_json(out_dir / "predictions_en.json", en_pred)
            _save_json(out_dir / "metrics_en.json", en_metrics)
            outputs["en"] = {"claims": len(en_pred), "metrics": en_metrics}

        if args.split in {"multi", "both"}:
            multi_rows = _load_cases(
                cases_path,
                "MULTI",
                args.num_claims,
                shuffle=bool(args.shuffle),
                seed=int(args.shuffle_seed) + 1,
            )
            # One shared multi pipeline to avoid loading duplicate model copies per language.
            multi_cfg = _pipeline_config("hi")
            if args.show_models:
                _print_model_summary("multi", multi_cfg)
            shared_multi = ClaimPipeline(multi_cfg)
            multi_pred = _analyze_rows_shared_multi(
                multi_rows,
                shared_multi,
                default_language="hi",
                max_in_flight=max(1, int(args.max_in_flight)),
                dispatch_delay_sec=max(0.0, float(args.dispatch_delay_ms) / 1000.0),
                retry_429=max(0, int(args.retry_429)),
                retry_backoff_sec=max(0.0, float(args.retry_backoff_sec)),
            )
            multi_metrics = _metrics(multi_pred)
            _save_json(out_dir / "predictions_multi.json", multi_pred)
            _save_json(out_dir / "metrics_multi.json", multi_metrics)
            outputs["multi"] = {"claims": len(multi_pred), "metrics": multi_metrics}

        run_summary = {
            "generated_at_utc": datetime.now(timezone.utc).isoformat(),
            "profile_name": args.profile,
            "profile_description": profile.get("description", ""),
            "profile_env": {} if args.env_only else profile.get("env", {}),
            "env_only": bool(args.env_only),
            "profile_file": str(profile_file.resolve()),
            "split": args.split,
            "num_claims_requested": args.num_claims,
            "claim_workers": max(1, int(args.claim_workers)),
            "max_in_flight": max(1, int(args.max_in_flight)),
            "dispatch_delay_ms": max(0, int(args.dispatch_delay_ms)),
            "retry_429": max(0, int(args.retry_429)),
            "retry_backoff_sec": max(0.0, float(args.retry_backoff_sec)),
            "shuffle": bool(args.shuffle),
            "shuffle_seed": int(args.shuffle_seed),
            "cases": str(cases_path.resolve()),
            "out_dir": str(out_dir.resolve()),
            "outputs": outputs,
        }
        _save_json(out_dir / "run_summary.json", run_summary)
        print(json.dumps(run_summary, ensure_ascii=False, indent=2))
    finally:
        if not args.env_only:
            _restore_env(old_env)


if __name__ == "__main__":
    main()
