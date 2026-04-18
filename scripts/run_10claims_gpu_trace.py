from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.routes.claim import _pipeline_config  # noqa: E402
from pipeline.claim_pipeline import ClaimPipeline  # noqa: E402


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


def _extract_parser_error(details: Dict[str, Any]) -> str:
    llm = (details or {}).get("llm_verifier") or {}
    reason = str(llm.get("reason", "") or "")
    if not reason:
        return ""
    low = reason.lower()
    if "parse" in low or "unterminated string" in low or "expecting" in low or "json" in low:
        return reason
    return ""


def _load_first_n_cases(cases_path: Path, n: int) -> List[Dict[str, Any]]:
    rows = json.loads(cases_path.read_text(encoding="utf-8"))
    return rows[:n]


def _attach_tavily_trace(pipeline: ClaimPipeline, tavily_events: List[Dict[str, Any]]) -> None:
    web = pipeline.evidence_gatherer.web_search
    tavily = getattr(web, "tavily", None)
    if tavily is None:
        return

    original_search = tavily.search
    original_next = tavily._next_client

    call_state: Dict[str, Any] = {"slots": []}

    def wrapped_next():
        nxt = original_next()
        if nxt:
            slot, _ = nxt
            call_state["slots"].append(int(slot) + 1)  # 1-based slot for readability
        return nxt

    def wrapped_search(query: str, max_results: int):
        call_state["slots"] = []
        before_idx = int(getattr(tavily, "_idx", 0))
        blocked_before = sorted(int(x) + 1 for x in getattr(tavily, "_blocked_slots", set()))
        t0 = time.time()
        ok = True
        err = ""
        result_count = 0
        try:
            out = original_search(query=query, max_results=max_results)
            result_count = len(out or [])
            return out
        except Exception as exc:
            ok = False
            err = str(exc)
            raise
        finally:
            after_idx = int(getattr(tavily, "_idx", 0))
            blocked_after = sorted(int(x) + 1 for x in getattr(tavily, "_blocked_slots", set()))
            tavily_events.append(
                {
                    "query": query,
                    "ok": ok,
                    "error": err,
                    "duration_ms": int((time.time() - t0) * 1000),
                    "result_count": result_count,
                    "client_count": len(getattr(tavily, "clients", []) or []),
                    "idx_before": before_idx,
                    "idx_after": after_idx,
                    "slot_attempt_order": list(call_state["slots"]),
                    "blocked_slots_before": blocked_before,
                    "blocked_slots_after": blocked_after,
                }
            )

    tavily._next_client = wrapped_next
    tavily.search = wrapped_search


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cases",
        default="tests/benchmarks/rfcs_benchmark_multi/benchmark_cases_multi.json",
    )
    parser.add_argument("--n", type=int, default=10)
    parser.add_argument("--out-dir", default="tests/benchmarks/rfcs_benchmark_multi/trace_10claims_gpu")
    parser.add_argument("--llm-mode", choices=["env", "on", "off"], default="env")
    args = parser.parse_args()

    # Let .env override inherited process vars unless caller explicitly forces mode.
    load_dotenv(override=True)
    if args.llm_mode == "on":
        os.environ["ENABLE_LLM_VERIFIER"] = "1"
    elif args.llm_mode == "off":
        os.environ["ENABLE_LLM_VERIFIER"] = "0"

    out_dir = ROOT / args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = _load_first_n_cases(ROOT / args.cases, args.n)
    run_rows: List[Dict[str, Any]] = []
    all_tavily_events: List[Dict[str, Any]] = []

    for i, case in enumerate(cases, start=1):
        claim = str(case.get("claim", ""))
        language = str(case.get("language", "hi"))
        expected = str(case.get("expected", "neutral")).lower()
        claim_id = case.get("id", i)

        cfg = _pipeline_config(language)
        pipeline = ClaimPipeline(cfg)
        claim_tavily_events: List[Dict[str, Any]] = []
        _attach_tavily_trace(pipeline, claim_tavily_events)

        started = datetime.now(timezone.utc)
        error = ""
        result = None
        try:
            result = pipeline.analyze(claim=claim, language=language)
        except Exception as exc:
            error = str(exc)
        ended = datetime.now(timezone.utc)

        verdict = "neutral"
        confidence = 0.0
        details: Dict[str, Any] = {}
        evidence_count = 0
        if result is not None:
            verdict = str(getattr(result, "verdict", "neutral") or "neutral").lower()
            confidence = float(getattr(result, "confidence", 0.0) or 0.0)
            details = getattr(result, "details", {}) or {}
            evidence_count = len(getattr(result, "evidence", []) or [])

        parser_error = _extract_parser_error(details)
        row = {
            "index": i,
            "id": claim_id,
            "language": language,
            "expected": expected,
            "claim": claim,
            "predicted": verdict,
            "correct": verdict == expected,
            "confidence": confidence,
            "evidence_count": evidence_count,
            "error": error,
            "parser_error": parser_error,
            "llm_verifier": _safe((details or {}).get("llm_verifier", {})),
            "started_utc": started.isoformat(),
            "ended_utc": ended.isoformat(),
            "duration_sec": (ended - started).total_seconds(),
            "tavily_events": claim_tavily_events,
        }
        run_rows.append(row)
        all_tavily_events.extend(claim_tavily_events)
        print(
            f"[{i}/{len(cases)}] id={claim_id} lang={language} expected={expected} "
            f"pred={verdict} ev={evidence_count} tavily_calls={len(claim_tavily_events)}"
        )

    slot_hist: Dict[str, int] = {}
    for ev in all_tavily_events:
        for slot in ev.get("slot_attempt_order", []):
            key = str(slot)
            slot_hist[key] = slot_hist.get(key, 0) + 1

    parser_fail_rows = [r for r in run_rows if r.get("parser_error")]
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "claims_run": len(run_rows),
        "correct": sum(1 for r in run_rows if r.get("correct")),
        "accuracy": (sum(1 for r in run_rows if r.get("correct")) / len(run_rows)) if run_rows else 0.0,
        "parser_fail_count": len(parser_fail_rows),
        "parser_fail_ids": [r.get("id") for r in parser_fail_rows],
        "tavily_slot_usage_histogram": slot_hist,
        "tavily_total_calls": len(all_tavily_events),
    }

    payload = {"summary": summary, "rows": run_rows}
    out_file = out_dir / "trace_10claims_gpu.json"
    out_file.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"Saved: {out_file}")


if __name__ == "__main__":
    main()
