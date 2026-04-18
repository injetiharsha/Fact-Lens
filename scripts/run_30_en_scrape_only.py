"""Run 30 EN benchmark claims with search APIs disabled and scraper-only evidence."""

from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Set, Tuple

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


def _dedupe_sources(sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen: Set[Tuple[str, str]] = set()
    out: List[Dict[str, Any]] = []
    for source in sources:
        key = (str(source.get("type")), str(source.get("subtype")))
        if key in seen:
            continue
        seen.add(key)
        out.append(source)
    return out


def _patch_scrape_only(pipeline: ClaimPipeline) -> None:
    gatherer = pipeline.evidence_gatherer

    def gather_scrape_only(
        claim: str,
        queries: List[str],
        sources: List[Dict[str, Any]],
        language: str = "en",
        max_evidence: int = 10,
    ) -> List[Dict[str, Any]]:
        urls: List[str] = []
        seen_urls: Set[str] = set()

        for source in _dedupe_sources(sources):
            source_type = str(source.get("type", ""))
            source_subtype = source.get("subtype")
            if source_type != "structured_api":
                continue
            try:
                if source_subtype and source_subtype not in gatherer.api_client.get_available_subtypes():
                    continue
                api_rows = gatherer.api_client.query(claim, queries, source_subtype, language)
            except Exception:
                continue
            for row in api_rows:
                url = str((row or {}).get("url", "")).strip()
                if not url or url in seen_urls:
                    continue
                seen_urls.add(url)
                urls.append(url)

        if not urls:
            return []

        scraped = gatherer.scraper.scrape_urls(urls, claim=claim, max_results=6)
        normalized = gatherer._normalize_evidence_list(scraped, default_type="scraping")
        if not normalized:
            return []
        unique = gatherer.aggregator.deduplicate(normalized)
        ranked = gatherer.aggregator.rank(unique, claim)
        return ranked[:max_evidence]

    gatherer.gather = gather_scrape_only


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
        default="tests/benchmarks/rfcs_benchmark_en_scrape_only",
    )
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

    rows = _load_cases(ROOT / args.cases, args.num_claims)
    pipeline = ClaimPipeline(_pipeline_config("en"))
    _patch_scrape_only(pipeline)

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
            "mode": "scrape_only_no_search_api",
        }
        predictions.append(row)
        print(
            f"[{idx}/{len(rows)}] id={row['id']} expected={expected} "
            f"predicted={verdict} ok={row['correct']} ev={evidence_count}"
        )

    metrics = _metrics(predictions)
    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "scrape_only_no_search_api",
        "claims": len(rows),
        "metrics": metrics,
        "outputs": {
            "predictions": str((out_dir / "predictions_en_30_scrape_only.json").resolve()),
            "metrics": str((out_dir / "metrics_en_30_scrape_only.json").resolve()),
        },
    }

    (out_dir / "predictions_en_30_scrape_only.json").write_text(
        json.dumps(predictions, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    (out_dir / "metrics_en_30_scrape_only.json").write_text(
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
