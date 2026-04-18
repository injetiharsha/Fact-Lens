"""Run RFCS benchmark from HTML file containing JS DATA array.

Outputs:
- metrics_en.json
- metrics_multi.json
- metrics_all.json
- predictions_en.json
- predictions_multi.json
- predictions_all.json
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from dotenv import load_dotenv
from sklearn.metrics import accuracy_score, confusion_matrix

from pipeline.claim_pipeline import ClaimPipeline

LABELS = ["support", "refute", "neutral"]
LANG_LABEL_TO_CODE = {
    "english": "en",
    "hindi": "hi",
    "telugu": "te",
    "tamil": "ta",
    "kannada": "kn",
    "malayalam": "ml",
}


def _env_bool(name: str, default: bool) -> bool:
    import os

    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    import os

    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else int(default)
    except Exception:
        value = int(default)
    return max(minimum, value)


def _select_checkpoint(component: str, language: str) -> str | None:
    import os

    lang_is_en = language.lower().startswith("en")
    upper = component.upper()

    en_enabled = _env_bool(f"ENABLE_{upper}_EN", True)
    multi_enabled = _env_bool(f"ENABLE_{upper}_MULTI", True)
    en_path = os.getenv(f"{upper}_EN_PATH", f"checkpoints/{component}/en")
    multi_path = os.getenv(f"{upper}_MULTI_PATH", f"checkpoints/{component}/multi")

    if lang_is_en and en_enabled:
        return en_path
    if (not lang_is_en) and multi_enabled:
        return multi_path
    if en_enabled:
        return en_path
    if multi_enabled:
        return multi_path
    return None


def _pipeline_config(language: str) -> dict:
    import os

    return {
        "claim_checkability_checkpoint": _select_checkpoint("checkability", language),
        "context_checkpoint": _select_checkpoint("context", language),
        "relevance_checkpoint": _select_checkpoint("relevance", language),
        "max_evidence": _env_int("PIPELINE_MAX_EVIDENCE", 5, minimum=1),
        "enable_two_stage_relevance": _env_bool("ENABLE_TWO_STAGE_RELEVANCE", True),
        "relevance_bi_encoder_model": os.getenv(
            "RELEVANCE_BI_ENCODER_MODEL", "intfloat/multilingual-e5-small"
        ),
        "relevance_shortlist_k": int(os.getenv("RELEVANCE_SHORTLIST_K", "20")),
        "relevance_top_k": (
            int(os.getenv("RELEVANCE_TOP_K", "0"))
            if int(os.getenv("RELEVANCE_TOP_K", "0")) > 0
            else None
        ),
        "relevance_drop_threshold": float(os.getenv("RELEVANCE_DROP_THRESHOLD", "0.30")),
        "stance_checkpoint": _select_checkpoint("stance", language),
        "enable_llm_verifier": _env_bool("ENABLE_LLM_VERIFIER", True),
        "llm_provider": os.getenv("LLM_VERIFIER_PROVIDER", "openai"),
        "llm_model": os.getenv("LLM_VERIFIER_MODEL", "gpt-4o-mini"),
        "llm_neutral_only": _env_bool("LLM_VERIFIER_NEUTRAL_ONLY", True),
        "llm_conf_threshold": float(os.getenv("LLM_VERIFIER_CONF_THRESHOLD", "0.55")),
    }


def _extract_data_array(html_text: str) -> List[Dict[str, Any]]:
    m = re.search(r"const\s+DATA\s*=\s*\[(.*?)\]\s*;\s*const\s+CONTEXT_COLORS", html_text, re.S)
    if not m:
        raise RuntimeError("Could not find `const DATA = [...]` in HTML.")
    js_arr_body = m.group(1)
    js_arr = "[" + js_arr_body + "]"
    js_arr = re.sub(r'([,{]\s*)([A-Za-z_][A-Za-z0-9_]*)\s*:', r'\1"\2":', js_arr)
    js_arr = re.sub(r",\s*\]\s*$", "]", js_arr, flags=re.S)
    return json.loads(js_arr)


def _normalize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for r in rows:
        lang = str(r.get("lang", "")).strip().upper()
        lang_label = str(r.get("lang_label", "")).strip()
        code = LANG_LABEL_TO_CODE.get(lang_label.lower(), "en" if lang == "EN" else "hi")
        expected = str(r.get("verdict", "")).strip().lower()
        if expected not in LABELS:
            continue
        claim = str(r.get("claim", "")).strip()
        if not claim:
            continue
        out.append(
            {
                "id": int(r.get("id", len(out) + 1)),
                "lang_bucket": "EN" if lang == "EN" else "MULTI",
                "lang_label": lang_label,
                "language": code,
                "context": str(r.get("ctx", "")).strip(),
                "type": str(r.get("type", "")).strip(),
                "claim": claim,
                "expected": expected,
            }
        )
    return out


def _build_metrics(pred_rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    y_true = [r["expected"] for r in pred_rows]
    y_pred = [r["predicted"] for r in pred_rows]
    total = len(pred_rows)
    correct = sum(1 for a, b in zip(y_true, y_pred) if a == b)
    neutral_pred = sum(1 for p in y_pred if p == "neutral")
    cm = confusion_matrix(y_true, y_pred, labels=LABELS).tolist() if total else [[0, 0, 0], [0, 0, 0], [0, 0, 0]]

    by_context = defaultdict(lambda: {"total": 0, "correct": 0})
    for row in pred_rows:
        ctx = row.get("context", "UNKNOWN")
        by_context[ctx]["total"] += 1
        if row["correct"]:
            by_context[ctx]["correct"] += 1

    by_context_acc = {
        ctx: {
            "total": d["total"],
            "correct": d["correct"],
            "accuracy": (d["correct"] / d["total"]) if d["total"] else 0.0,
        }
        for ctx, d in sorted(by_context.items())
    }

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
        "context_accuracy": by_context_acc,
    }


def _run(rows: List[Dict[str, Any]], out_dir: Path) -> None:
    en_pipeline = ClaimPipeline(_pipeline_config("en"))
    multi_pipeline = ClaimPipeline(_pipeline_config("hi"))

    predictions: Dict[str, List[Dict[str, Any]]] = {
        "en": [],
        "multi": [],
        "all": [],
    }

    for idx, item in enumerate(rows, start=1):
        lang_bucket = item["lang_bucket"].lower()
        language_code = item["language"]
        pipeline = en_pipeline if lang_bucket == "en" else multi_pipeline

        started = datetime.now(timezone.utc)
        err = None
        verdict = "neutral"
        confidence = 0.0
        evidence_count = 0
        details = {}
        try:
            result = pipeline.analyze(claim=item["claim"], language=language_code)
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
            **item,
            "predicted": verdict,
            "correct": verdict == item["expected"],
            "confidence": confidence,
            "evidence_count": evidence_count,
            "error": err,
            "details": details,
            "started_utc": started.isoformat(),
            "ended_utc": ended.isoformat(),
            "duration_sec": (ended - started).total_seconds(),
            "index": idx,
        }

        predictions["all"].append(row)
        predictions[lang_bucket].append(row)
        print(
            f"[{idx}/{len(rows)}] {item['lang_bucket']} {item['id']} "
            f"expected={item['expected']} predicted={verdict} ok={row['correct']} ev={evidence_count}"
        )

    for key in ("en", "multi", "all"):
        pred_rows = predictions[key]
        metrics = _build_metrics(pred_rows)
        metrics["split"] = key
        metrics["generated_at_utc"] = datetime.now(timezone.utc).isoformat()

        (out_dir / f"predictions_{key}.json").write_text(
            json.dumps(pred_rows, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        (out_dir / f"metrics_{key}.json").write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--html",
        default=r"C:\Users\chint\Downloads\rfcs_benchmark_claims.html",
        help="Path to benchmark HTML file",
    )
    parser.add_argument(
        "--out-dir",
        default="tests/benchmarks/rfcs_benchmark_run",
        help="Output directory for metrics/predictions",
    )
    parser.add_argument(
        "--split",
        choices=["all", "en", "multi"],
        default="all",
        help="Run only EN or MULTI subset, or all",
    )
    args = parser.parse_args()

    load_dotenv(override=True)

    html_path = Path(args.html)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    html_text = html_path.read_text(encoding="utf-8")
    raw_rows = _extract_data_array(html_text)
    rows = _normalize_rows(raw_rows)
    if not rows:
        raise RuntimeError("No valid benchmark rows parsed from HTML.")

    (out_dir / "benchmark_cases.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    en_rows = [r for r in rows if r["lang_bucket"] == "EN"]
    multi_rows = [r for r in rows if r["lang_bucket"] == "MULTI"]
    (out_dir / "benchmark_cases_en.json").write_text(
        json.dumps(en_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "benchmark_cases_multi.json").write_text(
        json.dumps(multi_rows, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    if args.split == "en":
        run_rows = en_rows
    elif args.split == "multi":
        run_rows = multi_rows
    else:
        run_rows = rows

    _run(run_rows, out_dir)

    summary = {
        "split": args.split,
        "cases": len(run_rows),
        "en_cases": sum(1 for r in rows if r["lang_bucket"] == "EN"),
        "multi_cases": sum(1 for r in rows if r["lang_bucket"] == "MULTI"),
        "outputs": {
            "metrics_en": str((out_dir / "metrics_en.json").resolve()),
            "metrics_multi": str((out_dir / "metrics_multi.json").resolve()),
            "metrics_all": str((out_dir / "metrics_all.json").resolve()),
            "predictions_en": str((out_dir / "predictions_en.json").resolve()),
            "predictions_multi": str((out_dir / "predictions_multi.json").resolve()),
            "predictions_all": str((out_dir / "predictions_all.json").resolve()),
        },
    }
    (out_dir / "run_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
