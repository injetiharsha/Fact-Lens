"""Run a 15-claim mini benchmark across the claim pipeline.

Outputs are written as Python files (no JSON/YAML):
- tests/benchmarks/mini_benchmark_15_results.py
"""

from __future__ import annotations

import argparse
import os
import pprint
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.routes.claim import _pipeline_config
from pipeline.claim_pipeline import ClaimPipeline

LABELS = ["support", "refute", "neutral"]

# 15 claims, balanced across expected verdicts.
BENCHMARK_15: List[Dict[str, str]] = [
    # support (5)
    {"id": "S1", "claim": "The Earth revolves around the Sun.", "expected": "support"},
    {"id": "S2", "claim": "At sea level, pure water boils at one hundred degrees Celsius.", "expected": "support"},
    {"id": "S3", "claim": "The capital city of India is New Delhi.", "expected": "support"},
    {"id": "S4", "claim": "An adult human typically has two hundred and six bones.", "expected": "support"},
    {"id": "S5", "claim": "The Great Wall is located in China.", "expected": "support"},
    # refute (5)
    {"id": "R1", "claim": "The Sun revolves around the Earth once every day.", "expected": "refute"},
    {"id": "R2", "claim": "Humans use only ten percent of their brain in daily life.", "expected": "refute"},
    {"id": "R3", "claim": "The Great Wall of China is clearly visible from the Moon.", "expected": "refute"},
    {"id": "R4", "claim": "COVID-19 vaccines cause infertility in all women.", "expected": "refute"},
    {"id": "R5", "claim": "NASA officially confirmed active microbial life on Mars in 2025.", "expected": "refute"},
    # neutral (5)
    {"id": "N1", "claim": "This restaurant serves the best biryani in Hyderabad.", "expected": "neutral"},
    {"id": "N2", "claim": "My neighbor's dog understands three different human languages.", "expected": "neutral"},
    {"id": "N3", "claim": "Most residents in my city prefer tea over coffee.", "expected": "neutral"},
    {"id": "N4", "claim": "The latest smartphone model is better than every previous model.", "expected": "neutral"},
    {"id": "N5", "claim": "A secret project currently controls global weather patterns.", "expected": "neutral"},
]


def _normalize_label(label: str) -> str:
    value = (label or "").strip().lower()
    if value in {"support", "supported", "true"}:
        return "support"
    if value in {"refute", "refuted", "false"}:
        return "refute"
    return "neutral"


def _ovr_error_stats(cm: List[List[int]], labels: List[str]) -> Dict[str, Dict[str, float]]:
    total = sum(sum(r) for r in cm)
    out: Dict[str, Dict[str, float]] = {}
    for i, label in enumerate(labels):
        tp = cm[i][i]
        fp = sum(cm[r][i] for r in range(len(labels)) if r != i)
        fn = sum(cm[i][c] for c in range(len(labels)) if c != i)
        tn = total - tp - fp - fn
        fpr = fp / (fp + tn) if (fp + tn) else 0.0
        fnr = fn / (fn + tp) if (fn + tp) else 0.0
        out[label] = {
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
            "tn": int(tn),
            "fpr": float(fpr),
            "fnr": float(fnr),
        }
    return out


def run(language: str, llm_mode: str, out_file: Path) -> Dict:
    if llm_mode not in {"env", "on", "off"}:
        raise ValueError("llm_mode must be one of: env, on, off")

    if llm_mode == "on":
        os.environ["ENABLE_LLM_VERIFIER"] = "1"
    elif llm_mode == "off":
        os.environ["ENABLE_LLM_VERIFIER"] = "0"

    pipeline = ClaimPipeline(_pipeline_config(language))

    rows = []
    y_true: List[str] = []
    y_pred: List[str] = []

    for item in BENCHMARK_15:
        claim = item["claim"]
        expected = item["expected"]
        result = pipeline.analyze(claim=claim, language=language)
        predicted = _normalize_label(result.verdict)
        details = result.details or {}
        rows.append(
            {
                "id": item["id"],
                "claim": claim,
                "expected": expected,
                "predicted": predicted,
                "correct": predicted == expected,
                "confidence": float(result.confidence),
                "evidence_count": len(result.evidence),
                "context": details.get("context"),
                "checkability": details.get("checkability"),
                "llm_verifier": details.get("llm_verifier"),
                "top_sources": [e.get("source") for e in result.evidence[:3]],
            }
        )
        y_true.append(expected)
        y_pred.append(predicted)

    acc = float(accuracy_score(y_true, y_pred))
    p, r, f1, support = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average=None, zero_division=0
    )
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="macro", zero_division=0
    )
    p_weighted, r_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=LABELS, average="weighted", zero_division=0
    )
    cm_arr = confusion_matrix(y_true, y_pred, labels=LABELS)
    cm = cm_arr.tolist()
    neutral_rate = float(sum(1 for x in y_pred if x == "neutral") / len(y_pred))

    per_class = {}
    for i, label in enumerate(LABELS):
        per_class[label] = {
            "precision": float(p[i]),
            "recall": float(r[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }

    ovr = _ovr_error_stats(cm, LABELS)

    summary = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "language": language,
        "llm_mode": llm_mode,
        "num_claims": len(BENCHMARK_15),
        "accuracy": acc,
        "neutral_rate": neutral_rate,
        "macro": {
            "precision": float(p_macro),
            "recall": float(r_macro),
            "f1": float(f1_macro),
        },
        "weighted": {
            "precision": float(p_weighted),
            "recall": float(r_weighted),
            "f1": float(f1_weighted),
        },
        "per_class": per_class,
        "confusion_matrix": {
            "labels": LABELS,
            "matrix": cm,
        },
        "error_stats_one_vs_rest": ovr,
        "rows": rows,
    }

    out_file.parent.mkdir(parents=True, exist_ok=True)
    payload = (
        "# Auto-generated mini benchmark report (Python only)\n\n"
        f"SUMMARY = {pprint.pformat(summary, width=120, sort_dicts=False)}\n"
    )
    out_file.write_text(payload, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--language", type=str, default="en")
    parser.add_argument("--llm-mode", type=str, default="env", choices=["env", "on", "off"])
    parser.add_argument(
        "--out",
        type=str,
        default="tests/benchmarks/mini_benchmark_15_results.py",
    )
    args = parser.parse_args()

    summary = run(language=args.language, llm_mode=args.llm_mode, out_file=Path(args.out))
    print("Mini benchmark complete")
    print(f"Output: {args.out}")
    print(
        {
            "accuracy": summary["accuracy"],
            "neutral_rate": summary["neutral_rate"],
            "macro_f1": summary["macro"]["f1"],
            "weighted_f1": summary["weighted"]["f1"],
        }
    )


if __name__ == "__main__":
    main()
