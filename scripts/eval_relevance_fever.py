"""Evaluate EN relevance checkpoint on FEVER-style claim-evidence pairs."""

from __future__ import annotations

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List, Tuple

from datasets import load_dataset
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.core.relevance import RelevanceScorer


def _extract_pairs(ds_split, max_positive: int, seed: int) -> List[Tuple[str, str]]:
    rng = random.Random(seed)
    positives: List[Tuple[str, str]] = []
    idxs = list(range(len(ds_split)))
    rng.shuffle(idxs)

    for i in idxs:
        row = ds_split[int(i)]
        ev = row.get("evidence")
        if not ev:
            continue
        first_set = ev[0]
        if not first_set or len(first_set) < 3:
            continue
        claim = str(row.get("claim", "")).strip()
        evidence_text = str(first_set[2]).strip()
        if not claim or not evidence_text:
            continue
        positives.append((claim, evidence_text))
        if len(positives) >= max_positive:
            break
    return positives


def _build_labeled_pairs(positives: List[Tuple[str, str]], seed: int) -> Tuple[List[int], List[Tuple[str, str]]]:
    rng = random.Random(seed)
    n = len(positives)
    claims = [c for c, _ in positives]
    evidences = [e for _, e in positives]

    # Positive pairs
    y_true: List[int] = [1] * n
    pairs: List[Tuple[str, str]] = list(positives)

    # Negative pairs by derangement-like shuffle
    idx = list(range(n))
    rng.shuffle(idx)
    for i in range(n):
        if idx[i] == i:
            j = (i + 1) % n
            idx[i], idx[j] = idx[j], idx[i]
    negatives = [(claims[i], evidences[idx[i]]) for i in range(n)]

    y_true.extend([0] * n)
    pairs.extend(negatives)
    return y_true, pairs


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="copenlu/fever_gold_evidence")
    p.add_argument("--split", type=str, default="validation")
    p.add_argument("--max-positive", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--threshold", type=float, default=0.05)
    p.add_argument("--model-path", type=str, default="checkpoints/relevance/en/v9_run1")
    p.add_argument(
        "--output-json",
        type=str,
        default="training/records/relevance_en_fever_eval_latest.json",
    )
    args = p.parse_args()

    ds = load_dataset(args.dataset, split=args.split)
    positives = _extract_pairs(ds, max_positive=args.max_positive, seed=args.seed)
    if not positives:
        raise RuntimeError("No FEVER positives extracted from dataset.")

    y_true, pairs = _build_labeled_pairs(positives, seed=args.seed)
    scorer = RelevanceScorer(model_path=args.model_path, enable_two_stage=False)

    y_score: List[float] = []
    for claim, evidence in pairs:
        y_score.append(float(scorer.score(claim, evidence)))

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)
    y_pred = [1 if s >= args.threshold else 0 for s in y_score]
    p1, r1, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary", zero_division=0)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "split": args.split,
        "max_positive": len(positives),
        "total_pairs": len(pairs),
        "model_path": args.model_path,
        "model_mode": "single_stage_en",
        "threshold": args.threshold,
        "metrics": {
            "auc": float(auc),
            "average_precision": float(ap),
            "precision_at_threshold": float(p1),
            "recall_at_threshold": float(r1),
            "f1_at_threshold": float(f1),
            "score_min": float(min(y_score)),
            "score_max": float(max(y_score)),
            "score_mean": float(sum(y_score) / len(y_score)),
        },
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    runs = out.parent / "relevance_en_fever_eval_runs.jsonl"
    with runs.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
