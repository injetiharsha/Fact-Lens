"""Prepare EN VitaminC stance data with strict cleaning and balancing."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset
from sklearn.model_selection import train_test_split


def _map_label(v) -> str | None:
    if isinstance(v, int):
        # Most NLI-style mappings: 0 entailment, 1 neutral, 2 contradiction
        return {0: "support", 1: "neutral", 2: "refute"}.get(v)
    s = str(v).strip().lower()
    if s in {"entailment", "supports", "support"}:
        return "support"
    if s in {"contradiction", "refutes", "refute"}:
        return "refute"
    if s in {"neutral", "not enough info", "nei"}:
        return "neutral"
    return None


def _norm_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _pick_text(row: Dict, keys: List[str]) -> str:
    for k in keys:
        if k in row and row[k] is not None:
            t = str(row[k]).strip()
            if t:
                return t
    return ""


def _convert_split(ds_split, source_name: str) -> List[Dict]:
    rows: List[Dict] = []
    for row in ds_split:
        label = _map_label(row.get("label", row.get("labels")))
        if label is None:
            continue
        claim = _pick_text(
            row,
            [
                "claim",
                "premise",
                "hypothesis",
                "evidence",
                "text",
                "sentence",
                "context",
            ],
        )
        evidence = _pick_text(
            row,
            [
                "evidence",
                "text",
                "sentence",
                "context",
                "hypothesis",
                "premise",
                "claim",
            ],
        )
        if not claim or not evidence:
            continue
        rows.append(
            {
                "claim": claim.strip(),
                "evidence": evidence.strip(),
                "label": label,
                "source": source_name,
            }
        )
    return rows


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _dedup_clean(rows: List[Dict], drop_same_pair: bool) -> tuple[List[Dict], Dict[str, int]]:
    stats = {
        "input_rows": len(rows),
        "dropped_same_claim_evidence": 0,
        "dropped_duplicates": 0,
    }
    out: List[Dict] = []
    seen = set()
    for r in rows:
        cn = _norm_text(r["claim"])
        en = _norm_text(r["evidence"])
        if drop_same_pair and cn == en:
            stats["dropped_same_claim_evidence"] += 1
            continue
        k = (cn, en, r["label"])
        if k in seen:
            stats["dropped_duplicates"] += 1
            continue
        seen.add(k)
        out.append(r)
    return out, stats


def _balance_equal(rows: List[Dict], seed: int) -> tuple[List[Dict], Dict[str, int], int]:
    by_label: Dict[str, List[Dict]] = {"support": [], "refute": [], "neutral": []}
    for r in rows:
        by_label[r["label"]].append(r)
    counts = {k: len(v) for k, v in by_label.items()}
    min_count = min(counts.values())
    rng = random.Random(seed)
    balanced: List[Dict] = []
    for lbl in ("support", "refute", "neutral"):
        items = by_label[lbl][:]
        rng.shuffle(items)
        balanced.extend(items[:min_count])
    rng.shuffle(balanced)
    return balanced, counts, min_count


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="tals/vitaminc")
    p.add_argument("--out-dir", type=str, default="data/processed/stance/en_vitaminc")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drop-same-claim-evidence", action="store_true")
    args = p.parse_args()

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    out_dir = Path(args.out_dir)
    ds = load_dataset(args.dataset)
    rows_all: List[Dict] = []
    split_used: Dict[str, str] = {}
    for split_name in ("train", "validation", "dev", "test"):
        if split_name in ds:
            split_used[split_name] = split_name
            rows_all.extend(_convert_split(ds[split_name], args.dataset))
    if not rows_all:
        # fallback for custom dataset layouts
        for k in ds.keys():
            split_used[k] = k
            rows_all.extend(_convert_split(ds[k], args.dataset))

    cleaned_rows, clean_stats = _dedup_clean(rows_all, args.drop_same_claim_evidence)
    balanced_rows, before_balance_counts, per_label_target = _balance_equal(cleaned_rows, args.seed)

    labels = [r["label"] for r in balanced_rows]
    train_rows, temp_rows = train_test_split(
        balanced_rows,
        test_size=(1.0 - args.train_ratio),
        random_state=args.seed,
        stratify=labels,
    )
    temp_labels = [r["label"] for r in temp_rows]
    val_share = args.val_ratio / (args.val_ratio + args.test_ratio)
    val_rows, test_rows = train_test_split(
        temp_rows,
        train_size=val_share,
        random_state=args.seed,
        stratify=temp_labels,
    )

    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "val.jsonl", val_rows)
    _write_jsonl(out_dir / "test.jsonl", test_rows)

    summary = {
        "dataset": args.dataset,
        "out_dir": str(out_dir),
        "split_used": split_used,
        "cleaning": clean_stats,
        "before_balance_label_counts": before_balance_counts,
        "balance_target_per_label": per_label_target,
        "after_balance_total_rows": len(balanced_rows),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "train_label_counts": dict(sorted(Counter([r["label"] for r in train_rows]).items())),
        "val_label_counts": dict(sorted(Counter([r["label"] for r in val_rows]).items())),
        "test_label_counts": dict(sorted(Counter([r["label"] for r in test_rows]).items())),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
