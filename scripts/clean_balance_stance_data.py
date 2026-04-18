"""Clean, deduplicate, normalize labels, and rebalance stance data."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from sklearn.model_selection import train_test_split


LABEL_STR_MAP = {
    "support": "support",
    "supports": "support",
    "entailment": "support",
    "refute": "refute",
    "refutes": "refute",
    "contradiction": "refute",
    "neutral": "neutral",
    "not enough info": "neutral",
    "nei": "neutral",
}

LABEL_INT_MAP = {
    0: "support",
    1: "neutral",
    2: "refute",
}


def _norm_text(s: str) -> str:
    s = s.strip().lower()
    s = re.sub(r"\s+", " ", s)
    return s


def _map_label(v) -> str | None:
    if isinstance(v, int):
        return LABEL_INT_MAP.get(v)
    s = str(v).strip().lower()
    return LABEL_STR_MAP.get(s)


def _read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _canonical_row(row: Dict) -> Dict | None:
    claim = str(row.get("claim", "")).strip()
    evidence = str(row.get("evidence", "")).strip()
    label = _map_label(row.get("label"))
    if not claim or not evidence or label is None:
        return None
    return {"claim": claim, "evidence": evidence, "label": label}


def _dedup_and_filter(rows: List[Dict], drop_same_pair: bool) -> Tuple[List[Dict], Dict[str, int]]:
    stats = {
        "input_rows": len(rows),
        "dropped_invalid": 0,
        "dropped_same_claim_evidence": 0,
        "dropped_duplicates": 0,
    }
    out: List[Dict] = []
    seen = set()
    for row in rows:
        c = _canonical_row(row)
        if c is None:
            stats["dropped_invalid"] += 1
            continue
        cn = _norm_text(c["claim"])
        en = _norm_text(c["evidence"])
        if drop_same_pair and cn == en:
            stats["dropped_same_claim_evidence"] += 1
            continue
        key = (cn, en, c["label"])
        if key in seen:
            stats["dropped_duplicates"] += 1
            continue
        seen.add(key)
        out.append(c)
    return out, stats


def _balance_equal(rows: List[Dict], seed: int) -> Tuple[List[Dict], Dict[str, int], int]:
    by_label: Dict[str, List[Dict]] = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)
    counts = {k: len(v) for k, v in by_label.items()}
    min_count = min(counts.values())
    rng = random.Random(seed)
    balanced: List[Dict] = []
    for lbl in sorted(by_label.keys()):
        items = by_label[lbl][:]
        rng.shuffle(items)
        balanced.extend(items[:min_count])
    rng.shuffle(balanced)
    return balanced, counts, min_count


def _split(rows: List[Dict], train_ratio: float, val_ratio: float, test_ratio: float, seed: int):
    labels = [r["label"] for r in rows]
    train_rows, temp_rows = train_test_split(
        rows, test_size=(1.0 - train_ratio), random_state=seed, stratify=labels
    )
    temp_labels = [r["label"] for r in temp_rows]
    val_share = val_ratio / (val_ratio + test_ratio)
    val_rows, test_rows = train_test_split(
        temp_rows, train_size=val_share, random_state=seed, stratify=temp_labels
    )
    return train_rows, val_rows, test_rows


def _counts(rows: List[Dict]) -> Dict[str, int]:
    return dict(sorted(Counter([r["label"] for r in rows]).items()))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=str, required=True, help="Directory with train/val/test jsonl")
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drop-same-claim-evidence", action="store_true")
    args = p.parse_args()

    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    in_files = [in_dir / "train.jsonl", in_dir / "val.jsonl", in_dir / "test.jsonl"]
    missing = [str(p) for p in in_files if not p.exists()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))

    raw_rows: List[Dict] = []
    for pth in in_files:
        raw_rows.extend(list(_read_jsonl(pth)))

    cleaned_rows, clean_stats = _dedup_and_filter(raw_rows, args.drop_same_claim_evidence)
    balanced_rows, pre_balance_counts, min_count = _balance_equal(cleaned_rows, args.seed)
    train_rows, val_rows, test_rows = _split(
        balanced_rows, args.train_ratio, args.val_ratio, args.test_ratio, args.seed
    )

    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "val.jsonl", val_rows)
    _write_jsonl(out_dir / "test.jsonl", test_rows)

    summary = {
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "cleaning": clean_stats,
        "before_balance_label_counts": pre_balance_counts,
        "balance_target_per_label": min_count,
        "after_balance_total_rows": len(balanced_rows),
        "split_ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "splits": {
            "train": {"rows": len(train_rows), "label_counts": _counts(train_rows)},
            "val": {"rows": len(val_rows), "label_counts": _counts(val_rows)},
            "test": {"rows": len(test_rows), "label_counts": _counts(test_rows)},
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

