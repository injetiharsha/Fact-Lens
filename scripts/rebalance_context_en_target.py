"""Rebalance EN context dataset to target total with fixed label percentages."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)

LABEL_PCT = {
    "GENERAL_FACTUAL": 0.16,
    "TECHNOLOGY": 0.12,
    "POLITICS_GOVERNMENT": 0.11,
    "ECONOMICS_BUSINESS": 0.10,
    "SPORTS": 0.09,
    "SCIENCE": 0.08,
    "HISTORY": 0.07,
    "LAW_CRIME": 0.07,
    "HEALTH": 0.06,
    "GEOGRAPHY": 0.05,
    "SPACE_ASTRONOMY": 0.04,
    "ENTERTAINMENT": 0.02,
    "SOCIETY_CULTURE": 0.015,
    "ENVIRONMENT_CLIMATE": 0.015,
}


@dataclass
class Row:
    text: str
    label: str
    source: str
    language: str


def _load_jsonl(path: Path) -> List[Row]:
    rows: List[Row] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            j = json.loads(line)
            rows.append(
                Row(
                    text=j["text"],
                    label=j["label"],
                    source=j.get("source", "unknown"),
                    language=j.get("language", "en"),
                )
            )
    return rows


def _write_jsonl(path: Path, rows: List[Row]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def _target_counts(total: int) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    running = 0
    labels = list(LABEL_PCT.keys())
    for label in labels[:-1]:
        n = int(round(total * LABEL_PCT[label]))
        counts[label] = n
        running += n
    counts[labels[-1]] = total - running
    return counts


def _counts(rows: List[Row]) -> Dict[str, int]:
    return dict(sorted(Counter([r.label for r in rows]).items()))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--total", type=int, default=10000, help="Total EN rows to build.")
    parser.add_argument("--source-dir", type=str, default="data/processed/context/en")
    parser.add_argument("--output-dir", type=str, default="data/processed/context/en_dist10k")
    args = parser.parse_args()

    src_dir = Path(args.source_dir)
    src_rows = (
        _load_jsonl(src_dir / "train.jsonl")
        + _load_jsonl(src_dir / "val.jsonl")
        + _load_jsonl(src_dir / "test.jsonl")
    )

    targets = _target_counts(args.total)
    by_label: Dict[str, List[Row]] = defaultdict(list)
    for row in src_rows:
        by_label[row.label].append(row)

    sampled: List[Row] = []
    sampling_meta: Dict[str, Dict[str, int | str]] = {}
    for label, need in targets.items():
        pool = by_label.get(label, [])
        if not pool:
            raise ValueError(f"No examples for label: {label}")
        if len(pool) >= need:
            chosen = random.sample(pool, need)
            mode = "without_replacement"
        else:
            chosen = [random.choice(pool) for _ in range(need)]
            mode = "with_replacement"
        sampled.extend(chosen)
        sampling_meta[label] = {"need": need, "source": len(pool), "mode": mode}

    random.shuffle(sampled)
    labels = [r.label for r in sampled]

    train_rows, temp_rows = train_test_split(
        sampled, test_size=0.2, random_state=SEED, stratify=labels
    )
    temp_labels = [r.label for r in temp_rows]
    val_rows, test_rows = train_test_split(
        temp_rows, test_size=0.5, random_state=SEED, stratify=temp_labels
    )

    out_dir = Path(args.output_dir)
    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "val.jsonl", val_rows)
    _write_jsonl(out_dir / "test.jsonl", test_rows)

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "source_dir": str(src_dir),
        "output_dir": str(out_dir),
        "target_total": args.total,
        "target_counts": targets,
        "sampling_meta": sampling_meta,
        "split_sizes": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
            "total": len(sampled),
        },
        "split_label_counts": {
            "train": _counts(train_rows),
            "val": _counts(val_rows),
            "test": _counts(test_rows),
            "all": _counts(sampled),
        },
    }

    records_dir = Path("training/records")
    records_dir.mkdir(parents=True, exist_ok=True)
    latest = records_dir / "context_en_rebalance_target_latest.json"
    runs = records_dir / "context_en_rebalance_target_runs.jsonl"
    with latest.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    with runs.open("a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")

    print(json.dumps(summary["split_sizes"], indent=2))
    print("all_label_counts:", summary["split_label_counts"]["all"])
    print("output_dir:", out_dir)


if __name__ == "__main__":
    main()
