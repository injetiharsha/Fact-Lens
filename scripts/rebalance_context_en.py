"""Rebalance EN context dataset to a target non-equal label distribution."""

from __future__ import annotations

import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List

from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)

TARGET_COUNTS = {
    "GENERAL_FACTUAL": 2240,
    "TECHNOLOGY": 1680,
    "POLITICS_GOVERNMENT": 1540,
    "ECONOMICS_BUSINESS": 1400,
    "SPORTS": 1260,
    "SCIENCE": 1120,
    "HISTORY": 980,
    "LAW_CRIME": 980,
    "HEALTH": 840,
    "GEOGRAPHY": 700,
    "SPACE_ASTRONOMY": 560,
    "ENTERTAINMENT": 280,
    "SOCIETY_CULTURE": 210,
    "ENVIRONMENT_CLIMATE": 210,
}


@dataclass
class Row:
    text: str
    label: str
    source: str
    language: str


def _load_jsonl(path: Path) -> List[Row]:
    rows: List[Row] = []
    with open(path, "r", encoding="utf-8") as f:
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
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(asdict(r), ensure_ascii=False) + "\n")


def main() -> None:
    src_dir = Path("data/processed/context/en")
    src_rows = (
        _load_jsonl(src_dir / "train.jsonl")
        + _load_jsonl(src_dir / "val.jsonl")
        + _load_jsonl(src_dir / "test.jsonl")
    )

    by_label: Dict[str, List[Row]] = defaultdict(list)
    for r in src_rows:
        by_label[r.label].append(r)

    sampled: List[Row] = []
    build_meta = {
        "label_sampling": {},
        "source_total": len(src_rows),
    }

    for label, target_n in TARGET_COUNTS.items():
        pool = by_label.get(label, [])
        if not pool:
            raise ValueError(f"No source examples for label: {label}")

        # If we have enough, sample without replacement; otherwise with replacement.
        if len(pool) >= target_n:
            chosen = random.sample(pool, target_n)
            mode = "without_replacement"
        else:
            chosen = [random.choice(pool) for _ in range(target_n)]
            mode = "with_replacement"

        sampled.extend(chosen)
        build_meta["label_sampling"][label] = {
            "source_count": len(pool),
            "target_count": target_n,
            "mode": mode,
        }

    random.shuffle(sampled)
    labels = [r.label for r in sampled]

    train_rows, temp_rows = train_test_split(
        sampled, test_size=0.2, random_state=SEED, stratify=labels
    )
    temp_labels = [r.label for r in temp_rows]
    val_rows, test_rows = train_test_split(
        temp_rows, test_size=0.5, random_state=SEED, stratify=temp_labels
    )

    out_dir = Path("data/processed/context/en_dist14k")
    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "val.jsonl", val_rows)
    _write_jsonl(out_dir / "test.jsonl", test_rows)

    def counts(rows: List[Row]) -> Dict[str, int]:
        return dict(sorted(Counter([r.label for r in rows]).items()))

    summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "target_counts": TARGET_COUNTS,
        "split_sizes": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
            "total": len(sampled),
        },
        "split_label_counts": {
            "train": counts(train_rows),
            "val": counts(val_rows),
            "test": counts(test_rows),
            "all": counts(sampled),
        },
        "build_meta": build_meta,
        "output_dir": str(out_dir),
    }

    records_dir = Path("training/records")
    records_dir.mkdir(parents=True, exist_ok=True)
    with open(records_dir / "context_en_rebalance_runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(summary, ensure_ascii=False) + "\n")
    with open(records_dir / "context_en_rebalance_latest.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary["split_sizes"], indent=2))
    print("all_label_counts:", summary["split_label_counts"]["all"])
    print("output_dir:", out_dir)


if __name__ == "__main__":
    main()

