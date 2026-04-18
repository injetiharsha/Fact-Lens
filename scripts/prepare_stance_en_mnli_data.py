"""Prepare EN stance warmup data from MNLI (entail/neutral/contradict -> support/neutral/refute)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from datasets import load_dataset


ID2STANCE = {
    0: "support",  # entailment
    1: "neutral",  # neutral
    2: "refute",   # contradiction
}


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _convert_split(ds_split, max_rows: int) -> List[Dict]:
    out: List[Dict] = []
    for row in ds_split:
        lbl = int(row.get("label", -1))
        if lbl not in ID2STANCE:
            continue
        out.append(
            {
                "claim": str(row["premise"]).strip(),
                "evidence": str(row["hypothesis"]).strip(),
                "label": ID2STANCE[lbl],
                "source": "mnli",
            }
        )
        if max_rows > 0 and len(out) >= max_rows:
            break
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--out-dir", type=str, default="data/processed/stance/en_mnli")
    p.add_argument("--max-train", type=int, default=120000)
    p.add_argument("--max-val", type=int, default=10000)
    p.add_argument("--max-test", type=int, default=10000)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    ds = load_dataset("multi_nli")

    train_rows = _convert_split(ds["train"], args.max_train)
    val_rows = _convert_split(ds["validation_matched"], args.max_val)
    test_rows = _convert_split(ds["validation_mismatched"], args.max_test)

    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "val.jsonl", val_rows)
    _write_jsonl(out_dir / "test.jsonl", test_rows)

    def counts(rows: List[Dict]) -> Dict[str, int]:
        return dict(sorted(Counter([r["label"] for r in rows]).items()))

    summary = {
        "out_dir": str(out_dir),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "train_label_counts": counts(train_rows),
        "val_label_counts": counts(val_rows),
        "test_label_counts": counts(test_rows),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
