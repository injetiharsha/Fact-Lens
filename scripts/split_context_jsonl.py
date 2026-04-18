"""Split mixed context JSONL into train/val/test with stratification."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List

from sklearn.model_selection import train_test_split


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _stratify_key(r: Dict, use_lang: bool) -> str:
    lbl = str(r.get("label", "UNK"))
    if use_lang:
        return f"{lbl}||{str(r.get('lang', 'unk'))}"
    return lbl


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", required=True, type=str, help="Path to mixed JSONL")
    p.add_argument("--output-dir", required=True, type=str, help="Output folder")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--stratify-by-lang", action="store_true")
    args = p.parse_args()

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train/val/test ratios must sum to 1.0")

    in_path = Path(args.input)
    rows = _read_jsonl(in_path)
    if not rows:
        raise ValueError(f"Input file empty: {in_path}")

    random.Random(args.seed).shuffle(rows)
    keys = [_stratify_key(r, use_lang=args.stratify_by_lang) for r in rows]

    train_rows, temp_rows = train_test_split(
        rows,
        test_size=(1.0 - args.train_ratio),
        random_state=args.seed,
        stratify=keys,
    )
    temp_keys = [_stratify_key(r, use_lang=args.stratify_by_lang) for r in temp_rows]
    val_share_of_temp = args.val_ratio / (args.val_ratio + args.test_ratio)
    val_rows, test_rows = train_test_split(
        temp_rows,
        test_size=(1.0 - val_share_of_temp),
        random_state=args.seed,
        stratify=temp_keys,
    )

    out_dir = Path(args.output_dir)
    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "val.jsonl", val_rows)
    _write_jsonl(out_dir / "test.jsonl", test_rows)

    label_counts = dict(sorted(Counter([str(r.get("label", "UNK")) for r in rows]).items()))
    lang_counts = dict(sorted(Counter([str(r.get("lang", "unk")) for r in rows]).items()))
    summary = {
        "input": str(in_path),
        "output_dir": str(out_dir),
        "n_total": len(rows),
        "n_train": len(train_rows),
        "n_val": len(val_rows),
        "n_test": len(test_rows),
        "label_counts_total": label_counts,
        "lang_counts_total": lang_counts,
        "stratify_by_lang": bool(args.stratify_by_lang),
        "seed": int(args.seed),
    }
    with (out_dir / "split_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
