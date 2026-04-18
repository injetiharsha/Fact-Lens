"""Prepare EN stance dataset (support/refute/neutral) from FEVER JSONL."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List

from sklearn.model_selection import train_test_split


LABEL_MAP = {
    "SUPPORTS": "support",
    "REFUTES": "refute",
    "NOT ENOUGH INFO": "neutral",
}


def _read_jsonl(path: Path) -> Iterable[Dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _extract_evidence_text(row: Dict) -> str:
    evidence = row.get("evidence") or []
    if not evidence:
        return ""
    first_set = evidence[0]
    if not first_set:
        return ""
    texts: List[str] = []
    # Variant A: first_set is a single triplet [page, sent_id, text]
    if isinstance(first_set, list) and len(first_set) >= 3 and not isinstance(first_set[0], list):
        txt = str(first_set[2]).strip()
        if txt:
            texts.append(txt)
    # Variant B: first_set is a list of triplets
    else:
        for item in first_set:
            if isinstance(item, list) and len(item) >= 3:
                txt = str(item[2]).strip()
                if txt:
                    texts.append(txt)
    return " ".join(texts).strip()


def _convert_row(row: Dict) -> Dict | None:
    raw_label = str(row.get("label", "")).strip()
    label = LABEL_MAP.get(raw_label)
    if not label:
        return None
    claim = str(row.get("claim", "")).strip()
    evidence_text = _extract_evidence_text(row)
    if not claim or not evidence_text:
        return None
    return {
        "claim": claim,
        "evidence": evidence_text,
        "label": label,
        "source": "fever_gold_evidence",
    }


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--src-dir", type=str, default="data/raw/fever_gold_evidence_dedup")
    p.add_argument("--out-dir", type=str, default="data/processed/stance/en_fever")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    src = Path(args.src_dir)
    out = Path(args.out_dir)

    total_ratio = args.train_ratio + args.val_ratio + args.test_ratio
    if abs(total_ratio - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    # Load all FEVER splits together, then global-dedup and stratify-split.
    all_rows: List[Dict] = []
    raw_rows = 0
    kept_rows = 0
    for src_split in ("train", "valid", "test"):
        src_file = src / f"{src_split}.jsonl"
        for row in _read_jsonl(src_file):
            raw_rows += 1
            c = _convert_row(row)
            if c is not None:
                all_rows.append(c)
                kept_rows += 1

    seen = set()
    dedup_rows: List[Dict] = []
    for r in all_rows:
        key = (r["claim"].strip(), r["evidence"].strip(), r["label"])
        if key in seen:
            continue
        seen.add(key)
        dedup_rows.append(r)

    labels = [r["label"] for r in dedup_rows]
    train_rows, temp_rows = train_test_split(
        dedup_rows,
        test_size=(1.0 - args.train_ratio),
        random_state=args.seed,
        stratify=labels,
    )
    temp_labels = [r["label"] for r in temp_rows]
    val_share_of_temp = args.val_ratio / (args.val_ratio + args.test_ratio)
    val_rows, test_rows = train_test_split(
        temp_rows,
        train_size=val_share_of_temp,
        random_state=args.seed,
        stratify=temp_labels,
    )

    _write_jsonl(out / "train.jsonl", train_rows)
    _write_jsonl(out / "val.jsonl", val_rows)
    _write_jsonl(out / "test.jsonl", test_rows)

    summary = {
        "src_dir": str(src),
        "out_dir": str(out),
        "raw_input_rows": raw_rows,
        "converted_rows": kept_rows,
        "global_unique_rows": len(dedup_rows),
        "removed_duplicates": kept_rows - len(dedup_rows),
        "split_ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "splits": {},
    }

    split_rows = {"train": train_rows, "valid": val_rows, "test": test_rows}
    for split_name, rows in split_rows.items():
        c = Counter([r["label"] for r in rows])
        summary["splits"][split_name] = {"rows": len(rows), "label_counts": dict(sorted(c.items()))}

    with (out / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
