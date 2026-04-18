"""Prepare IndicXNLI stance data with strict balancing and deduplication.

Output schema: claim, evidence, label, lang, source
Label set: support, refute, neutral
Balancing:
  - equal rows per language
  - equal labels within each language
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from datasets import DatasetDict, load_dataset
from sklearn.model_selection import train_test_split


LABEL_INT_MAP = {
    0: "support",   # entailment
    1: "neutral",   # neutral
    2: "refute",    # contradiction
}

LABEL_STR_MAP = {
    "entailment": "support",
    "supports": "support",
    "support": "support",
    "neutral": "neutral",
    "contradiction": "refute",
    "refutes": "refute",
    "refute": "refute",
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _map_label(v) -> str | None:
    if isinstance(v, int):
        return LABEL_INT_MAP.get(v)
    s = str(v).strip().lower()
    return LABEL_STR_MAP.get(s)


def _pick(row: Dict, keys: List[str]) -> str:
    for k in keys:
        if k in row and row[k] is not None:
            t = str(row[k]).strip()
            if t:
                return t
    return ""


def _rows_from_dataset(ds: DatasetDict) -> Iterable[Dict]:
    for split_name in ds.keys():
        split = ds[split_name]
        for row in split:
            r = dict(row)
            r["_split"] = split_name
            yield r


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _snapshot_candidates(dataset_id: str) -> List[Path]:
    parts = dataset_id.split("/")
    if len(parts) != 2:
        return []
    owner, name = parts
    root = Path.home() / ".cache" / "huggingface" / "hub" / f"datasets--{owner}--{name}" / "snapshots"
    if not root.exists():
        return []
    return [p for p in root.iterdir() if p.is_dir()]


def _counts(rows: List[Dict], key: str) -> Dict[str, int]:
    return dict(sorted(Counter([str(r[key]) for r in rows]).items()))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", type=str, default="Divyanshu/indicxnli")
    p.add_argument("--revision", type=str, default="refs/convert/parquet")
    p.add_argument("--out-dir", type=str, default="data/processed/stance/indic_xnli")
    p.add_argument("--langs", nargs="+", default=["hi", "ta", "te", "kn", "ml"])
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--drop-same-claim-evidence", action="store_true")
    args = p.parse_args()

    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    rng = random.Random(args.seed)
    target_langs = {x.strip().lower() for x in args.langs}

    # Preferred path: local HF snapshot parquet files with explicit language folders.
    all_rows: List[Dict] = []
    load_mode = "cache_snapshot_parquet"
    per_lang_loaded = []
    snapshots = _snapshot_candidates(args.dataset)
    snapshots_sorted = sorted(snapshots, key=lambda p: p.stat().st_mtime, reverse=True)
    if snapshots_sorted:
        snap = snapshots_sorted[0]
        try:
            for lg in sorted(target_langs):
                data_files = {}
                for sp in ("train", "validation", "test"):
                    sp_dir = snap / lg / sp
                    files = sorted([str(x) for x in sp_dir.glob("*.parquet")]) if sp_dir.exists() else []
                    if files:
                        data_files[sp] = files
                if not data_files:
                    continue
                ds_lg = load_dataset("parquet", data_files=data_files)
                c = 0
                for r in _rows_from_dataset(ds_lg):
                    r["lang"] = lg
                    all_rows.append(r)
                    c += 1
                per_lang_loaded.append({"lang": lg, "rows": c})
        except Exception:
            all_rows = []

    if not all_rows:
        # Fallback path: try loading the dataset id directly.
        load_mode = "fallback_full_dataset"
        ds = load_dataset(args.dataset, revision=args.revision)
        all_rows = list(_rows_from_dataset(ds))

    parsed_rows: List[Dict] = []
    stats = {
        "raw_rows": len(all_rows),
        "dropped_missing": 0,
        "dropped_bad_label": 0,
        "dropped_lang": 0,
        "dropped_same_claim_evidence": 0,
        "dropped_duplicates": 0,
    }

    seen = set()
    for row in all_rows:
        lang = _pick(row, ["lang", "language", "locale"]).lower()
        if lang not in target_langs:
            stats["dropped_lang"] += 1
            continue

        label = _map_label(row.get("label", row.get("labels")))
        if label is None:
            stats["dropped_bad_label"] += 1
            continue

        claim = _pick(row, ["claim", "premise", "sentence1", "text_a"])
        evidence = _pick(row, ["evidence", "hypothesis", "sentence2", "text_b"])
        if not claim or not evidence:
            stats["dropped_missing"] += 1
            continue
        if args.drop_same_claim_evidence and _norm(claim) == _norm(evidence):
            stats["dropped_same_claim_evidence"] += 1
            continue

        item = {
            "claim": claim,
            "evidence": evidence,
            "label": label,
            "lang": lang,
            "source": args.dataset,
        }
        key = (_norm(claim), _norm(evidence), label, lang)
        if key in seen:
            stats["dropped_duplicates"] += 1
            continue
        seen.add(key)
        parsed_rows.append(item)

    # Buckets by (lang, label)
    by_pair: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
    for r in parsed_rows:
        by_pair[(r["lang"], r["label"])].append(r)

    required_labels = ("support", "refute", "neutral")
    missing_pairs = []
    for lg in sorted(target_langs):
        for lb in required_labels:
            if len(by_pair[(lg, lb)]) == 0:
                missing_pairs.append((lg, lb))
    if missing_pairs:
        raise RuntimeError(f"Missing lang/label buckets: {missing_pairs}")

    # Equal language + equal label => same K for every (lang,label) bucket
    per_bucket_target = min(len(by_pair[(lg, lb)]) for lg in target_langs for lb in required_labels)

    balanced_rows: List[Dict] = []
    for lg in sorted(target_langs):
        for lb in required_labels:
            rows = by_pair[(lg, lb)][:]
            rng.shuffle(rows)
            balanced_rows.extend(rows[:per_bucket_target])
    rng.shuffle(balanced_rows)

    strat_keys = [f"{r['lang']}|{r['label']}" for r in balanced_rows]
    train_rows, temp_rows = train_test_split(
        balanced_rows,
        test_size=(1.0 - args.train_ratio),
        random_state=args.seed,
        stratify=strat_keys,
    )
    temp_keys = [f"{r['lang']}|{r['label']}" for r in temp_rows]
    val_share = args.val_ratio / (args.val_ratio + args.test_ratio)
    val_rows, test_rows = train_test_split(
        temp_rows,
        train_size=val_share,
        random_state=args.seed,
        stratify=temp_keys,
    )

    out_dir = Path(args.out_dir)
    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "val.jsonl", val_rows)
    _write_jsonl(out_dir / "test.jsonl", test_rows)

    summary = {
        "dataset": args.dataset,
        "revision": args.revision,
        "out_dir": str(out_dir),
        "target_langs": sorted(target_langs),
        "load_mode": load_mode,
        "per_lang_loaded": per_lang_loaded,
        "cleaning": stats,
        "parsed_rows": len(parsed_rows),
        "per_bucket_target": per_bucket_target,
        "balanced_total_rows": len(balanced_rows),
        "balanced_label_counts": _counts(balanced_rows, "label"),
        "balanced_lang_counts": _counts(balanced_rows, "lang"),
        "split_ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "splits": {
            "train": {
                "rows": len(train_rows),
                "label_counts": _counts(train_rows, "label"),
                "lang_counts": _counts(train_rows, "lang"),
            },
            "val": {
                "rows": len(val_rows),
                "label_counts": _counts(val_rows, "label"),
                "lang_counts": _counts(val_rows, "lang"),
            },
            "test": {
                "rows": len(test_rows),
                "label_counts": _counts(test_rows, "label"),
                "lang_counts": _counts(test_rows, "lang"),
            },
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
