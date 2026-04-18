"""Final cleanup for stance splits: dedup + label consistency + re-split."""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

from sklearn.model_selection import train_test_split


ALLOWED = {"support", "refute", "neutral"}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


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
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _counts(rows: List[Dict], key: str) -> Dict[str, int]:
    return dict(sorted(Counter([str(r[key]) for r in rows]).items()))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input-dir", type=str, required=True)
    p.add_argument("--output-dir", type=str, required=True)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--balance-lang-label", action="store_true")
    args = p.parse_args()

    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    in_dir = Path(args.input_dir)
    out_dir = Path(args.output_dir)
    files = [in_dir / "train.jsonl", in_dir / "val.jsonl", in_dir / "test.jsonl"]
    missing = [str(x) for x in files if not x.exists()]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))

    raw: List[Dict] = []
    for f in files:
        raw.extend(_read_jsonl(f))

    stats = {
        "input_rows": len(raw),
        "dropped_bad_schema_or_label": 0,
        "dropped_exact_duplicates": 0,
        "conflicting_pairs": 0,
        "rows_removed_by_conflict_resolution": 0,
    }

    # 1) schema/label filter + exact dedup
    seen_exact = set()
    rows: List[Dict] = []
    for r in raw:
        claim = str(r.get("claim", "")).strip()
        evidence = str(r.get("evidence", "")).strip()
        label = str(r.get("label", "")).strip().lower()
        lang = str(r.get("lang", "")).strip().lower()
        if not claim or not evidence or label not in ALLOWED or not lang:
            stats["dropped_bad_schema_or_label"] += 1
            continue
        k_exact = (_norm(claim), _norm(evidence), label, lang)
        if k_exact in seen_exact:
            stats["dropped_exact_duplicates"] += 1
            continue
        seen_exact.add(k_exact)
        rows.append(
            {
                "claim": claim,
                "evidence": evidence,
                "label": label,
                "lang": lang,
                "source": r.get("source", "unknown"),
            }
        )

    # 2) enforce label consistency per (claim,evidence,lang)
    pair_groups: Dict[Tuple[str, str, str], List[Dict]] = defaultdict(list)
    for r in rows:
        key = (_norm(r["claim"]), _norm(r["evidence"]), r["lang"])
        pair_groups[key].append(r)

    consistent_rows: List[Dict] = []
    for _, group in pair_groups.items():
        label_counts = Counter([g["label"] for g in group])
        if len(label_counts) > 1:
            stats["conflicting_pairs"] += 1
            # keep majority label deterministically
            best_label = sorted(label_counts.items(), key=lambda x: (-x[1], x[0]))[0][0]
            keep = [g for g in group if g["label"] == best_label]
            stats["rows_removed_by_conflict_resolution"] += (len(group) - len(keep))
            # keep one unique row for this pair
            consistent_rows.append(keep[0])
        else:
            # one label only; keep one unique row
            consistent_rows.append(group[0])

    balanced_rows = consistent_rows
    balance_meta = None
    if args.balance_lang_label:
        rng = random.Random(args.seed)
        by_pair: Dict[Tuple[str, str], List[Dict]] = defaultdict(list)
        for r in consistent_rows:
            by_pair[(r["lang"], r["label"])].append(r)
        langs = sorted(set([r["lang"] for r in consistent_rows]))
        labels = ["support", "refute", "neutral"]
        min_count = min(len(by_pair[(lg, lb)]) for lg in langs for lb in labels)
        balanced_rows = []
        for lg in langs:
            for lb in labels:
                items = by_pair[(lg, lb)][:]
                rng.shuffle(items)
                balanced_rows.extend(items[:min_count])
        rng.shuffle(balanced_rows)
        balance_meta = {
            "enabled": True,
            "per_lang_label_target": min_count,
            "balanced_total_rows": len(balanced_rows),
            "balanced_label_counts": _counts(balanced_rows, "label"),
            "balanced_lang_counts": _counts(balanced_rows, "lang"),
        }

    # 3) re-split stratified by lang|label
    strat = [f"{r['lang']}|{r['label']}" for r in balanced_rows]
    train_rows, temp_rows = train_test_split(
        balanced_rows,
        test_size=(1.0 - args.train_ratio),
        random_state=args.seed,
        stratify=strat,
    )
    strat_temp = [f"{r['lang']}|{r['label']}" for r in temp_rows]
    val_share = args.val_ratio / (args.val_ratio + args.test_ratio)
    val_rows, test_rows = train_test_split(
        temp_rows,
        train_size=val_share,
        random_state=args.seed,
        stratify=strat_temp,
    )

    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "val.jsonl", val_rows)
    _write_jsonl(out_dir / "test.jsonl", test_rows)

    # 4) overlap checks
    def _set_key(rs: List[Dict]):
        return {
            (_norm(r["claim"]), _norm(r["evidence"]), r["label"], r["lang"])
            for r in rs
        }

    s_tr, s_va, s_te = _set_key(train_rows), _set_key(val_rows), _set_key(test_rows)

    summary = {
        "input_dir": str(in_dir),
        "output_dir": str(out_dir),
        "cleaning": stats,
        "final_total_rows_before_optional_balance": len(consistent_rows),
        "balance": balance_meta,
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
        "overlap_checks": {
            "train_val": len(s_tr & s_va),
            "train_test": len(s_tr & s_te),
            "val_test": len(s_va & s_te),
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
