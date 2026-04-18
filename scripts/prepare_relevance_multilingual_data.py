"""Prepare multilingual relevance data from FEVER translated files.

Builds binary relevance pairs:
- relevant (1): SUPPORTS / REFUTES
- not_relevant (0): sampled negatives from NOT ENOUGH INFO and random mismatches
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List

from sklearn.model_selection import train_test_split


POS_LABELS = {"SUPPORTS", "REFUTES"}
NEG_LABELS = {"NOT ENOUGH INFO"}


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


def _dedup(rows: List[Dict]) -> List[Dict]:
    out: List[Dict] = []
    seen = set()
    for r in rows:
        k = (_norm(r["claim"]), _norm(r["evidence"]), int(r["label"]), r.get("lang", ""))
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--inputs",
        nargs="+",
        default=[
            "data/processed/relevance/fever_te_train.jsonl",
            "data/processed/relevance/fever_ta_train.jsonl",
            "data/processed/relevance/fever_ml_train.jsonl",
            "data/processed/relevance/fever_hi_train.jsonl",
            "data/processed/relevance/fever_kn_train.jsonl",
        ],
    )
    p.add_argument("--out-dir", type=str, default="data/processed/relevance/multilingual")
    p.add_argument("--negative-ratio", type=float, default=1.0, help="negatives per positive")
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    rng = random.Random(args.seed)
    raw_rows: List[Dict] = []
    for fp in args.inputs:
        path = Path(fp)
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")
        raw_rows.extend(_read_jsonl(path))

    by_lang = defaultdict(list)
    for r in raw_rows:
        claim = str(r.get("claim", "")).strip()
        evidence = str(r.get("evidence", "")).strip()
        lab = str(r.get("label", "")).strip().upper()
        lang = str(r.get("lang", "")).strip().lower()
        if not claim or not evidence or not lang:
            continue
        by_lang[lang].append({"claim": claim, "evidence": evidence, "src_label": lab, "lang": lang})

    pairs: List[Dict] = []
    for lang, rows in by_lang.items():
        positives = [r for r in rows if r["src_label"] in POS_LABELS]
        nei_rows = [r for r in rows if r["src_label"] in NEG_LABELS]
        pool_evidence = [r["evidence"] for r in rows]
        needed_neg = int(len(positives) * float(args.negative_ratio))

        # 1) NEI negatives first (hard negatives)
        negs: List[Dict] = []
        rng.shuffle(nei_rows)
        for r in nei_rows:
            if len(negs) >= needed_neg:
                break
            negs.append({"claim": r["claim"], "evidence": r["evidence"], "label": 0, "lang": lang})

        # 2) Random mismatch negatives (claim with unrelated evidence)
        if len(negs) < needed_neg:
            pos_claims = positives[:]
            rng.shuffle(pos_claims)
            for r in pos_claims:
                if len(negs) >= needed_neg:
                    break
                c_norm = _norm(r["claim"])
                # Try a few samples to avoid identical evidence pairing
                chosen = None
                for _ in range(10):
                    e = pool_evidence[rng.randrange(len(pool_evidence))]
                    if _norm(e) != _norm(r["evidence"]):
                        chosen = e
                        break
                if chosen is None:
                    continue
                negs.append({"claim": r["claim"], "evidence": chosen, "label": 0, "lang": lang})

        # positives
        for r in positives:
            pairs.append({"claim": r["claim"], "evidence": r["evidence"], "label": 1, "lang": lang})
        pairs.extend(negs)

    pairs = _dedup(pairs)
    labels = [int(r["label"]) for r in pairs]
    strat_keys = [f"{r['lang']}|{r['label']}" for r in pairs]

    train_rows, temp_rows = train_test_split(
        pairs,
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

    def _counts(rows: List[Dict], key: str) -> Dict[str, int]:
        return dict(sorted(Counter([str(r[key]) for r in rows]).items()))

    summary = {
        "inputs": args.inputs,
        "out_dir": str(out_dir),
        "total_rows": len(pairs),
        "negative_ratio": float(args.negative_ratio),
        "total_label_counts": _counts(pairs, "label"),
        "total_lang_counts": _counts(pairs, "lang"),
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "test_rows": len(test_rows),
        "train_label_counts": _counts(train_rows, "label"),
        "val_label_counts": _counts(val_rows, "label"),
        "test_label_counts": _counts(test_rows, "label"),
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

