"""Build mixed stance dataset from IndicXNLI + translated FEVER.

Targets:
- IndicXNLI target total (default 150k), equal per language
- FEVER target total (default 150k), equal per language; if a language has fewer rows,
  all languages are clipped to that minimum (so total may be < target)
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


LANGS = ["hi", "kn", "ml", "ta", "te"]

FEVER_LABEL_MAP = {
    "SUPPORTS": "support",
    "REFUTES": "refute",
    "NOT ENOUGH INFO": "neutral",
    "support": "support",
    "refute": "refute",
    "neutral": "neutral",
}


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", str(s).strip().lower())


def _read_jsonl(path: Path) -> List[Dict]:
    out = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def _write_jsonl(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def _counts(rows: List[Dict], key: str) -> Dict[str, int]:
    return dict(sorted(Counter([str(r[key]) for r in rows]).items()))


def _load_indicxnli(indic_dir: Path) -> List[Dict]:
    rows = []
    for split in ("train", "val", "test"):
        rows.extend(_read_jsonl(indic_dir / f"{split}.jsonl"))
    out = []
    for r in rows:
        claim = str(r.get("claim", "")).strip()
        evidence = str(r.get("evidence", "")).strip()
        label = str(r.get("label", "")).strip().lower()
        lang = str(r.get("lang", "")).strip().lower()
        if claim and evidence and label in {"support", "refute", "neutral"} and lang in LANGS:
            out.append(
                {
                    "claim": claim,
                    "evidence": evidence,
                    "label": label,
                    "lang": lang,
                    "source": "indicxnli",
                }
            )
    return out


def _load_fever(relevance_dir: Path) -> List[Dict]:
    rows = []
    for lg in LANGS:
        rows.extend(_read_jsonl(relevance_dir / f"fever_{lg}_train.jsonl"))
    out = []
    for r in rows:
        claim = str(r.get("claim", "")).strip()
        evidence = str(r.get("evidence", "")).strip()
        raw_label = str(r.get("label", "")).strip()
        label = FEVER_LABEL_MAP.get(raw_label)
        lang = str(r.get("lang", "")).strip().lower()
        if claim and evidence and label and lang in LANGS:
            out.append(
                {
                    "claim": claim,
                    "evidence": evidence,
                    "label": label,
                    "lang": lang,
                    "source": "fever_translated",
                }
            )
    return out


def _dedup(rows: List[Dict]) -> List[Dict]:
    out = []
    seen = set()
    for r in rows:
        k = (_norm(r["claim"]), _norm(r["evidence"]), r["label"], r["lang"], r["source"])
        if k in seen:
            continue
        seen.add(k)
        out.append(r)
    return out


def _sample_equal_lang(rows: List[Dict], target_total: int, seed: int) -> tuple[List[Dict], Dict]:
    rng = random.Random(seed)
    by_lang = defaultdict(list)
    for r in rows:
        by_lang[r["lang"]].append(r)

    per_lang_target = target_total // len(LANGS)
    min_avail = min(len(by_lang[l]) for l in LANGS)
    per_lang = min(per_lang_target, min_avail)

    sampled = []
    for l in LANGS:
        items = by_lang[l][:]
        rng.shuffle(items)
        sampled.extend(items[:per_lang])
    rng.shuffle(sampled)

    meta = {
        "target_total": target_total,
        "per_lang_target_floor": per_lang_target,
        "min_available_per_lang": min_avail,
        "used_per_lang": per_lang,
        "actual_total": len(sampled),
        "lang_counts": _counts(sampled, "lang"),
        "label_counts": _counts(sampled, "label"),
    }
    return sampled, meta


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--indic-dir", type=str, default="data/processed/stance/indic_xnli")
    p.add_argument("--fever-dir", type=str, default="data/processed/relevance")
    p.add_argument("--out-dir", type=str, default="data/processed/stance/indic_fever_mix")
    p.add_argument("--indic-target", type=int, default=150000)
    p.add_argument("--fever-target", type=int, default=150000)
    p.add_argument("--train-ratio", type=float, default=0.8)
    p.add_argument("--val-ratio", type=float, default=0.1)
    p.add_argument("--test-ratio", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    indic_rows = _dedup(_load_indicxnli(Path(args.indic_dir)))
    fever_rows = _dedup(_load_fever(Path(args.fever_dir)))

    indic_sampled, indic_meta = _sample_equal_lang(indic_rows, args.indic_target, args.seed)
    fever_sampled, fever_meta = _sample_equal_lang(fever_rows, args.fever_target, args.seed + 1)

    mixed = indic_sampled + fever_sampled
    mixed = _dedup(mixed)

    strat = [f"{r['source']}|{r['lang']}|{r['label']}" for r in mixed]
    train_rows, temp_rows = train_test_split(
        mixed, test_size=(1.0 - args.train_ratio), random_state=args.seed, stratify=strat
    )
    strat_temp = [f"{r['source']}|{r['lang']}|{r['label']}" for r in temp_rows]
    val_share = args.val_ratio / (args.val_ratio + args.test_ratio)
    val_rows, test_rows = train_test_split(
        temp_rows, train_size=val_share, random_state=args.seed, stratify=strat_temp
    )

    out_dir = Path(args.out_dir)
    _write_jsonl(out_dir / "train.jsonl", train_rows)
    _write_jsonl(out_dir / "val.jsonl", val_rows)
    _write_jsonl(out_dir / "test.jsonl", test_rows)

    summary = {
        "indic_meta": indic_meta,
        "fever_meta": fever_meta,
        "mixed_total": len(mixed),
        "mixed_source_counts": _counts(mixed, "source"),
        "mixed_lang_counts": _counts(mixed, "lang"),
        "mixed_label_counts": _counts(mixed, "label"),
        "splits": {
            "train": {
                "rows": len(train_rows),
                "source_counts": _counts(train_rows, "source"),
                "lang_counts": _counts(train_rows, "lang"),
                "label_counts": _counts(train_rows, "label"),
            },
            "val": {
                "rows": len(val_rows),
                "source_counts": _counts(val_rows, "source"),
                "lang_counts": _counts(val_rows, "lang"),
                "label_counts": _counts(val_rows, "label"),
            },
            "test": {
                "rows": len(test_rows),
                "source_counts": _counts(test_rows, "source"),
                "lang_counts": _counts(test_rows, "lang"),
                "label_counts": _counts(test_rows, "label"),
            },
        },
    }
    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

