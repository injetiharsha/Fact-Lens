"""Normalize stance JSONL schema to claim/evidence/label."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict


VALID = {"support", "refute", "neutral"}


def _normalize_row(row: Dict) -> Dict:
    claim = (
        row.get("claim")
        or row.get("premise")
        or row.get("text_a")
        or row.get("text")
        or ""
    )
    evidence = (
        row.get("evidence")
        or row.get("hypothesis")
        or row.get("text_b")
        or row.get("context")
        or ""
    )
    lbl = row.get("label")
    if isinstance(lbl, int):
        lbl = {0: "support", 1: "neutral", 2: "refute"}.get(lbl)
    lbl = str(lbl).strip().lower() if lbl is not None else ""
    if lbl in {"supports", "entailment"}:
        lbl = "support"
    elif lbl in {"refutes", "contradiction"}:
        lbl = "refute"
    elif lbl in {"not enough info", "nei"}:
        lbl = "neutral"

    if lbl not in VALID:
        raise ValueError(f"Unknown stance label: {row.get('label')}")
    return {
        "claim": str(claim).strip(),
        "evidence": str(evidence).strip(),
        "label": lbl,
    }


def normalize_file(path: Path) -> tuple[int, int]:
    rows = []
    in_n = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            in_n += 1
            row = json.loads(line)
            n = _normalize_row(row)
            if n["claim"] and n["evidence"]:
                rows.append(n)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    return in_n, len(rows)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument(
        "--dirs",
        nargs="+",
        default=["data/processed/stance/en_mnli", "data/processed/stance/en_fever", "data/processed/stance/en_vitaminc"],
    )
    args = p.parse_args()

    for d in args.dirs:
        base = Path(d)
        if not base.exists():
            continue
        for name in ("train.jsonl", "val.jsonl", "test.jsonl"):
            fp = base / name
            if not fp.exists():
                continue
            in_n, out_n = normalize_file(fp)
            print(f"{fp}: {in_n} -> {out_n}")


if __name__ == "__main__":
    main()
