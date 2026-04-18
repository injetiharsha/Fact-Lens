"""Normalize stance JSONL files to claim/evidence/label_id schema.

Mapping used (aligned with training label order [support, refute, neutral]):
- support -> 0
- refute  -> 1
- neutral -> 2
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Dict, List


LABEL2ID = {"support": 0, "refute": 1, "neutral": 2}
ID2LABEL = {v: k for k, v in LABEL2ID.items()}


def _norm_label(v) -> int:
    if isinstance(v, int):
        if v in ID2LABEL:
            return v
        raise ValueError(f"Unknown label id: {v}")
    s = str(v).strip().lower()
    if s not in LABEL2ID:
        raise ValueError(f"Unknown label value: {v}")
    return LABEL2ID[s]


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
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--dir", required=True, type=str, help="Dataset dir with train/val/test jsonl")
    p.add_argument("--backup-dir", required=False, type=str, default=None)
    p.add_argument("--replace", action="store_true")
    args = p.parse_args()

    base = Path(args.dir)
    files = [base / "train.jsonl", base / "val.jsonl", base / "test.jsonl"]
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        raise FileNotFoundError("Missing files:\n" + "\n".join(missing))

    if args.replace:
        bdir = Path(args.backup_dir) if args.backup_dir else base.parent / f"{base.name}_textlabel_backup"
        if bdir.exists():
            raise FileExistsError(f"Backup dir already exists: {bdir}")
        shutil.copytree(base, bdir)

    summary = {"dataset_dir": str(base), "mapping": LABEL2ID, "splits": {}}
    for f in files:
        rows = _read_jsonl(f)
        out = []
        for r in rows:
            claim = str(r.get("claim", "")).strip()
            evidence = str(r.get("evidence", "")).strip()
            lid = _norm_label(r.get("label"))
            if not claim or not evidence:
                continue
            out.append({"claim": claim, "evidence": evidence, "label": lid})
        _write_jsonl(f, out)
        c = {0: 0, 1: 0, 2: 0}
        for r in out:
            c[r["label"]] += 1
        summary["splits"][f.stem] = {"rows": len(out), "label_id_counts": c}

    with (base / "label_id_mapping.json").open("w", encoding="utf-8") as fp:
        json.dump({"label2id": LABEL2ID, "id2label": ID2LABEL}, fp, indent=2)
    with (base / "normalization_summary.json").open("w", encoding="utf-8") as fp:
        json.dump(summary, fp, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

