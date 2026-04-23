"""Build thesis dataset seed by merging EN + MULTI benchmark cases.

This creates a larger seed set for manual curation.

Usage:
  .\\.venv-gpu\\Scripts\\python.exe scripts\\build_thesis_dataset_seed.py --target 200
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
EN_CASES = ROOT / "tests/benchmarks/rfcs_benchmark_en/benchmark_cases_en.json"
MULTI_CASES = ROOT / "tests/benchmarks/rfcs_benchmark_multi/benchmark_cases_multi.json"
OUT_DEFAULT = ROOT / "tests/benchmarks/thesis_dataset_v1.seed.json"


def _load(path: Path) -> List[Dict[str, Any]]:
    return json.loads(path.read_text(encoding="utf-8"))


def _normalize(rows: List[Dict[str, Any]], bucket: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for i, r in enumerate(rows, start=1):
        out.append(
            {
                "id": r.get("id", f"{bucket.lower()}_{i}"),
                "claim": r.get("claim", ""),
                "expected": str(r.get("expected", "neutral")).lower(),
                "language": r.get("language", "en" if bucket == "EN" else "hi"),
                "lang_bucket": bucket,
                "claim_type": r.get("claim_type", "general"),
                "time_bucket": r.get("time_bucket", "mixed"),
                "expected_evidence_urls": r.get("expected_evidence_urls", []),
                "notes": "",
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--target", type=int, default=200, help="Target total rows in output")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out", default=str(OUT_DEFAULT))
    args = parser.parse_args()

    random.seed(args.seed)
    en = _normalize(_load(EN_CASES), "EN")
    multi = _normalize(_load(MULTI_CASES), "MULTI")

    merged = en + multi
    random.shuffle(merged)

    if args.target > 0 and len(merged) > args.target:
        merged = merged[: args.target]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(merged, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved {len(merged)} seed rows to {out_path}")


if __name__ == "__main__":
    main()
