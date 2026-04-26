"""Build per-language benchmark case files from 252_eval-style dataset."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        rows = obj
    elif isinstance(obj, dict):
        if isinstance(obj.get("claims"), list):
            rows = obj["claims"]
        elif isinstance(obj.get("cases"), list):
            rows = obj["cases"]
        else:
            raise ValueError("Unsupported JSON shape: expected top-level list or dict with claims/cases.")
    else:
        raise ValueError("Unsupported JSON type.")
    out = [r for r in rows if isinstance(r, dict)]
    if not out:
        raise ValueError("No valid claim rows found.")
    return out


def _norm_lang(row: Dict[str, Any]) -> str:
    return str(row.get("language") or row.get("lang") or row.get("lang_bucket") or "").strip().lower()


def _to_case(row: Dict[str, Any], expected: str) -> Dict[str, Any]:
    return {
        "id": row.get("id"),
        "claim": row.get("claim"),
        "language": row.get("language"),
        "lang_bucket": row.get("lang_bucket"),
        "label": row.get("label"),
        "subtype": row.get("subtype"),
        "checkability_label": row.get("checkability_label"),
        "recency_class": row.get("recency_class"),
        "expected_verdict": expected,
        "expected_verdict_3way": row.get("expected_verdict_3way"),
        "expected_verdict_4way": row.get("expected_verdict_4way"),
        "gloss_en": row.get("gloss_en"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Input dataset JSON (e.g., 252_eval.json)")
    ap.add_argument("--out-dir", required=True, help="Output directory for per-language files")
    ap.add_argument(
        "--include-uncheckable",
        action="store_true",
        help="Use 4-way expected verdicts and include uncheckable rows.",
    )
    args = ap.parse_args()

    in_path = Path(args.input).resolve()
    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = _load_rows(in_path)
    grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        lang = _norm_lang(row)
        if not lang:
            continue
        exp3 = str(row.get("expected_verdict_3way") or "").strip().lower()
        exp4 = str(row.get("expected_verdict_4way") or "").strip().lower()
        if args.include_uncheckable:
            expected = exp4
        else:
            expected = exp3
        if expected not in {"support", "refute", "neutral", "uncheckable"}:
            continue
        if (not args.include_uncheckable) and expected == "uncheckable":
            continue
        grouped[lang].append(_to_case(row, expected))

    summary: Dict[str, Any] = {
        "input": str(in_path),
        "out_dir": str(out_dir),
        "include_uncheckable": bool(args.include_uncheckable),
        "languages": {},
    }

    for lang, cases in sorted(grouped.items()):
        cases.sort(key=lambda r: str(r.get("id") or ""))
        out_path = out_dir / f"benchmark_cases_{lang}_v2_from_252.json"
        out_path.write_text(json.dumps(cases, ensure_ascii=False, indent=2), encoding="utf-8")

        lbl = Counter(str(c.get("label") or "") for c in cases)
        ver = Counter(str(c.get("expected_verdict") or "") for c in cases)
        summary["languages"][lang] = {
            "count": len(cases),
            "verdict_distribution": dict(ver),
            "label_distribution": dict(lbl),
            "file": str(out_path),
        }

    summary_path = out_dir / "dataset_summary_by_language.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

