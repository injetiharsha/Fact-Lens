"""Run OCR + locked EN/MULTI pipeline over an image folder."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from api.routes.claim import _auto_detect_language, _pipeline_config
from pipeline.claim_pipeline import ClaimPipeline
from pipeline.ingestion.image import ImageInputPipeline


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _print_lock_snapshot() -> None:
    keys = [
        "ENABLE_LLM_VERIFIER",
        "WEB_SEARCH_ENABLE_DDG",
        "WEB_SEARCH_ENABLE_TAVILY",
        "WEB_SEARCH_ENABLE_SERPAPI",
        "CHECKABILITY_BYPASS_LANGS",
        "CHECKABILITY_EN_PATH",
        "CHECKABILITY_MULTI_PATH",
        "CONTEXT_EN_PATH",
        "CONTEXT_MULTI_PATH",
        "RELEVANCE_EN_PATH",
        "RELEVANCE_MULTI_PATH",
        "STANCE_EN_PATH",
        "STANCE_MULTI_PATH",
    ]
    print("[lock-check] effective env:")
    for k in keys:
        print(f"[lock-check] {k}={os.getenv(k, '')}")


def _print_model_banner() -> None:
    en_cfg = _pipeline_config("en")
    multi_cfg = _pipeline_config("hi")
    print("[models:en] pipeline_language=en")
    print(f"[models:en] checkability={en_cfg.get('claim_checkability_checkpoint')}")
    print(f"[models:en] context={en_cfg.get('context_checkpoint')}")
    print(f"[models:en] relevance={en_cfg.get('relevance_checkpoint')}")
    print(f"[models:en] stance={en_cfg.get('stance_checkpoint')}")
    print(f"[models:en] llm_verifier_enabled={int(bool(en_cfg.get('enable_llm_verifier', True)))}")
    print("[models:multi] pipeline_language=multi(shared-hi)")
    print(f"[models:multi] checkability={multi_cfg.get('claim_checkability_checkpoint')}")
    print(f"[models:multi] context={multi_cfg.get('context_checkpoint')}")
    print(f"[models:multi] relevance={multi_cfg.get('relevance_checkpoint')}")
    print(f"[models:multi] stance={multi_cfg.get('stance_checkpoint')}")
    print(f"[models:multi] llm_verifier_enabled={int(bool(multi_cfg.get('enable_llm_verifier', True)))}")


def _collect_images(folder: Path) -> List[Path]:
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".tif", ".tiff"}
    files = [p for p in folder.iterdir() if p.is_file() and p.suffix.lower() in exts]
    return sorted(files, key=lambda p: p.name.lower())


def run(folder: Path, out_json: Path, out_csv: Path) -> Dict[str, Any]:
    _print_lock_snapshot()
    _print_model_banner()
    image_pipe = ImageInputPipeline()
    en_pipe = ClaimPipeline(_pipeline_config("en"))
    multi_pipe = ClaimPipeline(_pipeline_config("hi"))

    rows: List[Dict[str, Any]] = []
    for idx, image_path in enumerate(_collect_images(folder), start=1):
        print(f"[image {idx}] {image_path.name}")
        ing = image_pipe.process(str(image_path), language="auto")
        claim_text = (ing.claim_text or "").strip()
        detected = _auto_detect_language(claim_text, "auto")
        bucket = "en" if detected == "en" else "multi"
        pipe = en_pipe if bucket == "en" else multi_pipe

        if not claim_text:
            rows.append(
                {
                    "image": str(image_path),
                    "claim_text": "",
                    "ocr_text": ing.ocr_text,
                    "ocr_engine": ing.ocr_engine,
                    "ocr_confidence": _safe_float(ing.ocr_confidence),
                    "detected_language": detected,
                    "pipeline_bucket": bucket,
                    "verdict": "neutral",
                    "confidence": 0.0,
                    "evidence_count": 0,
                    "checkability": "unknown",
                    "checkability_blocked": False,
                    "warnings": list(ing.warnings or []),
                    "error": "no_claim_text_extracted",
                }
            )
            continue

        error_text = ""
        try:
            result = pipe.analyze(claim=claim_text, language=detected)
            details = dict(getattr(result, "details", {}) or {})
            evidence = list(getattr(result, "evidence", []) or [])
            checkability = str(details.get("checkability", "unknown"))
            rows.append(
                {
                    "image": str(image_path),
                    "claim_text": claim_text,
                    "ocr_text": ing.ocr_text,
                    "ocr_engine": ing.ocr_engine,
                    "ocr_confidence": _safe_float(ing.ocr_confidence),
                    "detected_language": detected,
                    "pipeline_bucket": bucket,
                    "verdict": str(getattr(result, "verdict", "neutral") or "neutral").lower(),
                    "confidence": _safe_float(getattr(result, "confidence", 0.0)),
                    "evidence_count": len(evidence),
                    "checkability": checkability,
                    "checkability_blocked": checkability.lower().startswith("uncheckable"),
                    "warnings": list(ing.warnings or []),
                    "error": error_text,
                }
            )
        except Exception as exc:
            error_text = str(exc)
            rows.append(
                {
                    "image": str(image_path),
                    "claim_text": claim_text,
                    "ocr_text": ing.ocr_text,
                    "ocr_engine": ing.ocr_engine,
                    "ocr_confidence": _safe_float(ing.ocr_confidence),
                    "detected_language": detected,
                    "pipeline_bucket": bucket,
                    "verdict": "neutral",
                    "confidence": 0.0,
                    "evidence_count": 0,
                    "checkability": "unknown",
                    "checkability_blocked": False,
                    "warnings": list(ing.warnings or []),
                    "error": error_text,
                }
            )

    summary = {
        "generated_at_utc": datetime.utcnow().isoformat() + "Z",
        "input_folder": str(folder),
        "total_images": len(rows),
        "rows": rows,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image",
                "detected_language",
                "pipeline_bucket",
                "ocr_engine",
                "ocr_confidence",
                "claim_text",
                "verdict",
                "confidence",
                "evidence_count",
                "checkability",
                "checkability_blocked",
                "error",
            ],
        )
        writer.writeheader()
        for r in rows:
            writer.writerow({k: r.get(k, "") for k in writer.fieldnames})

    print(f"[done] json={out_json}")
    print(f"[done] csv={out_csv}")
    return summary


def main() -> None:
    load_dotenv(ROOT / ".env")
    ap = argparse.ArgumentParser(description="OCR + locked EN/MULTI pipeline runner")
    ap.add_argument("--input-dir", default=str(ROOT / "test_images"))
    ap.add_argument("--output-json", default="")
    ap.add_argument("--output-csv", default="")
    args = ap.parse_args()

    in_dir = Path(args.input_dir)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_json = Path(args.output_json) if args.output_json else in_dir / f"locked_pipeline_image_run_{ts}.json"
    out_csv = Path(args.output_csv) if args.output_csv else in_dir / f"locked_pipeline_image_run_{ts}.csv"

    if not in_dir.exists() or not in_dir.is_dir():
        raise FileNotFoundError(f"Input folder not found: {in_dir}")
    run(in_dir, out_json, out_csv)


if __name__ == "__main__":
    main()
