from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

def main() -> int:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from api.routes.claim import _pipeline_config
    from pipeline.core.checkability import CheckabilityClassifier
    from pipeline.core.normalizer import ClaimNormalizer

    load_dotenv(root / ".env")

    claims_path = root / "tests/benchmarks/rfcs_benchmark_multi/benchmark_cases_multi.json"
    out_path = root / "checkpoints/diagnostics/multi_claims_translation_checkability_audit_v2.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    rows = json.loads(claims_path.read_text(encoding="utf-8"))
    rows = [r for r in rows if str(r.get("lang_bucket", "")).upper() == "MULTI"]

    cfg = _pipeline_config("hi")
    ckpt = cfg.get("claim_checkability_checkpoint")
    clf = CheckabilityClassifier(ckpt)
    norm = ClaimNormalizer()

    results = []
    summary = {
        "total": 0,
        "translated_non_empty": 0,
        "checkable_native": 0,
        "checkable_english": 0,
        "uncheckable_english": 0,
        "changed_after_translation": 0,
        "model_loaded": bool(clf.model is not None),
        "model_device": clf.device,
    }
    by_lang = Counter()
    by_lang_uncheck_en = Counter()

    for r in rows:
        claim = str(r.get("claim", "")).strip()
        lang = str(r.get("language", "")).lower()
        cid = int(r.get("id"))

        en = claim if lang == "en" else (norm._translate_indic_to_english(claim, language=lang) or "")
        if en:
            summary["translated_non_empty"] += 1

        native_check, native_reason = clf.classify(claim)
        en_target = en if en else claim
        en_check, en_reason = clf.classify(en_target)

        summary["total"] += 1
        by_lang[lang] += 1
        if native_check:
            summary["checkable_native"] += 1
        if en_check:
            summary["checkable_english"] += 1
        else:
            summary["uncheckable_english"] += 1
            by_lang_uncheck_en[lang] += 1
        if native_check != en_check:
            summary["changed_after_translation"] += 1

        results.append(
            {
                "id": cid,
                "language": lang,
                "expected": r.get("expected"),
                "claim_native": claim,
                "claim_english": en,
                "checkable_native": bool(native_check),
                "reason_native": native_reason,
                "checkable_english": bool(en_check),
                "reason_english": en_reason,
            }
        )

    payload = {
        "claims_file": str(claims_path),
        "checkability_checkpoint": ckpt,
        "summary": {
            **summary,
            "by_language_total": dict(sorted(by_lang.items())),
            "by_language_uncheckable_english": dict(sorted(by_lang_uncheck_en.items())),
        },
        "results": results,
    }
    out_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    bad = [r for r in results if (not r["checkable_native"]) or (not r["checkable_english"])]
    lines = [
        f"Saved: {out_path}",
        f"model_loaded={summary['model_loaded']} device={summary['model_device']}",
        f"Total={summary['total']} translated_non_empty={summary['translated_non_empty']}",
        f"Checkable(native)={summary['checkable_native']} Checkable(english)={summary['checkable_english']}",
        f"Uncheckable(english)={summary['uncheckable_english']} changed_after_translation={summary['changed_after_translation']}",
        f"bad_count={len(bad)}",
    ]
    for r in bad:
        lines.append(
            f"ID {r['id']} lang={r['language']} native={r['checkable_native']} "
            f"en={r['checkable_english']} rn={r['reason_native']} re={r['reason_english']}"
        )
    sys.stdout.buffer.write(("\n".join(lines) + "\n").encode("utf-8"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
