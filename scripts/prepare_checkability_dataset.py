"""Prepare multilingual checkability dataset with dedupe and rule-based synthesis.

Example:
  .\\.venv-gpu\\Scripts\\python.exe scripts\\prepare_checkability_dataset.py ^
    --input "C:\\Users\\chint\\Downloads\\balanced_multilingual_claims_252_full_v2.json" ^
    --out-dir "data/processed/checkability/multilingual" ^
    --target-per-class-lang 80
"""

from __future__ import annotations

import argparse
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

LABELS_ORDER = [
    "FACTUAL_CLAIM",
    "PERSONAL_STATEMENT",
    "OPINION",
    "QUESTION_OR_REWRITE",
    "OTHER_UNCHECKABLE",
]
LABEL2ID = {name: idx for idx, name in enumerate(LABELS_ORDER)}
SCRIPT_RANGES = {
    "hi": (0x0900, 0x097F),
    "ta": (0x0B80, 0x0BFF),
    "te": (0x0C00, 0x0C7F),
    "kn": (0x0C80, 0x0CFF),
    "ml": (0x0D00, 0x0D7F),
}


def _load_rows(path: Path) -> List[Dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if isinstance(payload, dict):
        rows = payload.get("claims", [])
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []
    return [r for r in rows if isinstance(r, dict)]


def _to_upper(value: Any) -> str:
    return str(value or "").strip().upper()


def _normalize_label(raw: Any) -> Optional[str]:
    text = _to_upper(raw)
    if not text:
        return None
    norm_map = {
        "FACTUAL_CLAIM": "FACTUAL_CLAIM",
        "FACTUAL": "FACTUAL_CLAIM",
        "CHECKABLE": "FACTUAL_CLAIM",
        "CLAIM_CHECKABLE": "FACTUAL_CLAIM",
        "PERSONAL_STATEMENT": "PERSONAL_STATEMENT",
        "PERSONAL": "PERSONAL_STATEMENT",
        "OPINION": "OPINION",
        "QUESTION_OR_REWRITE": "QUESTION_OR_REWRITE",
        "QUESTION": "QUESTION_OR_REWRITE",
        "REWRITE": "QUESTION_OR_REWRITE",
        "Q_OR_REWRITE": "QUESTION_OR_REWRITE",
        "OTHER_UNCHECKABLE": "OTHER_UNCHECKABLE",
        "UNCHECKABLE": "OTHER_UNCHECKABLE",
    }
    if text in norm_map:
        return norm_map[text]
    if text.isdigit():
        idx = int(text)
        if 0 <= idx < len(LABELS_ORDER):
            return LABELS_ORDER[idx]
    return None


def _extract_numeric_id(raw_id: str) -> Optional[int]:
    m = re.search(r"(\d+)$", str(raw_id or "").strip())
    if not m:
        return None
    return int(m.group(1))


def _c_id_override(raw_id: str) -> Optional[str]:
    """Optional overrides for legacy C001..C150 plans."""
    rid = str(raw_id or "").strip().upper()
    if not re.fullmatch(r"C\d{3}", rid):
        return None
    n = int(rid[1:])

    clean_support = {
        1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20, 22, 23, 25, 26, 28, 29,
        31, 32, 34, 35, 37, 38, 40, 41, 43, 44, 46, 47, 49, 50
    }
    amb_support = {3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42, 45, 48}
    recent_amb_rumor = {102, 108, 111, 114, 117, 120, 123, 126, 129, 132, 135, 138, 141, 144, 147, 150}
    opinion_social = {107, 110, 131, 134, 137}
    meta_reporting = {101, 103, 106, 109, 112, 121}
    misconception_negation = {113, 115, 116, 118, 119, 122, 125, 127, 128, 130, 133, 136, 139, 140, 143, 146, 148, 149}
    overgeneralized = {104, 105, 124, 142, 145}

    if n in clean_support or (51 <= n <= 100):
        return "FACTUAL_CLAIM"
    if n in amb_support:
        return "OTHER_UNCHECKABLE"
    if n in recent_amb_rumor:
        return "QUESTION_OR_REWRITE"
    if n in opinion_social:
        return "PERSONAL_STATEMENT"
    if n in meta_reporting:
        return "OPINION"
    if n in misconception_negation:
        return "QUESTION_OR_REWRITE"
    if n in overgeneralized:
        return "PERSONAL_STATEMENT"
    return None


def _fallback_label(row: Dict[str, Any]) -> str:
    claim = str(row.get("claim", "")).strip()
    claim_l = claim.lower()
    verdict = str(
        row.get("expected_verdict_4way")
        or row.get("expected_verdict")
        or row.get("expected")
        or row.get("verdict")
        or ""
    ).strip().lower()

    if verdict in {"support", "refute"}:
        return "FACTUAL_CLAIM"
    if verdict == "uncheckable":
        return "OTHER_UNCHECKABLE"
    if claim.endswith("?") or "is it true" in claim_l:
        return "QUESTION_OR_REWRITE"
    if any(x in claim_l for x in ("i think", "in my opinion", "from my perspective")):
        return "PERSONAL_STATEMENT"
    if any(x in claim_l for x in ("people say", "it is said", "widely believed")):
        return "OPINION"
    return "OTHER_UNCHECKABLE"


def _normalize_text_for_dedupe(text: str) -> str:
    t = str(text or "").strip().lower()
    t = re.sub(r"\s+", " ", t)
    t = re.sub(r"[\"'`]+", "", t)
    return t


def _count_script_chars(text: str, language: str) -> int:
    rng = SCRIPT_RANGES.get(language)
    if not rng:
        return 0
    lo, hi = rng
    return sum(1 for ch in text if lo <= ord(ch) <= hi)


def _count_latin_chars(text: str) -> int:
    return sum(1 for ch in text if ("a" <= ch.lower() <= "z"))


def _clean_claim_text(text: str) -> str:
    t = str(text or "").strip()
    # Drop common generation artifact suffixes like:
    # "This claim is provided in Hindi."
    t = re.sub(
        r"\s+(this\s+claim\s+is\s+(provided|presented)\s+in\s+[a-z]+\.?)\s*$",
        "",
        t,
        flags=re.IGNORECASE,
    ).strip()
    return t


def _maybe_demojibake(text: str) -> str:
    """Best-effort repair for UTF-8 text decoded as latin-1/cp1252."""
    t = str(text or "")
    # Common mojibake markers for Indic scripts.
    if not any(marker in t for marker in ("à¤", "à®", "à°", "à²", "à´")):
        return t
    try:
        fixed = t.encode("latin-1", errors="ignore").decode("utf-8", errors="ignore")
    except Exception:
        return t
    return fixed.strip() if fixed.strip() else t


def _language_text_ok(language: str, text: str, strict: bool = True) -> bool:
    lang = str(language or "").lower()
    t = str(text or "").strip()
    if not t:
        return False
    if not strict:
        return True
    if lang == "en":
        # English should mostly be latin letters.
        latin = _count_latin_chars(t)
        return latin >= 8
    if lang in SCRIPT_RANGES:
        native = _count_script_chars(t, lang)
        latin = _count_latin_chars(t)
        # Keep non-EN claims only if native script clearly dominates.
        # This removes mixed rows like "English sentence + translated note".
        allowed_latin = max(8, int(native * 0.25))
        return native >= 12 and latin <= allowed_latin
    return True


def _label_templates(label: str, language: str) -> List[str]:
    lang = str(language or "en").lower()
    per_lang = {
        "en": {
            "FACTUAL_CLAIM": ["Fact check statement: {claim}", "Verified factual claim: {claim}"],
            "PERSONAL_STATEMENT": ["I think that {claim}", "In my view, {claim}"],
            "OPINION": ["Many people believe that {claim}", "A common opinion is: {claim}"],
            "QUESTION_OR_REWRITE": ["Is it true that {claim}?", "Can this be verified: {claim}?"],
            "OTHER_UNCHECKABLE": ["Reports suggest that {claim}", "It is being said that {claim}"],
        },
        "hi": {
            "FACTUAL_CLAIM": ["तथ्य-जांच कथन: {claim}", "सत्यापित तथ्य दावा: {claim}"],
            "PERSONAL_STATEMENT": ["मेरा मानना है कि {claim}", "मेरी राय में, {claim}"],
            "OPINION": ["कई लोग मानते हैं कि {claim}", "एक आम राय है: {claim}"],
            "QUESTION_OR_REWRITE": ["क्या यह सच है कि {claim}?", "क्या इसकी पुष्टि हो सकती है: {claim}?"],
            "OTHER_UNCHECKABLE": ["ऐसी रिपोर्टें हैं कि {claim}", "ऐसा कहा जा रहा है कि {claim}"],
        },
        "ta": {
            "FACTUAL_CLAIM": ["உண்மைச் சரிபார்ப்பு கூற்று: {claim}", "சரிபார்க்கப்பட்ட உண்மை கூற்று: {claim}"],
            "PERSONAL_STATEMENT": ["என் கருத்தில் {claim}", "நான் நினைப்பது {claim}"],
            "OPINION": ["பலர் நினைப்பது {claim}", "பொதுவான கருத்து: {claim}"],
            "QUESTION_OR_REWRITE": ["{claim} என்பது உண்மையா?", "இதை சரிபார்க்க முடியுமா: {claim}?"],
            "OTHER_UNCHECKABLE": ["அறிக்கைகள் கூறுவது {claim}", "சொல்லப்படுவது {claim}"],
        },
        "te": {
            "FACTUAL_CLAIM": ["వాస్తవ నిర్ధారణ వాక్యం: {claim}", "నిర్ధారిత వాస్తవ ప్రకటన: {claim}"],
            "PERSONAL_STATEMENT": ["నా అభిప్రాయం ప్రకారం {claim}", "నేను అనుకుంటున్నది {claim}"],
            "OPINION": ["చాలామంది నమ్మేది {claim}", "సాధారణ అభిప్రాయం: {claim}"],
            "QUESTION_OR_REWRITE": ["{claim} నిజమా?", "దీన్ని ధృవీకరించగలమా: {claim}?"],
            "OTHER_UNCHECKABLE": ["కొన్ని నివేదికలు చెబుతున్నది {claim}", "ఇలా అంటున్నారు: {claim}"],
        },
        "kn": {
            "FACTUAL_CLAIM": ["ತಥ್ಯ ಪರಿಶೀಲನೆ ಹೇಳಿಕೆ: {claim}", "ಪರಿಶೀಲಿತ ತಥ್ಯ ಹೇಳಿಕೆ: {claim}"],
            "PERSONAL_STATEMENT": ["ನನ್ನ ಅಭಿಪ್ರಾಯದಲ್ಲಿ {claim}", "ನಾನು ಭಾವಿಸುವುದು {claim}"],
            "OPINION": ["ಅನೇಕರ ಅಭಿಪ್ರಾಯ {claim}", "ಸಾಮಾನ್ಯ ಅಭಿಪ್ರಾಯ: {claim}"],
            "QUESTION_OR_REWRITE": ["{claim} ಸತ್ಯವೇ?", "ಇದನ್ನು ಪರಿಶೀಲಿಸಬಹುದೇ: {claim}?"],
            "OTHER_UNCHECKABLE": ["ವರದಿಗಳು ಹೇಳುವುದೇನೆಂದರೆ {claim}", "ಹೀಗೆ ಹೇಳಲಾಗುತ್ತಿದೆ: {claim}"],
        },
        "ml": {
            "FACTUAL_CLAIM": ["വസ്തുത പരിശോധന വാക്യം: {claim}", "സ്ഥിരീകരിച്ച വസ്തുത അവകാശവാദം: {claim}"],
            "PERSONAL_STATEMENT": ["എന്റെ അഭിപ്രായത്തിൽ {claim}", "ഞാൻ കരുതുന്നത് {claim}"],
            "OPINION": ["പലരും വിശ്വസിക്കുന്നത് {claim}", "ഒരു പൊതുവായ അഭിപ്രായം: {claim}"],
            "QUESTION_OR_REWRITE": ["{claim} ശരിയാണോ?", "ഇത് പരിശോധിക്കാമോ: {claim}?"],
            "OTHER_UNCHECKABLE": ["റിപ്പോർട്ടുകൾ പറയുന്നത് {claim}", "ഇങ്ങനെ പറയപ്പെടുന്നു: {claim}"],
        },
    }
    if lang in per_lang and label in per_lang[lang]:
        return per_lang[lang][label]
    # fallback english
    if label == "FACTUAL_CLAIM":
        return [
            "Fact check statement: {claim}",
            "Verified factual claim: {claim}",
            "Official record states: {claim}",
        ]
    if label == "PERSONAL_STATEMENT":
        return [
            "I think that {claim}",
            "From my personal perspective, {claim}",
            "In my view, {claim}",
        ]
    if label == "OPINION":
        return [
            "Many people believe that {claim}",
            "It is often considered that {claim}",
            "A common opinion is: {claim}",
        ]
    if label == "QUESTION_OR_REWRITE":
        return [
            "Is it true that {claim}?",
            "Can this be verified: {claim}?",
            "Please rewrite and verify: {claim}",
        ]
    return [
        "Reports suggest that {claim}",
        "It is being said that {claim}",
        "Some sources claim that {claim}",
    ]


def _build_base_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in rows:
        claim = _maybe_demojibake(_clean_claim_text(row.get("claim", "")))
        if not claim:
            continue
        raw_id = str(row.get("id", f"row_{len(out)+1}")).strip()
        lang = str(row.get("language", "en")).strip().lower() or "en"

        label = _normalize_label(
            row.get("checkability_label")
            or row.get("suggested_checkability_label")
            or row.get("checkability")
        )
        if not label:
            label = _c_id_override(raw_id) or _fallback_label(row)
        if label not in LABEL2ID:
            continue

        out.append(
            {
                "id": raw_id,
                "orig_id": raw_id,
                "language": lang,
                "text": claim,
                "label": label,
                "label_id": LABEL2ID[label],
                "source": "original",
            }
        )
    return out


def _dedupe_rows(rows: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    out: List[Dict[str, Any]] = []
    for row in rows:
        key = (row["language"], _normalize_text_for_dedupe(row["text"]), row["label"])
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _augment_rows(
    rows: List[Dict[str, Any]],
    target_per_class_lang: int,
    seed: int,
    strict_language: bool,
) -> List[Dict[str, Any]]:
    rng = random.Random(seed)
    grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    lang_pool: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[(row["language"], row["label"])].append(row)
        lang_pool[row["language"]].append(row)

    out = list(rows)
    dedupe_keys = {(r["language"], _normalize_text_for_dedupe(r["text"]), r["label"]) for r in rows}

    languages = sorted(lang_pool.keys())
    targets: List[Tuple[str, str]] = []
    for lang in languages:
        for label in LABELS_ORDER:
            targets.append((lang, label))

    for (lang, label) in targets:
        bucket = list(grouped.get((lang, label), []))
        if len(bucket) >= target_per_class_lang:
            continue
        templates = _label_templates(label, lang)
        # Use full language pool for synthesis so minority labels can scale.
        seed_bucket = list(lang_pool.get(lang, []))
        if not seed_bucket:
            continue
        attempts = 0
        created = 0
        need = target_per_class_lang - len(bucket)
        while created < need and attempts < (need * 40):
            attempts += 1
            base = rng.choice(seed_bucket)
            tpl = rng.choice(templates)
            candidate = tpl.format(claim=base["text"]).strip()
            if not _language_text_ok(lang, candidate, strict=strict_language):
                continue
            key = (lang, _normalize_text_for_dedupe(candidate), label)
            if key in dedupe_keys:
                continue
            dedupe_keys.add(key)
            synth_id = f"SYN_{lang}_{label}_{len(out)+1:06d}"
            out.append(
                {
                    "id": synth_id,
                    "orig_id": base["orig_id"],
                    "language": lang,
                    "text": candidate,
                    "label": label,
                    "label_id": LABEL2ID[label],
                    "source": "synthetic_rule",
                }
            )
            grouped[(lang, label)].append(out[-1])
            created += 1
    return out


def _split_groupwise(
    rows: List[Dict[str, Any]],
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> Dict[str, List[Dict[str, Any]]]:
    rng = random.Random(seed)
    groups: Dict[Tuple[str, str], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[(row["language"], row["label"])].append(row)

    out = {"train": [], "val": [], "test": []}
    for key, bucket in groups.items():
        items = list(bucket)
        rng.shuffle(items)
        n = len(items)
        if n == 1:
            out["train"].extend(items)
            continue
        if n == 2:
            out["train"].append(items[0])
            out["test"].append(items[1])
            continue
        n_val = max(1, int(round(n * val_ratio)))
        n_test = max(1, int(round(n * test_ratio)))
        n_train = n - n_val - n_test
        if n_train < 1:
            n_train = 1
            if n_val >= n_test and n_val > 1:
                n_val -= 1
            elif n_test > 1:
                n_test -= 1
        out["train"].extend(items[:n_train])
        out["val"].extend(items[n_train:n_train + n_val])
        out["test"].extend(items[n_train + n_val:n_train + n_val + n_test])
    return out


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _count(rows: List[Dict[str, Any]], key: str) -> Dict[str, int]:
    return dict(sorted(Counter(str(r.get(key, "")) for r in rows).items()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, type=str, help="Primary JSON file")
    ap.add_argument("--extra-input", action="append", default=[], help="Optional extra JSON files")
    ap.add_argument("--out-dir", default="data/processed/checkability/multilingual", type=str)
    ap.add_argument("--target-per-class-lang", type=int, default=80)
    ap.add_argument("--train-ratio", type=float, default=0.8)
    ap.add_argument("--val-ratio", type=float, default=0.1)
    ap.add_argument("--test-ratio", type=float, default=0.1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--strict-language",
        action="store_true",
        default=True,
        help="Require non-EN claims to contain native script and avoid mixed-English artifacts.",
    )
    ap.add_argument(
        "--no-strict-language",
        action="store_false",
        dest="strict_language",
    )
    args = ap.parse_args()

    if abs((args.train_ratio + args.val_ratio + args.test_ratio) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    in_paths = [Path(args.input)] + [Path(p) for p in args.extra_input]
    raw_rows: List[Dict[str, Any]] = []
    for p in in_paths:
        raw_rows.extend(_load_rows(p))

    base_pre = _build_base_rows(raw_rows)
    base = _dedupe_rows(
        [r for r in base_pre if _language_text_ok(r["language"], r["text"], strict=bool(args.strict_language))]
    )
    expanded = _dedupe_rows(
        _augment_rows(
            rows=base,
            target_per_class_lang=max(1, int(args.target_per_class_lang)),
            seed=int(args.seed),
            strict_language=bool(args.strict_language),
        )
    )
    splits = _split_groupwise(
        rows=expanded,
        train_ratio=float(args.train_ratio),
        val_ratio=float(args.val_ratio),
        test_ratio=float(args.test_ratio),
        seed=int(args.seed),
    )

    out_dir = Path(args.out_dir)
    _write_jsonl(out_dir / "train.jsonl", splits["train"])
    _write_jsonl(out_dir / "val.jsonl", splits["val"])
    _write_jsonl(out_dir / "test.jsonl", splits["test"])

    summary = {
        "input_files": [str(p) for p in in_paths],
        "labels_order": LABELS_ORDER,
        "label2id": LABEL2ID,
        "seed": int(args.seed),
        "target_per_class_lang": int(args.target_per_class_lang),
        "strict_language": bool(args.strict_language),
        "counts": {
            "base_rows_raw": len(base_pre),
            "base_rows": len(base),
            "expanded_rows": len(expanded),
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
        "split_label_counts": {
            "train": _count(splits["train"], "label"),
            "val": _count(splits["val"], "label"),
            "test": _count(splits["test"], "label"),
        },
        "split_language_counts": {
            "train": _count(splits["train"], "language"),
            "val": _count(splits["val"], "language"),
            "test": _count(splits["test"], "language"),
        },
        "source_counts_expanded": _count(expanded, "source"),
    }
    (out_dir / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"Saved: {out_dir / 'train.jsonl'}")
    print(f"Saved: {out_dir / 'val.jsonl'}")
    print(f"Saved: {out_dir / 'test.jsonl'}")
    print(f"Saved: {out_dir / 'dataset_summary.json'}")


if __name__ == "__main__":
    main()
