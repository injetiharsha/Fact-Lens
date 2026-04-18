"""Build EN + Indic context datasets (JSONL) with weak supervision labels."""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from datasets import load_dataset
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from pipeline.core.context_classifier import ContextClassifier

SEED = 42
random.seed(SEED)

LABELS = [
    "SCIENCE",
    "HEALTH",
    "TECHNOLOGY",
    "HISTORY",
    "POLITICS_GOVERNMENT",
    "ECONOMICS_BUSINESS",
    "GEOGRAPHY",
    "SPACE_ASTRONOMY",
    "ENVIRONMENT_CLIMATE",
    "SOCIETY_CULTURE",
    "LAW_CRIME",
    "SPORTS",
    "ENTERTAINMENT",
    "GENERAL_FACTUAL",
]


@dataclass
class Sample:
    text: str
    label: str
    source: str
    language: str


def _weak_labeler() -> ContextClassifier:
    # model_path None => keyword fallback classifier
    return ContextClassifier(model_path=None)


def _label_text(labeler: ContextClassifier, text: str) -> str:
    l1, _l2, _c1, _c2 = labeler.classify(text)
    return l1 if l1 in LABELS else "GENERAL_FACTUAL"


def _collect_ag_news(labeler: ContextClassifier, max_items: int = 18000) -> List[Sample]:
    ds = load_dataset("ag_news")
    rows = []
    for split in ("train", "test"):
        for item in ds[split]:
            if "text" in item:
                text = (item.get("text") or "").strip()
            else:
                text = f"{item.get('title', '')} {item.get('description', '')}".strip()
            if not text:
                continue
            rows.append(
                Sample(
                    text=text,
                    label=_label_text(labeler, text),
                    source=f"ag_news:{split}",
                    language="en",
                )
            )
            if len(rows) >= max_items:
                return rows
    return rows


def _collect_xnli_hi(labeler: ContextClassifier, max_items: int = 9000) -> List[Sample]:
    ds = load_dataset("xnli", "hi")
    rows = []
    for split in ("train", "validation", "test"):
        for item in ds[split]:
            for text in (item.get("premise", ""), item.get("hypothesis", "")):
                text = (text or "").strip()
                if not text:
                    continue
                rows.append(
                    Sample(
                        text=text,
                        label=_label_text(labeler, text),
                        source=f"xnli_hi:{split}",
                        language="hi",
                    )
                )
                if len(rows) >= max_items:
                    return rows
    return rows


def _collect_indic_nli_variants(labeler: ContextClassifier, max_items: int = 16000) -> List[Sample]:
    rows: List[Sample] = []
    candidates = [
        ("ai4bharat/IndicNLI", None),
        ("Divyanshu/indicxnli", None),
    ]
    for ds_name, ds_conf in candidates:
        try:
            ds = load_dataset(ds_name, ds_conf) if ds_conf else load_dataset(ds_name)
        except Exception:
            continue
        for split in ds.keys():
            for item in ds[split]:
                lang = item.get("language") or item.get("lang") or "indic"
                if lang not in {"hi", "ta", "te", "ml", "kn"}:
                    continue
                for key in ("premise", "hypothesis", "sentence1", "sentence2", "text"):
                    text = (item.get(key) or "").strip()
                    if not text:
                        continue
                    rows.append(
                        Sample(
                            text=text,
                            label=_label_text(labeler, text),
                            source=f"{ds_name}:{split}",
                            language=lang,
                        )
                    )
                    if len(rows) >= max_items:
                        return rows
        if rows:
            break
    return rows


def _balance(samples: List[Sample], per_label_cap: int) -> List[Sample]:
    by_label: Dict[str, List[Sample]] = defaultdict(list)
    for s in samples:
        by_label[s.label].append(s)
    out: List[Sample] = []
    for label in LABELS:
        arr = by_label.get(label, [])
        random.shuffle(arr)
        out.extend(arr[:per_label_cap])
    random.shuffle(out)
    return out


def _split(samples: List[Sample]) -> Tuple[List[Sample], List[Sample], List[Sample]]:
    labels = [s.label for s in samples]
    counts = Counter(labels)
    stratify_labels = labels if all(v >= 3 for v in counts.values()) else None
    train, temp = train_test_split(
        samples, test_size=0.2, random_state=SEED, stratify=stratify_labels
    )
    temp_labels = [s.label for s in temp]
    temp_counts = Counter(temp_labels)
    stratify_temp = temp_labels if all(v >= 2 for v in temp_counts.values()) else None
    val, test = train_test_split(
        temp, test_size=0.5, random_state=SEED, stratify=stratify_temp
    )
    return train, val, test


def _write_jsonl(path: Path, samples: Iterable[Sample]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for s in samples:
            f.write(
                json.dumps(
                    {"text": s.text, "label": s.label, "source": s.source, "language": s.language},
                    ensure_ascii=False,
                )
                + "\n"
            )
            n += 1
    return n


def _counts(samples: List[Sample]) -> Dict[str, int]:
    return dict(sorted(Counter([s.label for s in samples]).items()))


def main() -> None:
    labeler = _weak_labeler()

    en = _collect_ag_news(labeler)
    en = _balance(en, per_label_cap=2000)
    en_train, en_val, en_test = _split(en)

    indic = _collect_indic_nli_variants(labeler)
    if not indic:
        indic = _collect_xnli_hi(labeler)
    indic = _balance(indic, per_label_cap=1200)
    indic_train, indic_val, indic_test = _split(indic)

    en_dir = Path("data/processed/context/en")
    in_dir = Path("data/processed/context/indic")
    _write_jsonl(en_dir / "train.jsonl", en_train)
    _write_jsonl(en_dir / "val.jsonl", en_val)
    _write_jsonl(en_dir / "test.jsonl", en_test)
    _write_jsonl(in_dir / "train.jsonl", indic_train)
    _write_jsonl(in_dir / "val.jsonl", indic_val)
    _write_jsonl(in_dir / "test.jsonl", indic_test)

    build_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "seed": SEED,
        "en_counts": {
            "train": len(en_train),
            "val": len(en_val),
            "test": len(en_test),
            "label_dist_train": _counts(en_train),
        },
        "indic_counts": {
            "train": len(indic_train),
            "val": len(indic_val),
            "test": len(indic_test),
            "label_dist_train": _counts(indic_train),
        },
        "paths": {
            "en": str(en_dir),
            "indic": str(in_dir),
        },
    }

    records_dir = Path("training/records")
    records_dir.mkdir(parents=True, exist_ok=True)
    with open(records_dir / "context_data_builds.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(build_summary, ensure_ascii=False) + "\n")
    with open(records_dir / "context_data_latest.json", "w", encoding="utf-8") as f:
        json.dump(build_summary, f, indent=2, ensure_ascii=False)

    print("Context data prepared")
    print(json.dumps(build_summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
