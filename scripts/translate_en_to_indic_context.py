"""Translate EN context dataset to 5 Indic languages with resume checkpoints."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from transformers.data import data_collator as _dc
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

# IndicTransToolkit expects this helper on some transformers versions.
if not hasattr(_dc, "pad_without_fast_tokenizer_warning"):
    def _pad_without_fast_tokenizer_warning(tokenizer, features, **kwargs):
        return tokenizer.pad(features, **kwargs)

    setattr(_dc, "pad_without_fast_tokenizer_warning", _pad_without_fast_tokenizer_warning)

try:
    from IndicTransToolkit import IndicProcessor
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "IndicTransToolkit is required. Install in venv-gpu:\n"
        "  pip install IndicTransToolkit sentencepiece sacremoses\n"
    ) from exc


LANG_MAP = {
    "hi": "hin_Deva",
    "te": "tel_Telu",
    "ta": "tam_Taml",
    "ml": "mal_Mlym",
    "kn": "kan_Knda",
}

SPLITS = ("train", "val", "test")


def _read_jsonl(path: Path) -> List[Dict]:
    rows: List[Dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _append_jsonl(path: Path, rows: Iterable[Dict]) -> None:
    with path.open("a", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _load_state(path: Path) -> Dict[str, int]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _save_state(path: Path, state: Dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)


def _translate_batch(
    texts: List[str],
    src_lang: str,
    tgt_lang: str,
    ip: IndicProcessor,
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    device: str,
    max_length: int,
) -> List[str]:
    pre = ip.preprocess_batch(texts, src_lang=src_lang, tgt_lang=tgt_lang)
    toks = tokenizer(
        pre,
        truncation=True,
        padding="longest",
        return_tensors="pt",
        max_length=max_length,
    )
    toks = {k: v.to(device) for k, v in toks.items()}
    with torch.no_grad():
        gen = model.generate(
            **toks,
            num_beams=5,
            max_length=max_length,
            min_length=0,
        )
    dec = tokenizer.batch_decode(gen, skip_special_tokens=True, clean_up_tokenization_spaces=True)
    return ip.postprocess_batch(dec, lang=tgt_lang)


def run(args: argparse.Namespace) -> None:
    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    out_files = {split: output_dir / f"{split}.jsonl" for split in SPLITS}
    state_file = output_dir / "translation_state.json"
    state = {} if args.reset else _load_state(state_file)

    if args.reset:
        for p in out_files.values():
            if p.exists():
                p.unlink()

    src_lang = "eng_Latn"
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"

    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model_name,
        trust_remote_code=True,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    ).to(device)
    model.eval()
    ip = IndicProcessor(inference=True)

    total_written = 0
    for split in SPLITS:
        src_rows = _read_jsonl(source_dir / f"{split}.jsonl")
        if args.max_rows > 0:
            src_rows = src_rows[: args.max_rows]

        for short_lang in args.langs:
            tgt_lang = LANG_MAP[short_lang]
            key = f"{split}:{short_lang}"
            start_idx = int(state.get(key, 0))

            if start_idx >= len(src_rows):
                continue

            i = start_idx
            while i < len(src_rows):
                chunk = src_rows[i : i + args.batch_size]
                texts = [row["text"] for row in chunk]
                translated = _translate_batch(
                    texts=texts,
                    src_lang=src_lang,
                    tgt_lang=tgt_lang,
                    ip=ip,
                    tokenizer=tokenizer,
                    model=model,
                    device=device,
                    max_length=args.max_length,
                )
                out_rows = []
                for j, row in enumerate(chunk):
                    out_rows.append(
                        {
                            "text": translated[j],
                            "label": row["label"],
                            "lang": short_lang,
                            "source_lang": "en",
                            "source_index": i + j,
                            "split": split,
                        }
                    )

                _append_jsonl(out_files[split], out_rows)
                i += len(chunk)
                state[key] = i
                _save_state(state_file, state)
                total_written += len(out_rows)
                print(f"[{split}][{short_lang}] {i}/{len(src_rows)} translated")

    summary = {
        "source_dir": str(source_dir),
        "output_dir": str(output_dir),
        "langs": args.langs,
        "model_name": args.model_name,
        "batch_size": args.batch_size,
        "max_rows": args.max_rows,
        "total_written_this_run": total_written,
    }
    with (output_dir / "translation_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print("Translation complete")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=str, default="data/processed/context/en_dist14k")
    parser.add_argument("--output-dir", type=str, default="data/processed/context/indic_mt")
    parser.add_argument(
        "--model-name",
        type=str,
        default="ai4bharat/indictrans2-en-indic-dist-200M",
    )
    parser.add_argument("--langs", type=str, nargs="+", default=["te", "ta", "ml", "hi", "kn"])
    parser.add_argument("--batch-size", type=int, default=24)
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument(
        "--max-rows",
        type=int,
        default=-1,
        help="For pilot run, cap rows per split from EN source. -1 means full split.",
    )
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--reset", action="store_true", help="Reset outputs and restart from 0.")
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
