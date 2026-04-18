"""Evaluate stance checkpoint by language and save per-language metrics/plots."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import yaml
from datasets import load_dataset
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _resolve_pair_columns(data_cfg: Dict[str, Any], ds_columns: List[str]) -> Tuple[str, str]:
    a_col = data_cfg.get("text_a_column")
    b_col = data_cfg.get("text_b_column")
    if a_col and b_col:
        return str(a_col), str(b_col)
    if "claim" in ds_columns and "evidence" in ds_columns:
        return "claim", "evidence"
    if "premise" in ds_columns and "hypothesis" in ds_columns:
        return "premise", "hypothesis"
    raise ValueError("Unable to resolve text pair columns from config/dataset")


def _compute_metrics(logits: np.ndarray, refs: np.ndarray) -> Dict[str, float]:
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(refs, preds)
    macro_f1 = f1_score(refs, preds, average="macro")
    weighted_f1 = f1_score(refs, preds, average="weighted")
    precision, recall, _, _ = precision_recall_fscore_support(
        refs, preds, average="macro", zero_division=0
    )
    return {
        "accuracy": float(acc),
        "macro_f1": float(macro_f1),
        "weighted_f1": float(weighted_f1),
        "macro_precision": float(precision),
        "macro_recall": float(recall),
    }


def _save_cm(refs: np.ndarray, preds: np.ndarray, labels: List[str], out_png: Path, title: str) -> None:
    cm = confusion_matrix(refs, preds, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title(title)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.close(fig)


def run(config_path: str, checkpoint_path: str, output_dir: str, split_mode: str = "both") -> None:
    cfg = _load_config(config_path)
    dcfg = cfg["data"]
    tcfg = cfg["training"]
    labels: List[str] = cfg["labels"]
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots_by_language"
    plots_dir.mkdir(parents=True, exist_ok=True)

    ds_raw = load_dataset(
        "json",
        data_files={
            "validation": dcfg["validation_file"],
            "test": dcfg["test_file"],
        },
    )
    text_a_col, text_b_col = _resolve_pair_columns(dcfg, ds_raw["validation"].column_names)
    label_col = dcfg["label_column"]

    tokenizer = None
    tokenizer_errors: List[str] = []
    for source in (checkpoint_path, cfg.get("model_name")):
        if not source:
            continue
        try:
            tokenizer = AutoTokenizer.from_pretrained(source)
            break
        except Exception as exc:
            tokenizer_errors.append(f"{source}: {exc}")
    if tokenizer is None:
        raise RuntimeError("Failed to load tokenizer: " + " | ".join(tokenizer_errors))

    def _to_label_id(x: Any) -> int:
        if isinstance(x, list):
            if not x:
                return 2
            x = x[0]
        if isinstance(x, str):
            return label2id.get(x, int(x) if x.isdigit() else 2)
        return int(x)

    def preprocess(batch):
        enc = tokenizer(
            batch[text_a_col],
            batch[text_b_col],
            truncation=True,
            max_length=int(tcfg.get("max_seq_length", 512)),
        )
        raw_labels = batch[label_col]
        enc["labels"] = [_to_label_id(x) for x in raw_labels]
        return enc

    ds = ds_raw.map(
        preprocess,
        batched=True,
        remove_columns=ds_raw["validation"].column_names,
    )
    collator = DataCollatorWithPadding(tokenizer=tokenizer)
    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    args = TrainingArguments(
        output_dir=str(out_dir / "tmp_eval_lang"),
        per_device_eval_batch_size=int(tcfg.get("per_device_eval_batch_size", 16)),
        dataloader_num_workers=0,
        report_to=[],
    )
    trainer = Trainer(model=model, args=args, tokenizer=tokenizer, data_collator=collator)

    result: Dict[str, Any] = {"config": config_path, "checkpoint": checkpoint_path, "labels": labels}

    if split_mode == "validation":
        splits = ("validation",)
    elif split_mode == "test":
        splits = ("test",)
    else:
        splits = ("validation", "test")

    for split in splits:
        split_raw = ds_raw[split]
        split_tok = ds[split]
        langs = sorted({str(x) for x in split_raw["lang"]}) if "lang" in split_raw.column_names else ["all"]
        split_out: Dict[str, Any] = {}

        for lang in langs:
            if lang == "all":
                idxs = list(range(len(split_raw)))
            else:
                idxs = [i for i, x in enumerate(split_raw["lang"]) if str(x) == lang]
            if not idxs:
                continue

            subset = split_tok.select(idxs)
            pred = trainer.predict(subset)
            refs = pred.label_ids
            logits = pred.predictions
            preds = np.argmax(logits, axis=-1)

            m = _compute_metrics(logits, refs)
            m["rows"] = len(idxs)
            split_out[lang] = m

            _save_cm(
                refs=refs,
                preds=preds,
                labels=labels,
                out_png=plots_dir / f"confusion_matrix_{split}_{lang}.png",
                title=f"Confusion Matrix ({split} - {lang})",
            )

        result[split] = split_out

    out_json = out_dir / "metrics_by_language.json"
    out_json.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps({"output_json": str(out_json), "splits": list(result.keys())}, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    parser.add_argument(
        "--split",
        choices=["both", "validation", "test"],
        default="both",
        help="Evaluate by language on validation/test/both.",
    )
    args = parser.parse_args()
    run(args.config, args.checkpoint, args.output_dir, split_mode=args.split)


if __name__ == "__main__":
    main()
