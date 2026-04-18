"""Plot relevance confusion matrices: overall + per-language."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from datasets import load_dataset
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)


def _load_config(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _save_cm(y_true: np.ndarray, y_pred: np.ndarray, out_png: Path, title: str) -> None:
    out_png.parent.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["not_relevant", "relevant"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, xticks_rotation=20, colorbar=False)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)


def _metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_binary": float(f1_score(y_true, y_pred, average="binary", zero_division=0)),
    }


def run(config_path: str, checkpoint: str, test_file: str | None, output_dir: str) -> None:
    cfg = _load_config(config_path)
    dcfg = cfg["data"]
    tcfg = cfg["training"]

    test_jsonl = test_file or dcfg["test_file"]
    raw = load_dataset("json", data_files={"test": test_jsonl})["test"]

    a_col = dcfg.get("text_a_column", "claim")
    b_col = dcfg.get("text_b_column", "evidence")
    y_col = dcfg.get("label_column", "label")
    lang_col = "lang" if "lang" in raw.column_names else None

    labels = np.array([int(x) for x in raw[y_col]])
    langs = np.array([str(x) for x in raw[lang_col]]) if lang_col else None

    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)

    def preprocess(batch):
        enc = tokenizer(
            batch[a_col],
            batch[b_col],
            truncation=True,
            max_length=int(tcfg["max_seq_length"]),
        )
        enc["labels"] = [int(x) for x in batch[y_col]]
        return enc

    tok = raw.map(preprocess, batched=True, remove_columns=raw.column_names)

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    args = TrainingArguments(
        output_dir=str(Path(output_dir) / "tmp_eval"),
        per_device_eval_batch_size=int(tcfg.get("per_device_eval_batch_size", 16)),
        report_to=[],
        dataloader_num_workers=0,
    )
    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
    )

    pred = trainer.predict(tok)
    y_pred = np.argmax(pred.predictions, axis=-1)

    out = Path(output_dir)
    plots = out / "plots"
    plots.mkdir(parents=True, exist_ok=True)

    # Overall
    _save_cm(labels, y_pred, plots / "confusion_matrix_test_overall.png", "Relevance CM (Overall)")
    overall_report = classification_report(
        labels, y_pred, target_names=["not_relevant", "relevant"], digits=4, zero_division=0
    )
    (out / "classification_report_test_overall.txt").write_text(overall_report + "\n", encoding="utf-8")

    metrics = {"overall": _metrics(labels, y_pred)}

    # Per-language
    if langs is not None:
        for lg in sorted(set(langs.tolist())):
            idx = np.where(langs == lg)[0]
            if len(idx) == 0:
                continue
            y_t = labels[idx]
            y_p = y_pred[idx]
            _save_cm(
                y_t,
                y_p,
                plots / f"confusion_matrix_test_{lg}.png",
                f"Relevance CM ({lg})",
            )
            rep = classification_report(
                y_t, y_p, target_names=["not_relevant", "relevant"], digits=4, zero_division=0
            )
            (out / f"classification_report_test_{lg}.txt").write_text(rep + "\n", encoding="utf-8")
            m = _metrics(y_t, y_p)
            m["n_samples"] = int(len(idx))
            metrics[lg] = m

    with (out / "metrics_test_by_language.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print(json.dumps({"output_dir": str(out), "languages": list(metrics.keys())}, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    p.add_argument("--checkpoint", required=True, type=str)
    p.add_argument("--test-file", type=str, default=None)
    p.add_argument("--output-dir", required=True, type=str)
    args = p.parse_args()
    run(args.config, args.checkpoint, args.test_file, args.output_dir)


if __name__ == "__main__":
    main()

