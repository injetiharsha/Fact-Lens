"""Evaluate a trained relevance checkpoint on val/test splits."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import yaml
from datasets import load_dataset
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
    roc_auc_score,
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


def _metrics(logits: np.ndarray, refs: np.ndarray) -> Dict[str, float]:
    probs = 1.0 / (1.0 + np.exp(-(logits[:, 1] - logits[:, 0])))
    preds = np.argmax(logits, axis=-1)
    acc = accuracy_score(refs, preds)
    f1 = f1_score(refs, preds, average="binary", zero_division=0)
    p, r, _, _ = precision_recall_fscore_support(refs, preds, average="binary", zero_division=0)
    try:
        auc = roc_auc_score(refs, probs)
    except Exception:
        auc = float("nan")
    try:
        ap = average_precision_score(refs, probs)
    except Exception:
        ap = float("nan")
    return {
        "accuracy": float(acc),
        "f1": float(f1),
        "precision": float(p),
        "recall": float(r),
        "auc": float(auc),
        "average_precision": float(ap),
    }


def _save_cm(refs: np.ndarray, preds: np.ndarray, out_dir: Path, name: str) -> None:
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(refs, preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(cm, display_labels=["not_relevant", "relevant"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, xticks_rotation=20, colorbar=False)
    plt.title(f"Confusion Matrix ({name})")
    plt.tight_layout()
    plt.savefig(plots / f"confusion_matrix_{name}.png")
    plt.close(fig)


def _save_loss_curve(checkpoint_dir: Path, out_dir: Path) -> None:
    state = checkpoint_dir / "trainer_state.json"
    if not state.exists():
        return
    with state.open("r", encoding="utf-8") as f:
        data = json.load(f)
    hist = data.get("log_history", [])
    tx, ty, ex, ey = [], [], [], []
    for row in hist:
        step = row.get("step")
        if step is None:
            continue
        if "loss" in row:
            tx.append(step); ty.append(row["loss"])
        if "eval_loss" in row:
            ex.append(step); ey.append(row["eval_loss"])
    if not tx and not ex:
        return
    plots = out_dir / "plots"
    plots.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    if tx:
        plt.plot(tx, ty, label="train_loss")
    if ex:
        plt.plot(ex, ey, label="eval_loss")
    plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("Loss Curve (from trainer_state)")
    plt.legend(); plt.tight_layout()
    plt.savefig(plots / "loss_curve_from_trainer_state.png")
    plt.close()


def run(config: str, checkpoint: str, out_dir: str) -> None:
    cfg = _load_config(config)
    dcfg = cfg["data"]
    tcfg = cfg["training"]

    ds = load_dataset(
        "json",
        data_files={
            "validation": dcfg["validation_file"],
            "test": dcfg["test_file"],
        },
    )
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)

    def preprocess(batch):
        enc = tokenizer(
            batch[dcfg.get("text_a_column", "claim")],
            batch[dcfg.get("text_b_column", "evidence")],
            truncation=True,
            max_length=int(tcfg["max_seq_length"]),
        )
        enc["labels"] = [int(x) for x in batch[dcfg.get("label_column", "label")]]
        return enc

    cols = ds["validation"].column_names
    ds = ds.map(preprocess, batched=True, remove_columns=cols)

    model = AutoModelForSequenceClassification.from_pretrained(checkpoint)
    args = TrainingArguments(
        output_dir=str(Path(out_dir) / "tmp_eval"),
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

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    eval_loss_raw = trainer.evaluate(ds["validation"])
    p_val = trainer.predict(ds["validation"])
    p_test = trainer.predict(ds["test"])

    m_eval = _metrics(p_val.predictions, p_val.label_ids)
    m_eval["eval_loss"] = float(eval_loss_raw.get("eval_loss", float("nan")))
    m_test = _metrics(p_test.predictions, p_test.label_ids)

    with (out_path / "metrics_eval.json").open("w", encoding="utf-8") as f:
        json.dump(m_eval, f, indent=2)
    with (out_path / "metrics_test.json").open("w", encoding="utf-8") as f:
        json.dump(m_test, f, indent=2)

    pred_eval = np.argmax(p_val.predictions, axis=-1)
    pred_test = np.argmax(p_test.predictions, axis=-1)
    _save_cm(p_val.label_ids, pred_eval, out_path, "validation")
    _save_cm(p_test.label_ids, pred_test, out_path, "test")
    _save_loss_curve(Path(checkpoint), out_path)

    rep = classification_report(
        p_test.label_ids, pred_test, target_names=["not_relevant", "relevant"], digits=4, zero_division=0
    )
    with (out_path / "classification_report_test.txt").open("w", encoding="utf-8") as f:
        f.write(rep + "\n")

    summary = {
        "config": config,
        "checkpoint": checkpoint,
        "output_dir": out_dir,
        "eval_metrics": m_eval,
        "test_metrics": m_test,
    }
    with (out_path / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True)
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--output-dir", required=True)
    args = p.parse_args()
    run(args.config, args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()

