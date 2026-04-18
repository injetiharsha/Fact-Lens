"""Evaluate a stance checkpoint and save metrics/plots without training."""

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
    classification_report,
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


def _resolve_pair_columns(data_cfg: Dict[str, Any], ds_train_columns: List[str]) -> Tuple[str, str]:
    a_col = data_cfg.get("text_a_column")
    b_col = data_cfg.get("text_b_column")
    if a_col and b_col:
        return str(a_col), str(b_col)
    if "claim" in ds_train_columns and "evidence" in ds_train_columns:
        return "claim", "evidence"
    if "premise" in ds_train_columns and "hypothesis" in ds_train_columns:
        return "premise", "hypothesis"
    raise ValueError("Unable to resolve text pair columns from config/dataset")


def _compute_metrics_from_preds(logits: np.ndarray, refs: np.ndarray) -> Dict[str, float]:
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


def _save_confusion_matrix(
    refs: np.ndarray, preds: np.ndarray, labels: List[str], output_dir: Path, split_name: str
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(refs, preds, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(ax=ax, xticks_rotation=45, colorbar=False)
    plt.title(f"Confusion Matrix ({split_name})")
    plt.tight_layout()
    plt.savefig(plots_dir / f"confusion_matrix_{split_name}.png")
    plt.close(fig)


def _save_per_class_f1(
    refs: np.ndarray, preds: np.ndarray, labels: List[str], output_dir: Path, split_name: str
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    _, _, f1s, _ = precision_recall_fscore_support(
        refs, preds, labels=list(range(len(labels))), average=None, zero_division=0
    )
    plt.figure(figsize=(8, 5))
    plt.bar(labels, f1s)
    plt.ylim(0.0, 1.0)
    plt.ylabel("F1")
    plt.title(f"Per-Class F1 ({split_name})")
    plt.xticks(rotation=25)
    plt.tight_layout()
    plt.savefig(plots_dir / f"per_class_f1_{split_name}.png")
    plt.close()


def _save_loss_curve_from_state(checkpoint_dir: Path, output_dir: Path) -> None:
    state_path = checkpoint_dir / "trainer_state.json"
    if not state_path.exists():
        return
    with state_path.open("r", encoding="utf-8") as f:
        state = json.load(f)
    history = state.get("log_history", [])
    train_x, train_y = [], []
    eval_x, eval_y = [], []
    for row in history:
        step = row.get("step")
        if step is None:
            continue
        if "loss" in row:
            train_x.append(step)
            train_y.append(row["loss"])
        if "eval_loss" in row:
            eval_x.append(step)
            eval_y.append(row["eval_loss"])
    if not train_x and not eval_x:
        return
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    if train_x:
        plt.plot(train_x, train_y, label="train_loss")
    if eval_x:
        plt.plot(eval_x, eval_y, label="eval_loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Loss Curve (from trainer_state)")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "loss_curve_from_trainer_state.png")
    plt.close()


def run(config_path: str, checkpoint_path: str, output_dir: str) -> None:
    cfg = _load_config(config_path)
    labels: List[str] = cfg["labels"]
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}

    dcfg = cfg["data"]
    tcfg = cfg["training"]

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = load_dataset(
        "json",
        data_files={
            "validation": dcfg["validation_file"],
            "test": dcfg["test_file"],
        },
    )
    text_a_col, text_b_col = _resolve_pair_columns(dcfg, ds["validation"].column_names)

    # Some checkpoints may not include full tokenizer assets (e.g., missing vocab/spm files).
    # Try checkpoint tokenizer first, then fall back to base model tokenizer from config.
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
        raise RuntimeError(
            "Failed to load tokenizer from checkpoint/model. "
            + " | ".join(tokenizer_errors)
        )

    def preprocess(batch):
        enc = tokenizer(
            batch[text_a_col],
            batch[text_b_col],
            truncation=True,
            max_length=int(tcfg["max_seq_length"]),
        )
        raw_labels = batch[dcfg["label_column"]]
        enc["labels"] = [label2id[str(x)] if isinstance(x, str) else int(x) for x in raw_labels]
        return enc

    ds = ds.map(preprocess, batched=True, remove_columns=ds["validation"].column_names)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        checkpoint_path,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )
    args = TrainingArguments(
        output_dir=str(out_dir / "tmp_eval"),
        per_device_eval_batch_size=int(tcfg["per_device_eval_batch_size"]),
        dataloader_num_workers=0,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=args,
        tokenizer=tokenizer,
        data_collator=collator,
    )

    eval_loss_metrics = trainer.evaluate(ds["validation"])
    pred_val = trainer.predict(ds["validation"])
    pred_test = trainer.predict(ds["test"])

    val_metrics = _compute_metrics_from_preds(pred_val.predictions, pred_val.label_ids)
    test_metrics = _compute_metrics_from_preds(pred_test.predictions, pred_test.label_ids)
    val_metrics["eval_loss"] = float(eval_loss_metrics.get("eval_loss", float("nan")))

    with (out_dir / "metrics_eval.json").open("w", encoding="utf-8") as f:
        json.dump(val_metrics, f, indent=2)
    with (out_dir / "metrics_test.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    val_preds = np.argmax(pred_val.predictions, axis=-1)
    test_preds = np.argmax(pred_test.predictions, axis=-1)
    _save_confusion_matrix(pred_val.label_ids, val_preds, labels, out_dir, "validation")
    _save_confusion_matrix(pred_test.label_ids, test_preds, labels, out_dir, "test")
    _save_per_class_f1(pred_val.label_ids, val_preds, labels, out_dir, "validation")
    _save_per_class_f1(pred_test.label_ids, test_preds, labels, out_dir, "test")
    _save_loss_curve_from_state(Path(checkpoint_path), out_dir)

    val_report = classification_report(
        pred_val.label_ids, val_preds, target_names=labels, digits=4, zero_division=0
    )
    test_report = classification_report(
        pred_test.label_ids, test_preds, target_names=labels, digits=4, zero_division=0
    )
    with (out_dir / "classification_report_eval.txt").open("w", encoding="utf-8") as f:
        f.write(val_report + "\n")
    with (out_dir / "classification_report_test.txt").open("w", encoding="utf-8") as f:
        f.write(test_report + "\n")

    summary = {
        "config": config_path,
        "checkpoint": checkpoint_path,
        "output_dir": str(out_dir),
        "labels": labels,
        "validation_examples": len(ds["validation"]),
        "test_examples": len(ds["test"]),
        "eval_metrics": val_metrics,
        "test_metrics": test_metrics,
    }
    with (out_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=str)
    parser.add_argument("--checkpoint", required=True, type=str)
    parser.add_argument("--output-dir", required=True, type=str)
    args = parser.parse_args()
    run(args.config, args.checkpoint, args.output_dir)


if __name__ == "__main__":
    main()
