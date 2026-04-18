"""Config-driven multilingual relevance training (binary)."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

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
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _check_data_files(cfg: Dict[str, Any]) -> None:
    d = cfg["data"]
    missing = [d[k] for k in ("train_file", "validation_file", "test_file") if not Path(d[k]).exists()]
    if missing:
        raise FileNotFoundError("Missing relevance dataset files:\n" + "\n".join(missing))


def _save_loss_plot(trainer: Trainer, output_dir: Path) -> None:
    history = trainer.state.log_history
    train_x, train_y, eval_x, eval_y = [], [], [], []
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
    p = output_dir / "plots"
    p.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    if train_x:
        plt.plot(train_x, train_y, label="train_loss")
    if eval_x:
        plt.plot(eval_x, eval_y, label="eval_loss")
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Relevance Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(p / "loss_curve.png")
    plt.close()


def _save_confusion_matrix(refs: np.ndarray, preds: np.ndarray, output_dir: Path, split_name: str) -> None:
    p = output_dir / "plots"
    p.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(refs, preds, labels=[0, 1])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["not_relevant", "relevant"])
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, xticks_rotation=20, colorbar=False)
    plt.title(f"Relevance Confusion Matrix ({split_name})")
    plt.tight_layout()
    plt.savefig(p / f"confusion_matrix_{split_name}.png")
    plt.close(fig)


def _build_metrics_fn():
    def compute_metrics(eval_pred):
        logits, refs = eval_pred
        probs = 1.0 / (1.0 + np.exp(-(logits[:, 1] - logits[:, 0])))
        preds = np.argmax(logits, axis=-1)
        acc = accuracy_score(refs, preds)
        f1 = f1_score(refs, preds, average="binary", zero_division=0)
        p, r, _, _ = precision_recall_fscore_support(refs, preds, average="binary", zero_division=0)
        try:
            auc = roc_auc_score(refs, probs)
        except ValueError:
            auc = float("nan")
        try:
            ap = average_precision_score(refs, probs)
        except ValueError:
            ap = float("nan")
        return {
            "accuracy": float(acc),
            "f1": float(f1),
            "precision": float(p),
            "recall": float(r),
            "auc": float(auc),
            "average_precision": float(ap),
        }

    return compute_metrics


def run_training(config_path: str) -> None:
    cfg = _load_config(config_path)
    _check_data_files(cfg)

    labels = ["not_relevant", "relevant"]
    label2id = {"not_relevant": 0, "relevant": 1}
    id2label = {0: "not_relevant", 1: "relevant"}

    tcfg = cfg["training"]
    scfg = cfg["strategy"]
    dcfg = cfg["data"]
    output_dir = Path(tcfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(int(tcfg.get("seed", 42)))

    ds = load_dataset(
        "json",
        data_files={
            "train": dcfg["train_file"],
            "validation": dcfg["validation_file"],
            "test": dcfg["test_file"],
        },
    )
    a_col = dcfg.get("text_a_column", "claim")
    b_col = dcfg.get("text_b_column", "evidence")
    y_col = dcfg.get("label_column", "label")

    model_name = str(cfg["model_name"])
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

    def preprocess(batch):
        enc = tokenizer(
            batch[a_col],
            batch[b_col],
            truncation=True,
            max_length=int(tcfg["max_seq_length"]),
        )
        enc["labels"] = [int(x) for x in batch[y_col]]
        return enc

    ds = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, id2label=id2label, label2id=label2id
    )

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(tcfg["num_train_epochs"]),
        learning_rate=float(tcfg["learning_rate"]),
        weight_decay=float(tcfg["weight_decay"]),
        warmup_ratio=float(tcfg["warmup_ratio"]),
        per_device_train_batch_size=int(tcfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(tcfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(tcfg["gradient_accumulation_steps"]),
        fp16=bool(tcfg.get("fp16", False)),
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", False)),
        lr_scheduler_type=str(tcfg.get("lr_scheduler_type", "linear")),
        dataloader_num_workers=0 if os.name == "nt" else int(tcfg.get("dataloader_num_workers", 2)),
        evaluation_strategy=str(scfg["evaluation_strategy"]),
        save_strategy=str(scfg["save_strategy"]),
        eval_steps=int(scfg.get("eval_steps", scfg.get("logging_steps", 100))),
        save_steps=int(scfg.get("save_steps", scfg.get("logging_steps", 100))),
        logging_strategy=str(scfg.get("logging_strategy", "steps")),
        logging_steps=int(scfg.get("logging_steps", 100)),
        logging_first_step=bool(scfg.get("logging_first_step", True)),
        load_best_model_at_end=bool(scfg["load_best_model_at_end"]),
        metric_for_best_model=str(scfg["metric_for_best_model"]),
        greater_is_better=bool(scfg["greater_is_better"]),
        save_total_limit=int(scfg["save_total_limit"]),
        report_to=["tensorboard"],
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=_build_metrics_fn(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(scfg["early_stopping_patience"]))],
    )

    trainer.train()
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))

    eval_metrics = trainer.evaluate(ds["validation"])
    test_metrics = trainer.evaluate(ds["test"], metric_key_prefix="test")
    with (output_dir / "metrics_eval.json").open("w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)
    with (output_dir / "metrics_test.json").open("w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    pred = trainer.predict(ds["test"])
    preds = np.argmax(pred.predictions, axis=-1)
    refs = pred.label_ids
    _save_confusion_matrix(refs, preds, output_dir, "test")
    _save_loss_plot(trainer, output_dir)

    report = classification_report(refs, preds, target_names=labels, digits=4, zero_division=0)
    with (output_dir / "classification_report_test.txt").open("w", encoding="utf-8") as f:
        f.write(report + "\n")

    run_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "model_name": model_name,
        "output_dir": str(output_dir),
        "best_model_dir": str(output_dir / "best_model"),
        "train_examples": len(ds["train"]),
        "validation_examples": len(ds["validation"]),
        "test_examples": len(ds["test"]),
        "eval_metrics": eval_metrics,
        "test_metrics": test_metrics,
    }
    with (output_dir / "run_summary.json").open("w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    records = Path("training/records")
    records.mkdir(parents=True, exist_ok=True)
    with (records / "relevance_multi_training_runs.jsonl").open("a", encoding="utf-8") as f:
        f.write(json.dumps(run_summary, ensure_ascii=False) + "\n")

    print("Relevance training complete")
    print(f"Best model: {output_dir / 'best_model'}")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, type=str)
    args = p.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()

