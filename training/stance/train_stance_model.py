"""Config-driven EN stance training (claim + evidence -> support/refute/neutral)."""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import torch
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
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)

# Safer defaults on Windows/laptops under mixed workloads.
# Keep override-friendly: user/system env still wins when already set.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("OMP_NUM_THREADS", "1")


class WeightedLossTrainer(Trainer):
    def __init__(self, *args, class_weights: List[float] | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        self._class_weights = (
            torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        )

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        if self._class_weights is None:
            loss = outputs.get("loss")
        else:
            loss_fct = torch.nn.CrossEntropyLoss(weight=self._class_weights.to(logits.device))
            loss = loss_fct(logits.view(-1, model.config.num_labels), labels.view(-1))
        return (loss, outputs) if return_outputs else loss


def _load_config(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def _check_data_files(cfg: Dict[str, Any]) -> None:
    data_cfg = cfg["data"]
    missing = []
    for key in ("train_file", "validation_file", "test_file"):
        if not Path(data_cfg[key]).exists():
            missing.append(data_cfg[key])
    if missing:
        raise FileNotFoundError("Missing stance dataset files:\n" + "\n".join(missing))


def _resolve_pair_columns(data_cfg: Dict[str, Any], ds_train_columns: List[str]) -> tuple[str, str]:
    a_col = data_cfg.get("text_a_column")
    b_col = data_cfg.get("text_b_column")
    if a_col and b_col:
        return str(a_col), str(b_col)

    # Backward-compatible keys
    legacy_a = data_cfg.get("claim_column")
    legacy_b = data_cfg.get("evidence_column")
    if legacy_a and legacy_b:
        return str(legacy_a), str(legacy_b)

    # Auto-detect common schema names
    if "premise" in ds_train_columns and "hypothesis" in ds_train_columns:
        return "premise", "hypothesis"
    if "claim" in ds_train_columns and "evidence" in ds_train_columns:
        return "claim", "evidence"
    raise ValueError(
        "Could not resolve pair columns. Set data.text_a_column and data.text_b_column "
        "in training config."
    )


def _build_metrics_fn():
    def compute_metrics(eval_pred):
        logits, refs = eval_pred
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

    return compute_metrics


def _save_loss_plot(trainer: Trainer, output_dir: Path) -> None:
    history = trainer.state.log_history
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
    plt.title("Stance Training Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "loss_curve.png")
    plt.close()


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


def run_training(config_path: str) -> None:
    cfg = _load_config(config_path)
    _check_data_files(cfg)

    labels: List[str] = cfg["labels"]
    label2id = {lbl: i for i, lbl in enumerate(labels)}
    id2label = {i: lbl for lbl, i in label2id.items()}

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
    text_a_col, text_b_col = _resolve_pair_columns(dcfg, ds["train"].column_names)

    model_init_path = str(cfg.get("model_init_path", cfg["model_name"]))
    tokenizer = AutoTokenizer.from_pretrained(model_init_path, use_fast=False)

    def preprocess(batch):
        claims = batch[text_a_col]
        evidences = batch[text_b_col]
        enc = tokenizer(
            claims,
            evidences,
            truncation=True,
            max_length=int(tcfg["max_seq_length"]),
        )
        raw_labels = batch[dcfg["label_column"]]
        enc["labels"] = [label2id[str(x)] if isinstance(x, str) else int(x) for x in raw_labels]
        return enc

    ds = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_init_path,
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

    dataloader_num_workers = int(tcfg.get("dataloader_num_workers", 2))
    if os.name == "nt" and dataloader_num_workers > 0:
        print(
            f"[info] Windows detected: forcing dataloader_num_workers=0 "
            f"(was {dataloader_num_workers}) to avoid spawn/pagefile issues."
        )
        dataloader_num_workers = 0

    args = TrainingArguments(
        output_dir=str(output_dir),
        num_train_epochs=float(tcfg["num_train_epochs"]),
        learning_rate=float(tcfg["learning_rate"]),
        weight_decay=float(tcfg["weight_decay"]),
        max_grad_norm=float(tcfg.get("max_grad_norm", 1.0)),
        warmup_ratio=float(tcfg["warmup_ratio"]),
        per_device_train_batch_size=int(tcfg["per_device_train_batch_size"]),
        per_device_eval_batch_size=int(tcfg["per_device_eval_batch_size"]),
        gradient_accumulation_steps=int(tcfg["gradient_accumulation_steps"]),
        fp16=bool(tcfg.get("fp16", False)),
        gradient_checkpointing=bool(tcfg.get("gradient_checkpointing", False)),
        lr_scheduler_type=str(tcfg.get("lr_scheduler_type", "linear")),
        dataloader_num_workers=dataloader_num_workers,
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

    class_weights = tcfg.get("class_weights")
    if class_weights is not None:
        if not isinstance(class_weights, list) or len(class_weights) != len(labels):
            raise ValueError(
                "training.class_weights must be a list with same length as labels. "
                f"Got {class_weights} for labels {labels}"
            )
        class_weights = [float(x) for x in class_weights]
        print(f"[info] Using class-weighted CE loss with label order {labels}: {class_weights}")

    trainer = WeightedLossTrainer(
        model=model,
        args=args,
        train_dataset=ds["train"],
        eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=collator,
        compute_metrics=_build_metrics_fn(),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(scfg["early_stopping_patience"]))],
        class_weights=class_weights,
    )

    resume_from_checkpoint = tcfg.get("resume_from_checkpoint")
    if resume_from_checkpoint:
        print(f"[info] Resuming training from checkpoint: {resume_from_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    trainer.save_model(str(output_dir / "best_model"))
    tokenizer.save_pretrained(str(output_dir / "best_model"))

    eval_metrics = trainer.evaluate(ds["validation"])
    test_metrics = trainer.evaluate(ds["test"], metric_key_prefix="test")
    with open(output_dir / "metrics_eval.json", "w", encoding="utf-8") as f:
        json.dump(eval_metrics, f, indent=2)
    with open(output_dir / "metrics_test.json", "w", encoding="utf-8") as f:
        json.dump(test_metrics, f, indent=2)

    pred_out = trainer.predict(ds["test"])
    preds = np.argmax(pred_out.predictions, axis=-1)
    refs = pred_out.label_ids
    _save_confusion_matrix(refs, preds, labels, output_dir, "test")
    _save_loss_plot(trainer, output_dir)

    report = classification_report(refs, preds, target_names=labels, digits=4, zero_division=0)
    with open(output_dir / "classification_report_test.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")

    run_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "experiment_name": cfg.get("experiment_name"),
        "model_name": cfg.get("model_name"),
        "model_init_path": model_init_path,
        "output_dir": str(output_dir),
        "best_model_dir": str(output_dir / "best_model"),
        "resume_from_checkpoint": resume_from_checkpoint,
        "train_examples": len(ds["train"]),
        "validation_examples": len(ds["validation"]),
        "test_examples": len(ds["test"]),
        "class_weights": class_weights,
        "eval_metrics": eval_metrics,
        "test_metrics": test_metrics,
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    records_dir = Path("training/records")
    records_dir.mkdir(parents=True, exist_ok=True)
    with open(records_dir / "stance_en_training_runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(run_summary, ensure_ascii=False) + "\n")

    print("Stance training complete")
    print(f"Best model: {output_dir / 'best_model'}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (e.g., configs/training/stance_en_fever.yaml)",
    )
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
