"""Config-driven context classifier training for EN and Indic models."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List

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
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
    set_seed,
)


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
        raise FileNotFoundError(
            "Missing dataset files:\n" + "\n".join(missing) + "\n"
            "Create jsonl files with columns: text,label"
        )


def _build_metrics_fn(id2label: Dict[int, str]):
    labels = list(id2label.keys())

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
    plt.title("Training and Evaluation Loss")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "loss_curve.png")
    plt.close()


def _save_eval_quality_plot(trainer: Trainer, output_dir: Path) -> None:
    history = trainer.state.log_history
    steps, eval_acc, eval_macro_f1, eval_weighted_f1 = [], [], [], []
    for row in history:
        step = row.get("step")
        if step is None:
            continue
        if "eval_accuracy" in row:
            steps.append(step)
            eval_acc.append(row["eval_accuracy"])
            eval_macro_f1.append(row.get("eval_macro_f1"))
            eval_weighted_f1.append(row.get("eval_weighted_f1"))

    if not steps:
        return

    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(9, 5))
    plt.plot(steps, eval_acc, label="eval_accuracy")
    if any(v is not None for v in eval_macro_f1):
        plt.plot(steps, eval_macro_f1, label="eval_macro_f1")
    if any(v is not None for v in eval_weighted_f1):
        plt.plot(steps, eval_weighted_f1, label="eval_weighted_f1")
    plt.xlabel("Step")
    plt.ylabel("Score")
    plt.title("Evaluation Metrics Over Steps")
    plt.legend()
    plt.tight_layout()
    plt.savefig(plots_dir / "eval_quality_curve.png")
    plt.close()


def _save_metric_history(trainer: Trainer, output_dir: Path) -> None:
    history_path = output_dir / "metrics_history.jsonl"
    with open(history_path, "w", encoding="utf-8") as f:
        for row in trainer.state.log_history:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _save_confusion_matrix(
    refs: np.ndarray, preds: np.ndarray, labels: List[str], output_dir: Path, split_name: str
) -> None:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    cm = confusion_matrix(refs, preds, labels=list(range(len(labels))))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    fig, ax = plt.subplots(figsize=(12, 10))
    disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
    plt.title(f"Confusion Matrix ({split_name})")
    plt.tight_layout()
    plt.savefig(plots_dir / f"confusion_matrix_{split_name}.png")
    plt.close(fig)


def _save_confusion_by_language(
    refs: np.ndarray,
    preds: np.ndarray,
    labels: List[str],
    langs: List[str],
    output_dir: Path,
    split_name: str,
) -> Dict[str, Dict[str, float]]:
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    lang_metrics: Dict[str, Dict[str, float]] = {}
    if not langs or len(langs) != len(refs):
        return lang_metrics

    unique_langs = sorted(set([str(x) for x in langs if x is not None and str(x).strip()]))
    for lang in unique_langs:
        mask = np.array([str(x) == lang for x in langs], dtype=bool)
        if mask.sum() == 0:
            continue
        r_lang = refs[mask]
        p_lang = preds[mask]
        cm = confusion_matrix(r_lang, p_lang, labels=list(range(len(labels))))
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
        fig, ax = plt.subplots(figsize=(12, 10))
        disp.plot(ax=ax, xticks_rotation=90, colorbar=False)
        plt.title(f"Confusion Matrix ({split_name}, lang={lang})")
        plt.tight_layout()
        plt.savefig(plots_dir / f"confusion_matrix_{split_name}_{lang}.png")
        plt.close(fig)

        acc = accuracy_score(r_lang, p_lang)
        macro_f1 = f1_score(r_lang, p_lang, average="macro", zero_division=0)
        weighted_f1 = f1_score(r_lang, p_lang, average="weighted", zero_division=0)
        lang_metrics[lang] = {
            "n_samples": int(mask.sum()),
            "accuracy": float(acc),
            "macro_f1": float(macro_f1),
            "weighted_f1": float(weighted_f1),
        }
    return lang_metrics


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
    lang_col = dcfg.get("lang_column")
    test_langs: List[str] = []
    if lang_col and lang_col in ds["test"].column_names:
        test_langs = [str(x) for x in ds["test"][lang_col]]

    # Some models (e.g., DeBERTa-v3 / MuRIL) may require sentencepiece and fail
    # fast-tokenizer conversion on some environments. Use slow tokenizer for stability.
    tokenizer = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=False)

    def preprocess(batch):
        texts = batch[dcfg["text_column"]]
        enc = tokenizer(
            texts,
            truncation=True,
            max_length=int(tcfg["max_seq_length"]),
        )
        raw_labels = batch[dcfg["label_column"]]
        enc["labels"] = [label2id[x] if isinstance(x, str) else int(x) for x in raw_labels]
        return enc

    ds = ds.map(preprocess, batched=True, remove_columns=ds["train"].column_names)
    collator = DataCollatorWithPadding(tokenizer=tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        cfg["model_name"],
        num_labels=len(labels),
        id2label=id2label,
        label2id=label2id,
    )

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
        dataloader_num_workers=int(tcfg.get("dataloader_num_workers", 2)),
        evaluation_strategy=str(scfg["evaluation_strategy"]),
        save_strategy=str(scfg["save_strategy"]),
        eval_steps=int(scfg.get("eval_steps", scfg.get("logging_steps", 50))),
        save_steps=int(scfg.get("save_steps", scfg.get("logging_steps", 50))),
        logging_strategy=str(scfg.get("logging_strategy", "steps")),
        logging_steps=int(scfg.get("logging_steps", 50)),
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
        compute_metrics=_build_metrics_fn(id2label),
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(scfg["early_stopping_patience"]))],
    )

    if bool(scfg.get("eval_on_start", True)):
        init_eval = trainer.evaluate(ds["validation"], metric_key_prefix="eval")
        init_row = {
            "step": 0,
            "epoch": 0.0,
            "eval_loss": init_eval.get("eval_loss"),
            "eval_accuracy": init_eval.get("eval_accuracy"),
            "eval_macro_f1": init_eval.get("eval_macro_f1"),
            "eval_weighted_f1": init_eval.get("eval_weighted_f1"),
            "eval_macro_precision": init_eval.get("eval_macro_precision"),
            "eval_macro_recall": init_eval.get("eval_macro_recall"),
        }
        trainer.state.log_history.append(init_row)

    trainer.train()
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
    lang_metrics = _save_confusion_by_language(
        refs=refs,
        preds=preds,
        labels=labels,
        langs=test_langs,
        output_dir=output_dir,
        split_name="test",
    )
    _save_loss_plot(trainer, output_dir)
    _save_eval_quality_plot(trainer, output_dir)
    _save_metric_history(trainer, output_dir)

    report = classification_report(
        refs, preds, target_names=labels, digits=4, zero_division=0
    )
    with open(output_dir / "classification_report_test.txt", "w", encoding="utf-8") as f:
        f.write(report + "\n")
    if lang_metrics:
        with open(output_dir / "metrics_test_by_language.json", "w", encoding="utf-8") as f:
            json.dump(lang_metrics, f, indent=2, ensure_ascii=False)

    run_summary = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "config_path": str(config_path),
        "experiment_name": cfg.get("experiment_name"),
        "model_name": cfg.get("model_name"),
        "output_dir": str(output_dir),
        "best_model_dir": str(output_dir / "best_model"),
        "train_examples": len(ds["train"]),
        "validation_examples": len(ds["validation"]),
        "test_examples": len(ds["test"]),
        "eval_metrics": eval_metrics,
        "test_metrics": test_metrics,
        "test_metrics_by_language": lang_metrics,
        "artifacts": {
            "metrics_eval": str(output_dir / "metrics_eval.json"),
            "metrics_test": str(output_dir / "metrics_test.json"),
            "classification_report_test": str(output_dir / "classification_report_test.txt"),
            "loss_plot": str(output_dir / "plots" / "loss_curve.png"),
            "eval_quality_plot": str(output_dir / "plots" / "eval_quality_curve.png"),
            "confusion_matrix_test": str(output_dir / "plots" / "confusion_matrix_test.png"),
            "metrics_test_by_language": str(output_dir / "metrics_test_by_language.json"),
            "metrics_history": str(output_dir / "metrics_history.jsonl"),
        },
    }
    with open(output_dir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump(run_summary, f, indent=2)

    records_dir = Path("training/records")
    records_dir.mkdir(parents=True, exist_ok=True)
    with open(records_dir / "context_training_runs.jsonl", "a", encoding="utf-8") as f:
        f.write(json.dumps(run_summary, ensure_ascii=False) + "\n")

    print("Training complete")
    print(f"Best model: {output_dir / 'best_model'}")
    print(f"Saved artifacts: metrics, report, plots/confusion matrix")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to YAML config (e.g., configs/training/context_en.yaml)",
    )
    args = parser.parse_args()
    run_training(args.config)


if __name__ == "__main__":
    main()
