"""Generate accuracy plots from stance evaluation artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_json(path: Path):
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def generate(eval_dir: Path) -> None:
    plots_dir = eval_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    eval_metrics = _read_json(eval_dir / "metrics_eval.json")
    test_metrics = _read_json(eval_dir / "metrics_test.json")

    # 1) Bar plot: eval vs test accuracy.
    eval_acc = float(eval_metrics.get("accuracy", 0.0))
    test_acc = float(test_metrics.get("accuracy", 0.0))
    plt.figure(figsize=(6, 4))
    plt.bar(["eval", "test"], [eval_acc, test_acc])
    plt.ylim(0.0, 1.0)
    plt.ylabel("Accuracy")
    plt.title("Accuracy (Eval vs Test)")
    plt.tight_layout()
    plt.savefig(plots_dir / "accuracy_eval_test_bar.png")
    plt.close()

    # 2) Curve: eval accuracy over steps if trainer_state exists.
    state_path = eval_dir / "trainer_state.json"
    if not state_path.exists():
        return

    state = _read_json(state_path)
    history = state.get("log_history", [])
    xs = []
    ys = []
    for row in history:
        if "eval_accuracy" in row and "step" in row:
            xs.append(row["step"])
            ys.append(row["eval_accuracy"])
    if not xs:
        return

    plt.figure(figsize=(8, 4.5))
    plt.plot(xs, ys)
    plt.ylim(0.0, 1.0)
    plt.xlabel("Step")
    plt.ylabel("Eval Accuracy")
    plt.title("Eval Accuracy Curve")
    plt.tight_layout()
    plt.savefig(plots_dir / "accuracy_curve_from_trainer_state.png")
    plt.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-dir", required=True, type=str)
    args = parser.parse_args()
    generate(Path(args.eval_dir))


if __name__ == "__main__":
    main()
