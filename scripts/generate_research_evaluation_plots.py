"""Generate paper-ready plots from Research_Evaluation canonical tables.

Outputs are written to Research_Evaluation/04_figures.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("Research_Evaluation")
TABLES = ROOT / "03_tables"
FIGS = ROOT / "04_figures"


LABELS = ["support", "refute", "neutral"]
LANG_ORDER = ["en", "hi", "ta", "te", "kn", "ml"]
LANG_DISPLAY = {
    "en": "EN",
    "hi": "HI",
    "ta": "TA",
    "te": "TE",
    "kn": "KN",
    "ml": "ML",
}


def _load_json(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _cm_from_obj(obj: Dict[str, Dict[str, int]]) -> np.ndarray:
    return np.array([[obj[r][c] for c in LABELS] for r in LABELS], dtype=int)


def _plot_cm(cm: np.ndarray, title: str, out: Path, *, annotate: bool = True) -> None:
    fig, ax = plt.subplots(figsize=(6.6, 5.6))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(LABELS)))
    ax.set_yticks(range(len(LABELS)))
    ax.set_xticklabels(LABELS, rotation=30, ha="right")
    ax.set_yticklabels(LABELS)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Expected")
    ax.set_title(title)
    if annotate:
        vmax = max(cm.max(), 1)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                v = cm[i, j]
                ax.text(
                    j,
                    i,
                    str(v),
                    ha="center",
                    va="center",
                    color="white" if v > vmax * 0.5 else "black",
                    fontsize=10,
                )
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=220)
    plt.close(fig)


def _plot_language_grid(lang_payload: Dict[str, dict], out: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.ravel()
    for idx, lg in enumerate(LANG_ORDER):
        cm = _cm_from_obj(lang_payload[lg]["confusion_matrix"])
        ax = axes[idx]
        im = ax.imshow(cm, cmap="Blues")
        ax.set_title(LANG_DISPLAY[lg])
        ax.set_xticks(range(len(LABELS)))
        ax.set_yticks(range(len(LABELS)))
        ax.set_xticklabels(LABELS, rotation=25, ha="right", fontsize=9)
        ax.set_yticklabels(LABELS, fontsize=9)
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                v = cm[i, j]
                vmax = max(cm.max(), 1)
                ax.text(
                    j,
                    i,
                    str(v),
                    ha="center",
                    va="center",
                    color="white" if v > vmax * 0.5 else "black",
                    fontsize=9,
                )
    fig.subplots_adjust(right=0.9, wspace=0.34, hspace=0.38)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.02, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    fig.suptitle("Confusion Matrix Grid (EN + 5 Indic)", fontsize=14)
    fig.savefig(out, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_accuracy_bars(metrics_df: pd.DataFrame, out_acc: Path, out_checkable: Path, out_neutral: Path) -> None:
    subset = metrics_df.loc[metrics_df["file"].str.contains("v2_from_252")].copy()
    keep = [
        "parallel_like_results_en_v2_from_252_scrape_upgrade_v1.json",
        "parallel_like_results_hi_v2_from_252_rerun.json",
        "parallel_like_results_ta_v2_from_252.json",
        "parallel_like_results_te_v2_from_252.json",
        "parallel_like_results_kn_v2_from_252_nock.json",
        "parallel_like_results_ml_v2_from_252_nock.json",
        "parallel_like_results_multi_v2_from_252_5lang.json",
    ]
    subset = subset.loc[subset["file"].isin(keep)].copy()
    name_map = {
        "parallel_like_results_en_v2_from_252_scrape_upgrade_v1.json": "EN",
        "parallel_like_results_hi_v2_from_252_rerun.json": "HI",
        "parallel_like_results_ta_v2_from_252.json": "TA",
        "parallel_like_results_te_v2_from_252.json": "TE",
        "parallel_like_results_kn_v2_from_252_nock.json": "KN",
        "parallel_like_results_ml_v2_from_252_nock.json": "ML",
        "parallel_like_results_multi_v2_from_252_5lang.json": "MULTI-175",
    }
    subset["run"] = subset["file"].map(name_map)
    subset["accuracy"] = pd.to_numeric(subset["accuracy"], errors="coerce")
    subset["neutral_rate"] = pd.to_numeric(subset["neutral_rate"], errors="coerce")
    subset["accuracy_checkable_only"] = pd.to_numeric(
        subset.get("accuracy_checkable_only"), errors="coerce"
    )

    order = ["EN", "HI", "TA", "TE", "KN", "ML", "MULTI-175"]
    subset = subset.set_index("run").reindex(order).reset_index()

    fig, ax = plt.subplots(figsize=(10, 5))
    vals = subset["accuracy"].fillna(0).to_numpy()
    bars = ax.bar(subset["run"], vals, color="#2d6a4f")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Accuracy")
    ax.set_title("Accuracy by Run")
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_acc, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    cvals = subset["accuracy_checkable_only"].fillna(np.nan).to_numpy()
    bars = ax.bar(subset["run"], np.nan_to_num(cvals, nan=0.0), color="#1d3557")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Checkable-only Accuracy")
    ax.set_title("Checkable-only Accuracy by Run")
    for b, v in zip(bars, cvals):
        if not np.isnan(v):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
        else:
            ax.text(b.get_x() + b.get_width() / 2, 0.03, "N/A", ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_checkable, dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 5))
    nvals = subset["neutral_rate"].fillna(0).to_numpy()
    bars = ax.bar(subset["run"], nvals, color="#6a4c93")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Neutral Rate")
    ax.set_title("Neutral Rate by Run")
    for b, v in zip(bars, nvals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.3f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_neutral, dpi=220)
    plt.close(fig)


def _plot_distributions(vcounts: dict, out_expected: Path, out_pred: Path) -> None:
    combined = vcounts["combined"]
    exp = [combined["expected_counts"][k] for k in LABELS]
    pred = [combined["predicted_counts"][k] for k in LABELS]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(LABELS, exp, color="#457b9d")
    ax.set_title("Expected Verdict Distribution (Combined 252)")
    ax.set_ylabel("Count")
    for b, v in zip(bars, exp):
        ax.text(b.get_x() + b.get_width() / 2, v + 1, str(v), ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_expected, dpi=220)
    plt.close(fig)

    langs = LANG_ORDER
    data = np.array([[vcounts["languages"][lg]["predicted_counts"][k] for lg in langs] for k in LABELS])
    fig, ax = plt.subplots(figsize=(10, 6))
    bottom = np.zeros(len(langs))
    colors = {"support": "#2a9d8f", "refute": "#e76f51", "neutral": "#8d99ae"}
    for i, label in enumerate(LABELS):
        vals = data[i]
        ax.bar([LANG_DISPLAY[x] for x in langs], vals, bottom=bottom, label=label, color=colors[label])
        bottom += vals
    ax.set_title("Predicted Verdict Distribution by Language")
    ax.set_ylabel("Count")
    ax.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(out_pred, dpi=220)
    plt.close(fig)


def _refresh_manifest() -> None:
    files = sorted([str(p).replace("/", "\\") for p in ROOT.rglob("*") if p.is_file()])
    payload = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "root": "Research_Evaluation",
        "file_count": len(files),
        "files": files,
    }
    out = ROOT / "manifest.json"
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    FIGS.mkdir(parents=True, exist_ok=True)

    vcounts = _load_json(TABLES / "verdict_counts_6lang_with_combined.json")
    cm_table = pd.read_csv(TABLES / "whole_252_confusion_matrix.csv")
    metrics_df = pd.read_csv(TABLES / "metrics_summary.csv")

    # Per-language and combined confusion matrices.
    for lg in LANG_ORDER:
        cm = _cm_from_obj(vcounts["languages"][lg]["confusion_matrix"])
        _plot_cm(cm, f"Confusion Matrix ({LANG_DISPLAY[lg]})", FIGS / f"confusion_matrix_{lg}.png")

    cm_combined = _cm_from_obj(vcounts["combined"]["confusion_matrix"])
    _plot_cm(cm_combined, "Confusion Matrix (Combined 252)", FIGS / "confusion_matrix_combined_all_6.png")
    _plot_language_grid(vcounts["languages"], FIGS / "confusion_matrix_all_6_grid.png")

    # Whole confusion (3-way) from CSV for consistency.
    cml = LABELS
    matrix = np.array(
        [
            [
                int(
                    cm_table.loc[
                        (cm_table["expected"] == r) & (cm_table["predicted"] == c),
                        "count",
                    ].iloc[0]
                )
                for c in cml
            ]
            for r in cml
        ],
        dtype=int,
    )
    _plot_cm(matrix, "Whole 252 Confusion Matrix (3-way)", FIGS / "whole_252_confusion_3way.png")

    # Accuracy and rate plots.
    _plot_accuracy_bars(
        metrics_df,
        FIGS / "accuracy_by_language_and_combined.png",
        FIGS / "checkable_accuracy_by_language_and_combined.png",
        FIGS / "neutral_rate_by_language_and_combined.png",
    )

    # Verdict distributions.
    _plot_distributions(
        vcounts,
        FIGS / "dist_expected_verdict_252.png",
        FIGS / "predicted_distribution_by_language_stacked.png",
    )

    _refresh_manifest()
    print(f"Figures generated in: {FIGS}")


if __name__ == "__main__":
    main()
