"""Generate LLM pre/post verifier analysis tables and plots for Research_Evaluation.

Inputs:
- tests/benchmarks/thesis_runs/20260424_003008_secondary_api_search_v1_both/failed_pre_post_llm_scores.csv
- tests/benchmarks/thesis_runs/20260424_003008_secondary_api_search_v1_both/passed_pre_post_llm_scores.csv

Outputs:
- Research_Evaluation/03_tables_llm_pre_post/*
- Research_Evaluation/04_figures_llm_pre_post/*
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path("Research_Evaluation")
TABLES_OUT = ROOT / "03_tables_llm_pre_post"
FIGS_OUT = ROOT / "04_figures_llm_pre_post"
FAILED_CSV = Path(
    "tests/benchmarks/thesis_runs/20260424_003008_secondary_api_search_v1_both/failed_pre_post_llm_scores.csv"
)
PASSED_CSV = Path(
    "tests/benchmarks/thesis_runs/20260424_003008_secondary_api_search_v1_both/passed_pre_post_llm_scores.csv"
)

VERDICTS = ["support", "refute", "neutral"]
LANG_ORDER = ["en", "hi", "ta", "te", "kn", "ml"]
LANG_DISPLAY = {
    "en": "EN",
    "hi": "HI",
    "ta": "TA",
    "te": "TE",
    "kn": "KN",
    "ml": "ML",
}


def _to_bool(v: object) -> bool:
    if isinstance(v, bool):
        return v
    if v is None:
        return False
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def _ensure_dirs() -> None:
    TABLES_OUT.mkdir(parents=True, exist_ok=True)
    FIGS_OUT.mkdir(parents=True, exist_ok=True)


def _load_rows() -> pd.DataFrame:
    failed = pd.read_csv(FAILED_CSV)
    passed = pd.read_csv(PASSED_CSV)
    df = pd.concat([failed, passed], ignore_index=True)
    df["language"] = df["language"].astype(str).str.lower()
    df["expected"] = df["expected"].astype(str).str.lower()
    df["pre_llm_verdict"] = df["pre_llm_verdict"].astype(str).str.lower()
    df["post_llm_verdict"] = df["post_llm_verdict"].astype(str).str.lower()
    df["llm_triggered"] = df["llm_triggered"].apply(_to_bool)
    df["llm_used"] = df["llm_used"].apply(_to_bool)
    df["pre_llm_confidence"] = pd.to_numeric(df["pre_llm_confidence"], errors="coerce")
    df["post_llm_confidence"] = pd.to_numeric(df["post_llm_confidence"], errors="coerce")
    df = df[df["expected"].isin(VERDICTS)].copy()
    df = df[df["pre_llm_verdict"].isin(VERDICTS)].copy()
    df = df[df["post_llm_verdict"].isin(VERDICTS)].copy()
    df["pre_correct"] = df["pre_llm_verdict"] == df["expected"]
    df["post_correct"] = df["post_llm_verdict"] == df["expected"]
    df["verdict_changed"] = df["pre_llm_verdict"] != df["post_llm_verdict"]
    df["changed_to_neutral"] = df["verdict_changed"] & (df["post_llm_verdict"] == "neutral")
    df["changed_from_neutral"] = df["verdict_changed"] & (df["pre_llm_verdict"] == "neutral")
    return df


def _accuracy_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lg, part in df.groupby("language"):
        total = int(len(part))
        pre_correct = int(part["pre_correct"].sum())
        post_correct = int(part["post_correct"].sum())
        rows.append(
            {
                "language": lg,
                "total": total,
                "pre_correct": pre_correct,
                "post_correct": post_correct,
                "pre_accuracy": pre_correct / total if total else np.nan,
                "post_accuracy": post_correct / total if total else np.nan,
                "delta_accuracy": (post_correct - pre_correct) / total if total else np.nan,
            }
        )
    full_total = int(len(df))
    pre_correct = int(df["pre_correct"].sum())
    post_correct = int(df["post_correct"].sum())
    rows.append(
        {
            "language": "all",
            "total": full_total,
            "pre_correct": pre_correct,
            "post_correct": post_correct,
            "pre_accuracy": pre_correct / full_total if full_total else np.nan,
            "post_accuracy": post_correct / full_total if full_total else np.nan,
            "delta_accuracy": (post_correct - pre_correct) / full_total if full_total else np.nan,
        }
    )
    out = pd.DataFrame(rows)
    order = LANG_ORDER + ["all"]
    out["sort_key"] = out["language"].apply(lambda x: order.index(x) if x in order else 999)
    out = out.sort_values(["sort_key", "language"]).drop(columns=["sort_key"]).reset_index(drop=True)
    return out


def _intervention_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lg, part in df.groupby("language"):
        rows.append(
            {
                "language": lg,
                "total": int(len(part)),
                "llm_triggered": int(part["llm_triggered"].sum()),
                "llm_used": int(part["llm_used"].sum()),
                "verdict_changed": int(part["verdict_changed"].sum()),
                "changed_to_neutral": int(part["changed_to_neutral"].sum()),
                "changed_from_neutral": int(part["changed_from_neutral"].sum()),
                "triggered_but_not_used": int((part["llm_triggered"] & ~part["llm_used"]).sum()),
            }
        )
    rows.append(
        {
            "language": "all",
            "total": int(len(df)),
            "llm_triggered": int(df["llm_triggered"].sum()),
            "llm_used": int(df["llm_used"].sum()),
            "verdict_changed": int(df["verdict_changed"].sum()),
            "changed_to_neutral": int(df["changed_to_neutral"].sum()),
            "changed_from_neutral": int(df["changed_from_neutral"].sum()),
            "triggered_but_not_used": int((df["llm_triggered"] & ~df["llm_used"]).sum()),
        }
    )
    out = pd.DataFrame(rows)
    order = LANG_ORDER + ["all"]
    out["sort_key"] = out["language"].apply(lambda x: order.index(x) if x in order else 999)
    out = out.sort_values(["sort_key", "language"]).drop(columns=["sort_key"]).reset_index(drop=True)
    return out


def _transition_table(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for lg, part in list(df.groupby("language")) + [("all", df)]:
        for exp in VERDICTS:
            for pre in VERDICTS:
                for post in VERDICTS:
                    cnt = int(
                        (
                            (part["expected"] == exp)
                            & (part["pre_llm_verdict"] == pre)
                            & (part["post_llm_verdict"] == post)
                        ).sum()
                    )
                    if cnt:
                        rows.append(
                            {
                                "language": lg,
                                "expected": exp,
                                "pre_verdict": pre,
                                "post_verdict": post,
                                "count": cnt,
                            }
                        )
    return pd.DataFrame(rows)


def _plot_pre_post_accuracy(acc: pd.DataFrame) -> None:
    part = acc[acc["language"] != "all"].copy()
    xs = np.arange(len(part))
    w = 0.36
    fig, ax = plt.subplots(figsize=(11, 6))
    b1 = ax.bar(xs - w / 2, part["pre_accuracy"], w, label="Pre-LLM", color="#457b9d")
    b2 = ax.bar(xs + w / 2, part["post_accuracy"], w, label="Post-LLM", color="#2a9d8f")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(xs)
    ax.set_xticklabels([LANG_DISPLAY.get(x, x.upper()) for x in part["language"]])
    ax.set_ylabel("Accuracy")
    ax.set_title("Pre-LLM vs Post-LLM Accuracy by Language")
    ax.legend()
    for i, row in part.iterrows():
        ax.text(
            xs[i] - w / 2,
            float(row["pre_accuracy"]) + 0.02,
            f'{row["pre_correct"]}/{row["total"]}',
            ha="center",
            fontsize=8,
        )
        ax.text(
            xs[i] + w / 2,
            float(row["post_accuracy"]) + 0.02,
            f'{row["post_correct"]}/{row["total"]}',
            ha="center",
            fontsize=8,
        )
    fig.tight_layout()
    fig.savefig(FIGS_OUT / "pre_post_accuracy_by_language.png", dpi=220)
    plt.close(fig)

    all_row = acc[acc["language"] == "all"].iloc[0]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    vals = [float(all_row["pre_accuracy"]), float(all_row["post_accuracy"])]
    labels = ["Pre-LLM", "Post-LLM"]
    bars = ax.bar(labels, vals, color=["#457b9d", "#2a9d8f"])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Accuracy")
    ax.set_title("Overall Accuracy Gain (All Languages)")
    for i, b in enumerate(bars):
        c = int(all_row["pre_correct"] if i == 0 else all_row["post_correct"])
        t = int(all_row["total"])
        ax.text(b.get_x() + b.get_width() / 2, vals[i] + 0.02, f"{vals[i]:.3f}\n({c}/{t})", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(FIGS_OUT / "pre_post_accuracy_overall.png", dpi=220)
    plt.close(fig)


def _plot_delta(acc: pd.DataFrame) -> None:
    part = acc[acc["language"] != "all"].copy()
    xs = np.arange(len(part))
    vals = part["delta_accuracy"].to_numpy()
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    colors = ["#2a9d8f" if v >= 0 else "#e76f51" for v in vals]
    bars = ax.bar(xs, vals, color=colors)
    ax.axhline(0.0, color="black", linewidth=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([LANG_DISPLAY.get(x, x.upper()) for x in part["language"]])
    ax.set_ylabel("Post - Pre Accuracy")
    ax.set_title("LLM Accuracy Delta by Language")
    for i, b in enumerate(bars):
        ax.text(
            b.get_x() + b.get_width() / 2,
            vals[i] + (0.01 if vals[i] >= 0 else -0.03),
            f"{vals[i]:+.3f}",
            ha="center",
            va="bottom" if vals[i] >= 0 else "top",
            fontsize=9,
        )
    fig.tight_layout()
    fig.savefig(FIGS_OUT / "llm_accuracy_delta_by_language.png", dpi=220)
    plt.close(fig)


def _plot_interventions(inter: pd.DataFrame) -> None:
    part = inter[inter["language"] != "all"].copy()
    xs = np.arange(len(part))
    w = 0.22
    fig, ax = plt.subplots(figsize=(12, 6))
    b1 = ax.bar(xs - w, part["llm_triggered"], width=w, label="Triggered", color="#457b9d")
    b2 = ax.bar(xs, part["llm_used"], width=w, label="Used", color="#2a9d8f")
    b3 = ax.bar(xs + w, part["verdict_changed"], width=w, label="Verdict changed", color="#e9c46a")
    ax.set_xticks(xs)
    ax.set_xticklabels([LANG_DISPLAY.get(x, x.upper()) for x in part["language"]])
    ax.set_ylabel("Count")
    ax.set_title("LLM Intervention Counts by Language")
    ax.legend()
    for bars in (b1, b2, b3):
        for bar in bars:
            v = int(bar.get_height())
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3, str(v), ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGS_OUT / "llm_intervention_counts_by_language.png", dpi=220)
    plt.close(fig)


def _plot_transition_heatmap(df: pd.DataFrame) -> None:
    part = df[df["language"] == "all"].copy()
    mat = np.zeros((3, 3), dtype=int)
    for i, pre in enumerate(VERDICTS):
        for j, post in enumerate(VERDICTS):
            mat[i, j] = int(((part["pre_llm_verdict"] == pre) & (part["post_llm_verdict"] == post)).sum())
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    im = ax.imshow(mat, cmap="Blues")
    ax.set_xticks(range(3))
    ax.set_yticks(range(3))
    ax.set_xticklabels(VERDICTS)
    ax.set_yticklabels(VERDICTS)
    ax.set_xlabel("Post-LLM verdict")
    ax.set_ylabel("Pre-LLM verdict")
    ax.set_title("Verdict Transition Heatmap (All Languages)")
    vmax = max(int(mat.max()), 1)
    for i in range(3):
        for j in range(3):
            v = mat[i, j]
            ax.text(j, i, str(v), ha="center", va="center", color="white" if v > vmax * 0.5 else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(FIGS_OUT / "verdict_transition_heatmap_overall.png", dpi=220)
    plt.close(fig)


def _plot_confidence(df: pd.DataFrame) -> None:
    long_rows = []
    for lg, part in df.groupby("language"):
        long_rows.extend(
            {
                "language": LANG_DISPLAY.get(lg, lg.upper()),
                "phase": "Pre-LLM",
                "confidence": float(v),
            }
            for v in part["pre_llm_confidence"].dropna().tolist()
        )
        long_rows.extend(
            {
                "language": LANG_DISPLAY.get(lg, lg.upper()),
                "phase": "Post-LLM",
                "confidence": float(v),
            }
            for v in part["post_llm_confidence"].dropna().tolist()
        )
    long_df = pd.DataFrame(long_rows)
    fig, ax = plt.subplots(figsize=(12, 6))
    phases = ["Pre-LLM", "Post-LLM"]
    langs = [LANG_DISPLAY.get(l, l.upper()) for l in LANG_ORDER if l in set(df["language"])]
    width = 0.35
    xs = np.arange(len(langs))
    for idx, ph in enumerate(phases):
        vals = []
        for lg in langs:
            vals.append(float(long_df[(long_df["language"] == lg) & (long_df["phase"] == ph)]["confidence"].mean()))
        ax.bar(xs + (-width / 2 if idx == 0 else width / 2), vals, width=width, label=ph)
    ax.set_xticks(xs)
    ax.set_xticklabels(langs)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Mean confidence")
    ax.set_title("Pre-LLM vs Post-LLM Mean Confidence")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIGS_OUT / "pre_post_confidence_mean_by_language.png", dpi=220)
    plt.close(fig)


def _write_summary_json(acc: pd.DataFrame, inter: pd.DataFrame, transition: pd.DataFrame) -> None:
    out = {
        "generated_at_utc": pd.Timestamp.now("UTC").isoformat(),
        "sources": [str(FAILED_CSV), str(PASSED_CSV)],
        "rows_total": int(int(acc.loc[acc["language"] == "all", "total"].iloc[0])),
        "accuracy": acc.to_dict(orient="records"),
        "interventions": inter.to_dict(orient="records"),
        "transition_rows": transition.to_dict(orient="records"),
    }
    with (TABLES_OUT / "llm_pre_post_summary.json").open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)


def _refresh_manifest() -> None:
    files = sorted([str(p).replace("/", "\\") for p in ROOT.rglob("*") if p.is_file()])
    payload = {
        "generated_at": pd.Timestamp.now("UTC").isoformat(),
        "root": "Research_Evaluation",
        "file_count": len(files),
        "files": files,
    }
    with (ROOT / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def main() -> None:
    _ensure_dirs()
    df = _load_rows()
    acc = _accuracy_table(df)
    inter = _intervention_table(df)
    transitions = _transition_table(df)

    acc.to_csv(TABLES_OUT / "pre_vs_post_accuracy_by_language.csv", index=False)
    inter.to_csv(TABLES_OUT / "llm_interventions_by_language.csv", index=False)
    transitions.to_csv(TABLES_OUT / "pre_post_transition_rows.csv", index=False)
    df.to_csv(TABLES_OUT / "pre_post_rows_combined.csv", index=False)
    _write_summary_json(acc, inter, transitions)

    _plot_pre_post_accuracy(acc)
    _plot_delta(acc)
    _plot_interventions(inter)
    _plot_transition_heatmap(df)
    _plot_confidence(df)
    _refresh_manifest()

    print(f"Wrote tables: {TABLES_OUT}")
    print(f"Wrote figures: {FIGS_OUT}")


if __name__ == "__main__":
    main()

