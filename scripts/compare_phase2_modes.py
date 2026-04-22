"""Run MULTI benchmark in Phase2 A/B modes and print metric delta."""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "tests/benchmarks/rfcs_benchmark_multi_full_pipeline/parallel_like_results_multi.json"
OUT3 = ROOT / "tests/benchmarks/rfcs_benchmark_multi_full_pipeline/parallel_like_results_multi_3claims.json"
OUT_CURRENT = ROOT / "tests/benchmarks/rfcs_benchmark_multi_full_pipeline/parallel_like_results_multi_phase2_current.json"
OUT_SCORED = ROOT / "tests/benchmarks/rfcs_benchmark_multi_full_pipeline/parallel_like_results_multi_phase2_scored.json"
OUT3_CURRENT = ROOT / "tests/benchmarks/rfcs_benchmark_multi_full_pipeline/parallel_like_results_multi_3claims_phase2_current.json"
OUT3_SCORED = ROOT / "tests/benchmarks/rfcs_benchmark_multi_full_pipeline/parallel_like_results_multi_3claims_phase2_scored.json"


def _run(mode: str, quick: bool) -> dict:
    env = os.environ.copy()
    env["MULTI_PHASE2_MODE"] = mode
    target = ROOT / ("scripts/run_benchmark_multi_3.py" if quick else "scripts/run_benchmark_multi.py")
    output_file = OUT3 if quick else OUT
    cmd = [str(ROOT / ".venv-gpu/Scripts/python.exe"), str(target)]
    subprocess.run(cmd, cwd=str(ROOT), env=env, check=True)
    data = json.loads(output_file.read_text(encoding="utf-8"))
    if quick:
        dest = OUT3_CURRENT if mode == "current" else OUT3_SCORED
    else:
        dest = OUT_CURRENT if mode == "current" else OUT_SCORED
    shutil.copyfile(output_file, dest)
    return data.get("benchmark_metrics", {})


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare MULTI Phase2 modes")
    parser.add_argument("--full", action="store_true", help="Run full 30-claim benchmark")
    args = parser.parse_args()
    quick = not args.full

    print(f"Running Phase2 mode=current ({'3-claim quick' if quick else '30-claim full'}) ...", flush=True)
    cur = _run("current", quick=quick)
    print(f"Running Phase2 mode=scored ({'3-claim quick' if quick else '30-claim full'}) ...", flush=True)
    scr = _run("scored", quick=quick)

    keys = [
        "accuracy",
        "correct_predictions",
        "neutral_rate",
        "neutral_precision",
        "neutral_recall",
        "neutral_error_rate",
        "false_negative_rate",
        "f1_true_class",
    ]
    print("\nPHASE2 A/B COMPARISON")
    for k in keys:
        a = cur.get(k)
        b = scr.get(k)
        try:
            d = float(b) - float(a)
            print(f"{k}: current={a} scored={b} delta={d:+.3f}")
        except Exception:
            print(f"{k}: current={a} scored={b}")

    if float(scr.get("accuracy", 0.0) or 0.0) >= float(cur.get("accuracy", 0.0) or 0.0):
        print("\nRecommended MULTI_PHASE2_MODE=scored")
    else:
        print("\nRecommended MULTI_PHASE2_MODE=current")


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Benchmark command failed with code {exc.returncode}")
        sys.exit(exc.returncode)
