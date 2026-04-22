"""Quick MULTI benchmark smoke run for 3 claims."""

from __future__ import annotations

from pathlib import Path

from dotenv import load_dotenv

from run_benchmark_multi import (
    DEFAULT_CASES,
    evaluate,
    run_benchmark,
    save_results,
    summarize_stage_timings,
    load_claim_batch,
)


def main() -> None:
    load_dotenv(override=True)
    rows = load_claim_batch(Path(DEFAULT_CASES), 3)
    results, claim_times, total_time = run_benchmark(rows)
    metrics = evaluate(results)

    print("\n==============================")
    print("BENCHMARK METRICS (3-claim smoke)")
    print("==============================")
    for key in [
        "total_claims",
        "correct_predictions",
        "accuracy",
        "neutral_rate",
        "neutral_precision",
        "neutral_recall",
        "neutral_error_rate",
        "tp",
        "tn",
        "fp",
        "fn",
        "false_positive_rate",
        "false_negative_rate",
        "precision_true_class",
        "recall_true_class",
        "f1_true_class",
    ]:
        print(f"{key} : {metrics.get(key)}")

    print("failed_by_expected_verdict :", metrics.get("failed_by_expected_verdict"))
    print("failed_by_predicted_verdict :", metrics.get("failed_by_predicted_verdict"))
    print("failed_by_language :", metrics.get("failed_by_language"))
    stage_summary = summarize_stage_timings(results)
    print("dominant_stage :", stage_summary.get("dominant_stage"))
    print("dominant_stage_seconds :", stage_summary.get("dominant_stage_seconds"))
    print("model_locked_total_seconds :", stage_summary.get("model_locked_total_seconds"))

    output = Path("tests/benchmarks/rfcs_benchmark_multi_full_pipeline/parallel_like_results_multi_3claims.json")
    save_results(results, metrics, claim_times, total_time, output)


if __name__ == "__main__":
    main()
