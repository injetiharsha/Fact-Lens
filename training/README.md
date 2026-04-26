# training

Training entrypoints, context trainer package, notebooks, and run records.

## Structure
- `train_context.py`: wrapper entrypoint for context training.
- `context/`: main configurable trainer implementation.
- `records/`: JSON/JSONL audit trail for data prep, rebalance, and training runs.
- `notebooks/`: experimentation notebooks.

## Recent Trainer Behavior
- Step-based evaluation/save supported.
- Initial evaluation at step 0 enabled via config.
- Metrics history and quality plots are exported per run.

## Current Priority: Stance Model

Architecture docs (`project_restructure/*.html`) suggest stance as top model priority.

### Suggested EN Stance Path (from docs)
- Baseline first: benchmark existing `stance_b3` on FEVER dev.
- New model target: `deberta_v3_base_v1`.
- Staged curriculum:
  1. `MNLI` warmup
  2. `FEVER` fine-tune
  3. `VitaminC` robustness fine-tune
- Suggested training config:
  - `lr=2e-5`
  - `batch_size=8`
  - `gradient_accumulation=4`
  - `max_len=512`
  - `fp16=true`
- Primary eval:
  - 3-way accuracy (support/refute/neutral)
  - macro-F1
  - confusion matrix

### Suggested Indic Stance Path (from docs)
- New model target: `mdeberta_v3_base_v1`.
- Data path:
  - `IndicNLI`
  - translated FEVER
- Suggested training config:
  - `lr=2e-5`
  - `batch_size=8`
  - `gradient_accumulation=4`
  - `max_len=256`
  - `fp16=true`
- Primary eval:
  - per-language metrics on `hi/ta/te/kn/ml`
  - macro-F1 and confusion matrices

### Recommended Execution Order
1. Lock EN baseline numbers with existing `stance_b3`.
2. Train EN staged model and compare to baseline.
3. Train Indic stance model and report per-language gaps.
4. Promote only if macro-F1 improves and weakest language does not regress badly.

## Scope Snapshot
- Path: `training/README.md`.
- Purpose: Training workspace guide for model development and run records.
- Audience: Engineers running, extending, or evaluating this module.

