# Context Training

Use config-driven training for EN and Indic context classifiers.

## Expected Data Format

JSONL files with fields:
- `text`: claim text
- `label`: one of configured labels (string) or class index (int)

Paths are set in:
- `configs/training/context_en.yaml`
- `configs/training/context_indic.yaml`
- `configs/training/context_indic_mt.yaml`

## Run

```bash
.venv-gpu\Scripts\python scripts\prepare_context_data.py
.venv-gpu\Scripts\python training\context\train_context_model.py --config configs/training/context_en.yaml
.venv-gpu\Scripts\python training\context\train_context_model.py --config configs/training/context_indic.yaml
.venv-gpu\Scripts\python training\context\train_context_model.py --config configs/training/context_indic_mt.yaml
```

## Saved Artifacts

- Best model: `.../best_model`
- Metrics: `metrics_eval.json`, `metrics_test.json`
- Per-language test metrics (if `lang` column exists): `metrics_test_by_language.json`
- Report: `classification_report_test.txt`
- Plots:
  - `plots/loss_curve.png`
  - `plots/eval_quality_curve.png`
  - `plots/confusion_matrix_test.png`
  - `plots/confusion_matrix_test_<lang>.png` (if multilingual test data has `lang`)

Training policy includes:
- `evaluation_strategy=steps`
- `save_strategy=steps`
- `save_total_limit=2`
- `load_best_model_at_end=true`
- early stopping via validation metric

## Records

Build and training records are appended to:
- `training/records/context_data_builds.jsonl`
- `training/records/context_training_runs.jsonl`

## Scope Snapshot
- Path: `training/context/README.md`.
- Purpose: Context-model training workflow, configs, and outputs.
- Audience: Engineers running, extending, or evaluating this module.

