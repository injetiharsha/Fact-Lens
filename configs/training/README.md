# Training Configs

YAML configs used by `training/context/train_context_model.py`.

## Files
- `context_en.yaml`: English context classifier training.
- `context_indic.yaml`: Indic context classifier training (existing Indic data).
- `context_indic_mt.yaml`: Indic context classifier training on translated EN data.

## Strategy Defaults
- `evaluation_strategy: steps`
- `eval_steps: 50`
- `save_strategy: steps`
- `save_steps: 50`
- `logging_first_step: true`
- `eval_on_start: true`

## Scope Snapshot
- Path: `configs/training/README.md`.
- Purpose: Training YAML config set and expected flags.
- Audience: Engineers running, extending, or evaluating this module.

