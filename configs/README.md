# configs

Primary configuration source for runtime and training.

## Structure
- `pipeline.yaml`: runtime checkpoint and pipeline defaults.
- `domains/`: routing domains, credibility tiers, source/domain controls.
- `scoring/`: diversity/temporal/verdict scoring settings.
- `training/`: context model training configs (EN and Indic variants).

## Recent Updates
- Added step-based training/evaluation defaults.
- Added early metric logging support via strategy flags.

## Scope Snapshot
- Path: `configs/README.md`.
- Purpose: Primary runtime/training configuration index.
- Audience: Engineers running, extending, or evaluating this module.

