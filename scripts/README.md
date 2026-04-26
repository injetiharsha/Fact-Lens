# scripts

Operational scripts for data prep, balancing, translation, and export.

## Key Scripts Added/Used
- `prepare_context_data.py`: builds initial EN/Indic context datasets.
- `rebalance_context_en.py`: builds EN 14k target distribution.
- `rebalance_context_en_target.py`: builds EN target total (e.g., 10k for 50k multilingual plan).
- `translate_en_to_indic_context.py`: EN -> `te ta ml hi kn` translation with resume checkpoints.
- `test_context_model.py`: quick inference smoke test on trained context model.
- `export_model.py`: export helper for model artifacts.

## Translation Notes
- Only `text` is translated.
- `label` is preserved.
- Resume uses `translation_state.json`.

## Scope Snapshot
- Path: `scripts/README.md`.
- Purpose: Operational script catalog for preparation, evaluation, and exports.
- Audience: Engineers running, extending, or evaluating this module.

