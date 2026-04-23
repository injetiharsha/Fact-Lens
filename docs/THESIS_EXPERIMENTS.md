# Thesis Experiment Workflow

## 1) Pick profile
Profiles live in `configs/research/thesis_profiles.json`.

- `base_thesis_v1`: frozen baseline for thesis
- `secondary_api_search_v1`: secondary improvement profile
- `research_score_v1`: baseline with research scoring equation

## 2) Run benchmark with profile
Example:

```powershell
.\.venv-gpu\Scripts\python.exe scripts\run_thesis_experiment.py --profile base_thesis_v1 --split both --num-claims 30
```

Outputs:
- `predictions_en.json` / `predictions_multi.json`
- `metrics_en.json` / `metrics_multi.json`
- `run_summary.json`

## 3) Compare two runs

```powershell
.\.venv-gpu\Scripts\python.exe scripts\compare_thesis_runs.py --a <runA>\run_summary.json --b <runB>\run_summary.json
```

## 4) Build larger dataset seed

```powershell
.\.venv-gpu\Scripts\python.exe scripts\build_thesis_dataset_seed.py --target 200
```

Then manually curate:
- true/false/misleading balance
- multilingual spread
- recent + evergreen balance
- optional `expected_evidence_urls`
