# Thesis Scoring: Current vs Research Mode

## Current scoring (heuristic mode)
Implemented in `pipeline/scoring/__init__.py`.

Current evidence weight uses multiplicative heuristics:

`weight = relevance * credibility * temporal_decay * source_tier_multiplier`

With extra heuristic behavior:
- high-relevance credibility uplift
- domain trust map (hard-coded host priors)
- temporal half-life decay by context
- source-tier boost/penalty for host tokens

Why this is strong for product, weak for thesis:
- many interacting priors are hard to isolate in ablation
- multiplicative coupling hides which factor caused final change
- harder statistical explanation in paper methods section

## Research scoring (new)
Enabled by env:
- `SCORING_MODE=research`
- `SCORING_RESEARCH_ALPHA`
- `SCORING_RESEARCH_BETA`
- `SCORING_RESEARCH_GAMMA`

Formula:

`research_weight = alpha*relevance + beta*stance_confidence + gamma*source_trust`

Where:
- `relevance`: relevance model output
- `stance_confidence`: max stance probability (support/refute/neutral)
- `source_trust`: domain credibility signal

Why better for thesis:
- explicit coefficients (`alpha,beta,gamma`) => controlled tuning
- linear decomposition => easy attribution + ablation
- reproducible profile switching across experiments

## How to compare in this repo
1. Run frozen baseline profile:
   - `scripts/run_thesis_experiment.py --profile base_thesis_v1 --split both --num-claims 30`
2. Run research score profile:
   - `scripts/run_thesis_experiment.py --profile research_score_v1 --split both --num-claims 30`
3. Compare:
   - `scripts/compare_thesis_runs.py --a <base_run>/run_summary.json --b <research_run>/run_summary.json`

Use same dataset and same claim order for valid comparison.
