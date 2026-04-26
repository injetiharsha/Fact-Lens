# Locked EN/MULTI Pipelines

This document is the canonical lock specification for current research/production runs.

## EN Lock

- `CHECKABILITY_EN_PATH=./checkpoints/checkability/multi/checkability_multi_v1/best_model`
- `CONTEXT_EN_PATH=./checkpoints/context/en/context_en_v1/checkpoint-1400`
- `RELEVANCE_EN_PATH=./checkpoints/relevance/en/v9_run1`
- `STANCE_EN_PATH=./checkpoints/stance/en/stance_en_deberta_v1_vitaminc/checkpoint-10000`

## MULTI Lock

- `CHECKABILITY_MULTI_PATH=./checkpoints/checkability/multi/checkability_multi_v1/best_model`
- `CONTEXT_MULTI_PATH=./checkpoints/context/indic/context_indic_mt_v1/checkpoint-6000`
- `RELEVANCE_MULTI_PATH=./checkpoints/relevance/multi/relevance_multi_v1/checkpoint-5000`
- `STANCE_MULTI_PATH=./checkpoints/stance/multi/multi-indic-fever/checkpoint-11000`

## Retrieval + Verifier Lock

- `ENABLE_LLM_VERIFIER=1`
- `WEB_SEARCH_ENABLE_DDG=1`
- `WEB_SEARCH_ENABLE_TAVILY=0`
- `WEB_SEARCH_ENABLE_SERPAPI=0`
- `MULTI_NEUTRAL_TAVILY_BOOST_ENABLE=0`
- `IMAGE_ENABLE_TAVILY_BOOST=0`

Rationale:

- LLM verifier remains enabled for higher empirical end-to-end accuracy.
- Retrieval provider set is intentionally constrained to DDG-only for stable cost/rate behavior.
- No new retrieval provider should be introduced unless a new lock revision is approved.

## Commit Policy

- Lock updates are committed locally first.
- Local-only commit policy applies by default (no automatic push to remote).
- Generated benchmark/report/plot artifacts are not part of lock commits.
