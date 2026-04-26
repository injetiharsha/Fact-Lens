# Model Inventory And 3-Month Delivery Timeline

This document is the detailed reference for:
- models currently used in locked runtime,
- models trained/fine-tuned in this repository,
- dataset/config lineage for those models,
- project progression from scratch to current final stage in 12 stages across ~3 months.

Use this with:
- `docs/LOCKED_PIPELINES.md`
- `docs/PROJECT_FULL_EXPLANATION.md`
- `Research_Evaluation/` (canonical metrics/plots/results)

## 1) Current Locked Runtime Model Stack

Source of truth: `.env` lock block and `docs/LOCKED_PIPELINES.md`.

### EN pipeline (locked)
- Checkability: `./checkpoints/checkability/multi/checkability_multi_v1/best_model`
- Context: `./checkpoints/context/en/context_en_v1/checkpoint-1400`
- Relevance: `./checkpoints/relevance/en/v9_run1`
- Stance: `./checkpoints/stance/en/stance_en_deberta_v1_vitaminc/checkpoint-10000`

### MULTI pipeline (locked)
- Checkability: `./checkpoints/checkability/multi/checkability_multi_v1/best_model`
- Context: `./checkpoints/context/indic/context_indic_mt_v1/checkpoint-6000`
- Relevance: `./checkpoints/relevance/multi/relevance_multi_v1/checkpoint-5000`
- Stance: `./checkpoints/stance/multi/multi-indic-fever/checkpoint-11000`

### Runtime support models/services
- LLM verifier: enabled (`ENABLE_LLM_VERIFIER=1`)
  - current provider rotation in `.env`: Fireworks + Cerebras
  - model target in lock runs: `accounts/fireworks/models/gpt-oss-120b`
- Translation LLM: Groq (`TRANSLATION_LLM_PROVIDER=groq`, `openai/gpt-oss-20b`)
- Retrieval bi-encoder (runtime ranking path): `intfloat/multilingual-e5-small`
- OCR: Tesseract-based ingestion path for image claims

## 2) Trained / Fine-Tuned Models And Config Lineage

## 2.1 Checkability (multilingual, 5-class)
- Config: `configs/training/checkability_multi.yaml`
- Base model: `xlm-roberta-base`
- Output: `checkpoints/checkability/multi/checkability_multi_v1`
- Runtime use: EN + MULTI
- Dataset files:
  - `data/processed/checkability/multilingual/train.jsonl`
  - `data/processed/checkability/multilingual/val.jsonl`
  - `data/processed/checkability/multilingual/test.jsonl`
- Labels:
  - `FACTUAL_CLAIM`
  - `PERSONAL_STATEMENT`
  - `OPINION`
  - `QUESTION_OR_REWRITE`
  - `OTHER_UNCHECKABLE`
- Training record file: `training/records/checkability_training_runs.jsonl`

## 2.2 Context EN (14-class)
- Config: `configs/training/context_en.yaml`
- Base model: `microsoft/deberta-v3-base`
- Output: `checkpoints/context/en/context_en_v1`
- Runtime use: EN path
- Dataset: `data/processed/context/en_dist14k/*`
- Training records: `training/records/context_training_runs.jsonl`

## 2.3 Context Indic MT (14-class multilingual)
- Config: `configs/training/context_indic_mt.yaml`
- Base model: `google/muril-base-cased`
- Output: `checkpoints/context/indic/context_indic_mt_v1`
- Runtime use: MULTI path
- Dataset: `data/processed/context/indic_mt_50k/*`
- Per-language eval snapshots are written in context training artifacts.

## 2.4 Relevance MULTI (binary)
- Config: `configs/training/relevance_multi.yaml`
- Base model: `xlm-roberta-base`
- Output: `checkpoints/relevance/multi/relevance_multi_v1`
- Runtime use: MULTI path
- Dataset: `data/processed/relevance/multilingual/*`
- Training/eval records:
  - `training/records/relevance_multi_training_runs.jsonl`

## 2.5 Relevance EN (serving checkpoint lineage)
- Runtime checkpoint: `checkpoints/relevance/en/v9_run1`
- This EN checkpoint is used in locked serving and appears in EN relevance eval records:
  - `training/records/relevance_en_fever_eval_runs.jsonl`
  - `training/records/relevance_en_single_stage_threshold_sweep_runs.jsonl`

## 2.6 Stance EN (staged fine-tune chain)
- Stage A config: `configs/training/stance_en_mnli.yaml`
  - Base/init: `microsoft/deberta-v3-base`
  - Output: `checkpoints/stance/en/stance_en_deberta_v1_mnli`
- Stage B config: `configs/training/stance_en_fever.yaml`
  - Init from Stage A checkpoint (`...mnli/checkpoint-2000`)
  - Output: `checkpoints/stance/en/stance_en_deberta_v1_fever`
- Stage C config: `configs/training/stance_en_vitaminc.yaml`
  - Init from FEVER stage checkpoint
  - Output: `checkpoints/stance/en/stance_en_deberta_v1_vitaminc`
- Locked runtime pick: `...vitaminc/checkpoint-10000`
- Training records: `training/records/stance_en_training_runs.jsonl`

## 2.7 Stance MULTI (Indic FEVER-style eval/train path)
- Config: `configs/training/stance_multi_indic_eval.yaml`
- Base model: `microsoft/mdeberta-v3-base`
- Output root: `checkpoints/stance/multi/multi-indic-fever`
- Locked runtime pick: `.../checkpoint-11000`

## 3) EN vs MULTI Runtime Routing Summary

- Claim enters same pipeline stages, then checkpoint routing selects EN or MULTI paths.
- EN and MULTI have separate context/relevance/stance checkpoints.
- Checkability checkpoint is shared across EN/MULTI in current lock.
- Translation/query augmentation is applied for non-English retrieval paths.
- LLM verifier remains enabled in lock profile and participates in verdict stabilization.

## 4) 3-Month Project Timeline (12 Stages)

This is the end-to-end progression from scratch to current locked production/research state.

### Month 1: Foundation And Baselines
1. Stage 1 (Week 1): Repository scaffold + API + initial pipeline skeleton
   - Basic claim flow, stage wiring, and environment configuration added.
2. Stage 2 (Week 1-2): Input ingestion foundation
   - Text/image/PDF entry paths, OCR integration, normalization and parsing.
3. Stage 3 (Week 2): Evidence retrieval v1
   - Structured + web + scrape adapters integrated with fallback behavior.
4. Stage 4 (Week 2-3): Initial scoring/verdict path
   - Relevance and stance hooks connected to final verdict aggregation.
5. Stage 5 (Week 3-4): First benchmark harnesses
   - EN and MULTI benchmark scripts/results collection stabilized.

### Month 2: Model Training And Multilingual Hardening
6. Stage 6 (Week 5): Context model training rollout
   - EN context and Indic context model training pipelines finalized.
7. Stage 7 (Week 5-6): Relevance model hardening
   - MULTI relevance training and EN relevance checkpoint selection/tuning.
8. Stage 8 (Week 6-7): Stance EN staged curriculum
   - MNLI -> FEVER -> VitaminC staged fine-tune chain completed.
9. Stage 9 (Week 7-8): Stance MULTI integration
   - mDeBERTa-based multi stance path promoted and benchmarked.

### Month 3: Production Lock, Error Analysis, Research Packaging
10. Stage 10 (Week 9): LLM verifier + translation fallback stabilization
    - verifier policy/trigger logic, provider handling, rate limiting controls.
11. Stage 11 (Week 10): Checkability hardening and language-specific controls
    - multilingual checkability training promoted; routing and calibration guard refinements.
12. Stage 12 (Week 11-12): Final lock + research package
    - EN/MULTI checkpoints locked in `.env`,
    - canonical `Research_Evaluation/` package prepared,
    - duplicate/legacy eval artifacts cleaned from active documentation path.

## 5) Final Current Stage (What Is Considered “Final” Now)

- Locked EN/MULTI checkpoints are fixed (see Section 1).
- Active benchmark/result references are canonicalized to `Research_Evaluation/`.
- Legacy architecture comparison artifacts were removed from active repo docs path to avoid metric drift.
- README files now intentionally avoid duplicated historical benchmark snapshots.

## 6) Files To Use For Paper/Academic References

- Runtime lock specification: `docs/LOCKED_PIPELINES.md`
- End-to-end implementation narrative: `docs/PROJECT_FULL_EXPLANATION.md`
- Model + timeline reference (this file): `docs/MODEL_INVENTORY_AND_3MONTH_TIMELINE.md`
- Canonical evaluated outputs/figures/tables: `Research_Evaluation/`
