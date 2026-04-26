# Project Full Explanation (Top-to-Bottom)

This document is the full technical explanation of the RFCS project architecture and runtime behavior.
It is focused on implementation flow, not benchmark storytelling.

For pinned runtime values, use [LOCKED_PIPELINES.md](F:\RFCS\docs\LOCKED_PIPELINES.md).

## 1) What This Project Does

RFCS is a multilingual claim verification system.  
Input can be:

- direct text claim
- image (OCR -> claim extraction -> verification pipeline)
- PDF (text extraction/OCR fallback -> sentence/claim extraction -> verification pipeline)

Output is a final verdict (`support`, `refute`, `neutral`) with confidence, evidence, and stage-level diagnostics.

## 2) Main Entry Points

Core files:

- [api/main.py](F:\RFCS\api\main.py): FastAPI app startup.
- [api/routes/claim.py](F:\RFCS\api\routes\claim.py): API handlers and pipeline config construction.
- [pipeline/claim_pipeline.py](F:\RFCS\pipeline\claim_pipeline.py): claim pipeline entry alias.
- [pipeline/orchestrator.py](F:\RFCS\pipeline\orchestrator.py): full stage orchestrator.
- [pipeline/document_pipeline.py](F:\RFCS\pipeline\document_pipeline.py): document/PDF-oriented processing.
- [pipeline/ingestion/image/image_input.py](F:\RFCS\pipeline\ingestion\image\image_input.py): image OCR path.

## 3) Configuration Resolution Model

The runtime config is assembled in:

- [claim.py::_pipeline_config](F:\RFCS\api\routes\claim.py)

It decides:

- pipeline language bucket (`en` vs multi)
- checkpoints (checkability/context/relevance/stance)
- evidence and relevance caps
- LLM verifier provider/model/policy

Important behavior:

- process env is primary.
- `.env` file fallback exists in [claim.py::_env_or_file](F:\RFCS\api\routes\claim.py) unless explicitly disabled.

## 4) EN vs MULTI Routing

Language routing rules:

- English-like request -> EN checkpoint paths.
- Non-English request -> MULTI checkpoint paths.
- Auto-detect script/language is handled in [claim.py::_auto_detect_language](F:\RFCS\api\routes\claim.py).

Checkpoint selection is in:

- [claim.py::_select_checkpoint](F:\RFCS\api\routes\claim.py)

## 5) Stage-by-Stage Pipeline (Orchestrator)

All major logic runs inside:

- [orchestrator.py::FactCheckingPipeline.analyze](F:\RFCS\pipeline\orchestrator.py)

Stages:

1. `stage1_normalize`
- normalize claim text, cleanup noisy framing.

2. `stage2_checkability`
- run checkability classifier.
- may early-exit as `neutral` with `Uncheckable(...)` reason.

3. `stage3_context`
- predict coarse/fine context (domain intent).

4. `stage4_routing`
- choose evidence source families using context + fallbacks.

5. `stage5_evidence_gather`
- gather evidence via configured source mode (typically `staged_fallback`).
- can include web search, structured APIs, and scraping enrichment.

6. `stage6_relevance`
- rank and keep top evidence (bi-encoder/cross-encoder flow + keep thresholds).
- optional Sarvam rerank block runs as `stage6b_sarvam_rerank`.

7. `stage7_stance`
- classify evidence stance against claim.

8. `stage8_evidence_scoring`
- aggregate relevance/stance/credibility into evidence-level scores.

9. `stage9_aggregate_verdict`
- produce provisional verdict from scored evidence.

10. `stage10_llm_verify`
- optional verifier pass (policy-controlled), may adjust verdict/confidence.
- includes optional subpaths:
  - multi tavily boost (if enabled)
  - image multi EN fallback
  - neutral recovery
  - neutral quality recovery

## 6) Evidence System Details

Main gatherer:

- [pipeline/evidence/core/gatherer.py](F:\RFCS\pipeline\evidence\core\gatherer.py)

Key mechanics:

- source mode: `staged_fallback` (common locked mode)
- stage order from env (`EVIDENCE_STAGE_ORDER`)
- early stage stop threshold (`EVIDENCE_STAGE_MIN_RESULTS`)
- translated query augmentation for multi claims (`EVIDENCE_ENABLE_TRANSLATED_QUERY_SEARCH`)
- scrape enrichment over discovered URLs when enabled

Web search provider logic:

- [pipeline/evidence/providers/web_search.py](F:\RFCS\pipeline\evidence\providers\web_search.py)

It supports:

- DDG/Tavily/SerpAPI toggles
- provider order and escalation
- domain/url blocking filters
- provider failover handling

## 7) Checkability Model Behavior

Wrapper:

- [pipeline/core/checkability.py](F:\RFCS\pipeline\core\checkability.py)

Important rules:

- fast heuristic blocks: question/opinion/short claim (short-claim strictness mainly when model unavailable)
- model labels mapped to checkable IDs
- optional multi relax mode (to reduce over-blocking)

## 8) LLM Verifier Behavior

Verifier adapter:

- [pipeline/verdict/llm_verifier.py](F:\RFCS\pipeline\verdict\llm_verifier.py)

Capabilities:

- provider chain support (`openai`, `groq`, `fireworks`, `nscale`, `cerebras`, `openrouter`, `sarvam`)
- key rotation across multiple keys
- verdict normalization and evidence update parsing
- neutral-only verification policy support
- optional verdict fallback usage when model confidence is high enough

## 9) Shared LLM Rate Limiting

Limiter:

- [pipeline/core/llm_rate_limiter.py](F:\RFCS\pipeline\core\llm_rate_limiter.py)

Features:

- cross-process RPM limiting (SQLite-backed when global mode enabled)
- in-process concurrency semaphore
- cooldown on 429-like pressure
- optional audit events in `.llm_verifier_rate_limit.sqlite`

This is why sustained batch runs can behave differently from tiny isolated runs.

## 10) OCR/Image Path

Primary components:

- [pipeline/ingestion/image/image_input.py](F:\RFCS\pipeline\ingestion\image\image_input.py)
- [pipeline/ingestion/image/ocr_selector.py](F:\RFCS\pipeline\ingestion\image\ocr_selector.py)
- [pipeline/ingestion/image/tesseract_wrapper.py](F:\RFCS\pipeline\ingestion\image\tesseract_wrapper.py)
- [pipeline/ingestion/image/easyocr_wrapper.py](F:\RFCS\pipeline\ingestion\image\easyocr_wrapper.py)

Flow:

- quality check -> OCR (Tesseract first, EasyOCR fallback) -> postprocess -> claim text
- then same orchestrator stages as text claims

## 11) Locked Runtime Contract (Current)

Canonical lock spec:

- [LOCKED_PIPELINES.md](F:\RFCS\docs\LOCKED_PIPELINES.md)

Operational meaning:

- pinned EN/MULTI checkpoints
- DDG-only retrieval (Tavily/SerpAPI off)
- LLM verifier on

## 12) Benchmark and Experiment Scripts

Common scripts:

- [scripts/run_benchmark_en.py](F:\RFCS\scripts\run_benchmark_en.py)
- [scripts/run_benchmark_multi.py](F:\RFCS\scripts\run_benchmark_multi.py)
- [scripts/run_thesis_experiment.py](F:\RFCS\scripts\run_thesis_experiment.py)
- [scripts/run_locked_image_folder.py](F:\RFCS\scripts\run_locked_image_folder.py)

These scripts emit:

- per-claim predictions
- metrics and confusion-oriented counters
- stage timing traces

## 13) Output/Debug Fields You Should Trust Most

In prediction rows, most useful fields for diagnosis:

- `details.checkability`
- `details.context`
- `details.timings`
- `details.llm_verifier`
- evidence list (`stance`, `relevance`, `credibility`, `source`, `url`)

For image runs:

- `ocr_engine`
- `ocr_confidence`
- `pipeline_bucket`
- `checkability_blocked`

## 14) Known Failure Modes (Engineering View)

- retrieval starvation: high evidence count can still be weak if relevance/credibility are low
- checkability thresholding can over-filter borderline claims if configured too strictly
- LLM quota/rate pressure changes verifier contribution
- staged retrieval order impacts domain diversity and final verdict tendency
- noisy OCR can distort query generation and context classification

## 15) Safe Change Policy

When changing behavior, do it in this order:

1. update lock spec/environment keys
2. run narrow language tests
3. run combined benchmark
4. inspect `details.llm_verifier` transitions and stage timings
5. only then promote changes into locked config

Do not mix architecture changes and benchmark dataset edits in the same validation cycle.
