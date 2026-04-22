# pipeline

Restructured claim/document analysis pipeline.

## Structure
- `claim_pipeline.py`: claim-level analysis flow.
- `document_pipeline.py`: document-level analysis flow.
- `orchestrator.py`: orchestration layer for pipeline execution.
- `ingestion/`: input extraction (image OCR + PDF text extraction).
- `core/`: shared NLP components (normalization, context, stance, etc.).
- `evidence/`: evidence collection components and providers.
- `scoring/`: credibility/temporal/confidence aggregation.
- `verdict/`: final verdict logic and LLM verifier integration.

## Recent Cleanup
- Removed duplicate/legacy nested folders from older layout.
- Consolidated evidence modules into clearer `core` and `providers` split.
- Added split search adapters:
  - `pipeline/evidence/search/tavily_search.py`
  - `pipeline/evidence/search/serpapi_search.py`
  - `pipeline/evidence/search/duckduckgo_search.py`
  - `pipeline/evidence/search/newsapi_search.py`
- Added tiered scraper stack:
  - `pipeline/evidence/scraper/trafilatura_scraper.py` (tier-1)
  - `pipeline/evidence/scraper/playwright_scraper.py` (tier-2)
  - `pipeline/evidence/scraper/beautifulsoup_scraper.py` (tier-3)
- Added publication-date extraction utility:
  - `utils/date_extractor.py`

## Recent Progression (April 2026)
- Stage5 (evidence gather) remains the dominant runtime bottleneck in benchmarks.
- EN pipeline currently reaches high benchmark stability (25/30 latest run).
- Multi pipeline accuracy remains lower due to retrieval + stance ambiguity on Indic claims.
- Recent pipeline-level updates include:
  - LLM verifier verdict fallback (when structured evidence updates are not returned),
  - MMR-style reranking support in gatherer (`EVIDENCE_MMR_*`),
  - tighter query variant strategy (EN and multi paths),
  - translated query cap control for multi (`EVIDENCE_TRANSLATED_QUERY_CAP`).

## Root-Cause Record

- Architecture comparison and failure root-cause summary:
  - `../ARCHITECTURE_ROOT_CAUSE_ANALYSIS.md`
