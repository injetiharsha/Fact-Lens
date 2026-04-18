# pipeline

Restructured claim/document analysis pipeline.

## Structure
- `claim_pipeline.py`: claim-level analysis flow.
- `document_pipeline.py`: document-level analysis flow.
- `orchestrator.py`: orchestration layer for pipeline execution.
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
