# pipeline/evidence

Evidence retrieval stack for structured APIs, web search, and scraping.

## Structure
- `core/`: gathering, aggregation, deduplication logic.
- `providers/`: pluggable evidence providers.

## Notes
- Tavily multi-key support is implemented through env-driven key list handling.
- Provider failures are handled with fallbacks when possible.

## Scope Snapshot
- Path: `pipeline/evidence/README.md`.
- Purpose: Evidence retrieval, provider fan-out, and dedup/enrichment behavior.
- Audience: Engineers running, extending, or evaluating this module.

