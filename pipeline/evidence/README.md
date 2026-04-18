# pipeline/evidence

Evidence retrieval stack for structured APIs, web search, and scraping.

## Structure
- `core/`: gathering, aggregation, deduplication logic.
- `providers/`: pluggable evidence providers.

## Notes
- Tavily multi-key support is implemented through env-driven key list handling.
- Provider failures are handled with fallbacks when possible.
