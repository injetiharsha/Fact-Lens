# API

FastAPI entrypoints and request/response wiring for claim/document analysis.

## What Is Here
- `routes/`: HTTP routes (claim analysis and document analysis endpoints).
- `schemas/`: Pydantic models for requests and responses.

## Notes
- Routes are wired to the restructured `pipeline` modules.
- API behavior depends on `.env` toggles (checkpoints, concurrency, verifier).
- PDF route warnings:
  - `/api/analyze-pdf` expects machine-readable PDFs; scanned PDFs may return empty text.
  - Extraction is capped by `PDF_MAX_PAGES` and `PDF_MAX_EXTRACTED_CHARS` for predictable runtime.
