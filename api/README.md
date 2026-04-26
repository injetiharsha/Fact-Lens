# API

FastAPI entrypoints and request/response wiring for claim/document analysis.

## What Is Here
- `routes/`: HTTP routes (claim analysis and document analysis endpoints).
- `schemas/`: Pydantic models for requests and responses.

## Notes
- Routes are wired to the restructured `pipeline` modules.
- API behavior depends on `.env` toggles (checkpoints, concurrency, verifier).
- Translation helper endpoints are available for UI:
  - `/api/translate-preview`
  - `/api/translate-batch`
- PDF route warnings:
  - `/api/analyze-pdf` expects machine-readable PDFs; scanned PDFs may return empty text.
  - Extraction is capped by `PDF_MAX_PAGES` and `PDF_MAX_EXTRACTED_CHARS` for predictable runtime.

## Progression Notes (April 2026)
- Image and PDF analysis modes are separated in pipeline behavior.
- Document checkability gating is configurable per mode through env.
- LLM verifier behavior is controlled via `LLM_VERIFIER_*` env flags, including verdict fallback.

## Scope Snapshot
- Path: `api/README.md`.
- Purpose: API service layer: routes, schemas, and runtime endpoint behavior.
- Audience: Engineers running, extending, or evaluating this module.

