# ingestion

Input extraction layer before claim pipeline.

## Modules
- `image/`: image preprocessing + OCR extraction pipeline.
- `pdf/`: PDF text extraction pipeline (machine-readable PDF first).

## Design
- Keep ingestion lightweight and deterministic.
- Return structured warnings instead of hard-failing when extraction quality is low.
- Do not run verdict logic here; only produce clean claim text for downstream pipeline.

## Environment knobs
- Image OCR knobs live in `.env` (`OCR_*`).
- PDF knobs:
  - `PDF_MAX_PAGES`
  - `PDF_MAX_EXTRACTED_CHARS`
  - `PDF_FORCE_SINGLE_CLAIM_WHEN_CLAIM_PROVIDED`

## Scope Snapshot
- Path: `pipeline/ingestion/README.md`.
- Purpose: Input ingestion layer for image/PDF preprocessing and extraction.
- Audience: Engineers running, extending, or evaluating this module.

