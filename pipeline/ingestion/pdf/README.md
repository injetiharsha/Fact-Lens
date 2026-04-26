# ingestion/pdf

PDF extraction entry for claim verification.

## What it does
- Reads uploaded PDF via `pypdf`.
- Extracts text from first `PDF_MAX_PAGES` pages.
- Truncates at `PDF_MAX_EXTRACTED_CHARS`.
- Cleans extracted text and returns:
  - `claim_text`
  - `extracted_text`
  - `page_count`
  - `warnings`

## Current scope
- Optimized for machine-readable PDFs.
- Scanned/image-only PDFs may return empty text.
- OCR fallback for scanned PDFs is not enabled yet (future phase).

## Output contract
- See `pipeline/ingestion/pdf/pdf_input.py` (`PDFInputResult`).
- API wrappers:
  - `POST /api/extract-pdf-preview`
  - `POST /api/analyze-pdf`

## Scope Snapshot
- Path: `pipeline/ingestion/pdf/README.md`.
- Purpose: PDF extraction module behavior and limits.
- Audience: Engineers running, extending, or evaluating this module.

