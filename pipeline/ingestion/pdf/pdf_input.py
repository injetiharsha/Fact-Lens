"""PDF input pipeline: extract text -> clean -> claim text."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import List

from pipeline.ingestion.image.ocr_postprocessor import OCRPostprocessor


@dataclass
class PDFInputResult:
    claim_text: str
    extracted_text: str
    page_count: int
    extraction_engine: str
    warnings: List[str] = field(default_factory=list)


class PDFInputPipeline:
    """Extract text from PDF and prepare claim text for pipeline."""

    def __init__(self):
        self.post = OCRPostprocessor()
        self.max_pages = max(1, int(os.getenv("PDF_MAX_PAGES", "5")))
        self.max_chars = max(1000, int(os.getenv("PDF_MAX_EXTRACTED_CHARS", "30000")))

    def process(self, pdf_path: str, claim_text: str = "") -> PDFInputResult:
        warnings: List[str] = []
        extracted = ""
        page_count = 0
        engine = "pypdf"

        try:
            from pypdf import PdfReader
        except Exception:
            return PDFInputResult(
                claim_text=self.post.clean(claim_text) if claim_text else "",
                extracted_text="",
                page_count=0,
                extraction_engine="unavailable",
                warnings=["pypdf dependency missing. Install pypdf to enable PDF extraction."],
            )

        try:
            reader = PdfReader(pdf_path)
            page_count = len(reader.pages)
            chunks: List[str] = []
            for page in reader.pages[: self.max_pages]:
                try:
                    txt = page.extract_text() or ""
                except Exception:
                    txt = ""
                if txt:
                    chunks.append(txt)
                if sum(len(c) for c in chunks) >= self.max_chars:
                    break
            extracted = "\n".join(chunks).strip()
        except Exception as exc:
            warnings.append(f"PDF extraction failed: {exc}")

        cleaned_extracted = self.post.clean(extracted)
        cleaned_claim = self.post.clean(claim_text) if claim_text else ""
        final_claim = cleaned_claim or cleaned_extracted

        if not cleaned_extracted:
            warnings.append("No machine-readable text extracted from PDF (possibly scanned/image-only).")
        if page_count > self.max_pages:
            warnings.append(f"Only first {self.max_pages} pages processed.")
        if len(cleaned_extracted) >= self.max_chars:
            warnings.append(f"Extracted text truncated at {self.max_chars} chars.")

        return PDFInputResult(
            claim_text=final_claim,
            extracted_text=cleaned_extracted,
            page_count=page_count,
            extraction_engine=engine,
            warnings=warnings,
        )
