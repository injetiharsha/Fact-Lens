"""PDF input pipeline: extract text -> clean -> claim text."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass, field
from typing import List, Tuple

from pipeline.ingestion.image.ocr_postprocessor import OCRPostprocessor


@dataclass
class PDFInputResult:
    claim_text: str
    extracted_text: str
    page_count: int
    extraction_engine: str
    selected_pages: List[int] = field(default_factory=list)
    selected_page_spec: str = ""
    warnings: List[str] = field(default_factory=list)


class PDFInputPipeline:
    """Extract text from PDF and prepare claim text for pipeline."""

    def __init__(self):
        self.post = OCRPostprocessor()
        self.max_pages = max(1, int(os.getenv("PDF_MAX_PAGES", "5")))
        self.max_chars = max(1000, int(os.getenv("PDF_MAX_EXTRACTED_CHARS", "30000")))
        self.enable_ocr_fallback = os.getenv("PDF_ENABLE_OCR_FALLBACK", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.ocr_langs = os.getenv("OCR_IMAGE_LANGS", "eng")

    def process(self, pdf_path: str, claim_text: str = "", page_spec: str = "") -> PDFInputResult:
        warnings: List[str] = []
        extracted = ""
        page_count = 0
        engine = "pypdf"
        selected_pages: List[int] = []
        selected_page_spec = ""

        try:
            from pypdf import PdfReader
        except Exception:
            return PDFInputResult(
                claim_text=self.post.clean(claim_text) if claim_text else "",
                extracted_text="",
                page_count=0,
                extraction_engine="unavailable",
                selected_pages=[],
                selected_page_spec="",
                warnings=["pypdf dependency missing. Install pypdf to enable PDF extraction."],
            )

        try:
            reader = PdfReader(pdf_path)
            page_count = len(reader.pages)
            page_indexes, selected_page_spec, spec_warnings = self._resolve_pages(
                page_spec=page_spec,
                page_count=page_count,
            )
            warnings.extend(spec_warnings)
            selected_pages = [i + 1 for i in page_indexes]

            chunks: List[str] = []
            for idx in page_indexes:
                page = reader.pages[idx]
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

        if not cleaned_extracted and self.enable_ocr_fallback and selected_pages:
            ocr_text, ocr_warnings = self._ocr_fallback_extract(
                pdf_path=pdf_path,
                page_numbers=selected_pages,
                max_chars=self.max_chars,
            )
            warnings.extend(ocr_warnings)
            cleaned_extracted = self.post.clean(ocr_text)

        cleaned_claim = self.post.clean(claim_text) if claim_text else ""
        final_claim = cleaned_claim or cleaned_extracted

        if not cleaned_extracted:
            warnings.append("No machine-readable text extracted from PDF (possibly scanned/image-only).")
        if not page_spec and page_count > self.max_pages:
            warnings.append(f"Only first {self.max_pages} pages processed.")
        if len(selected_pages) == 4:
            warnings.append("WARN: 4 pages selected. This run may be time-extensive.")
        if len(cleaned_extracted) >= self.max_chars:
            warnings.append(f"Extracted text truncated at {self.max_chars} chars.")

        return PDFInputResult(
            claim_text=final_claim,
            extracted_text=cleaned_extracted,
            page_count=page_count,
            extraction_engine=engine,
            selected_pages=selected_pages,
            selected_page_spec=selected_page_spec,
            warnings=warnings,
        )

    def _ocr_fallback_extract(self, pdf_path: str, page_numbers: List[int], max_chars: int) -> Tuple[str, List[str]]:
        warnings: List[str] = []
        try:
            import fitz  # pymupdf
        except Exception:
            warnings.append("WARN: OCR fallback unavailable (pymupdf not installed).")
            return "", warnings

        try:
            import pytesseract
            from PIL import Image
            import io
        except Exception:
            warnings.append("WARN: OCR fallback unavailable (pytesseract/Pillow missing).")
            return "", warnings

        chunks: List[str] = []
        try:
            doc = fitz.open(pdf_path)
            for pn in page_numbers:
                idx = max(0, int(pn) - 1)
                if idx >= len(doc):
                    continue
                page = doc[idx]
                pix = page.get_pixmap(matrix=fitz.Matrix(2, 2), alpha=False)
                img = Image.open(io.BytesIO(pix.tobytes("png")))
                try:
                    txt = pytesseract.image_to_string(img, lang=self.ocr_langs) or ""
                except Exception:
                    txt = ""
                if txt:
                    chunks.append(txt)
                if sum(len(c) for c in chunks) >= max_chars:
                    break
            doc.close()
        except Exception as exc:
            warnings.append(f"WARN: OCR fallback failed: {exc}")
            return "", warnings

        out = "\n".join(chunks).strip()
        if out:
            warnings.append("WARN: Used OCR fallback for scanned PDF pages.")
        return out[:max_chars], warnings

    def _resolve_pages(self, page_spec: str, page_count: int) -> Tuple[List[int], str, List[str]]:
        warnings: List[str] = []
        if page_count <= 0:
            return [], "", warnings

        policy_max_page = min(self.max_pages, page_count)
        raw = str(page_spec or "").strip()
        if not raw:
            end = policy_max_page
            idx = list(range(0, end))
            return idx, f"1-{end}" if end > 1 else "1", warnings

        single = re.fullmatch(r"\d+", raw)
        range_m = re.fullmatch(r"(\d+)\s*-\s*(\d+)", raw)
        pages: List[int] = []

        if single:
            n = int(single.group(0))
            pages = [n]
        elif range_m:
            a = int(range_m.group(1))
            b = int(range_m.group(2))
            if a > b:
                a, b = b, a
            pages = list(range(a, b + 1))
        else:
            warnings.append("WARN: Invalid page selector. Using default page window.")
            end = min(page_count, self.max_pages)
            idx = list(range(0, end))
            return idx, f"1-{end}" if end > 1 else "1", warnings

        in_bounds = [p for p in pages if 1 <= p <= page_count]
        if not in_bounds:
            warnings.append("WARN: Selected pages out of bounds. Using default page window.")
            end = policy_max_page
            idx = list(range(0, end))
            return idx, f"1-{end}" if end > 1 else "1", warnings

        # Enforce product/runtime policy: page selector cannot exceed configured max page window.
        over_policy = [p for p in in_bounds if p > policy_max_page]
        if over_policy:
            warnings.append(
                f"HIGH: Page selector exceeds max allowed page {policy_max_page}. "
                f"Allowed range is 1-{policy_max_page}."
            )
            in_bounds = [p for p in in_bounds if p <= policy_max_page]
            if not in_bounds:
                in_bounds = [policy_max_page]

        if len(in_bounds) > self.max_pages:
            warnings.append(
                f"HIGH: Selected {len(in_bounds)} pages exceeds max limit {self.max_pages}. "
                f"Using first {self.max_pages} selected pages."
            )
            in_bounds = in_bounds[: self.max_pages]

        # keep order + unique
        seen = set()
        out = []
        for p in in_bounds:
            if p in seen:
                continue
            seen.add(p)
            out.append(p)

        if len(out) == 1:
            spec = str(out[0])
        else:
            spec = f"{out[0]}-{out[-1]}"
        return [p - 1 for p in out], spec, warnings
