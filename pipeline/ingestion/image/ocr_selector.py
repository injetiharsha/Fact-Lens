"""OCR engine selection logic."""

from __future__ import annotations

import os
from typing import List

from .easyocr_wrapper import EasyOCRWrapper
from .tesseract_wrapper import OCRResult, TesseractWrapper


class OCRSelector:
    """Select OCR engine based on confidence thresholds."""

    def __init__(self):
        self.tesseract = TesseractWrapper()
        self.easyocr = EasyOCRWrapper()
        self.min_tesseract_conf = float(os.getenv("OCR_TESSERACT_MIN_CONF", "0.70"))
        self.auto_lang_candidates = self._build_auto_lang_candidates()

    def _build_auto_lang_candidates(self) -> List[str]:
        raw = os.getenv("OCR_AUTO_LANG_CANDIDATES", "eng,hin,tam,tel,kan,mal")
        vals = [x.strip() for x in raw.split(",") if x.strip()]
        return vals or ["eng"]

    def _resolve_tesseract_languages(self, language: str) -> str:
        lang = (language or "").strip().lower()
        mapping = {
            "en": "eng",
            "hi": "hin",
            "ta": "tam",
            "te": "tel",
            "kn": "kan",
            "ml": "mal",
        }
        if not lang or lang == "auto":
            return ""
        return mapping.get(lang, lang)

    def _extract_auto(self, image_path: str) -> OCRResult:
        best = OCRResult(text="", confidence=0.0, engine="tesseract", metadata={})
        for lang in self.auto_lang_candidates:
            out = self.tesseract.extract(image_path=image_path, languages=lang)
            if out.confidence > best.confidence:
                best = out
        return best

    def extract(self, image_path: str, language: str = "en") -> OCRResult:
        tess_lang = self._resolve_tesseract_languages(language)
        if tess_lang:
            primary = self.tesseract.extract(image_path=image_path, languages=tess_lang)
        else:
            primary = self._extract_auto(image_path=image_path)

        if primary.text and primary.confidence >= self.min_tesseract_conf:
            return primary

        fallback = self.easyocr.extract(image_path=image_path, language=language)
        if fallback.text and fallback.confidence > primary.confidence:
            return fallback
        return primary
