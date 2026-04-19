"""Fallback OCR engine using EasyOCR."""

from __future__ import annotations

import logging
import os
from typing import Dict, List

from .tesseract_wrapper import OCRResult

logger = logging.getLogger(__name__)


LANG_HINTS = {
    "en": "en",
    "hi": "hi",
    "te": "te",
    "ta": "ta",
    "kn": "kn",
    "ml": "ml",
}


class EasyOCRWrapper:
    """Optional OCR fallback engine."""

    def __init__(self):
        self._reader_cache: Dict[str, object] = {}
        self._enabled = os.getenv("OCR_ENABLE_EASYOCR", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }

    def _get_reader(self, language: str):
        if not self._enabled:
            return None
        lang = LANG_HINTS.get((language or "en").lower(), "en")
        key = f"{lang}+en"
        if key in self._reader_cache:
            return self._reader_cache[key]
        try:
            import easyocr

            reader = easyocr.Reader([lang, "en"], gpu=False)
            self._reader_cache[key] = reader
            return reader
        except Exception as exc:
            logger.warning("EasyOCR unavailable: %s", exc)
            return None

    def extract(self, image_path: str, language: str = "en") -> OCRResult:
        reader = self._get_reader(language)
        if reader is None:
            return OCRResult(
                text="",
                confidence=0.0,
                engine="easyocr",
                metadata={"error": "easyocr_unavailable"},
            )

        try:
            rows: List = reader.readtext(image_path, detail=1, paragraph=False)
            texts = []
            confs = []
            for row in rows:
                if len(row) < 3:
                    continue
                text = str(row[1] or "").strip()
                if not text:
                    continue
                texts.append(text)
                try:
                    confs.append(float(row[2]))
                except Exception:
                    pass
            confidence = (sum(confs) / len(confs)) if confs else 0.0
            confidence = max(0.0, min(1.0, confidence))
            return OCRResult(
                text=" ".join(texts).strip(),
                confidence=confidence,
                engine="easyocr",
                metadata={"rows": len(rows), "language": language},
            )
        except Exception as exc:
            logger.warning("EasyOCR extraction failed: %s", exc)
            return OCRResult(
                text="",
                confidence=0.0,
                engine="easyocr",
                metadata={"error": str(exc)},
            )

