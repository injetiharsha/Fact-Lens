"""Primary OCR engine using Tesseract."""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Dict

logger = logging.getLogger(__name__)


@dataclass
class OCRResult:
    text: str
    confidence: float
    engine: str
    metadata: Dict[str, object] = field(default_factory=dict)


class TesseractWrapper:
    """Thin wrapper for Tesseract OCR."""

    def __init__(self, languages: str | None = None):
        self.languages = languages or os.getenv("OCR_IMAGE_LANGS", "eng")
        self._ready = False

    def _initialize(self) -> None:
        if self._ready:
            return
        import pytesseract
        from PIL import Image

        self.pytesseract = pytesseract
        self.Image = Image
        self._ready = True

    def extract(self, image_path: str, languages: str | None = None) -> OCRResult:
        try:
            self._initialize()
            lang = languages or self.languages
            img = self.Image.open(image_path)

            data = self.pytesseract.image_to_data(
                img,
                lang=lang,
                output_type=self.pytesseract.Output.DICT,
            )
            words = []
            confs = []
            for i, token in enumerate(data.get("text", [])):
                token = str(token or "").strip()
                if not token:
                    continue
                conf_raw = str(data.get("conf", ["-1"])[i]).strip()
                try:
                    conf_val = float(conf_raw)
                except Exception:
                    conf_val = -1.0
                if conf_val >= 0:
                    confs.append(conf_val)
                words.append(token)

            text = " ".join(words).strip()
            confidence = (sum(confs) / len(confs) / 100.0) if confs else 0.0
            confidence = max(0.0, min(1.0, confidence))

            return OCRResult(
                text=text,
                confidence=confidence,
                engine="tesseract",
                metadata={
                    "languages": lang,
                    "word_count": len(words),
                    "mean_conf_raw": (sum(confs) / len(confs)) if confs else 0.0,
                },
            )
        except Exception as exc:
            logger.warning("Tesseract OCR failed: %s", exc)
            return OCRResult(
                text="",
                confidence=0.0,
                engine="tesseract",
                metadata={"error": str(exc)},
            )

