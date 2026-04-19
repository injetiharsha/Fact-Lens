"""Image input pipeline: quality check -> OCR -> cleaned claim text."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from typing import Dict, List

from .ocr_postprocessor import OCRPostprocessor
from .ocr_selector import OCRSelector, OCRResult


@dataclass
class ImageInputResult:
    claim_text: str
    ocr_text: str
    ocr_engine: str
    ocr_confidence: float
    image_quality: Dict[str, object] = field(default_factory=dict)
    warnings: List[str] = field(default_factory=list)


class ImageInputPipeline:
    """Run image preprocessing and OCR for downstream claim analysis."""

    def __init__(self):
        self.selector = OCRSelector()
        self.post = OCRPostprocessor()
        self.min_width = int(os.getenv("IMAGE_MIN_WIDTH", "320"))
        self.min_height = int(os.getenv("IMAGE_MIN_HEIGHT", "180"))

    def process(self, image_path: str, claim_text: str = "", language: str = "en") -> ImageInputResult:
        warnings: List[str] = []
        quality = self._assess_quality(image_path)
        if quality.get("low_quality"):
            warnings.append("Image quality is low; OCR confidence may drop.")

        ocr: OCRResult = self.selector.extract(image_path=image_path, language=language)
        cleaned_ocr = self.post.clean(ocr.text)
        cleaned_claim = self.post.clean(claim_text) if claim_text else ""

        final_claim = cleaned_claim or cleaned_ocr
        if not final_claim:
            warnings.append("No text extracted from image.")

        return ImageInputResult(
            claim_text=final_claim,
            ocr_text=cleaned_ocr,
            ocr_engine=ocr.engine,
            ocr_confidence=float(ocr.confidence),
            image_quality=quality,
            warnings=warnings,
        )

    def _assess_quality(self, image_path: str) -> Dict[str, object]:
        out: Dict[str, object] = {"path": image_path, "low_quality": False}
        if not os.path.exists(image_path):
            out["low_quality"] = True
            out["error"] = "file_not_found"
            return out
        try:
            from PIL import Image

            with Image.open(image_path) as img:
                width, height = img.size
                out["width"] = width
                out["height"] = height
                low = width < self.min_width or height < self.min_height
                out["low_quality"] = low
                out["mode"] = str(img.mode)
                return out
        except Exception as exc:
            out["low_quality"] = True
            out["error"] = str(exc)
            return out

