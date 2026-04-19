"""OCR text cleanup utilities."""

from __future__ import annotations

import re


class OCRPostprocessor:
    """Normalize OCR output into cleaner claim text."""

    def clean(self, text: str) -> str:
        if not text:
            return ""

        out = text
        out = out.replace("\r", "\n")
        out = re.sub(r"[ \t]+", " ", out)
        out = re.sub(r"\n{2,}", "\n", out)
        out = re.sub(r"\s+([,.;:!?])", r"\1", out)
        out = re.sub(r"([,.;:!?])([^\s])", r"\1 \2", out)
        out = re.sub(r"\s{2,}", " ", out)
        return out.strip()

