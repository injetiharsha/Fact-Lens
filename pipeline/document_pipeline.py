"""Document-level pipeline entrypoint."""

import re
from dataclasses import dataclass
from typing import Dict, List

from pipeline.claim_pipeline import ClaimPipeline


@dataclass
class DocumentResult:
    """Aggregate output for a document run."""
    claims: List[Dict]
    summary_verdict: str
    summary_confidence: float


class DocumentPipeline:
    """Runs claim pipeline over extracted document claims."""

    def __init__(self, config: Dict):
        self.claim_pipeline = ClaimPipeline(config)

    def _extract_claim_candidates(self, text: str) -> List[str]:
        if not text:
            return []

        cleaned = text.replace("\r", "\n")
        # Normalize common OCR abbreviation spacing.
        cleaned = re.sub(r"\b([A-Za-z])\.\s+([A-Za-z])\.\b", r"\1.\2.", cleaned)
        cleaned = re.sub(r"\s+", " ", cleaned).strip()

        # Protect dotted abbreviations (e.g., U.S., U.K.) during sentence split.
        abbr_map = {}
        def _protect_abbr(match: re.Match) -> str:
            token = match.group(0)
            key = f"__ABBR_{len(abbr_map)}__"
            abbr_map[key] = token
            return key
        cleaned = re.sub(r"\b(?:[A-Za-z]\.){2,}", _protect_abbr, cleaned)

        # Split on strong sentence boundaries and line breaks.
        raw_chunks = re.split(r"[\n]+|(?<=[.!?])\s+", cleaned)

        out: List[str] = []
        seen = set()
        for chunk in raw_chunks:
            s = chunk.strip(" \t\n\r-–—•*")
            if not s:
                continue
            for key, token in abbr_map.items():
                s = s.replace(key, token)

            # Drop known OCR overlay/watermark prefixes.
            s = re.sub(r"^(?:py\)|@o\d+|ocr|ai)\s*[:\-)]?\s*", "", s, flags=re.IGNORECASE).strip()
            # Drop OCR artifact fragments.
            if len(s) < 20:
                continue
            if len(s.split()) < 5:
                continue
            if re.fullmatch(r"[\W_]+", s):
                continue
            # Require at least one alphabetic character.
            if not re.search(r"[A-Za-z\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0CFF\u0D00-\u0D7F]", s):
                continue

            key = s.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(s)

        return out[:20]

    def analyze_text(self, text: str, language: str = "en") -> DocumentResult:
        candidates = self._extract_claim_candidates(text)
        results: List[Dict] = []
        for claim in candidates:
            out = self.claim_pipeline.analyze(claim=claim, language=language)
            results.append(
                {
                    "claim": claim,
                    "verdict": out.verdict,
                    "confidence": out.confidence,
                    "evidence_count": len(out.evidence),
                }
            )

        if not results:
            return DocumentResult(claims=[], summary_verdict="neutral", summary_confidence=0.0)

        best = max(results, key=lambda r: r["confidence"])
        return DocumentResult(
            claims=results,
            summary_verdict=best["verdict"],
            summary_confidence=float(best["confidence"]),
        )
