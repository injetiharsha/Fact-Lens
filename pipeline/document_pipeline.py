"""Document-level pipeline entrypoint."""

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
        chunks = [c.strip() for c in text.split(".") if c.strip()]
        # Keep simple deterministic extraction for now.
        return chunks[:20]

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

