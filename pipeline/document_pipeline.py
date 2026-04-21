"""Document-level pipeline entrypoint."""

import re
from dataclasses import dataclass
from typing import Dict, List

from pipeline.claim_pipeline import ClaimPipeline
from pipeline.core.checkability import CheckabilityClassifier


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
        self.config = dict(config or {})
        self.min_words = int(self.config.get("document_min_words", 15))
        self.max_words = int(self.config.get("document_max_words", 80))
        if self.max_words < self.min_words:
            self.max_words = self.min_words
        self.enable_checkability_filter = bool(
            self.config.get("enable_document_checkability_filter", True)
        )
        self.checkability = CheckabilityClassifier(
            model_path=self.config.get("claim_checkability_checkpoint")
        )

    def _extract_claim_candidates(self, text: str) -> List[str]:
        if not text:
            return []

        cleaned = text.replace("\r", "\n")
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
        raw_chunks = re.split(r"[\n]+|(?<=[.!?।])\s+", cleaned)

        sentences: List[str] = []
        for chunk in raw_chunks:
            s = chunk.strip(" \t\n\r-–—•*")
            if not s:
                continue
            for key, token in abbr_map.items():
                s = s.replace(key, token)

            s = re.sub(r"^(?:py\)|@o\d+|ocr|ai)\s*[:\-)]?\s*", "", s, flags=re.IGNORECASE).strip()
            wc = len(s.split())
            if len(s) < 20 or wc < self.min_words or wc > self.max_words:
                continue
            if re.fullmatch(r"[\W_]+", s):
                continue
            if not re.search(r"[A-Za-z\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0CFF\u0D00-\u0D7F]", s):
                continue
            if not self._looks_like_sentence(s):
                continue
            sentences.append(s)

        # Add context windows of 2 sentences when possible.
        windowed: List[str] = list(sentences)
        for i in range(len(sentences) - 1):
            merged = f"{sentences[i]} {sentences[i + 1]}".strip()
            wc = len(merged.split())
            if self.min_words <= wc <= self.max_words:
                windowed.append(merged)

        out: List[str] = []
        seen = set()
        for s in windowed:
            if not self._is_verifiable_claim(s):
                continue
            key = " ".join(s.lower().split())
            if key in seen:
                continue
            seen.add(key)
            out.append(s)
        return out[:20]

    def rank_claim_candidates(self, text: str, language: str = "en") -> List[str]:
        """Return claim candidates sorted by verifiability-oriented score."""
        cands = self._extract_claim_candidates(text)
        cands = self._filter_by_checkability(cands)
        if not cands:
            return []

        def _score(c: str) -> float:
            low = c.lower()
            score = 0.0
            if re.search(r"\b(19|20)\d{2}\b", c):
                score += 2.0
            if re.search(r"\b\d+\b|%", c):
                score += 1.5
            if "according to" in low or "reported" in low or "states" in low:
                score += 1.0
            if re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", c):
                score += 1.0
            # Boost concrete incident-impact language.
            if any(k in low for k in ("killed", "affected", "flooding", "struck", "surge", "deaths", "injured", "displaced")):
                score += 1.8
            if re.search(r"\b(on|in)\s+\d{1,2}\s+[A-Za-z]{3,9}\s+\d{4}\b", c):
                score += 1.2
            # Penalize header/title-like lines.
            if any(k in low for k in ("introduction", "figure:", "table:", "contents", "environmental impact assessment")):
                score -= 2.0
            if re.match(r"^[A-Z][A-Za-z\s:,-]{0,120}$", c.strip()):
                score -= 1.2
            wc = len(c.split())
            score += max(0.0, 1.0 - abs(wc - 28) / 40.0)
            return score

        return sorted(cands, key=_score, reverse=True)

    def _filter_by_checkability(self, candidates: List[str]) -> List[str]:
        """Filter candidates using checkability classifier/heuristics."""
        if not candidates:
            return []
        if not self.enable_checkability_filter:
            return candidates

        out: List[str] = []
        for cand in candidates:
            is_checkable, _reason = self.checkability.classify(cand)
            if is_checkable:
                out.append(cand)
        return out

    def _looks_like_sentence(self, s: str) -> bool:
        """Heuristic sentence-shape filter to drop non-sentence fragments."""
        text = str(s or "").strip()
        if not text:
            return False
        low = text.lower()
        if re.match(r"^(figure|fig\.|table|chart|source|note|page|section)\b", low):
            return False
        if re.search(r"[.!?।]$", text):
            return True
        verb_cues = {
            "is",
            "are",
            "was",
            "were",
            "has",
            "have",
            "had",
            "will",
            "can",
            "caused",
            "causes",
            "lead",
            "leads",
            "resulted",
            "result",
            "reported",
            "shows",
            "showed",
            "indicates",
            "indicated",
            "specifies",
            "states",
            "includes",
        }
        words = re.findall(r"[A-Za-z']+", low)
        if any(w in verb_cues for w in words):
            return True
        if re.search(r"[,:;]", text):
            return True
        return False

    def _is_verifiable_claim(self, s: str) -> bool:
        """Require at least one factual anchor."""
        low = s.lower()
        has_number = bool(re.search(r"\b\d{1,4}\b|%|₹|\$", s))
        has_year = bool(re.search(r"\b(19|20)\d{2}\b", s))
        factual_verbs = (
            "is",
            "are",
            "was",
            "were",
            "has",
            "have",
            "had",
            "includes",
            "include",
            "specifies",
            "states",
            "reported",
            "caused",
            "occurred",
            "recorded",
            "announced",
            "killed",
            "affected",
        )
        has_factual_verb = any(v in low for v in factual_verbs)
        has_entity_like = bool(re.search(r"\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3}\b", s))
        return bool(has_number or has_year or has_factual_verb or has_entity_like)

    def select_best_claim_candidate(self, text: str, language: str = "en") -> str:
        """Pick best verifiable candidate from extracted text."""
        ranked = self.rank_claim_candidates(text=text, language=language)
        if ranked:
            return ranked[0]

        # Soft fallback if strict extraction yields none.
        soft = re.split(r"[\n]+|(?<=[.!?।])\s+", str(text or "").strip())
        for chunk in soft:
            s = chunk.strip()
            if len(s.split()) >= self.min_words and self._looks_like_sentence(s):
                return s
        return ""

    def analyze_text(self, text: str, language: str = "en") -> DocumentResult:
        candidates = self.rank_claim_candidates(text=text, language=language)
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
