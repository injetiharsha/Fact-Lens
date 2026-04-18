"""Evidence aggregation, deduplication, and ranking."""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class EvidenceAggregator:
    """Merge, deduplicate, and rank evidence."""

    def deduplicate(self, evidence_list: List[Dict]) -> List[Dict]:
        """
        Remove near-duplicate evidence.
        
        Uses URL dedup + semantic similarity check.
        """
        if not evidence_list:
            return []
        
        seen_urls = set()
        unique_evidence = []
        
        for ev in evidence_list:
            url = ev.get("url", "")
            
            # Skip if URL already seen
            if url and url in seen_urls:
                logger.debug(f"Duplicate URL skipped: {url}")
                continue
            
            # Check text similarity with existing evidence
            is_duplicate = False
            if ev.get("text"):
                for existing in unique_evidence:
                    similarity = self._text_similarity(ev["text"], existing.get("text", ""))
                    if similarity > 0.90:  # 90% similar = duplicate
                        is_duplicate = True
                        # Keep higher credibility source
                        if ev.get("score", 0) > existing.get("score", 0):
                            unique_evidence.remove(existing)
                            unique_evidence.append(ev)
                        break
            
            if not is_duplicate:
                unique_evidence.append(ev)
                if url:
                    seen_urls.add(url)
        
        logger.info(f"Deduplication: {len(evidence_list)} → {len(unique_evidence)} items")
        return unique_evidence
    
    def rank(self, evidence_list: List[Dict], claim: str) -> List[Dict]:
        """
        Rank evidence by quality indicators.
        
        Sorts by: source credibility > relevance > recency
        """
        if not evidence_list:
            return []
        
        # Score each evidence
        for ev in evidence_list:
            quality_score = 0.0
            
            # Source credibility (0-40 points)
            source_type = ev.get("type", "web_search")
            if source_type == "structured_api":
                quality_score += 40
            elif source_type == "web_search":
                quality_score += 25
            else:  # scraping
                quality_score += 15
            
            # Relevance score (0-30 points)
            relevance = ev.get("relevance", ev.get("score", 0.5))
            quality_score += relevance * 30
            
            # Evidence length (0-20 points)
            text_length = len(ev.get("text", ""))
            if text_length > 100:
                quality_score += 20
            elif text_length > 50:
                quality_score += 15
            elif text_length > 20:
                quality_score += 10
            
            # Has URL (0-10 points)
            if ev.get("url"):
                quality_score += 10
            
            ev["quality_score"] = quality_score
        
        # Sort by quality score (descending)
        ranked = sorted(evidence_list, key=lambda x: x.get("quality_score", 0), reverse=True)
        
        logger.info(f"Ranked {len(ranked)} evidence items")
        return ranked
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity (Jaccard + word overlap)."""
        if not text1 or not text2:
            return 0.0
        
        # Tokenize
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        # Jaccard similarity
        intersection = words1 & words2
        union = words1 | words2
        
        jaccard = len(intersection) / len(union) if union else 0.0
        
        # Word containment (one text contains most of the other)
        containment1 = len(intersection) / len(words1) if words1 else 0.0
        containment2 = len(intersection) / len(words2) if words2 else 0.0
        
        # Combine metrics
        return max(jaccard, containment1, containment2)
