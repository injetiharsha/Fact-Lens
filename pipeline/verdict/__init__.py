"""Final verdict aggregation and scoring."""

import logging
from typing import Dict, List

logger = logging.getLogger(__name__)


class VerdictEngine:
    """Aggregate evidence scores into final verdict."""
    
    def compute(self, evidence_list: List[Dict], claim: str) -> Dict:
        """
        Compute final verdict from all evidence.
        
        Args:
            evidence_list: List of scored evidence
            claim: Original claim text
            
        Returns:
            Dict with verdict, confidence, reasoning
        """
        if not evidence_list:
            return {
                "verdict": "neutral",
                "confidence": 0.3,
                "reasoning": "No evidence found to support or refute the claim."
            }
        
        # Aggregate weighted stance across all evidence
        agg_support = 0.0
        agg_refute = 0.0
        agg_neutral = 0.0
        total_weight = 0.0
        
        for ev in evidence_list:
            ws = ev.get("weighted_stance", {})
            ew = ev.get("evidence_weight", 0.5)
            
            # weighted_stance is already stance_probs * evidence_weight from stage 8.
            # Do not multiply by ew again here (that would square-weight evidence).
            agg_support += ws.get("support", 0.33)
            agg_refute += ws.get("refute", 0.33)
            agg_neutral += ws.get("neutral", 0.34)
            total_weight += ew
        
        # Normalize
        if total_weight > 0:
            agg_support /= total_weight
            agg_refute /= total_weight
            agg_neutral /= total_weight
        
        # Conflict detection
        conflict = (agg_support > 0.40 and agg_refute > 0.40)
        
        # Evidence count adjustment
        evidence_count = len(evidence_list)
        count_multiplier = self._get_count_multiplier(evidence_count)
        
        # Determine verdict
        verdict, confidence = self._determine_verdict(
            agg_support, agg_refute, agg_neutral,
            conflict, count_multiplier
        )
        
        # Generate reasoning
        reasoning = self._generate_reasoning(
            verdict, confidence, agg_support, agg_refute, agg_neutral,
            evidence_count, conflict
        )
        
        return {
            "verdict": verdict,
            "confidence": confidence,
            "reasoning": reasoning,
            "agg_support": agg_support,
            "agg_refute": agg_refute,
            "agg_neutral": agg_neutral,
            "conflict_detected": conflict,
            "evidence_count": evidence_count
        }
    
    def _get_count_multiplier(self, count: int) -> float:
        """Adjust confidence based on evidence count."""
        if count < 2:
            return 0.75  # Low confidence with few sources
        elif count <= 3:
            return 0.85  # Moderate confidence
        else:
            return 1.0   # Full confidence with 4+ sources
    
    def _determine_verdict(
        self,
        agg_support: float,
        agg_refute: float,
        agg_neutral: float,
        conflict: bool,
        count_multiplier: float
    ) -> tuple:
        """Determine final verdict and confidence."""
        
        # If conflict detected, return neutral
        if conflict:
            return "neutral", min(max(agg_neutral, 0.5), 0.7) * count_multiplier
        
        # Get max stance
        max_stance = max(agg_support, agg_refute, agg_neutral)
        
        if agg_support > agg_refute and agg_support > agg_neutral:
            verdict = "support"
            confidence = agg_support
        elif agg_refute > agg_support and agg_refute > agg_neutral:
            verdict = "refute"
            confidence = agg_refute
        else:
            verdict = "neutral"
            confidence = agg_neutral
        
        # Apply count adjustment
        confidence *= count_multiplier
        
        # Clamp to 0-1
        confidence = max(0.0, min(confidence, 1.0))
        
        return verdict, confidence
    
    def _generate_reasoning(
        self,
        verdict: str,
        confidence: float,
        agg_support: float,
        agg_refute: float,
        agg_neutral: float,
        evidence_count: int,
        conflict: bool
    ) -> str:
        """Generate human-readable reasoning."""
        
        parts = []
        
        # Evidence summary
        parts.append(f"Analyzed {evidence_count} evidence sources.")
        
        # Conflict warning
        if conflict:
            parts.append("Conflicting evidence detected from different sources.")
        
        # Stance breakdown
        parts.append(
            f"Support: {agg_support:.0%}, Refute: {agg_refute:.0%}, Neutral: {agg_neutral:.0%}."
        )
        
        # Verdict explanation
        if verdict == "support":
            if confidence >= 0.75:
                parts.append("Strong evidence supports this claim.")
            elif confidence >= 0.55:
                parts.append("Moderate evidence supports this claim.")
            else:
                parts.append("Weak evidence leans toward supporting this claim.")
        
        elif verdict == "refute":
            if confidence >= 0.75:
                parts.append("Strong evidence refutes this claim.")
            elif confidence >= 0.55:
                parts.append("Moderate evidence refutes this claim.")
            else:
                parts.append("Weak evidence suggests this claim may be inaccurate.")
        
        else:  # neutral
            if conflict:
                parts.append("Evidence is contradictory, no clear conclusion.")
            elif evidence_count < 2:
                parts.append("Insufficient evidence to reach a conclusion.")
            else:
                parts.append("Evidence is inconclusive or mixed.")
        
        # Confidence level
        if confidence >= 0.75:
            parts.append(f"High confidence ({confidence:.0%}).")
        elif confidence >= 0.55:
            parts.append(f"Moderate confidence ({confidence:.0%}).")
        else:
            parts.append(f"Low confidence ({confidence:.0%}). Further verification recommended.")
        
        return " ".join(parts)
