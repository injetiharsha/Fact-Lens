"""Credibility weighting and composite scoring."""

import logging
import math
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)

# Domain credibility tiers (higher = more credible)
DOMAIN_CREDIBILITY = {
    # Tier 1: Government/Official (1.0)
    "gov": 1.0,
    "edu": 0.95,
    "int": 0.9,  # International orgs (UN, WHO, etc)
    
    # Tier 2: Established news (0.85)
    "reuters": 0.9,
    "ap": 0.9,
    "bbc": 0.85,
    "nytimes": 0.85,
    
    # Tier 3: Regional/Established (0.70)
    "wikipedia": 0.75,
    "medium": 0.6,
    "blog": 0.5,
    
    # Tier 4: Social/Unverified (0.25)
    "twitter": 0.25,
    "facebook": 0.25,
    "reddit": 0.35,
    "youtube": 0.4,
}

# Temporal decay rates by context (half-life in months)
TEMPORAL_DECAY = {
    "HEALTH": 12,        # Health info expires fast
    "TECHNOLOGY": 18,    # Tech changes quickly
    "POLITICS_GOVERNMENT": 48,  # Politics lasts medium
    "ECONOMICS_BUSINESS": 24,   # Economic data updates
    "SCIENCE": 36,       # Science lasts longer
    "HISTORY": 999,      # History doesn't expire
    "GEOGRAPHY": 999,    # Geography stable
    "SPACE_ASTRONOMY": 60,  # Space knowledge lasts
    "ENVIRONMENT_CLIMATE": 36,
    "SOCIETY_CULTURE": 60,
    "LAW_CRIME": 60,
    "SPORTS": 12,        # Sports records update
    "ENTERTAINMENT": 24,
    "GENERAL_FACTUAL": 999,
}


class EvidenceScorer:
    """Calculate composite evidence scores."""
    
    def calculate_weight(self, evidence: Dict) -> float:
        """
        Calculate evidence weight = relevance × credibility × temporal.
        
        Args:
            evidence: Evidence dict with relevance, source, date, context
            
        Returns:
            Evidence weight 0.0-1.0
        """
        relevance = evidence.get("relevance", 0.5)
        credibility = self._get_credibility(evidence)
        temporal = self._get_temporal_weight(evidence)
        
        # Composite score
        weight = relevance * credibility * temporal
        
        evidence["credibility"] = credibility
        evidence["temporal_weight"] = temporal
        evidence["evidence_weight"] = weight
        
        return weight
    
    def weight_stance(self, evidence: Dict) -> Dict[str, float]:
        """
        Weight stance probabilities by evidence weight.
        
        Returns:
            Weighted stance {support, refute, neutral}
        """
        ew = evidence.get("evidence_weight", 0.5)
        stance_probs = evidence.get("stance_probs", {"support": 0.33, "refute": 0.33, "neutral": 0.34})
        
        # Weight stance by evidence quality
        weighted_stance = {
            "support": stance_probs.get("support", 0.33) * ew,
            "refute": stance_probs.get("refute", 0.33) * ew,
            "neutral": stance_probs.get("neutral", 0.34) * ew
        }
        
        evidence["weighted_stance"] = weighted_stance
        return weighted_stance
    
    def _get_credibility(self, evidence: Dict) -> float:
        """Get credibility weight based on source domain."""
        source = evidence.get("source", "").lower()
        url = evidence.get("url", "").lower()
        source_type = evidence.get("type", "web_search")
        
        # Structured APIs are most credible
        if source_type == "structured_api":
            return 0.9
        
        # Check domain credibility
        for domain, cred in DOMAIN_CREDIBILITY.items():
            if domain in source or domain in url:
                return cred
        
        # Default credibility
        if source_type == "web_search":
            return 0.7
        
        return 0.5  # Scraping/fallback
    
    def _get_temporal_weight(self, evidence: Dict) -> float:
        """
        Calculate temporal weight based on recency.
        
        Uses exponential decay based on context type.
        """
        context = evidence.get("context", "GENERAL_FACTUAL")
        date_str = evidence.get("date")
        
        # Get half-life for this context
        half_life_months = TEMPORAL_DECAY.get(context, 999)
        
        # If no date or very long half-life, return 1.0
        if not date_str or half_life_months > 500:
            return 1.0
        
        try:
            # Parse date (assume ISO format)
            if isinstance(date_str, str):
                pub_date = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
            else:
                pub_date = date_str
            
            # Calculate months since publication
            months_ago = (datetime.now() - pub_date.replace(tzinfo=None)).days / 30.44
            
            # Exponential decay: weight = 0.5^(months_ago / half_life)
            weight = 0.5 ** (months_ago / half_life_months)
            
            # Clamp to 0.4-1.0 range
            return max(weight, 0.4)
        
        except Exception as e:
            logger.warning(f"Failed to calculate temporal weight: {e}")
            return 0.8  # Default


class ScoringCalculator:
    """Backward-compatible simple scoring utility for tests."""

    def calculate_weighted_score(self, scores, weights):
        total_weight = sum(weights.values()) or 1.0
        weighted = sum(scores.get(k, 0.0) * weights.get(k, 0.0) for k in weights)
        result = weighted / total_weight
        return max(0.0, min(1.0, float(result)))

    def apply_domain_weight(self, score, domain):
        d = (domain or "").lower()
        factor = 1.0
        if "science" in d:
            factor = 1.1
        elif "social" in d:
            factor = 0.8
        return max(0.0, min(1.0, float(score) * factor))
