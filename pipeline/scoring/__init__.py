"""Credibility weighting and composite scoring."""

import logging
import math
import os
import re
from urllib.parse import urlparse
from datetime import datetime
from typing import Dict

logger = logging.getLogger(__name__)

# Domain credibility tiers (higher = more credible)
DOMAIN_CREDIBILITY = {
    # Tier 1: Government/Official (1.0)
    "gov": 1.0,
    "edu": 0.95,
    "int": 0.9,  # International orgs (UN, WHO, etc)
    "india.gov.in": 1.0,
    "pib.gov.in": 0.98,
    "parliament.uk": 0.98,
    "whitehouse.gov": 0.98,
    "europa.eu": 0.98,
    "indiacode.nic.in": 1.0,
    "sci.gov.in": 0.98,
    "ecourts.gov.in": 0.98,
    "justice.gov": 0.96,
    "archives.gov": 0.96,
    "nationalarchives.gov.uk": 0.96,
    "loc.gov": 0.95,
    "rbi.org.in": 0.96,
    "federalreserve.gov": 0.96,
    "census.gov": 0.95,
    "mospi.gov.in": 0.95,
    "isro.gov.in": 0.98,
    "nasa.gov": 0.98,
    "jpl.nasa.gov": 0.98,
    "esa.int": 0.96,
    "who.int": 0.98,
    "cdc.gov": 0.98,
    "nih.gov": 0.98,
    "ncbi.nlm.nih.gov": 0.98,
    "ipcc.ch": 0.98,
    "unep.org": 0.95,
    "epa.gov": 0.96,
    "noaa.gov": 0.96,
    "usgs.gov": 0.96,
    "unesco.org": 0.95,
    "unicef.org": 0.95,
    "un.org": 0.95,
    
    # Tier 2: Established news (0.85)
    "reuters": 0.9,
    "ap": 0.9,
    "bbc": 0.85,
    "nytimes": 0.85,
    "apnews.com": 0.9,
    "reuters.com": 0.9,
    "ft.com": 0.88,
    "wsj.com": 0.88,
    "economist.com": 0.86,
    
    # Tier 3: Regional/Established (0.70)
    "wikipedia": 0.75,
    "wikidata.org": 0.8,
    "britannica.com": 0.9,
    "nature.com": 0.92,
    "science.org": 0.92,
    "sciencedirect.com": 0.88,
    "springer.com": 0.87,
    "arxiv.org": 0.86,
    "ieee.org": 0.9,
    "nist.gov": 0.95,
    "cisa.gov": 0.95,
    "owasp.org": 0.86,
    "developer.mozilla.org": 0.85,
    "history.com": 0.78,
    "worldbank.org": 0.95,
    "imf.org": 0.95,
    "oecd.org": 0.93,
    "wto.org": 0.93,
    "iucn.org": 0.9,
    "pewresearch.org": 0.9,
    "law.cornell.edu": 0.9,
    "olympics.com": 0.9,
    "fifa.com": 0.86,
    "icc-cricket.com": 0.86,
    "worldathletics.org": 0.86,
    "imdb.com": 0.82,
    "rottentomatoes.com": 0.8,
    "metacritic.com": 0.78,
    "boxofficemojo.com": 0.8,
    "bfi.org.uk": 0.82,
    "geonames.org": 0.88,
    "mayoclinic.org": 0.9,
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
    
    def calculate_weight(self, evidence: Dict, recency_policy: Dict | None = None) -> float:
        """
        Calculate evidence weight = relevance × credibility × temporal.
        
        Args:
            evidence: Evidence dict with relevance, source, date, context
            
        Returns:
            Evidence weight 0.0-1.0
        """
        relevance = evidence.get("relevance", 0.5)
        credibility = self._get_credibility(evidence)
        temporal = self._get_temporal_weight(evidence, recency_policy=recency_policy)

        # High relevance should not be over-penalized by medium/low credibility sources.
        # Useful for local/regional factual reporting.
        try:
            rel_f = float(relevance)
            cred_f = float(credibility)
            if rel_f >= 0.95 and cred_f < 0.82:
                credibility = min(0.82, cred_f + 0.15)
        except Exception:
            pass

        # Source trust tier (host/source heuristics) acts as a quality prior.
        source_tier = self._get_source_tier_multiplier(evidence)

        # Keep recency internal via temporal decay only (no extra bonus/compare boost).
        recency_bonus = 1.0
        
        # Composite score
        weight = relevance * credibility * temporal * recency_bonus * source_tier
        
        evidence["credibility"] = credibility
        evidence["temporal_weight"] = temporal
        evidence["source_tier_multiplier"] = source_tier
        evidence["recency_bonus"] = recency_bonus
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

    def _get_source_tier_multiplier(self, evidence: Dict) -> float:
        """Additional trust prior based on host/source quality bands."""
        source = str(evidence.get("source", "")).lower()
        url = str(evidence.get("url", "")).strip().lower()
        host = ""
        try:
            host = (urlparse(url).netloc or "").lower().split(":")[0].strip(".")
        except Exception:
            host = ""
        text = f"{source} {host} {url}"

        if os.getenv("SCORING_ENABLE_SOURCE_TIER_MULTIPLIER", "1").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return 1.0

        high_trust = {
            "gov.in", "gov", "edu", "who.int", "cdc.gov", "nih.gov", "nasa.gov", "isro.gov.in",
            "reuters.com", "apnews.com", "bbc.com", "ft.com", "wsj.com", "economist.com",
            "nature.com", "science.org", "worldbank.org", "imf.org",
        }
        low_trust_tokens = {
            "blogspot", "wordpress", "substack", "reddit", "quora", "facebook", "instagram",
            "twitter", "x.com", "youtube", "dailymotion", "pinterest", "tiktok",
            "rumor", "gossip", "forums", "forum", "thread",
        }

        for d in high_trust:
            if host == d or host.endswith("." + d) or d in text:
                return 1.05

        if any(tok in text for tok in low_trust_tokens):
            return 0.80

        # Penalize question-only / UGC-heavy URLs lightly.
        if re.search(r"/questions?/|/qna/|/forum/|/community/|/comments?/|/threads?/", url):
            return 0.85
        return 1.0
    
    def _get_temporal_weight(self, evidence: Dict, recency_policy: Dict | None = None) -> float:
        """
        Calculate temporal weight based on recency.
        
        Uses exponential decay based on context type.
        """
        context = evidence.get("context", "GENERAL_FACTUAL")
        date_str = evidence.get("date") or evidence.get("published_at")
        relevance = float(evidence.get("relevance", 0.0) or 0.0)
        
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

            # Recency balance policy:
            # higher relevance softens recency penalty.
            # thresholds requested: >95, >90, >80.
            if relevance >= 0.95:
                weight = min(1.0, weight + 0.35)
            elif relevance >= 0.90:
                weight = min(1.0, weight + 0.25)
            elif relevance >= 0.80:
                weight = min(1.0, weight + 0.10)
            
            # Clamp to 0.4-1.0 range
            return max(weight, 0.4)
        
        except Exception as e:
            logger.warning(f"Failed to calculate temporal weight: {e}")
            return 0.8  # Default

    def _get_recency_bonus(self, evidence: Dict, recency_policy: Dict | None = None) -> float:
        """Deprecated path retained for compatibility; bonus disabled by policy."""
        return 1.0


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
