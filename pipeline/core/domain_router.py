"""Domain-based evidence source routing (hierarchical + inclusive)."""

import logging
from typing import Dict, List, Set, Tuple

logger = logging.getLogger(__name__)

# Level 1 → Level 2 → Evidence sources
# More granular routing based on subcategory
HIERARCHICAL_SOURCES = {
    # SCIENCE
    "SCIENCE": {
        "physics": ["structured_api:arxiv", "web_search:physics", "scraping:physics"],
        "chemistry": ["structured_api:pubchem", "web_search:chemistry", "scraping:chemistry"],
        "biology": ["structured_api:pubmed", "web_search:biology", "scraping:biology"],
        "earth_science": ["structured_api:usgs", "web_search:geology", "scraping:geology"],
        "environmental_science": ["structured_api:epa", "web_search:environment", "scraping:environment"],
        "materials_science": ["structured_api:arxiv", "web_search:materials", "scraping:materials"],
        "scientific_consensus": ["structured_api:pubmed", "web_search:meta-analysis", "structured_api:wikipedia"],
        "_default": ["web_search:science", "structured_api:wikipedia", "scraping:general"]
    },
    
    # HEALTH
    "HEALTH": {
        "medicine": ["structured_api:pubmed", "structured_api:openfda", "web_search:medical"],
        "public_health": ["structured_api:who", "structured_api:cdc", "web_search:health"],
        "nutrition": ["structured_api:usda", "web_search:nutrition", "scraping:nutrition"],
        "epidemiology": ["structured_api:who", "structured_api:cdc", "web_search:epidemic"],
        "toxicology": ["structured_api:openfda", "web_search:toxicology", "scraping:toxicology"],
        "disease_treatment": ["structured_api:pubmed", "structured_api:openfda", "web_search:treatment"],
        "mental_health": ["structured_api:pubmed", "web_search:mental_health", "scraping:psychology"],
        "_default": ["structured_api:openfda", "web_search:health", "scraping:medical"]
    },
    
    # TECHNOLOGY
    "TECHNOLOGY": {
        "telecom": ["web_search:telecom", "structured_api:ieee", "scraping:telecom"],
        "internet": ["web_search:internet", "structured_api:wikipedia", "scraping:tech"],
        "software_ai": ["structured_api:arxiv", "web_search:ai", "scraping:tech"],
        "hardware": ["web_search:hardware", "scraping:tech"],
        "cybersecurity": ["web_search:cybersecurity", "structured_api:cve", "scraping:security"],
        "social_media": ["web_search:social_media", "scraping:social"],
        "_default": ["web_search:tech", "scraping:techcrunch"]
    },
    
    # HISTORY
    "HISTORY": {
        "ancient_history": ["structured_api:wikipedia", "web_search:ancient", "scraping:history"],
        "modern_history": ["structured_api:wikipedia", "web_search:modern_history", "scraping:history"],
        "wars_conflicts": ["structured_api:wikipedia", "web_search:military", "scraping:military"],
        "historical_events": ["structured_api:wikipedia", "web_search:history", "scraping:history"],
        "diplomacy_treaties": ["structured_api:un", "web_search:treaty", "scraping:diplomacy"],
        "historical_figures": ["structured_api:wikipedia", "web_search:biography", "scraping:biography"],
        "_default": ["web_search:history", "structured_api:wikipedia", "scraping:history"]
    },
    
    # POLITICS_GOVERNMENT
    "POLITICS_GOVERNMENT": {
        "elections": ["web_search:news", "scraping:news", "structured_api:gov_api"],
        "public_policy": ["structured_api:gov_api", "web_search:policy", "scraping:policy"],
        "foreign_affairs": ["structured_api:un", "web_search:diplomacy", "scraping:foreign"],
        "legislation": ["structured_api:gov_api", "web_search:legislation", "scraping:legal"],
        "governance": ["structured_api:gov_api", "web_search:governance", "scraping:politics"],
        "political_statements": ["web_search:news", "scraping:news"],
        "_default": ["web_search:news", "structured_api:gov_api", "scraping:news"]
    },
    
    # ECONOMICS_BUSINESS
    "ECONOMICS_BUSINESS": {
        "macroeconomics": ["structured_api:worldbank", "structured_api:imf", "web_search:economy"],
        "finance": ["web_search:finance", "structured_api:sec", "scraping:finance"],
        "trade": ["structured_api:wto", "web_search:trade", "scraping:trade"],
        "corporate_claims": ["web_search:business", "structured_api:sec", "scraping:business"],
        "labor_inflation": ["structured_api:bureau_labor_stats", "web_search:inflation", "scraping:economy"],
        "markets": ["web_search:markets", "scraping:finance"],
        "_default": ["web_search:business", "structured_api:worldbank", "scraping:business"]
    },
    
    # GEOGRAPHY
    "GEOGRAPHY": {
        "countries": ["structured_api:geonames", "structured_api:wikipedia", "web_search:geography"],
        "continents": ["structured_api:wikipedia", "web_search:geography"],
        "capitals_borders": ["structured_api:geonames", "structured_api:wikipedia", "web_search:geography"],
        "rivers_lakes": ["structured_api:usgs", "structured_api:wikipedia", "web_search:geography"],
        "mountains": ["structured_api:geonames", "structured_api:wikipedia", "web_search:mountains"],
        "climate_regions": ["structured_api:noaa", "web_search:climate", "scraping:climate"],
        "_default": ["web_search", "structured_api:wikipedia", "structured_api:geonames"]
    },
    
    # SPACE_ASTRONOMY
    "SPACE_ASTRONOMY": {
        "planets": ["structured_api:nasa", "web_search:space", "scraping:space"],
        "moons": ["structured_api:nasa", "web_search:space", "scraping:space"],
        "stars": ["structured_api:nasa", "web_search:astronomy", "scraping:astronomy"],
        "space_missions": ["structured_api:nasa", "structured_api:isro", "web_search:space", "scraping:space"],
        "planetary_science": ["structured_api:nasa", "web_search:space", "scraping:space"],
        "cosmology": ["structured_api:nasa", "structured_api:arxiv", "web_search:cosmology"],
        "_default": ["structured_api:nasa", "web_search:space", "scraping:space"]
    },
    
    # ENVIRONMENT_CLIMATE
    "ENVIRONMENT_CLIMATE": {
        "climate_change": ["structured_api:ipcc", "structured_api:noaa", "web_search:climate"],
        "biodiversity": ["structured_api:iucn", "web_search:biodiversity", "scraping:conservation"],
        "pollution": ["structured_api:epa", "web_search:pollution", "scraping:environment"],
        "disasters_weather": ["structured_api:noaa", "web_search:weather", "scraping:weather"],
        "sustainability": ["structured_api:unep", "web_search:sustainability", "scraping:sustainability"],
        "ecological_impacts": ["structured_api:epa", "web_search:ecology", "scraping:ecology"],
        "_default": ["web_search:climate", "structured_api:epa", "scraping:general"]
    },
    
    # SOCIETY_CULTURE
    "SOCIETY_CULTURE": {
        "religion": ["structured_api:wikipedia", "web_search:religion", "scraping:religion"],
        "education": ["structured_api:unesco", "web_search:education", "scraping:education"],
        "demographics": ["structured_api:worldbank", "web_search:demographics", "scraping:census"],
        "social_issues": ["web_search:social", "scraping:social"],
        "language_identity": ["structured_api:wikipedia", "web_search:language", "scraping:linguistics"],
        "customs_traditions": ["structured_api:wikipedia", "web_search:culture", "scraping:culture"],
        "_default": ["web_search", "structured_api:wikipedia", "scraping:culture"]
    },
    
    # LAW_CRIME
    "LAW_CRIME": {
        "courts": ["web_search:legal", "scraping:legal", "structured_api:court_api"],
        "regulation": ["structured_api:gov_api", "web_search:regulation", "scraping:regulatory"],
        "constitutional_issues": ["web_search:legal", "scraping:constitutional", "structured_api:court_api"],
        "criminal_cases": ["web_search:crime", "scraping:crime"],
        "rights_compliance": ["web_search:legal", "structured_api:un", "scraping:legal"],
        "_default": ["web_search:legal", "scraping:legal"]
    },
    
    # SPORTS
    "SPORTS": {
        "teams": ["web_search:sports", "structured_api:sports_api", "scraping:sports"],
        "athletes": ["web_search:sports", "structured_api:sports_api", "scraping:sports"],
        "tournaments": ["web_search:sports", "structured_api:sports_api", "scraping:sports"],
        "records": ["web_search:sports", "structured_api:guinness", "scraping:sports"],
        "rules": ["web_search:sports", "scraping:sports"],
        "_default": ["web_search:sports", "structured_api:sports_api", "scraping:sports"]
    },
    
    # ENTERTAINMENT
    "ENTERTAINMENT": {
        "film": ["structured_api:imdb", "web_search:film", "scraping:entertainment"],
        "television": ["structured_api:imdb", "web_search:tv", "scraping:entertainment"],
        "music": ["web_search:music", "scraping:music"],
        "celebrity": ["web_search:celebrity", "scraping:entertainment"],
        "gaming": ["web_search:gaming", "scraping:gaming"],
        "streaming_media": ["web_search:streaming", "scraping:streaming"],
        "_default": ["web_search:entertainment", "structured_api:imdb", "scraping:entertainment"]
    },
    
    # GENERAL_FACTUAL
    "GENERAL_FACTUAL": {
        "encyclopedic": ["structured_api:wikipedia", "web_search", "scraping:general"],
        "entity_property": ["structured_api:wikipedia", "structured_api:wikidata", "web_search"],
        "general_news": ["web_search:news", "scraping:news"],
        "_default": ["web_search", "structured_api:wikipedia", "scraping:general"]
    }
}

# Source priority
SOURCE_PRIORITY = {
    "structured_api": 1,  # Highest priority (most credible)
    "web_search": 2,      # Medium priority
    "scraping": 3,        # Lowest priority (fallback)
}

# Inclusive baseline sources that are always appended (deduplicated).
# This ensures domain routing is additive, not restrictive.
GLOBAL_INCLUSIVE_SOURCES = [
    "structured_api:wikipedia",
    "web_search:general",
    "scraping:general",
]


class DomainRouter:
    """Route claims to appropriate evidence sources based on hierarchical context."""

    def route(self, level1: str, level2: str = "general") -> List[Dict]:
        """
        Return ordered list of evidence sources for context.
        
        Args:
            level1: Level-1 context label (e.g., "HEALTH")
            level2: Level-2 subcategory (e.g., "medicine")
            
        Returns:
            List of source dicts with type, subtype, and priority
        """
        # Try to get level-2 specific sources first
        level1_sources = HIERARCHICAL_SOURCES.get(level1, HIERARCHICAL_SOURCES["GENERAL_FACTUAL"])
        routed_sources = level1_sources.get(level2, level1_sources.get("_default", []))
        sources = self._merge_inclusive_sources(routed_sources, GLOBAL_INCLUSIVE_SOURCES)
        
        # Parse source strings into structured format
        parsed_sources = self._parse_sources(sources, level1, level2)
        
        # Sort by priority (structured APIs first)
        parsed_sources.sort(key=lambda x: x["priority"])
        
        logger.info(f"Routed context '{level1}/{level2}' to {len(parsed_sources)} sources")
        return parsed_sources

    def _merge_inclusive_sources(self, primary: List[str], inclusive: List[str]) -> List[str]:
        """Merge primary and inclusive sources while preserving order and uniqueness."""
        merged: List[str] = []
        seen: Set[str] = set()
        for source in list(primary) + list(inclusive):
            if source not in seen:
                seen.add(source)
                merged.append(source)
        return merged

    def _parse_sources(self, sources: List[str], level1: str, level2: str) -> List[Dict]:
        """Parse serialized source strings into source dicts."""
        parsed_sources: List[Dict] = []
        for source_str in sources:
            parts = source_str.split(":", 1)
            source_type = parts[0]
            source_subtype = parts[1] if len(parts) > 1 else None
            parsed_sources.append({
                "type": source_type,
                "subtype": source_subtype,
                "priority": SOURCE_PRIORITY.get(source_type, 99),
                "level1": level1,
                "level2": level2,
            })
        return parsed_sources
