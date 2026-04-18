"""Context classification for routing."""

import logging
from typing import Tuple, Dict

logger = logging.getLogger(__name__)

# Hierarchical context taxonomy (Level 1 → Level 2)
CONTEXT_TAXONOMY = {
    "SCIENCE": ["physics", "chemistry", "biology", "earth_science", "environmental_science", "materials_science", "scientific_consensus"],
    "HEALTH": ["medicine", "public_health", "nutrition", "epidemiology", "toxicology", "disease_treatment", "mental_health"],
    "TECHNOLOGY": ["telecom", "internet", "software_ai", "hardware", "cybersecurity", "social_media"],
    "HISTORY": ["ancient_history", "modern_history", "wars_conflicts", "historical_events", "diplomacy_treaties", "historical_figures"],
    "POLITICS_GOVERNMENT": ["elections", "public_policy", "foreign_affairs", "legislation", "governance", "political_statements"],
    "ECONOMICS_BUSINESS": ["macroeconomics", "finance", "trade", "corporate_claims", "labor_inflation", "markets"],
    "GEOGRAPHY": ["countries", "continents", "capitals_borders", "rivers_lakes", "mountains", "climate_regions"],
    "SPACE_ASTRONOMY": ["planets", "moons", "stars", "space_missions", "planetary_science", "cosmology"],
    "ENVIRONMENT_CLIMATE": ["climate_change", "biodiversity", "pollution", "disasters_weather", "sustainability", "ecological_impacts"],
    "SOCIETY_CULTURE": ["religion", "education", "demographics", "social_issues", "language_identity", "customs_traditions"],
    "LAW_CRIME": ["courts", "regulation", "constitutional_issues", "criminal_cases", "rights_compliance"],
    "SPORTS": ["teams", "athletes", "tournaments", "records", "rules"],
    "ENTERTAINMENT": ["film", "television", "music", "celebrity", "gaming", "streaming_media"],
    "GENERAL_FACTUAL": ["encyclopedic", "entity_property", "general_news"],
}

# Level 2 keyword mapping for classification
LEVEL2_KEYWORDS = {
    # SCIENCE
    "physics": ["physics", "quantum", "particle", "energy", "force", "relativity", "newton", "einstein"],
    "chemistry": ["chemistry", "chemical", "molecule", "reaction", "element", "compound", "periodic"],
    "biology": ["biology", "cell", "gene", "dna", "evolution", "organism", "species"],
    "earth_science": ["geology", "earthquake", "volcano", "tectonic", "mineral", "fossil"],
    "environmental_science": ["ecosystem", "habitat", "conservation", "biodiversity"],
    "materials_science": ["nanotechnology", "polymer", "alloy", "crystal", "semiconductor"],
    "scientific_consensus": ["peer review", "scientific consensus", "research agreement", "meta-analysis"],
    
    # HEALTH
    "medicine": ["medical", "doctor", "hospital", "surgery", "diagnosis", "treatment", "prescription"],
    "public_health": ["vaccination", "epidemic", "pandemic", "who", "cdc", "health campaign", "immunization"],
    "nutrition": ["diet", "vitamin", "calorie", "protein", "obesity", "malnutrition", "supplement"],
    "epidemiology": ["disease spread", "infection rate", "r0 value", "outbreak", "contact tracing"],
    "toxicology": ["toxic", "poison", "chemical exposure", "ld50", "carcinogen"],
    "disease_treatment": ["cancer", "diabetes", "heart disease", "therapy", "chemotherapy", "remedy"],
    "mental_health": ["depression", "anxiety", "psychology", "therapy", "ptsd", "bipolar"],
    
    # TECHNOLOGY
    "telecom": ["5g", "6g", "mobile network", "broadband", "fiber optic", "satellite communication"],
    "internet": ["web", "browser", "cloud", "bandwidth", "dns", "http", "protocol"],
    "software_ai": ["ai", "machine learning", "neural network", "algorithm", "software", "coding", "llm", "gpt"],
    "hardware": ["processor", "cpu", "gpu", "ram", "chip", "semiconductor", "transistor"],
    "cybersecurity": ["hacker", "malware", "ransomware", "encryption", "data breach", "phishing"],
    "social_media": ["facebook", "twitter", "instagram", "tiktok", "social network", "viral"],
    
    # HISTORY
    "ancient_history": ["ancient", "bc", "before christ", "mesopotamia", "egypt", "roman empire", "greek"],
    "modern_history": ["20th century", "world war", "cold war", "independence", "revolution"],
    "wars_conflicts": ["battle", "military", "war", "conflict", "invasion", "army", "navy"],
    "historical_events": ["historical event", "discovery", "invention", "treaty signed"],
    "diplomacy_treaties": ["treaty", "diplomacy", "accord", "peace agreement", "alliance"],
    "historical_figures": ["leader", "president", "king", "emperor", "dictator", "revolutionary"],
    
    # POLITICS_GOVERNMENT
    "elections": ["election", "vote", "campaign", "candidate", "ballot", "poll", "electoral"],
    "public_policy": ["policy", "government program", "welfare", "subsidy", "reform"],
    "foreign_affairs": ["diplomatic relations", "embassy", "sanctions", "nato", "un resolution"],
    "legislation": ["bill passed", "law enacted", "amendment", "congress", "parliament", "legislature"],
    "governance": ["administration", "bureaucracy", "transparency", "corruption", "accountability"],
    "political_statements": ["politician claimed", "party statement", "political promise"],
    
    # ECONOMICS_BUSINESS
    "macroeconomics": ["gdp", "inflation", "unemployment", "fiscal policy", "monetary policy", "recession"],
    "finance": ["bank", "loan", "interest rate", "stock market", "investment", "bond"],
    "trade": ["import", "export", "tariff", "trade deal", "wto", "free trade"],
    "corporate_claims": ["company revenue", "profit", "merger", "acquisition", "startup"],
    "labor_inflation": ["wage", "labor market", "strike", "minimum wage", "cost of living"],
    "markets": ["bull market", "bear market", "nasdaq", "s&p", "dow jones", "crypto"],
    
    # GEOGRAPHY
    "countries": ["india", "china", "usa", "brazil", "country border", "nation state"],
    "continents": ["asia", "europe", "africa", "americas", "oceania", "antarctica"],
    "capitals_borders": ["capital city", "borders with", "neighboring country"],
    "rivers_lakes": ["river", "lake", "ocean", "sea", "waterfall", "dam"],
    "mountains": ["mountain", "peak", "mountain range", "himalayas", "andes", "everest"],
    "climate_regions": ["tropical", "arctic", "desert", "temperate", "monsoon"],
    
    # SPACE_ASTRONOMY
    "planets": ["mercury", "venus", "mars", "jupiter", "saturn", "uranus", "neptune", "exoplanet"],
    "moons": ["moon", "lunar", "europa", "titan", "ganymede"],
    "stars": ["star", "supernova", "black hole", "nebula", "constellation"],
    "space_missions": ["nasa", "isro", "spacex", "apollo", "mars mission", "iss", "launch"],
    "planetary_science": ["asteroid", "comet", "meteor", "planetary formation"],
    "cosmology": ["big bang", "dark matter", "dark energy", "universe expansion", "cosmic microwave"],
    
    # ENVIRONMENT_CLIMATE
    "climate_change": ["global warming", "greenhouse gas", "carbon emission", "paris agreement", "ipcc"],
    "biodiversity": ["endangered species", "extinction", "wildlife", "conservation area"],
    "pollution": ["air pollution", "water pollution", "plastic waste", "smog", "toxic waste"],
    "disasters_weather": ["hurricane", "cyclone", "flood", "drought", "wildfire", "tornado"],
    "sustainability": ["renewable energy", "solar", "wind power", "sustainable", "green technology"],
    "ecological_impacts": ["deforestation", "habitat loss", "soil erosion", "coral bleaching"],
    
    # SOCIETY_CULTURE
    "religion": ["hinduism", "islam", "christianity", "buddhism", "sikhism", "temple", "mosque", "church"],
    "education": ["school", "university", "literacy", "curriculum", "exam", "degree"],
    "demographics": ["population", "census", "age distribution", "migration", "urbanization"],
    "social_issues": ["poverty", "inequality", "discrimination", "gender", "caste", "human rights"],
    "language_identity": ["language", "dialect", "script", "cultural identity", "indigenous"],
    "customs_traditions": ["festival", "ritual", "wedding", "tradition", "heritage", "cuisine"],
    
    # LAW_CRIME
    "courts": ["supreme court", "high court", "judge", "verdict", "appeal", "trial"],
    "regulation": ["regulation", "compliance", "sebi", "rbi", "fda", "regulatory body"],
    "constitutional_issues": ["constitution", "fundamental rights", "article", "amendment", "fundamental duties"],
    "criminal_cases": ["arrest", "crime rate", "murder", "theft", "fraud", "investigation"],
    "rights_compliance": ["civil rights", "freedom", "privacy", "labor rights", "consumer protection"],
    
    # SPORTS
    "teams": ["team", "club", "franchise", "ipl", "premier league", "nba"],
    "athletes": ["player", "athlete", "cricketer", "footballer", "olympian"],
    "tournaments": ["world cup", "olympics", "championship", "open", "tournament"],
    "records": ["world record", "highest score", "fastest", "most wins"],
    "rules": ["rule change", "foul", "penalty", "referee", "doping"],
    
    # ENTERTAINMENT
    "film": ["movie", "film", "cinema", "actor", "director", "oscar", "bollywood", "hollywood"],
    "television": ["tv show", "series", "netflix", "episode", "broadcast"],
    "music": ["song", "album", "singer", "concert", "grammy", "band"],
    "celebrity": ["celebrity", "famous", "influencer", "star", "actor personal life"],
    "gaming": ["video game", "esports", "gaming", "playstation", "xbox", "nintendo"],
    "streaming_media": ["youtube", "spotify", "streaming", "subscriber", "view count"],
    
    # GENERAL_FACTUAL
    "encyclopedic": ["encyclopedia", "defined as", "known for", "fact about"],
    "entity_property": ["born", "founded", "established", "located", "population of"],
    "general_news": ["news report", "according to", "recently", "today"],
}


class ContextClassifier:
    """Classify claim context for domain routing (hierarchical)."""

    def __init__(self, model_path: str = None):
        """Initialize with model checkpoint."""
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = "cpu"
        
        if model_path:
            self._load_model()
    
    def _load_model(self):
        """Load trained model from checkpoint."""
        try:
            import torch
            from transformers import AutoTokenizer, AutoModelForSequenceClassification

            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            # Level 1 model
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path, use_fast=False)
            self.model = AutoModelForSequenceClassification.from_pretrained(
                self.model_path,
                num_labels=14  # 14 level-1 categories
            ).to(self.device)
            self.model.eval()
            
            # TODO: Load level-2 model if available
            # self.model_l2 = AutoModelForSequenceClassification.from_pretrained(...)
            
            logger.info(f"Context classifier loaded from {self.model_path} on {self.device}")
        except Exception as e:
            logger.warning(f"Failed to load context model: {e}. Using keyword fallback.")
            self.model = None

    def classify(self, claim: str) -> Tuple[str, str, float, float]:
        """
        Classify claim context hierarchically.
        
        Returns:
            (level1_label, level2_label, level1_confidence, level2_confidence)
        """
        # Level 1 classification
        if self.model:
            l1_label, l1_conf = self._classify_level1(claim)
        else:
            l1_label, l1_conf = self._classify_level1_keywords(claim)
        
        # Level 2 classification (keyword-based for now, model later)
        l2_label, l2_conf = self._classify_level2(claim, l1_label)
        
        # If level2 confidence low, fallback to level1 only
        if l2_conf < 0.4:
            l2_label = "general"
            l2_conf = 0.3
        
        return l1_label, l2_label, l1_conf, l2_conf
    
    def _classify_level1(self, claim: str) -> Tuple[str, float]:
        """Use trained model for level-1 classification."""
        import torch
        
        inputs = self.tokenizer(claim, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
            predicted = torch.argmax(probs, dim=-1).item()
            confidence = probs[0][predicted].item()
        
        label = list(CONTEXT_TAXONOMY.keys())[predicted]
        return label, confidence
    
    def _classify_level1_keywords(self, claim: str) -> Tuple[str, float]:
        """Keyword-based level-1 classification (fallback)."""
        claim_lower = claim.lower()
        
        keyword_map = {
            "SCIENCE": ["science", "research", "study", "experiment", "discovery", "physics", "chemistry", "biology"],
            "HEALTH": ["health", "medical", "disease", "hospital", "doctor", "medicine", "cure", "treatment", "fda"],
            "TECHNOLOGY": ["technology", "tech", "software", "computer", "ai", "app", "digital", "internet", "startup"],
            "HISTORY": ["history", "historical", "ancient", "century", "war", "empire", "dynasty", "independence"],
            "POLITICS_GOVERNMENT": ["politics", "government", "president", "minister", "parliament", "election", "policy", "law"],
            "ECONOMICS_BUSINESS": ["economy", "business", "market", "stock", "gdp", "inflation", "trade", "company", "revenue"],
            "GEOGRAPHY": ["geography", "country", "city", "river", "mountain", "continent", "region", "capital"],
            "SPACE_ASTRONOMY": ["space", "nasa", "planet", "star", "galaxy", "moon", "mars", "orbit", "telescope", "astronomy"],
            "ENVIRONMENT_CLIMATE": ["environment", "climate", "pollution", "carbon", "emission", "global warming", "renewable"],
            "SOCIETY_CULTURE": ["society", "culture", "tradition", "religion", "festival", "community", "population"],
            "LAW_CRIME": ["law", "crime", "court", "judge", "police", "arrest", "trial", "legal", "criminal", "supreme court"],
            "SPORTS": ["sports", "cricket", "football", "soccer", "olympics", "tournament", "match", "player", "team"],
            "ENTERTAINMENT": ["entertainment", "movie", "film", "music", "actor", "celebrity", "show", "concert"],
        }
        
        scores = {}
        for label, keywords in keyword_map.items():
            score = sum(1 for kw in keywords if kw in claim_lower)
            if score > 0:
                scores[label] = score
        
        if scores:
            best_label = max(scores, key=scores.get)
            confidence = min(scores[best_label] / 3.0, 0.85)
            return best_label, confidence
        
        return "GENERAL_FACTUAL", 0.5
    
    def _classify_level2(self, claim: str, level1: str) -> Tuple[str, float]:
        """Classify level-2 subcategory within level-1."""
        claim_lower = claim.lower()
        
        # Get level-2 candidates for this level-1 category
        level2_candidates = CONTEXT_TAXONOMY.get(level1, [])
        
        # Score each level-2 subcategory
        scores = {}
        for l2_label in level2_candidates:
            keywords = LEVEL2_KEYWORDS.get(l2_label, [])
            score = sum(1 for kw in keywords if kw in claim_lower)
            if score > 0:
                scores[l2_label] = score
        
        if scores:
            best_l2 = max(scores, key=scores.get)
            confidence = min(scores[best_l2] / 2.0, 0.9)  # Normalize
            return best_l2, confidence
        
        # No level-2 match
        return "general", 0.3
