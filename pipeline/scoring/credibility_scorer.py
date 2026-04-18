from pipeline.scoring import DOMAIN_CREDIBILITY


class CredibilityScorer:
    def get_weight(self, source: str, url: str = "") -> float:
        src = (source or "").lower()
        link = (url or "").lower()
        for domain, weight in DOMAIN_CREDIBILITY.items():
            if domain in src or domain in link:
                return weight
        return 0.7
