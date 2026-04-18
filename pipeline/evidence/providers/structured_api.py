"""Structured API sources (NASA, OpenFDA, Wikipedia, etc)."""

import logging
import os
from typing import Callable, Dict, List, Set, Tuple
import requests
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv(override=True)
logger = logging.getLogger(__name__)


class StructuredAPIClient:
    """Query structured APIs for evidence."""
    
    def __init__(self, config: Dict):
        """Initialize with config."""
        self.config = config
        self.nasa_api_key = os.getenv("NASA_API_KEY")
        self.nasa_timeout_s = int(os.getenv("NASA_API_TIMEOUT_SECONDS", "20"))
        self.openfda_timeout_s = int(os.getenv("OPENFDA_API_TIMEOUT_SECONDS", "20"))
        self.session = requests.Session()
        self.session.headers.update(
            {
                # Wikimedia requires a descriptive User-Agent for API access.
                "User-Agent": "FactLens/1.0 (evidence-retrieval; contact: admin@example.com)"
            }
        )
        self.api_map: Dict[str, Callable[[str, int], List[Dict]]] = {
            "nasa": self._query_nasa,
            "openfda": self._query_openfda,
            "wikipedia": self._query_wikipedia,
            "arxiv": self._query_arxiv,
            "worldbank": self._query_worldbank,
        }
        self.enabled_subtypes = self._resolve_enabled_subtypes()
    
    def query(
        self,
        claim: str,
        queries: List[str],
        api_subtype: str,
        language: str = "en",
        max_results: int = 3
    ) -> List[Dict]:
        """Query structured API for evidence."""
        subtype = (api_subtype or "wikipedia").lower()
        if subtype not in self.enabled_subtypes:
            logger.info("Structured subtype '%s' unavailable; fallback to wikipedia", subtype)
            subtype = "wikipedia"
            if subtype not in self.enabled_subtypes:
                return []

        query_fn = self.api_map.get(subtype)
        if not query_fn:
            logger.warning("Unknown API subtype: %s; falling back to wikipedia", subtype)
            return self._query_wikipedia(queries[0], max_results)
        
        try:
            q = queries[0] if queries else claim
            return query_fn(q, max_results)
        except Exception as e:
            logger.error(f"API query failed for {subtype}: {e}")
            return []

    def get_available_subtypes(self) -> Set[str]:
        """Return structured API subtypes currently considered healthy."""
        return set(self.enabled_subtypes)
    
    def _query_nasa(self, query: str, max_results: int) -> List[Dict]:
        """Query NASA API."""
        url = "https://images-api.nasa.gov/search"
        params = {"q": query, "media_type": "image"}
        try:
            response = self.session.get(url, params=params, timeout=self.nasa_timeout_s)
            response.raise_for_status()
            data = response.json()
            items = data.get("collection", {}).get("items", [])[:max_results]
            out: List[Dict] = []
            for item in items:
                data_items = item.get("data", [])
                links = item.get("links", [])
                if not data_items:
                    continue
                meta = data_items[0]
                out.append(
                    {
                        "text": meta.get("description") or meta.get("title", ""),
                        "source": f"NASA: {meta.get('title', 'result')}",
                        "url": links[0].get("href") if links else "",
                        "score": 0.92,
                        "type": "structured_api",
                    }
                )
            return out
        except Exception as e:
            logger.error(f"NASA query failed: {e}")
            return []
    
    def _query_openfda(self, query: str, max_results: int) -> List[Dict]:
        """Query OpenFDA API."""
        url = "https://api.fda.gov/drug/label.json"
        params = {"search": query, "limit": max_results}
        
        try:
            response = self.session.get(url, params=params, timeout=self.openfda_timeout_s)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("results", []):
                results.append({
                    "text": item.get("purpose", [{}])[0].get("description", "") if item.get("purpose") else "",
                    "source": "OpenFDA",
                    "url": f"https://api.fda.gov/drug/label.json",
                    "score": 0.9,  # High credibility for government API
                    "type": "structured_api"
                })
            
            return results
        except Exception as e:
            logger.error(f"OpenFDA query failed: {e}")
            return []
    
    def _query_wikipedia(self, query: str, max_results: int) -> List[Dict]:
        """Query Wikipedia API."""
        url = f"https://en.wikipedia.org/w/api.php"
        params = {
            "action": "query",
            "list": "search",
            "srsearch": query,
            "format": "json",
            "srlimit": max_results
        }
        
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            
            results = []
            for item in data.get("query", {}).get("search", []):
                title = item.get("title", "")
                snippet = item.get("snippet", "")
                snippet = BeautifulSoup(snippet, "html.parser").get_text(" ", strip=True)
                page_id = item.get("pageid", "")
                
                results.append({
                    "text": snippet,
                    "source": f"Wikipedia: {title}",
                    "url": f"https://en.wikipedia.org/?curid={page_id}",
                    "score": 0.75,  # Medium-high credibility
                    "type": "structured_api"
                })
            
            return results
        except Exception as e:
            logger.error(f"Wikipedia query failed: {e}")
            return []
    
    def _query_arxiv(self, query: str, max_results: int) -> List[Dict]:
        """Query arXiv API."""
        url = f"http://export.arxiv.org/api/query?search_query=all:{query}&max_results={max_results}"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            # Parse XML response (simplified)
            return []
        except Exception as e:
            logger.error(f"arXiv query failed: {e}")
            return []
    
    def _query_worldbank(self, query: str, max_results: int) -> List[Dict]:
        """Query World Bank API."""
        url = f"https://api.worldbank.org/v2/country/all/indicator/all?format=json&per_page={max_results}"
        
        try:
            response = self.session.get(url, timeout=10)
            response.raise_for_status()
            return []
        except Exception as e:
            logger.error(f"World Bank query failed: {e}")
            return []

    def _resolve_enabled_subtypes(self) -> Set[str]:
        """
        Resolve available structured APIs:
        1) Optional allowlist from env/config
        2) Optional health checks, keeping only working endpoints
        """
        allowlist_raw = os.getenv("STRUCTURED_API_ALLOWLIST", "").strip()
        allowlist_cfg = self.config.get("structured_api_allowlist", [])

        allowlist: Set[str] = set()
        if allowlist_raw:
            allowlist.update(s.strip().lower() for s in allowlist_raw.split(",") if s.strip())
        if isinstance(allowlist_cfg, list):
            allowlist.update(str(s).strip().lower() for s in allowlist_cfg if str(s).strip())

        candidates = set(self.api_map.keys())
        if allowlist:
            candidates = {s for s in candidates if s in allowlist}

        do_ping = bool(self.config.get("structured_api_ping", True))
        if str(os.getenv("STRUCTURED_API_PING", "1")).strip().lower() in {"0", "false", "no"}:
            do_ping = False

        if not do_ping:
            logger.info("Structured API ping disabled; using candidates: %s", sorted(candidates))
            return candidates

        healthy, unhealthy = self._health_check(candidates)
        if unhealthy:
            logger.warning("Structured APIs unavailable: %s", sorted(unhealthy))
        logger.info("Structured APIs enabled: %s", sorted(healthy))

        # Never return empty: keep wikipedia as a last fallback candidate.
        if not healthy and "wikipedia" in candidates:
            return {"wikipedia"}
        return healthy or candidates

    def _health_check(self, candidates: Set[str]) -> Tuple[Set[str], Set[str]]:
        """Best-effort health check for structured APIs."""
        probes = {
            "wikipedia": self._ping_wikipedia,
            "openfda": self._ping_openfda,
            "nasa": self._ping_nasa,
            # Placeholder handlers kept disabled by default until fully implemented.
            "arxiv": lambda: False,
            "worldbank": lambda: False,
        }
        healthy: Set[str] = set()
        unhealthy: Set[str] = set()
        for subtype in candidates:
            probe = probes.get(subtype)
            if not probe:
                unhealthy.add(subtype)
                continue
            try:
                if probe():
                    healthy.add(subtype)
                else:
                    unhealthy.add(subtype)
            except Exception:
                unhealthy.add(subtype)
        return healthy, unhealthy

    def _ping_wikipedia(self) -> bool:
        url = "https://en.wikipedia.org/w/api.php"
        params = {"action": "query", "list": "search", "srsearch": "earth", "format": "json", "srlimit": 1}
        response = self.session.get(url, params=params, timeout=self.openfda_timeout_s)
        response.raise_for_status()
        data = response.json()
        return "query" in data

    def _ping_openfda(self) -> bool:
        url = "https://api.fda.gov/drug/label.json"
        params = {"search": "aspirin", "limit": 1}
        response = self.session.get(url, params=params, timeout=self.nasa_timeout_s)
        response.raise_for_status()
        data = response.json()
        return "results" in data or "meta" in data

    def _ping_nasa(self) -> bool:
        url = "https://images-api.nasa.gov/search"
        params = {"q": "moon", "media_type": "image"}
        response = self.session.get(url, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        return "collection" in data
