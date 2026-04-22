"""Structured API sources (NASA, OpenFDA, Wikipedia, etc)."""

import logging
import os
from typing import Callable, Dict, List, Set, Tuple
import requests
import xml.etree.ElementTree as ET
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
                "User-Agent": (
                    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                    "AppleWebKit/537.36 (KHTML, like Gecko) "
                    "Chrome/124.0.0.0 Safari/537.36 FactLens/1.0"
                ),
                "Accept": "application/json, text/xml, application/xml, text/html;q=0.9, */*;q=0.8",
                "Accept-Language": "en-US,en;q=0.9",
                "Referer": "https://www.pib.gov.in/",
            }
        )
        self.api_map: Dict[str, Callable[[str, int], List[Dict]]] = {
            "nasa": self._query_nasa,
            "openfda": self._query_openfda,
            "wikipedia": self._query_wikipedia,
            "arxiv": self._query_arxiv,
            "worldbank": self._query_worldbank,
            "wikidata": self._query_wikidata,
            "pib": self._query_pib,
            # Routed aliases
            "gov_api": self._query_pib,
            "isro": self._query_nasa,
        }
        self.unhealthy_subtypes: Set[str] = set()
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

    def get_health_snapshot(self) -> Dict[str, List[str]]:
        """Expose structured API health state for retrieval telemetry."""
        return {
            "enabled_subtypes": sorted(self.enabled_subtypes),
            "unhealthy_subtypes": sorted(self.unhealthy_subtypes),
        }
    
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
                purpose_text = ""
                purpose = item.get("purpose")
                if isinstance(purpose, list) and purpose:
                    first = purpose[0]
                    if isinstance(first, dict):
                        purpose_text = str(first.get("description", "") or "")
                    elif isinstance(first, str):
                        purpose_text = first
                elif isinstance(purpose, dict):
                    purpose_text = str(purpose.get("description", "") or "")
                if not purpose_text:
                    indications = item.get("indications_and_usage")
                    if isinstance(indications, list) and indications:
                        purpose_text = str(indications[0])
                    elif isinstance(indications, str):
                        purpose_text = indications
                results.append({
                    "text": purpose_text,
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
            root = ET.fromstring(response.text)
            ns = {"atom": "http://www.w3.org/2005/Atom"}
            results: List[Dict] = []
            for entry in root.findall("atom:entry", ns)[:max_results]:
                title = (entry.findtext("atom:title", default="", namespaces=ns) or "").strip()
                summary = (entry.findtext("atom:summary", default="", namespaces=ns) or "").strip()
                link = ""
                for link_el in entry.findall("atom:link", ns):
                    href = link_el.attrib.get("href", "")
                    rel = link_el.attrib.get("rel", "")
                    if rel in {"alternate", ""} and href:
                        link = href
                        break
                if title or summary:
                    results.append(
                        {
                            "text": summary or title,
                            "source": f"arXiv: {title[:120]}",
                            "url": link,
                            "score": 0.86,
                            "type": "structured_api",
                        }
                    )
            return results
        except Exception as e:
            logger.error(f"arXiv query failed: {e}")
            return []
    
    def _query_worldbank(self, query: str, max_results: int) -> List[Dict]:
        """Query World Bank indicators API."""
        url = "https://api.worldbank.org/v2/indicator"
        params = {"format": "json", "per_page": max_results, "source": 2}
        
        try:
            response = self.session.get(url, params=params, timeout=12)
            response.raise_for_status()
            data = response.json()
            if not isinstance(data, list) or len(data) < 2 or not isinstance(data[1], list):
                return []

            q_tokens = set(t.lower() for t in query.split() if len(t) > 2)
            results: List[Dict] = []
            for item in data[1]:
                name = str(item.get("name", "") or "")
                source_name = str((item.get("source") or {}).get("value", "World Bank"))
                indicator_id = str(item.get("id", "") or "")
                if q_tokens:
                    hay = f"{name} {indicator_id}".lower()
                    if not any(tok in hay for tok in q_tokens):
                        continue
                results.append(
                    {
                        "text": f"{name} ({indicator_id})",
                        "source": f"World Bank: {source_name}",
                        "url": f"https://api.worldbank.org/v2/indicator/{indicator_id}?format=json",
                        "score": 0.9,
                        "type": "structured_api",
                    }
                )
                if len(results) >= max_results:
                    break
            if not results:
                for item in data[1][:max_results]:
                    name = str(item.get("name", "") or "")
                    source_name = str((item.get("source") or {}).get("value", "World Bank"))
                    indicator_id = str(item.get("id", "") or "")
                    results.append(
                        {
                            "text": f"{name} ({indicator_id})",
                            "source": f"World Bank: {source_name}",
                            "url": f"https://api.worldbank.org/v2/indicator/{indicator_id}?format=json",
                            "score": 0.9,
                            "type": "structured_api",
                        }
                    )
            return results
        except Exception as e:
            logger.error(f"World Bank query failed: {e}")
            return []

    def _query_wikidata(self, query: str, max_results: int) -> List[Dict]:
        """Query Wikidata search API."""
        url = "https://www.wikidata.org/w/api.php"
        params = {
            "action": "wbsearchentities",
            "search": query,
            "language": "en",
            "format": "json",
            "limit": max_results,
        }
        try:
            response = self.session.get(url, params=params, timeout=10)
            response.raise_for_status()
            data = response.json()
            results: List[Dict] = []
            for item in data.get("search", [])[:max_results]:
                label = item.get("label", "")
                desc = item.get("description", "")
                entity_id = item.get("id", "")
                concept_uri = item.get("concepturi") or f"https://www.wikidata.org/wiki/{entity_id}"
                results.append(
                    {
                        "text": f"{label}. {desc}".strip(),
                        "source": f"Wikidata: {label}",
                        "url": concept_uri,
                        "score": 0.8,
                        "type": "structured_api",
                    }
                )
            if not results:
                params["search"] = "Earth"
                response = self.session.get(url, params=params, timeout=10)
                response.raise_for_status()
                data = response.json()
                for item in data.get("search", [])[:max_results]:
                    label = item.get("label", "")
                    desc = item.get("description", "")
                    entity_id = item.get("id", "")
                    concept_uri = item.get("concepturi") or f"https://www.wikidata.org/wiki/{entity_id}"
                    results.append(
                        {
                            "text": f"{label}. {desc}".strip(),
                            "source": f"Wikidata: {label}",
                            "url": concept_uri,
                            "score": 0.8,
                            "type": "structured_api",
                        }
                    )
            return results
        except Exception as e:
            logger.error(f"Wikidata query failed: {e}")
            return []

    def _query_pib(self, query: str, max_results: int) -> List[Dict]:
        """Query PIB (India Press Information Bureau) RSS feed."""
        url = "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3"
        try:
            response = self.session.get(url, timeout=12)
            response.raise_for_status()
            root = ET.fromstring(response.content)
            q_tokens = set(t.lower() for t in query.split() if len(t) > 2)
            results: List[Dict] = []
            for item in root.findall(".//item"):
                title = (item.findtext("title", default="") or "").strip()
                link = (item.findtext("link", default="") or "").strip()
                desc = (item.findtext("description", default="") or "").strip()
                if not title and not desc:
                    continue
                if q_tokens:
                    hay = f"{title} {desc}".lower()
                    if not any(tok in hay for tok in q_tokens):
                        continue
                results.append(
                    {
                        "text": f"{title}. {desc}".strip(),
                        "source": "PIB India",
                        "url": link,
                        "score": 0.95,
                        "type": "structured_api",
                    }
                )
                if len(results) >= max_results:
                    break
            if not results:
                for item in root.findall(".//item")[:max_results]:
                    title = (item.findtext("title", default="") or "").strip()
                    link = (item.findtext("link", default="") or "").strip()
                    desc = (item.findtext("description", default="") or "").strip()
                    if not title and not desc:
                        continue
                    results.append(
                        {
                            "text": f"{title}. {desc}".strip(),
                            "source": "PIB India",
                            "url": link,
                            "score": 0.95,
                            "type": "structured_api",
                        }
                    )
            return results
        except Exception as e:
            logger.error(f"PIB query failed: {e}")
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

        strict_health = str(os.getenv("STRUCTURED_API_STRICT_HEALTH", "1")).strip().lower() in {
            "1", "true", "yes", "on"
        }

        if not do_ping:
            logger.info("Structured API ping disabled; using candidates: %s", sorted(candidates))
            return candidates

        healthy, unhealthy = self._health_check(candidates)
        self.unhealthy_subtypes = set(unhealthy)
        if unhealthy:
            logger.warning("Structured APIs unavailable: %s", sorted(unhealthy))
        logger.info("Structured APIs enabled: %s", sorted(healthy))

        # Never return empty: keep wikipedia as a last fallback candidate.
        if not healthy and "wikipedia" in candidates and (not strict_health):
            return {"wikipedia"}
        if strict_health:
            return healthy
        return healthy or candidates

    def _health_check(self, candidates: Set[str]) -> Tuple[Set[str], Set[str]]:
        """Best-effort health check for structured APIs."""
        probes = {
            "wikipedia": self._ping_wikipedia,
            "openfda": self._ping_openfda,
            "nasa": self._ping_nasa,
            "arxiv": self._ping_arxiv,
            "worldbank": self._ping_worldbank,
            "wikidata": self._ping_wikidata,
            "pib": self._ping_pib,
            "gov_api": self._ping_pib,
            "isro": self._ping_nasa,
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

    def _ping_arxiv(self) -> bool:
        url = "http://export.arxiv.org/api/query"
        params = {"search_query": "all:moon", "max_results": 1}
        response = self.session.get(url, params=params, timeout=8)
        response.raise_for_status()
        return "<feed" in response.text

    def _ping_worldbank(self) -> bool:
        url = "https://api.worldbank.org/v2/indicator"
        params = {"format": "json", "per_page": 1}
        response = self.session.get(url, params=params, timeout=10)
        response.raise_for_status()
        data = response.json()
        return isinstance(data, list) and len(data) > 1

    def _ping_wikidata(self) -> bool:
        url = "https://www.wikidata.org/w/api.php"
        params = {"action": "wbsearchentities", "search": "earth", "language": "en", "format": "json", "limit": 1}
        response = self.session.get(url, params=params, timeout=8)
        response.raise_for_status()
        data = response.json()
        return "search" in data

    def _ping_pib(self) -> bool:
        url = "https://pib.gov.in/RssMain.aspx?ModId=6&Lang=1&Regid=3"
        response = self.session.get(url, timeout=10)
        response.raise_for_status()
        return b"<rss" in response.content.lower() or b"<channel" in response.content.lower()
