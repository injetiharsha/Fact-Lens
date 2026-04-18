"""Evidence gathering from multiple sources."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple

from pipeline.evidence.providers.web_search import WebSearchEngine
from pipeline.evidence.providers.structured_api import StructuredAPIClient
from pipeline.evidence.providers.scraper import EvidenceScraper
from pipeline.evidence.core.aggregator import EvidenceAggregator

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else int(default)
    except Exception:
        value = int(default)
    return max(minimum, value)


class EvidenceGatherer:
    """Gather evidence from structured APIs, web search, and scraping."""
    
    def __init__(self, config: Dict):
        """Initialize evidence gathering components."""
        self.web_search = WebSearchEngine(config)
        self.api_client = StructuredAPIClient(config)
        self.scraper = EvidenceScraper(config)
        self.aggregator = EvidenceAggregator()
        self.parallel_sources = _env_bool("EVIDENCE_PARALLEL_SOURCES", True)
        self.source_workers = _env_int("EVIDENCE_PARALLEL_SOURCE_WORKERS", 4, minimum=1)
        self.enable_scraper_enrichment = _env_bool("EVIDENCE_ENABLE_SCRAPER_ENRICHMENT", True)
        self.live_progress = _env_bool("EVIDENCE_LIVE_PROGRESS", False)

    def _emit(self, msg: str) -> None:
        if self.live_progress:
            print(f"[evidence] {msg}", flush=True)
    
    def gather(
        self,
        claim: str,
        queries: List[str],
        sources: List[Dict],
        language: str = "en",
        max_evidence: int = 10
    ) -> List[Dict]:
        """
        Gather evidence from multiple sources.
        
        Args:
            claim: Normalized claim text
            queries: Search queries
            sources: Routed evidence sources
            language: Language code
            max_evidence: Maximum evidence items to return
            
        Returns:
            List of evidence dicts with text, source, url, metadata
        """
        logger.info(f"Gathering evidence for claim: {claim[:50]}...")
        self._emit(
            f"start lang={language} max_evidence={max_evidence} queries={len(queries)} routed_sources={len(sources)}"
        )
        
        all_evidence = []
        
        # Keep unique source pairs in-order.
        deduped_sources = self._dedupe_sources(sources)
        self._emit(f"deduped_sources={len(deduped_sources)}")

        if self.parallel_sources and len(deduped_sources) > 1:
            workers = min(self.source_workers, len(deduped_sources))
            self._emit(f"mode=parallel workers={workers}")
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [
                    pool.submit(self._gather_one_source, source, claim, queries, language)
                    for source in deduped_sources
                ]
                for future in as_completed(futures):
                    try:
                        normalized = future.result()
                        all_evidence.extend(normalized)
                        self._emit(f"source_done items={len(normalized)} running_total={len(all_evidence)}")
                    except Exception as e:
                        logger.error("Parallel source gather failed: %s", e)
                        self._emit(f"source_error err={e}")
        else:
            # Gather from each source type (sequential fallback mode)
            self._emit("mode=sequential")
            for source in deduped_sources:
                try:
                    normalized = self._gather_one_source(source, claim, queries, language)
                    all_evidence.extend(normalized)
                    self._emit(
                        f"source_done type={source.get('type')} subtype={source.get('subtype')} "
                        f"items={len(normalized)} running_total={len(all_evidence)}"
                    )
                except Exception as e:
                    source_type = source.get("type")
                    source_subtype = source.get("subtype")
                    logger.error(f"Error gathering from {source_type}:{source_subtype}: {e}")
                    self._emit(f"source_error type={source_type} subtype={source_subtype} err={e}")

        # Scrape discovered URLs (inclusive enrichment).
        discovered_urls = self._collect_urls(all_evidence)
        self._emit(
            f"scrape_enrichment enabled={self.enable_scraper_enrichment} discovered_urls={len(discovered_urls)}"
        )
        if self.enable_scraper_enrichment and discovered_urls:
            scraped = self.scraper.scrape_urls(discovered_urls, claim=claim, max_results=3)
            normalized_scraped = self._normalize_evidence_list(scraped, default_type="scraping")
            all_evidence.extend(normalized_scraped)
            self._emit(
                f"scrape_enrichment_done raw={len(scraped)} normalized={len(normalized_scraped)} total={len(all_evidence)}"
            )
            logger.info(
                "URL scraping enrichment returned %d items (%d after normalization)",
                len(scraped),
                len(normalized_scraped),
            )
        
        # Deduplicate and rank
        if not all_evidence:
            logger.warning("No evidence found from any source")
            self._emit("done no_evidence")
            return []
        
        # Remove duplicates
        unique_evidence = self.aggregator.deduplicate(all_evidence)
        
        # Rank by quality indicators
        ranked_evidence = self.aggregator.rank(unique_evidence, claim)
        out = ranked_evidence[:max_evidence]
        self._emit(
            f"done raw_total={len(all_evidence)} unique={len(unique_evidence)} ranked={len(ranked_evidence)} returned={len(out)}"
        )
        
        # Return top N
        return out

    def _gather_one_source(
        self,
        source: Dict[str, Any],
        claim: str,
        queries: List[str],
        language: str,
    ) -> List[Dict[str, Any]]:
        source_type = source["type"]
        source_subtype = source.get("subtype")

        if source_type == "structured_api":
            if source_subtype and source_subtype not in self.api_client.get_available_subtypes():
                logger.info("Skipping unhealthy structured API subtype: %s", source_subtype)
                return []
            evidence = self._gather_from_api(claim, queries, source_subtype, language)
        elif source_type == "web_search":
            evidence = self._gather_from_web_search(claim, queries, source_subtype, language)
        elif source_type == "scraping":
            evidence = self._gather_from_scraping(claim, queries, source_subtype, language)
        else:
            logger.warning(f"Unknown source type: {source_type}")
            evidence = []

        normalized = self._normalize_evidence_list(evidence, default_type=source_type)
        logger.info(
            "Source %s:%s returned %d items (%d after normalization)",
            source_type,
            source_subtype,
            len(evidence),
            len(normalized),
        )
        self._emit(
            f"source_result type={source_type} subtype={source_subtype} raw={len(evidence)} normalized={len(normalized)}"
        )
        return normalized
    
    def _gather_from_api(
        self,
        claim: str,
        queries: List[str],
        api_subtype: str,
        language: str
    ) -> List[Dict]:
        """Gather evidence from structured API."""
        return self.api_client.query(claim, queries, api_subtype, language)
    
    def _gather_from_web_search(
        self,
        claim: str,
        queries: List[str],
        search_subtype: str,
        language: str
    ) -> List[Dict]:
        """Gather evidence from web search."""
        return self.web_search.search(claim, queries, search_subtype, language)
    
    def _gather_from_scraping(
        self,
        claim: str,
        queries: List[str],
        scrape_subtype: str,
        language: str
    ) -> List[Dict]:
        """Gather evidence from web scraping."""
        return self.scraper.scrape(claim, queries, scrape_subtype, language)

    def _dedupe_sources(self, sources: List[Dict]) -> List[Dict]:
        """Deduplicate routed sources by (type, subtype) preserving order."""
        seen: Set[Tuple[str, str]] = set()
        output: List[Dict] = []
        for source in sources:
            source_type = str(source.get("type"))
            source_subtype = str(source.get("subtype"))

            # Avoid duplicate web-search fanout across near-identical subtypes
            # (tech/general/regulation). Keep news-like routes distinct.
            if source_type == "web_search":
                sub = source_subtype.lower()
                if "news" in sub:
                    key = ("web_search", "news")
                else:
                    key = ("web_search", "generic")
            else:
                key = (source_type, source_subtype)

            if key in seen:
                continue
            seen.add(key)
            output.append(source)
        return output

    def _collect_urls(self, evidence: List[Dict]) -> List[str]:
        """Collect candidate URLs from already gathered evidence."""
        urls: List[str] = []
        seen: Set[str] = set()
        for ev in evidence:
            url = str(ev.get("url") or "").strip()
            if not url or url in seen:
                continue
            seen.add(url)
            urls.append(url)
        return urls

    def _normalize_evidence_list(self, rows: List[Dict], default_type: str) -> List[Dict]:
        """Normalize provider-specific rows into canonical evidence schema."""
        if not isinstance(rows, list):
            return []

        out: List[Dict] = []
        for row in rows:
            normalized = self._normalize_evidence_row(row, default_type=default_type)
            if normalized:
                out.append(normalized)
        return out

    def _normalize_evidence_row(self, row: Dict[str, Any], default_type: str) -> Optional[Dict[str, Any]]:
        """Map loose retrieval formats into {text,url,title,source,...}."""
        if not isinstance(row, dict):
            return None

        text = self._pick_text(
            row,
            keys=["text", "content", "snippet", "body", "description", "summary", "abstract"],
        )
        title = self._pick_text(row, keys=["title", "headline", "name"])
        snippet = self._pick_text(row, keys=["snippet", "description", "summary", "content", "body"])
        if not text:
            text = " ".join(x for x in [title, snippet] if x).strip()
        if not text:
            logger.debug("Dropping evidence row with empty text after normalization: keys=%s", sorted(row.keys()))
            return None

        url = self._pick_text(row, keys=["url", "source_url", "link", "href", "permalink"])
        source = self._pick_text(
            row,
            keys=["source", "source_name", "publisher", "provider", "site_name"],
        )
        if not source:
            source = title or "unknown"

        published_at = self._pick_text(
            row,
            keys=["published_at", "published", "date", "pub_date", "timestamp", "time"],
        )

        score = self._to_float(row.get("score"))
        if score is None:
            score = self._to_float(row.get("relevance"))
        if score is None:
            score = self._to_float(row.get("confidence"))
        if score is None:
            score = 0.5

        normalized: Dict[str, Any] = {
            "text": text[:4000],
            "url": url or "",
            "title": title or source,
            "source": source,
            "score": max(0.0, min(1.0, float(score))),
            "type": str(row.get("type") or default_type or "web_search"),
            "published_at": published_at or "",
            "lang": self._pick_text(row, keys=["lang", "language"]) or "",
            "metadata": {"raw_keys": sorted(row.keys())},
        }
        return normalized

    def _pick_text(self, row: Dict[str, Any], keys: List[str]) -> str:
        for key in keys:
            value = row.get(key)
            if value is None:
                continue
            text = str(value).strip()
            if text:
                return text
        return ""

    def _to_float(self, value: Any) -> Optional[float]:
        try:
            if value is None:
                return None
            return float(value)
        except Exception:
            return None
