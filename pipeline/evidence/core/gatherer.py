"""Evidence gathering from multiple sources."""

import logging
import os
import re
import math
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Set, Tuple
from urllib.parse import urlparse, urlunparse

from pipeline.evidence.providers.web_search import WebSearchEngine
from pipeline.evidence.providers.structured_api import StructuredAPIClient
from pipeline.evidence.providers.scraper import EvidenceScraper
from pipeline.evidence.core.aggregator import EvidenceAggregator
from pipeline.core.normalizer import ClaimNormalizer

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
        self.scraper_enrich_max_results = _env_int("EVIDENCE_SCRAPER_ENRICH_MAX_RESULTS", 6, minimum=1)
        self.enable_translated_query_search = _env_bool("EVIDENCE_ENABLE_TRANSLATED_QUERY_SEARCH", True)
        self.translated_query_cap = _env_int("EVIDENCE_TRANSLATED_QUERY_CAP", 1, minimum=1)
        self.domain_diversity_enabled = _env_bool("EVIDENCE_DOMAIN_DIVERSITY_ENABLED", True)
        self.domain_max_per_host = _env_int("EVIDENCE_DOMAIN_MAX_PER_HOST", 2, minimum=1)
        self.mmr_enabled = _env_bool("EVIDENCE_MMR_ENABLED", True)
        self.mmr_lambda = max(0.0, min(1.0, float(os.getenv("EVIDENCE_MMR_LAMBDA", "0.75"))))
        self.mmr_candidates_mult = max(1, _env_int("EVIDENCE_MMR_CANDIDATES_MULT", 3, minimum=1))
        self.normalizer = ClaimNormalizer()
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
        max_evidence: int = 10,
        scraper_enrichment_override: Optional[bool] = None,
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
        queries = self._augment_queries_with_translation(claim, queries, language)
        enrichment_enabled = (
            self.enable_scraper_enrichment
            if scraper_enrichment_override is None
            else bool(scraper_enrichment_override)
        )

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
            f"scrape_enrichment enabled={enrichment_enabled} discovered_urls={len(discovered_urls)}"
        )
        if enrichment_enabled and discovered_urls:
            scraped = self.scraper.scrape_urls(
                discovered_urls,
                claim=claim,
                max_results=self.scraper_enrich_max_results,
            )
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
        unique_evidence = self._dedupe_canonical_evidence(all_evidence)
        unique_evidence = self.aggregator.deduplicate(unique_evidence)
        
        # Rank by quality indicators
        ranked_evidence = self.aggregator.rank(unique_evidence, claim)
        if self.mmr_enabled:
            ranked_evidence = self._apply_mmr(
                rows=ranked_evidence,
                max_items=max_evidence,
                lambda_coeff=self.mmr_lambda,
                candidate_mult=self.mmr_candidates_mult,
            )
        if self.domain_diversity_enabled:
            ranked_evidence = self._apply_domain_diversity(
                ranked_evidence,
                max_items=max_evidence,
                per_host_cap=self.domain_max_per_host,
            )
        out = ranked_evidence[:max_evidence]
        self._emit(
            f"done raw_total={len(all_evidence)} unique={len(unique_evidence)} ranked={len(ranked_evidence)} returned={len(out)}"
        )
        
        # Return top N
        return out

    def _augment_queries_with_translation(self, claim: str, queries: List[str], language: str) -> List[str]:
        """For non-English claims, add a small set of translated English queries for retrieval breadth."""
        base = [q for q in (queries or []) if str(q).strip()]
        if not self.enable_translated_query_search:
            return base
        lang = (language or "en").strip().lower()
        if lang.startswith("en"):
            return base
        try:
            translated = self.normalizer._translate_indic_to_english(claim, language=lang)  # pylint: disable=protected-access
        except Exception:
            translated = ""
        translated = str(translated or "").strip()
        if not translated:
            return base
        extra = [translated]
        extra = [q for q in extra if str(q).strip()][: self.translated_query_cap]
        # Keep translated queries first so WEB_SEARCH_MAX_QUERIES slicing still includes them.
        out: List[str] = []
        seen = set()
        for q in extra + base:
            key = str(q).strip().lower()
            if not key or key in seen:
                continue
            seen.add(key)
            out.append(str(q).strip())
        return out

    def _canonicalize_url(self, url: str) -> str:
        raw = str(url or "").strip()
        if not raw:
            return ""
        try:
            p = urlparse(raw)
            cleaned = p._replace(query="", fragment="")
            return urlunparse(cleaned).rstrip("/").lower()
        except Exception:
            return raw.lower()

    def _dedupe_canonical_evidence(self, rows: List[Dict]) -> List[Dict]:
        """Strong in-gathering dedupe: canonical URL first, fallback to normalized text key."""
        out: List[Dict] = []
        seen = set()
        for row in rows or []:
            url_key = self._canonicalize_url(row.get("url"))
            text_key = " ".join(str(row.get("text") or "").strip().lower().split())[:240]
            key = url_key or text_key
            if not key:
                continue
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    def _host_from_url(self, url: str) -> str:
        try:
            return (urlparse(str(url or "")).netloc or "").lower().split(":")[0].strip(".")
        except Exception:
            return ""

    def _title_key(self, row: Dict[str, Any]) -> str:
        title = str(row.get("title") or "").strip().lower()
        title = re.sub(r"\s+", " ", title)
        if len(title) < 24:
            return ""
        return title[:200]

    def _apply_domain_diversity(self, rows: List[Dict], max_items: int, per_host_cap: int) -> List[Dict]:
        """Prefer host diversity in top evidence while preserving quality order as much as possible."""
        if not rows:
            return rows
        selected: List[Dict] = []
        host_counts: Dict[str, int] = {}
        seen_title: Set[str] = set()

        # First pass: pick high quality rows with host cap and title dedupe.
        for row in rows:
            if len(selected) >= max_items:
                break
            host = self._host_from_url(row.get("url"))
            if host and host_counts.get(host, 0) >= per_host_cap:
                continue
            tkey = self._title_key(row)
            if tkey and tkey in seen_title:
                continue
            selected.append(row)
            if host:
                host_counts[host] = host_counts.get(host, 0) + 1
            if tkey:
                seen_title.add(tkey)

        # Second pass: fill if we under-filled because of strict diversity.
        if len(selected) < max_items:
            for row in rows:
                if len(selected) >= max_items:
                    break
                if row in selected:
                    continue
                tkey = self._title_key(row)
                if tkey and tkey in seen_title:
                    continue
                selected.append(row)
                if tkey:
                    seen_title.add(tkey)

        return selected

    def _tokenize_for_similarity(self, text: str) -> Set[str]:
        tokens = re.findall(r"[A-Za-z\u0900-\u0D7F]{3,}", str(text or "").lower())
        return set(tokens)

    def _jaccard(self, a: Set[str], b: Set[str]) -> float:
        if not a or not b:
            return 0.0
        inter = len(a.intersection(b))
        union = len(a.union(b))
        if union <= 0:
            return 0.0
        return inter / union

    def _apply_mmr(
        self,
        rows: List[Dict],
        max_items: int,
        lambda_coeff: float = 0.75,
        candidate_mult: int = 3,
    ) -> List[Dict]:
        """MMR rerank to balance relevance with diversity."""
        if not rows:
            return rows
        if max_items <= 0:
            return rows

        cap = min(len(rows), max_items * max(1, candidate_mult))
        candidates = rows[:cap]
        if len(candidates) <= 1:
            return candidates

        toks = [
            self._tokenize_for_similarity(
                f"{r.get('title','')} {r.get('source','')} {r.get('text','')}"
            )
            for r in candidates
        ]
        selected_idx: List[int] = []
        remaining = list(range(len(candidates)))

        while remaining and len(selected_idx) < max_items:
            best_idx = None
            best_score = -math.inf
            for i in remaining:
                rel = float(candidates[i].get("score", candidates[i].get("relevance", 0.0)) or 0.0)
                sim_penalty = 0.0
                if selected_idx:
                    sim_penalty = max(self._jaccard(toks[i], toks[j]) for j in selected_idx)
                mmr = (lambda_coeff * rel) - ((1.0 - lambda_coeff) * sim_penalty)
                if mmr > best_score:
                    best_score = mmr
                    best_idx = i
            if best_idx is None:
                break
            selected_idx.append(best_idx)
            remaining.remove(best_idx)

        selected_rows = [candidates[i] for i in selected_idx]
        leftovers = [candidates[i] for i in range(len(candidates)) if i not in selected_idx]
        out = selected_rows + leftovers + rows[cap:]
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
        date_source = "explicit" if published_at else "none"
        if not published_at:
            published_at = self._extract_date_from_text(
                " ".join(x for x in [title, snippet, text] if x)
            )
            if published_at:
                date_source = "inferred"
        published_at = self._normalize_published_at(published_at)
        if not published_at:
            date_source = "none"

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
        normalized["metadata"]["date_source"] = date_source
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

    def _extract_date_from_text(self, text: str) -> str:
        """Best-effort date extraction from snippet/title text."""
        src = str(text or "")
        if not src:
            return ""

        # December 11, 2022 / Dec 11, 2022
        m = re.search(
            r"\b(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|Aug(?:ust)?|"
            r"Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{1,2},\s+\d{4}\b",
            src,
            flags=re.IGNORECASE,
        )
        if m:
            return m.group(0)

        # 11 December 2022
        m = re.search(
            r"\b\d{1,2}\s+(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|Jul(?:y)?|"
            r"Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)\s+\d{4}\b",
            src,
            flags=re.IGNORECASE,
        )
        if m:
            return m.group(0)

        # ISO-like formats: 2022-12-11 or 2022/12/11
        m = re.search(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b", src)
        if m:
            return m.group(0)

        return ""

    def _normalize_published_at(self, value: str) -> str:
        """Normalize many date strings to ISO datetime."""
        raw = str(value or "").strip()
        if not raw:
            return ""

        # try common direct parse formats first
        formats = [
            "%Y-%m-%d",
            "%Y/%m/%d",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%SZ",
            "%Y-%m-%d %H:%M:%S",
            "%B %d, %Y",
            "%b %d, %Y",
            "%d %B %Y",
            "%d %b %Y",
        ]
        for fmt in formats:
            try:
                dt = datetime.strptime(raw, fmt)
                return dt.isoformat()
            except Exception:
                continue

        # fallback: keep original if it already looks like ISO-ish
        if re.match(r"^\d{4}-\d{1,2}-\d{1,2}", raw):
            return raw
        return ""
