"""Web scraping for evidence."""

import logging
import os
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set
from urllib.parse import urlparse
import requests
from pipeline.evidence.policy import RetrievalPolicy
from pipeline.evidence.scraper import (
    BeautifulSoupScraper,
    PlaywrightScraper,
    TrafilaturaScraper,
)

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


class EvidenceScraper:
    """Scrape web pages for evidence."""
    
    def __init__(self, config: Dict):
        """Initialize with config."""
        self.config = config
        self.session = requests.Session()
        self.session.headers.update({
            "User-Agent": "Mozilla/5.0 (Fact-Checking Bot)"
        })
        self.trafilatura = TrafilaturaScraper(session=self.session)
        self.playwright = PlaywrightScraper()
        self.bs4 = BeautifulSoupScraper(session=self.session)
        self.parallel_methods = _env_bool("SCRAPER_PARALLEL_METHODS", True)
        self.enable_playwright_fallback = _env_bool("SCRAPER_ENABLE_PLAYWRIGHT_FALLBACK", False)
        self.playwright_on_short_text = _env_int("SCRAPER_PLAYWRIGHT_SHORT_TEXT_THRESHOLD", 300, minimum=1)
        self.parallel_urls = _env_bool("SCRAPER_PARALLEL_URLS", True)
        self.url_workers = _env_int("SCRAPER_PARALLEL_WORKERS", 4, minimum=1)
        self.url_candidate_mult = _env_int("SCRAPER_URL_CANDIDATE_MULT", 1, minimum=1)
        self.min_text_chars = _env_int("SCRAPER_MIN_TEXT_CHARS", 120, minimum=1)
        self.min_claim_overlap = float(os.getenv("SCRAPER_MIN_CLAIM_OVERLAP", "0.06") or 0.06)
        self.boilerplate_penalty_weight = float(
            os.getenv("SCRAPER_BOILERPLATE_PENALTY_WEIGHT", "0.35") or 0.35
        )
        self.structured_quality_threshold = float(
            os.getenv("SCRAPE_STRUCTURED_QUALITY_THRESHOLD", "0.62") or 0.62
        )
        self.skip_file_exts = {
            ext.strip().lower().lstrip(".")
            for ext in str(
                os.getenv("SCRAPER_SKIP_FILE_EXTENSIONS", "pdf,json,csv,xml,zip")
            ).split(",")
            if ext.strip()
        }
        self.skip_api_hosts = {
            h.strip().lower()
            for h in str(
                os.getenv("SCRAPER_SKIP_API_HOSTS", "api.worldbank.org")
            ).split(",")
            if h.strip()
        }
        self.skip_api_path_tokens = {
            tok.strip().lower()
            for tok in str(
                os.getenv("SCRAPER_SKIP_API_PATH_TOKENS", "/api/,/v1/,/v2/,/graphql")
            ).split(",")
            if tok.strip()
        }
        self.policy = RetrievalPolicy.from_env(
            domain_env="SCRAPER_BAD_DOMAINS",
            token_env="SCRAPER_BLOCKED_URL_TOKENS",
            block_wordpress_env="SCRAPER_BLOCK_WORDPRESS",
            block_explicit_env="SCRAPER_BLOCK_EXPLICIT",
            default_bad_domains={
                "dailymotion.com",
                "testbook.com",
                "soundcloud.com",
                "spotify.com",
                "gaana.com",
                "jiosaavn.com",
                "wynk.in",
                "pornhub.com",
                "xvideos.com",
                "xnxx.com",
                "xhamster.com",
                "redtube.com",
                "youporn.com",
                "beeg.com",
                "adultfriendfinder.com",
            },
            default_tokens={
                "/questions/",
                "/question/",
                "/questiosn/",
                "/mcq",
                "mcq/",
                "quiz",
                "quizzes",
                "podcast",
                "audio",
                "/categories/44/",
            },
        )
    
    def scrape(
        self,
        claim: str,
        queries: List[str],
        subtype: str = None,
        language: str = "en",
        max_results: int = 3
    ) -> List[Dict]:
        """
        Scrape web pages for evidence.
        
        Returns:
            List of evidence dicts
        """
        # Query-only scraping is not reliable without search result URLs.
        # Use scrape_urls(...) from gathered web/API results.
        logger.info("Scraping needs URLs; call scrape_urls with discovered links")
        return []

    def scrape_urls(
        self,
        urls: List[str],
        claim: str,
        max_results: int = 3
    ) -> List[Dict]:
        """Scrape a list of URLs and return extracted evidence."""
        evidence: List[Dict] = []
        seen: Set[str] = set()
        candidates: List[str] = []
        for url in urls:
            if not url or url in seen:
                continue
            seen.add(url)
            if self._is_http_url(url):
                candidates.append(url)
        # Hard-cap candidate fanout so scraper enrichment does not dominate stage5 latency.
        candidate_cap = max(1, int(max_results)) * self.url_candidate_mult
        candidates = candidates[:candidate_cap]

        if self.parallel_urls and len(candidates) > 1 and max_results > 1:
            workers = min(self.url_workers, len(candidates))
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(self._scrape_url, url, claim) for url in candidates]
                for future in as_completed(futures):
                    try:
                        item = future.result()
                    except Exception as exc:
                        logger.debug("Parallel scrape task failed: %s", exc)
                        continue
                    if item and item.get("text"):
                        evidence.append(item)
                    if len(evidence) >= max_results:
                        break
        else:
            for url in candidates:
                item = self._scrape_url(url, claim)
                if item and item.get("text"):
                    evidence.append(item)
                if len(evidence) >= max_results:
                    break

        logger.info("Scraper extracted %d items from %d urls", len(evidence), len(urls))
        return evidence
    
    def _scrape_url(self, url: str, claim: str) -> Dict:
        """Scrape a single URL with fast parallel extraction first, optional heavy fallback."""
        primary_rows: List[Dict] = []
        primary_methods = [
            ("trafilatura", self.trafilatura),
            ("beautifulsoup", self.bs4),
        ]

        if self.parallel_methods:
            with ThreadPoolExecutor(max_workers=len(primary_methods)) as pool:
                futures = {
                    pool.submit(scraper.scrape_url, url): name
                    for name, scraper in primary_methods
                    if getattr(scraper, "enabled", True)
                }
                for future in as_completed(futures):
                    name = futures[future]
                    try:
                        row = future.result()
                    except Exception as exc:
                        logger.debug("Scraper method %s failed for %s: %s", name, url, exc)
                        continue
                    if row and row.get("text"):
                        row.setdefault("scrape_tier", name)
                        primary_rows.append(row)
        else:
            for name, scraper in primary_methods:
                if not getattr(scraper, "enabled", True):
                    continue
                try:
                    row = scraper.scrape_url(url)
                except Exception as exc:
                    logger.debug("Scraper method %s failed for %s: %s", name, url, exc)
                    continue
                if row and row.get("text"):
                    row.setdefault("scrape_tier", name)
                    primary_rows.append(row)

        best = self._pick_best_scrape_row(primary_rows, claim)
        if best:
            # Heavy fallback only when explicitly enabled and best text is too short.
            text_len = len(str(best.get("text") or ""))
            if (
                self.enable_playwright_fallback
                and text_len < self.playwright_on_short_text
                and getattr(self.playwright, "enabled", True)
            ):
                try:
                    pw_row = self.playwright.scrape_url(url)
                    if pw_row and pw_row.get("text"):
                        pw_row.setdefault("scrape_tier", "playwright")
                        if self._scrape_row_quality(pw_row, claim) > self._scrape_row_quality(best, claim):
                            self._enrich_scrape_row(pw_row, claim)
                            return pw_row
                except Exception as exc:
                    logger.debug("Playwright fallback failed for %s: %s", url, exc)
            self._enrich_scrape_row(best, claim)
            return best

        if self.enable_playwright_fallback and getattr(self.playwright, "enabled", True):
            try:
                row = self.playwright.scrape_url(url)
                if row and row.get("text"):
                    row.setdefault("scrape_tier", "playwright")
                    self._enrich_scrape_row(row, claim)
                    return row
            except Exception as exc:
                logger.debug("Playwright fallback failed for %s: %s", url, exc)

        logger.error("Failed to scrape %s with all enabled methods", url)
        return {}

    def _pick_best_scrape_row(self, rows: List[Dict], claim: str = "") -> Dict:
        if not rows:
            return {}
        filtered = [r for r in rows if self._is_row_usable(r, claim)]
        pool = filtered if filtered else rows
        best = max(pool, key=lambda r: self._scrape_row_quality(r, claim))
        return best or {}

    def _scrape_row_quality(self, row: Dict, claim: str = "") -> float:
        text = str(row.get("text") or "")
        title = str(row.get("title") or "")
        score = float(row.get("score", 0.0) or 0.0)
        overlap = self._claim_overlap_ratio(text, claim)
        boilerplate_ratio = self._boilerplate_ratio(text)
        row["scrape_claim_overlap"] = overlap
        row["scrape_boilerplate_ratio"] = boilerplate_ratio
        row["scrape_text_len"] = len(text)
        # Prefer richer extract and stable parser score.
        return (
            score
            + min(len(text), 4000) / 4000.0
            + (0.05 if title else 0.0)
            + (0.8 * overlap)
            - (self.boilerplate_penalty_weight * boilerplate_ratio)
        )

    def _is_row_usable(self, row: Dict, claim: str = "") -> bool:
        text = str(row.get("text") or "").strip()
        if not text:
            return False
        text_len = len(text)
        overlap = self._claim_overlap_ratio(text, claim)
        # Keep short rows only if they strongly overlap with claim tokens.
        if text_len < self.min_text_chars and overlap < self.min_claim_overlap:
            return False
        return True

    def _claim_overlap_ratio(self, text: str, claim: str) -> float:
        t_tokens = self._tokenize(text)
        c_tokens = self._tokenize(claim)
        if not t_tokens or not c_tokens:
            return 0.0
        return float(len(t_tokens.intersection(c_tokens))) / float(max(1, len(c_tokens)))

    def _tokenize(self, text: str) -> Set[str]:
        toks = re.findall(r"[A-Za-z\u0900-\u0D7F]{3,}", str(text or "").lower())
        return set(toks)

    def _boilerplate_ratio(self, text: str) -> float:
        s = str(text or "").lower()
        if not s:
            return 0.0
        markers = [
            "cookie",
            "privacy policy",
            "terms of service",
            "subscribe",
            "newsletter",
            "sign in",
            "advertisement",
            "all rights reserved",
            "javascript",
            "enable cookies",
            "accept all",
        ]
        hits = sum(s.count(m) for m in markers)
        # normalized soft ratio: 0..1-ish
        return min(1.0, hits / 12.0)

    def _enrich_scrape_row(self, row: Dict, claim: str = "") -> None:
        """Attach structured hints to scraped evidence for safer downstream scoring."""
        text = str(row.get("text") or "").strip()
        if not text:
            return
        fact_span = self._extract_fact_span(text, claim)
        entity = self._extract_entity(fact_span or text)
        date_str = self._extract_date(fact_span or text)
        numeric = self._extract_numeric_claim(fact_span or text)
        overlap = self._claim_overlap_ratio(fact_span or text, claim)
        has_date = 1.0 if date_str else 0.0
        has_num = 1.0 if numeric else 0.0
        has_entity = 1.0 if entity else 0.0
        quality = (0.55 * overlap) + (0.20 * has_date) + (0.15 * has_num) + (0.10 * has_entity)

        row["fact_span"] = fact_span
        row["entity"] = entity
        row["date"] = date_str
        row["numeric_claim"] = numeric
        row["source_type"] = "scraped_web"
        row["scrape_struct_quality"] = max(0.0, min(1.0, quality))
        row["structured_from_scrape"] = bool(quality >= self.structured_quality_threshold)
        # Keep focused snippet to reduce boilerplate noise in ranking/stance.
        if fact_span and len(fact_span) >= 60:
            row["text"] = fact_span

    def _extract_fact_span(self, text: str, claim: str = "") -> str:
        chunks = [c.strip() for c in re.split(r"(?<=[.!?])\s+|\n+", str(text or "")) if c.strip()]
        if not chunks:
            return ""
        claim_tokens = self._tokenize(claim)
        if not claim_tokens:
            return chunks[0][:700]
        keep: List[str] = []
        for sent in chunks:
            toks = self._tokenize(sent)
            overlap = len(claim_tokens.intersection(toks))
            has_date = bool(self._extract_date(sent))
            has_num = bool(self._extract_numeric_claim(sent))
            if overlap > 0 or has_date or has_num:
                keep.append(sent)
            if len(" ".join(keep)) > 700:
                break
        return (" ".join(keep) if keep else chunks[0])[:900]

    def _extract_entity(self, text: str) -> str:
        # Lightweight NER proxy: consecutive capitalized tokens.
        m = re.search(r"\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+){0,3})\b", str(text or ""))
        return m.group(1).strip() if m else ""

    def _extract_date(self, text: str) -> str:
        s = str(text or "")
        pats = [
            r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b",
            r"\b\d{1,2}\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{4}\b",
            r"\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)[a-z]*\s+\d{1,2},\s+\d{4}\b",
            r"\b(?:20\d{2}|19\d{2})\b",
        ]
        for p in pats:
            m = re.search(p, s, flags=re.IGNORECASE)
            if m:
                return m.group(0)
        return ""

    def _extract_numeric_claim(self, text: str) -> str:
        m = re.search(r"\b\d[\d,]*(?:\.\d+)?(?:\s?(?:%|million|billion|crore|lakh|km|kg|m|years?))?\b", str(text or ""), flags=re.IGNORECASE)
        return m.group(0).strip() if m else ""

    def _is_http_url(self, url: str) -> bool:
        """Allow only http(s) URLs for scraping."""
        try:
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                return False
            host = (parsed.netloc or "").lower().split(":")[0].strip(".")
            if not host:
                return False
            path_l = str(parsed.path or "").lower()
            query_l = str(parsed.query or "").lower()
            if self._is_nonscrapable_endpoint(host, path_l, query_l):
                logger.info("Skipping non-scrapable endpoint: %s", url)
                return False
            if self.policy.is_wordpress_like(url, host):
                logger.info("Skipping wordpress-like domain/url: %s", host or url)
                return False
            url_l = url.lower()
            if self._is_explicit_url(url_l, host):
                logger.info("Skipping explicit/unsafe scrape URL: %s", url)
                return False
            if self.policy.is_blocked_url_pattern(url_l):
                logger.info("Skipping blocked scrape URL pattern: %s", url)
                return False
            if self._is_bad_domain(host):
                logger.info("Skipping bad scrape domain: %s", host)
                return False
            return True
        except Exception:
            return False

    def _is_explicit_url(self, url: str, host: str = "") -> bool:
        return self.policy.is_explicit_url(url, host)

    def _is_bad_domain(self, host: str) -> bool:
        """Match blocked domain or any of its subdomains."""
        return self.policy.is_bad_domain(host)

    def _is_nonscrapable_endpoint(self, host: str, path_l: str, query_l: str) -> bool:
        # Skip hard API hosts.
        if host in self.skip_api_hosts:
            return True
        # Skip obvious file types not suitable for HTML scrapers.
        for ext in self.skip_file_exts:
            if path_l.endswith("." + ext):
                return True
        # Skip JSON/data API style URLs.
        if "format=json" in query_l or "output=json" in query_l or "alt=json" in query_l:
            return True
        if "/api/" in path_l or path_l.startswith("/api/"):
            return True
        for tok in self.skip_api_path_tokens:
            if tok and tok in path_l:
                return True
        return False
