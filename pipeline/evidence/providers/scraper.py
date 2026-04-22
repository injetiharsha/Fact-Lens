"""Web scraping for evidence."""

import logging
import os
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

        best = self._pick_best_scrape_row(primary_rows)
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
                        if self._scrape_row_quality(pw_row) > self._scrape_row_quality(best):
                            return pw_row
                except Exception as exc:
                    logger.debug("Playwright fallback failed for %s: %s", url, exc)
            return best

        if self.enable_playwright_fallback and getattr(self.playwright, "enabled", True):
            try:
                row = self.playwright.scrape_url(url)
                if row and row.get("text"):
                    row.setdefault("scrape_tier", "playwright")
                    return row
            except Exception as exc:
                logger.debug("Playwright fallback failed for %s: %s", url, exc)

        logger.error("Failed to scrape %s with all enabled methods", url)
        return {}

    def _pick_best_scrape_row(self, rows: List[Dict]) -> Dict:
        if not rows:
            return {}
        best = max(rows, key=self._scrape_row_quality)
        return best or {}

    def _scrape_row_quality(self, row: Dict) -> float:
        text = str(row.get("text") or "")
        title = str(row.get("title") or "")
        score = float(row.get("score", 0.0) or 0.0)
        # Prefer richer extract and stable parser score.
        return score + min(len(text), 4000) / 4000.0 + (0.05 if title else 0.0)

    def _is_http_url(self, url: str) -> bool:
        """Allow only http(s) URLs for scraping."""
        try:
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                return False
            host = (parsed.netloc or "").lower().split(":")[0].strip(".")
            if not host:
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
