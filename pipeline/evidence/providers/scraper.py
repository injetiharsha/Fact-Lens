"""Web scraping for evidence."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Dict, List, Set
from urllib.parse import urlparse
import requests
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
        self.tier_order = [
            ("trafilatura", self.trafilatura),
            ("playwright", self.playwright),
            ("beautifulsoup", self.bs4),
        ]
        self.parallel_urls = _env_bool("SCRAPER_PARALLEL_URLS", True)
        self.url_workers = _env_int("SCRAPER_PARALLEL_WORKERS", 4, minimum=1)
        self.bad_domains = self._load_bad_domains()
    
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
        """Scrape a single URL with tiered fallback stack."""
        for name, scraper in self.tier_order:
            if not getattr(scraper, "enabled", True):
                continue
            try:
                row = scraper.scrape_url(url)
                if row and row.get("text"):
                    row.setdefault("scrape_tier", name)
                    return row
            except Exception as exc:
                logger.debug("Scraper tier %s failed for %s: %s", name, url, exc)
        logger.error("Failed to scrape %s with all tiers", url)
        return {}

    def _is_http_url(self, url: str) -> bool:
        """Allow only http(s) URLs for scraping."""
        try:
            parsed = urlparse(url)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                return False
            host = (parsed.netloc or "").lower().split(":")[0].strip(".")
            if not host:
                return False
            if self._is_bad_domain(host):
                logger.info("Skipping bad scrape domain: %s", host)
                return False
            return True
        except Exception:
            return False

    def _load_bad_domains(self) -> Set[str]:
        """Load blocked scrape domains from env with safe defaults."""
        defaults = {"dailymotion.com"}
        raw = os.getenv("SCRAPER_BAD_DOMAINS", "")
        if not raw.strip():
            return defaults
        out: Set[str] = set()
        for token in raw.split(","):
            d = token.strip().lower().strip(".")
            if d:
                out.add(d)
        return out or defaults

    def _is_bad_domain(self, host: str) -> bool:
        """Match blocked domain or any of its subdomains."""
        for domain in self.bad_domains:
            if host == domain or host.endswith("." + domain):
                return True
        return False
