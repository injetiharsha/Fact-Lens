"""JS-rendered fallback scraper using Playwright."""

import os
from typing import Dict

from bs4 import BeautifulSoup

from utils.date_extractor import extract_publication_date


class PlaywrightScraper:
    def __init__(self):
        self.enabled = True
        self.heavy_mode = str(os.getenv("SCRAPER_PLAYWRIGHT_HEAVY_MODE", "0")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        try:
            self.heavy_wait_ms = max(0, int(os.getenv("SCRAPER_PLAYWRIGHT_HEAVY_WAIT_MS", "2000")))
        except Exception:
            self.heavy_wait_ms = 2000
        try:
            from playwright.sync_api import sync_playwright  # noqa: F401
        except Exception:
            self.enabled = False

    def scrape_url(self, url: str) -> Dict:
        if not self.enabled:
            return {}
        try:
            from playwright.sync_api import sync_playwright

            with sync_playwright() as p:
                browser = p.chromium.launch(headless=True)
                page = browser.new_page()
                page.goto(url, timeout=25000, wait_until="domcontentloaded")
                if self.heavy_mode:
                    try:
                        page.wait_for_load_state("networkidle", timeout=8000)
                    except Exception:
                        pass
                    try:
                        page.evaluate("window.scrollTo(0, document.body.scrollHeight)")
                        page.wait_for_timeout(self.heavy_wait_ms)
                        page.evaluate("window.scrollTo(0, 0)")
                    except Exception:
                        pass
                html = page.content()
                browser.close()

            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text(" ", strip=True)
            if not text:
                return {}
            title = (soup.title.string.strip() if soup.title and soup.title.string else url)
            return {
                "text": text[:3000],
                "url": url,
                "source": title,
                "title": title,
                "score": 0.52,
                "type": "scraping",
                "published_at": extract_publication_date(html) or "",
                "scrape_method": "playwright",
            }
        except Exception:
            return {}
