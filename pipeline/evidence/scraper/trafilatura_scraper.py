"""Primary article extraction using trafilatura."""

from typing import Dict

import requests

from utils.date_extractor import extract_publication_date


class TrafilaturaScraper:
    def __init__(self, session: requests.Session):
        self.session = session
        self.enabled = True
        try:
            import trafilatura  # noqa: F401
        except Exception:
            self.enabled = False

    def scrape_url(self, url: str) -> Dict:
        if not self.enabled:
            return {}
        try:
            import trafilatura

            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            html = response.text
            text = trafilatura.extract(html, include_comments=False, include_tables=False) or ""
            if not text.strip():
                return {}
            return {
                "text": text[:3500],
                "url": url,
                "source": url,
                "title": url,
                "score": 0.55,
                "type": "scraping",
                "published_at": extract_publication_date(html) or "",
                "scrape_method": "trafilatura",
            }
        except Exception:
            return {}

