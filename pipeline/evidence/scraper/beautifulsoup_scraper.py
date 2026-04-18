"""HTML fallback scraper using requests + BeautifulSoup."""

from typing import Dict

import requests
from bs4 import BeautifulSoup

from utils.date_extractor import extract_publication_date


class BeautifulSoupScraper:
    def __init__(self, session: requests.Session):
        self.session = session
        self.enabled = True

    def scrape_url(self, url: str) -> Dict:
        try:
            response = self.session.get(url, timeout=15)
            response.raise_for_status()
            content_type = (response.headers.get("content-type") or "").lower()
            if "text/html" not in content_type and "application/xhtml+xml" not in content_type:
                return {}

            html = response.text
            soup = BeautifulSoup(html, "html.parser")
            for tag in soup(["script", "style", "noscript"]):
                tag.decompose()
            text = soup.get_text(" ", strip=True)
            if not text:
                return {}
            title = (soup.title.string.strip() if soup.title and soup.title.string else url)
            return {
                "text": text[:2500],
                "url": url,
                "source": title,
                "title": title,
                "score": 0.5,
                "type": "scraping",
                "published_at": extract_publication_date(html) or "",
                "scrape_method": "beautifulsoup",
            }
        except Exception:
            return {}

