"""SerpAPI search adapter."""

from typing import Dict, List

import requests
import logging

logger = logging.getLogger(__name__)


class SerpApiSearchAdapter:
    def __init__(self, api_key: str):
        self.api_key = (api_key or "").strip()
        self.enabled = bool(self.api_key)
        self.disable_on_429 = True
        self._rate_limited = False

    def search(self, query: str, max_results: int, language: str = "en") -> List[Dict]:
        if not self.enabled:
            return []

        url = "https://serpapi.com/search.json"
        params = {
            "engine": "google",
            "q": query,
            "api_key": self.api_key,
            "hl": language or "en",
            "num": max_results,
        }
        response = requests.get(url, params=params, timeout=15)
        if response.status_code == 429:
            self._rate_limited = True
            if self.disable_on_429:
                self.enabled = False
            logger.warning("SerpAPI returned 429; disabling SerpAPI for this run.")
            return []
        response.raise_for_status()
        data = response.json()
        if isinstance(data, dict):
            err = str(data.get("error") or data.get("message") or "").lower()
            if "rate limit" in err or "quota" in err or "too many requests" in err:
                self._rate_limited = True
                if self.disable_on_429:
                    self.enabled = False
                logger.warning("SerpAPI quota/rate-limit error; disabling SerpAPI for this run: %s", err)
                return []

        out: List[Dict] = []
        for item in data.get("organic_results", [])[:max_results]:
            out.append(
                {
                    "text": item.get("snippet", ""),
                    "title": item.get("title", "SerpAPI"),
                    "source": item.get("title", "SerpAPI"),
                    "url": item.get("link", ""),
                    "score": 0.55,
                    "type": "web_search",
                }
            )
        return out
