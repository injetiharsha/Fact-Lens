"""SerpAPI search adapter."""

from typing import Dict, List

import requests


class SerpApiSearchAdapter:
    def __init__(self, api_key: str):
        self.api_key = (api_key or "").strip()
        self.enabled = bool(self.api_key)

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
        response.raise_for_status()
        data = response.json()

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

