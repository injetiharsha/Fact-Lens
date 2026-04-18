"""News API adapter (NewsAPI.org and NewsData.io)."""

from typing import Dict, List

import requests


class NewsApiSearchAdapter:
    def __init__(self, api_key: str):
        self.api_key = (api_key or "").strip()
        self.enabled = bool(self.api_key)

    def search(self, query: str, max_results: int, language: str = "en") -> List[Dict]:
        if not self.enabled:
            return []
        if self.api_key.startswith("pub_"):
            return self._search_newsdata(query, max_results, language)
        return self._search_newsapi_org(query, max_results, language)

    def _search_newsapi_org(self, query: str, max_results: int, language: str) -> List[Dict]:
        url = "https://newsapi.org/v2/everything"
        params = {
            "q": query,
            "apiKey": self.api_key,
            "language": (language or "en")[:2],
            "pageSize": max_results,
            "sortBy": "relevancy",
        }
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        out: List[Dict] = []
        for item in data.get("articles", [])[:max_results]:
            text = ((item.get("description") or "") + " " + (item.get("content") or "")).strip()
            out.append(
                {
                    "text": text,
                    "title": item.get("title", "NewsAPI"),
                    "source": item.get("source", {}).get("name", "NewsAPI"),
                    "url": item.get("url", ""),
                    "score": 0.5,
                    "published_at": item.get("publishedAt", ""),
                    "type": "web_search",
                }
            )
        return out

    def _search_newsdata(self, query: str, max_results: int, language: str) -> List[Dict]:
        url = "https://newsdata.io/api/1/news"
        params = {"apikey": self.api_key, "q": query, "language": (language or "en")[:2]}
        response = requests.get(url, params=params, timeout=15)
        response.raise_for_status()
        data = response.json()

        out: List[Dict] = []
        for item in data.get("results", [])[:max_results]:
            text = ((item.get("description") or "") + " " + (item.get("content") or "")).strip()
            out.append(
                {
                    "text": text,
                    "title": item.get("title", "NewsData"),
                    "source": item.get("source_id", "NewsData"),
                    "url": item.get("link", ""),
                    "score": 0.5,
                    "published_at": item.get("pubDate", ""),
                    "type": "web_search",
                }
            )
        return out

