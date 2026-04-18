"""Search adapters for web evidence retrieval."""

from pipeline.evidence.search.tavily_search import TavilySearchAdapter
from pipeline.evidence.search.serpapi_search import SerpApiSearchAdapter
from pipeline.evidence.search.duckduckgo_search import DuckDuckGoSearchAdapter
from pipeline.evidence.search.newsapi_search import NewsApiSearchAdapter

__all__ = [
    "TavilySearchAdapter",
    "SerpApiSearchAdapter",
    "DuckDuckGoSearchAdapter",
    "NewsApiSearchAdapter",
]

