"""Evidence source provider clients."""

from pipeline.evidence.providers.structured_api import StructuredAPIClient
from pipeline.evidence.providers.web_search import WebSearchEngine
from pipeline.evidence.providers.scraper import EvidenceScraper

__all__ = ["StructuredAPIClient", "WebSearchEngine", "EvidenceScraper"]

