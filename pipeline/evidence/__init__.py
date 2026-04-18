"""Evidence package exports."""

from pipeline.evidence.core.gatherer import EvidenceGatherer
from pipeline.evidence.core.aggregator import EvidenceAggregator
from pipeline.evidence.core.deduplicator import EvidenceDeduplicator
from pipeline.evidence.providers.structured_api import StructuredAPIClient
from pipeline.evidence.providers.web_search import WebSearchEngine
from pipeline.evidence.providers.scraper import EvidenceScraper

__all__ = [
    "EvidenceGatherer",
    "EvidenceAggregator",
    "EvidenceDeduplicator",
    "StructuredAPIClient",
    "WebSearchEngine",
    "EvidenceScraper",
]
