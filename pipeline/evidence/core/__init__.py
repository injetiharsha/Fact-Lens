"""Core evidence orchestration and ranking."""

from pipeline.evidence.core.gatherer import EvidenceGatherer
from pipeline.evidence.core.aggregator import EvidenceAggregator
from pipeline.evidence.core.deduplicator import EvidenceDeduplicator

__all__ = ["EvidenceGatherer", "EvidenceAggregator", "EvidenceDeduplicator"]

