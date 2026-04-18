"""Unit tests for verdict aggregation."""

import pytest
from pipeline.verdict import VerdictAggregator

def test_aggregate_verdict():
    aggregator = VerdictAggregator()
    scores = {"support": 0.7, "refute": 0.2, "neutral": 0.1}
    result = aggregator.aggregate(scores)
    assert isinstance(result, str)

def test_get_confidence():
    aggregator = VerdictAggregator()
    scores = {"support": 0.9, "refute": 0.05, "neutral": 0.05}
    confidence = aggregator.get_confidence(scores)
    assert 0 <= confidence <= 1
