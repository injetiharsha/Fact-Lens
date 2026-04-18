"""Unit tests for scoring calculations."""

import pytest
from pipeline.scoring import ScoringCalculator

def test_calculate_weighted_score():
    calculator = ScoringCalculator()
    scores = {"relevance": 0.8, "stance": 0.9}
    weights = {"relevance": 0.5, "stance": 0.5}
    result = calculator.calculate_weighted_score(scores, weights)
    assert 0 <= result <= 1

def test_apply_domain_weight():
    calculator = ScoringCalculator()
    result = calculator.apply_domain_weight(0.8, "science")
    assert isinstance(result, float)
