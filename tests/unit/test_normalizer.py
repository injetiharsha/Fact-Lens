"""Unit tests for claim normalization."""

import pytest
from pipeline.core.normalizer import ClaimNormalizer

def test_normalize_basic():
    normalizer = ClaimNormalizer()
    result = normalizer.normalize("The Moon is Made of Cheese!!!")
    assert isinstance(result, str)

def test_rephrase_for_search():
    normalizer = ClaimNormalizer()
    result = normalizer.rephrase_for_search("Is water liquid?")
    assert isinstance(result, str)
