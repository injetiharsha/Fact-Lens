"""Integration tests for retrieval-budget and provider escalation behavior."""

from __future__ import annotations

from typing import Dict, List

import pytest

from pipeline.evidence.providers.structured_api import StructuredAPIClient
from pipeline.evidence.providers.web_search import WebSearchEngine


@pytest.mark.integration
def test_web_search_ddg_only_when_quality_gate_passes(monkeypatch):
    """Paid providers should not be called when DDG pool is already strong."""
    monkeypatch.setenv("WEB_SEARCH_PROVIDER_ORDER", "ddg")
    monkeypatch.setenv("WEB_SEARCH_ESCALATION_PROVIDER_ORDER", "serpapi,tavily")
    monkeypatch.setenv("WEB_SEARCH_PAID_ESCALATION_ENABLE", "1")
    monkeypatch.setenv("WEB_SEARCH_ENABLE_DDG", "1")
    monkeypatch.setenv("WEB_SEARCH_ENABLE_SERPAPI", "1")
    monkeypatch.setenv("WEB_SEARCH_ENABLE_TAVILY", "1")
    monkeypatch.setenv("WEB_SEARCH_MAX_QUERIES_EN", "3")
    monkeypatch.setenv("WEB_SEARCH_MAX_TOTAL_RESULTS_EN", "30")
    monkeypatch.setenv("WEB_SEARCH_ESCALATION_MIN_UNIQUE_DOMAINS", "3")
    monkeypatch.setenv("WEB_SEARCH_ESCALATION_MIN_TRUSTED_HITS", "1")

    engine = WebSearchEngine(config={})
    called: List[str] = []

    def fake_search_provider(provider: str, queries: List[str], max_results: int, language: str) -> List[Dict]:
        called.append(provider)
        if provider != "ddg":
            return []
        return [
            {"url": "https://www.who.int/a", "title": "A", "text": "earth facts"},
            {"url": "https://www.reuters.com/b", "title": "B", "text": "earth facts"},
            {"url": "https://en.wikipedia.org/wiki/Earth", "title": "C", "text": "earth facts"},
        ]

    monkeypatch.setattr(engine, "_search_provider", fake_search_provider)

    rows = engine.search(
        claim="The Earth is round",
        queries=["The Earth is round", "earth shape", "is earth round"],
        language="en",
        max_results=5,
    )
    assert len(rows) >= 3
    assert called == ["ddg"]


@pytest.mark.integration
def test_web_search_escalates_to_paid_when_pool_is_weak(monkeypatch):
    """Paid escalation should trigger only when DDG pool fails quality gate."""
    monkeypatch.setenv("WEB_SEARCH_PROVIDER_ORDER", "ddg")
    monkeypatch.setenv("WEB_SEARCH_ESCALATION_PROVIDER_ORDER", "serpapi,tavily")
    monkeypatch.setenv("WEB_SEARCH_PAID_ESCALATION_ENABLE", "1")
    monkeypatch.setenv("WEB_SEARCH_ENABLE_DDG", "1")
    monkeypatch.setenv("WEB_SEARCH_ENABLE_SERPAPI", "1")
    monkeypatch.setenv("WEB_SEARCH_ENABLE_TAVILY", "1")
    monkeypatch.setenv("WEB_SEARCH_ESCALATION_MIN_UNIQUE_DOMAINS", "3")
    monkeypatch.setenv("WEB_SEARCH_ESCALATION_MIN_TRUSTED_HITS", "1")
    monkeypatch.setenv("WEB_SEARCH_ESCALATION_MIN_AVG_PRESCORE", "0.2")

    engine = WebSearchEngine(config={})
    called: List[str] = []

    def fake_search_provider(provider: str, queries: List[str], max_results: int, language: str) -> List[Dict]:
        called.append(provider)
        if provider == "ddg":
            # weak: one low-signal domain only
            return [{"url": "https://example.com/low", "title": "x", "text": "random content"}]
        if provider == "serpapi":
            return [{"url": "https://www.who.int/fact", "title": "y", "text": "fact check result"}]
        return []

    monkeypatch.setattr(engine, "_search_provider", fake_search_provider)

    rows = engine.search(
        claim="Vitamin C prevents common cold",
        queries=["Vitamin C prevents common cold"],
        language="en",
        max_results=5,
    )
    assert len(rows) >= 1
    assert "ddg" in called
    assert "serpapi" in called


@pytest.mark.integration
def test_structured_api_uses_multiple_queries(monkeypatch):
    """Structured API client should try multiple query rewrites, not only first query."""
    monkeypatch.setenv("STRUCTURED_API_MAX_QUERIES", "3")
    client = StructuredAPIClient(config={"structured_api_ping": False})

    calls: List[str] = []

    def fake_wikipedia(q: str, max_results: int) -> List[Dict]:
        calls.append(q)
        return [{"url": f"https://example.org/{q}", "text": q, "score": 0.7}]

    client.api_map["wikipedia"] = fake_wikipedia
    client.enabled_subtypes = {"wikipedia"}

    out = client.query(
        claim="The Earth is round",
        queries=["The Earth is round", "earth shape", "is earth round"],
        api_subtype="wikipedia",
        language="en",
        max_results=3,
    )
    assert len(calls) == 3
    assert len(out) == 3

