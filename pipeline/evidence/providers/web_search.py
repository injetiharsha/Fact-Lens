"""Web search orchestration via split adapters."""

import logging
import os
from typing import Dict, List
from dotenv import load_dotenv
from pipeline.evidence.search import (
    DuckDuckGoSearchAdapter,
    NewsApiSearchAdapter,
    SerpApiSearchAdapter,
    TavilySearchAdapter,
)

load_dotenv(override=True)
logger = logging.getLogger(__name__)

def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else int(default)
    except Exception:
        value = int(default)
    return max(minimum, value)


class WebSearchEngine:
    """Query web search APIs."""
    
    def __init__(self, config: Dict):
        """Initialize with config."""
        self.config = config
        keys_raw = os.getenv("TAVILY_API_KEYS", "").strip()
        tavily_keys = [k.strip() for k in keys_raw.split(",") if k.strip()]
        single_tavily = os.getenv("TAVILY_API_KEY", "").strip()
        if not tavily_keys and single_tavily:
            tavily_keys = [single_tavily]

        self.tavily = TavilySearchAdapter(api_keys=tavily_keys)
        self.serpapi = SerpApiSearchAdapter(api_key=os.getenv("SERPAPI_KEY", "").strip())
        self.newsapi = NewsApiSearchAdapter(api_key=os.getenv("NEWS_API_KEY", "").strip())
        self.ddg = DuckDuckGoSearchAdapter()

        # Search behavior controls with fast defaults.
        # Env can still override, but old-style users can keep .env minimal.
        self.max_queries = _env_int("WEB_SEARCH_MAX_QUERIES", 1, minimum=1)
        self.min_results_before_fallback = _env_int("WEB_SEARCH_MIN_RESULTS_BEFORE_FALLBACK", 5, minimum=0)
        self.max_total_results = _env_int("WEB_SEARCH_MAX_TOTAL_RESULTS", 5, minimum=0)
        self.enable_tavily = _env_bool("WEB_SEARCH_ENABLE_TAVILY", True)
        self.enable_serpapi = _env_bool("WEB_SEARCH_ENABLE_SERPAPI", True)
        self.enable_ddg = _env_bool("WEB_SEARCH_ENABLE_DDG", True)
        self.enable_newsapi = _env_bool("WEB_SEARCH_ENABLE_NEWSAPI", False)
        self.provider_fail_fast = _env_bool("WEB_SEARCH_PROVIDER_FAIL_FAST", True)
        self.provider_order = self._resolve_provider_order()
        self.live_progress = _env_bool("EVIDENCE_LIVE_PROGRESS", False)
        logger.info(
            "WebSearch config: order=%s max_queries=%d min_results_before_fallback=%d max_total_results=%d fail_fast=%s",
            self.provider_order,
            self.max_queries,
            self.min_results_before_fallback,
            self.max_total_results,
            self.provider_fail_fast,
        )

    def _emit(self, msg: str) -> None:
        if self.live_progress:
            print(f"[web-search] {msg}", flush=True)
    
    def search(
        self,
        claim: str,
        queries: List[str],
        subtype: str = None,
        language: str = "en",
        max_results: int = 5
    ) -> List[Dict]:
        """
        Search web for evidence.
        
        Returns:
            List of evidence dicts
        """
        all_results = []
        is_news_route = self._is_news_subtype(subtype)
        query_slice = queries[: self.max_queries]
        self._emit(
            f"start subtype={subtype} lang={language} queries={len(query_slice)} order={','.join(self.provider_order)}"
        )

        def _apply_result_cap(rows: List[Dict]) -> List[Dict]:
            if self.max_total_results <= 0:
                return rows
            return rows[: self.max_total_results]

        # News API is only for news-routed claims (not generic search fallback).
        if is_news_route and self.newsapi.enabled and self.enable_newsapi:
            self._emit("newsapi:start")
            for query in query_slice:
                try:
                    results = self.newsapi.search(query=query, max_results=max_results, language=language)
                    all_results.extend(results)
                    self._emit(f"newsapi:query_done added={len(results)} total={len(all_results)}")
                    all_results = _apply_result_cap(all_results)
                    if self.max_total_results > 0 and len(all_results) >= self.max_total_results:
                        break
                except Exception as e:
                    logger.error(f"News API search failed: {e}")
                    self._emit(f"newsapi:error err={e}")

        for provider in self.provider_order:
            if len(all_results) >= self.min_results_before_fallback:
                self._emit(
                    f"fallback_stop reason=min_results_reached total={len(all_results)} min={self.min_results_before_fallback}"
                )
                break
            if self.max_total_results > 0 and len(all_results) >= self.max_total_results:
                self._emit(
                    f"fallback_stop reason=max_total_reached total={len(all_results)} cap={self.max_total_results}"
                )
                break
            before = len(all_results)
            all_results.extend(
                self._search_provider(
                    provider=provider,
                    queries=query_slice,
                    max_results=max_results,
                    language=language,
                )
            )
            all_results = _apply_result_cap(all_results)
            self._emit(f"provider_done provider={provider} added={len(all_results)-before} total={len(all_results)}")

        all_results = self._dedupe_by_url(all_results)
        all_results = _apply_result_cap(all_results)
        self._emit(f"done deduped_total={len(all_results)}")
        
        logger.info(f"Web search returned {len(all_results)} results")
        return all_results

    def _resolve_provider_order(self) -> List[str]:
        raw = os.getenv("WEB_SEARCH_PROVIDER_ORDER", "ddg,tavily,serpapi")
        allowed = {"ddg", "tavily", "serpapi"}
        out: List[str] = []
        seen = set()
        for part in raw.split(","):
            token = part.strip().lower()
            if token not in allowed or token in seen:
                continue
            seen.add(token)
            out.append(token)
        if not out:
            out = ["ddg", "tavily", "serpapi"]
        return out

    def _search_provider(self, provider: str, queries: List[str], max_results: int, language: str) -> List[Dict]:
        rows: List[Dict] = []
        for query in queries:
            try:
                self._emit(f"provider_query provider={provider} q='{query[:80]}'")
                if provider == "tavily":
                    if not (self.enable_tavily and self.tavily.enabled):
                        self._emit("provider_skip tavily disabled")
                        continue
                    rows.extend(self.tavily.search(query=query, max_results=max_results))
                elif provider == "serpapi":
                    if not (self.enable_serpapi and self.serpapi.enabled):
                        self._emit("provider_skip serpapi disabled")
                        continue
                    rows.extend(self.serpapi.search(query=query, max_results=max_results, language=language))
                elif provider == "ddg":
                    if not self.enable_ddg:
                        self._emit("provider_skip ddg disabled")
                        continue
                    rows.extend(self.ddg.search(query=query, max_results=max_results))
            except Exception as e:
                logger.error("%s search failed: %s", provider.upper(), e)
                self._emit(f"provider_error provider={provider} err={e}")
                if self.provider_fail_fast:
                    logger.warning(
                        "%s fail-fast enabled; moving to next provider after query failure.",
                        provider.upper(),
                    )
                    self._emit(f"provider_fail_fast provider={provider} action=break")
                    break
        return rows
    
    def _dedupe_by_url(self, rows: List[Dict]) -> List[Dict]:
        """Deduplicate evidence rows by URL while preserving order."""
        out: List[Dict] = []
        seen = set()
        for row in rows:
            key = str(row.get("url") or "").strip() or str(row.get("text") or "").strip()[:120]
            if key in seen:
                continue
            seen.add(key)
            out.append(row)
        return out

    def _is_news_subtype(self, subtype: str) -> bool:
        """Return True when route subtype indicates news-style retrieval."""
        s = (subtype or "").strip().lower()
        if not s:
            return False
        news_tokens = {"news", "general_news", "politics_news", "breaking_news"}
        return s in news_tokens or "news" in s
