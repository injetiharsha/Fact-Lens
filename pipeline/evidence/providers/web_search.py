"""Web search orchestration via split adapters."""

import logging
import os
import re
import sys
from typing import Dict, List
from urllib.parse import urlparse
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
        self.min_providers_before_stop = _env_int("WEB_SEARCH_MIN_PROVIDERS_BEFORE_STOP", 2, minimum=1)
        self.provider_order = self._resolve_provider_order()
        self.multi_phase2_mode = os.getenv("MULTI_PHASE2_MODE", "current").strip().lower()
        self.multi_phase2_candidate_cap = _env_int("MULTI_PHASE2_CANDIDATE_CAP", 10, minimum=1)
        self.bad_domains = self._load_bad_domains()
        self.block_wordpress = _env_bool("WEB_SEARCH_BLOCK_WORDPRESS", True)
        self.block_explicit = _env_bool("WEB_SEARCH_BLOCK_EXPLICIT", True)
        self.blocked_url_tokens = self._load_blocked_url_tokens()
        self.live_progress = _env_bool("EVIDENCE_LIVE_PROGRESS", False)
        self.query_log_truncate_chars = _env_int("WEB_SEARCH_QUERY_LOG_TRUNCATE_CHARS", 0, minimum=0)
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
            line = f"[web-search] {msg}"
            try:
                print(line, flush=True)
            except UnicodeEncodeError:
                buf = getattr(sys.stdout, "buffer", None)
                if buf is not None:
                    buf.write((line + "\n").encode("utf-8", errors="replace"))
                    buf.flush()
                else:
                    print(line.encode("ascii", errors="replace").decode("ascii"), flush=True)
    
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
        max_total_cap = self._max_total_results_for_language(language)
        is_news_route = self._is_news_subtype(subtype)
        query_slice = queries[: self.max_queries]
        self._emit(
            f"start subtype={subtype} lang={language} queries={len(query_slice)} order={','.join(self.provider_order)}"
        )

        def _apply_result_cap(rows: List[Dict]) -> List[Dict]:
            if max_total_cap <= 0:
                return rows
            return rows[: max_total_cap]

        # News API is only for news-routed claims (not generic search fallback).
        if is_news_route and self.newsapi.enabled and self.enable_newsapi:
            self._emit("newsapi:start")
            for query in query_slice:
                try:
                    results = self.newsapi.search(query=query, max_results=max_results, language=language)
                    all_results.extend(results)
                    self._emit(f"newsapi:query_done added={len(results)} total={len(all_results)}")
                    all_results = _apply_result_cap(all_results)
                    if max_total_cap > 0 and len(all_results) >= max_total_cap:
                        break
                except Exception as e:
                    logger.error(f"News API search failed: {e}")
                    self._emit(f"newsapi:error err={e}")

        attempted_providers = 0
        for provider in self.provider_order:
            if (
                len(all_results) >= self.min_results_before_fallback
                and attempted_providers >= self.min_providers_before_stop
            ):
                self._emit(
                    f"fallback_stop reason=min_results_reached total={len(all_results)} min={self.min_results_before_fallback}"
                )
                break
            if max_total_cap > 0 and len(all_results) >= max_total_cap:
                self._emit(
                    f"fallback_stop reason=max_total_reached total={len(all_results)} cap={max_total_cap}"
                )
                break
            before = len(all_results)
            attempted_providers += 1
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
        all_results = self._filter_bad_domains(all_results)
        if (language or "en").strip().lower() != "en" and self.multi_phase2_mode == "scored":
            all_results = self._phase2_rank_candidates(
                claim=claim,
                queries=query_slice,
                rows=all_results,
                cap=min(max_total_cap if max_total_cap > 0 else len(all_results), self.multi_phase2_candidate_cap),
            )
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

    def _max_total_results_for_language(self, language: str) -> int:
        lang = (language or "en").strip().lower()
        if lang.startswith("en"):
            return _env_int("WEB_SEARCH_MAX_TOTAL_RESULTS_EN", self.max_total_results, minimum=0)
        return _env_int("WEB_SEARCH_MAX_TOTAL_RESULTS_MULTI", self.max_total_results, minimum=0)

    def _search_provider(self, provider: str, queries: List[str], max_results: int, language: str) -> List[Dict]:
        rows: List[Dict] = []
        for query in queries:
            try:
                q = str(query or "")
                if self.query_log_truncate_chars > 0:
                    q_show = q[: self.query_log_truncate_chars]
                else:
                    q_show = q
                self._emit(
                    f"provider_query provider={provider} q_len={len(q)} q='{q_show}'"
                )
                batch: List[Dict] = []
                if provider == "tavily":
                    if not (self.enable_tavily and self.tavily.enabled):
                        self._emit("provider_skip tavily disabled")
                        continue
                    batch = self.tavily.search(query=query, max_results=max_results)
                elif provider == "serpapi":
                    if not (self.enable_serpapi and self.serpapi.enabled):
                        self._emit("provider_skip serpapi disabled")
                        continue
                    batch = self.serpapi.search(query=query, max_results=max_results, language=language)
                elif provider == "ddg":
                    if not self.enable_ddg:
                        self._emit("provider_skip ddg disabled")
                        continue
                    batch = self.ddg.search(query=query, max_results=max_results)
                for row in batch:
                    if isinstance(row, dict):
                        row["search_provider"] = provider
                        row["search_query"] = str(query or "")
                rows.extend(batch)
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

    def _load_bad_domains(self) -> set[str]:
        """Load blocked domains for web-search evidence rows."""
        raw = os.getenv("WEB_SEARCH_BAD_DOMAINS", "").strip() or os.getenv("SCRAPER_BAD_DOMAINS", "").strip()
        # Default low-signal domains for fact-check retrieval.
        defaults = {
            "testbook.com",
            "soundcloud.com",
            "spotify.com",
            "gaana.com",
            "jiosaavn.com",
            "wynk.in",
            "hungama.com",
            "apple.com",  # apple podcasts pages are mostly audio metadata
            "brainly.com",
            "quora.com",
            "answers.com",
            "wikihow.com",
            # Explicit/adult domains (safety hard-block).
            "pornhub.com",
            "xvideos.com",
            "xnxx.com",
            "xhamster.com",
            "redtube.com",
            "youporn.com",
            "beeg.com",
            "adultfriendfinder.com",
        }
        if not raw:
            return defaults
        out = set()
        for tok in raw.split(","):
            d = tok.strip().lower().strip(".")
            if d:
                out.add(d)
        out.update(defaults)
        return out

    def _load_blocked_url_tokens(self) -> List[str]:
        raw = os.getenv(
            "WEB_SEARCH_BLOCKED_URL_TOKENS",
            "/questions/,/question/,/questiosn/,/qna/,/qa/,/mcq,mcq/,quiz,quizzes,podcast,audio,/shorts/,/reel/,/reels/,/watch?,/video/,/categories/44/",
        )
        out: List[str] = []
        for tok in str(raw).split(","):
            t = tok.strip().lower()
            if t:
                out.append(t)
        return out

    def _is_explicit_url(self, url: str, host: str = "") -> bool:
        if not self.block_explicit:
            return False
        value = f"{host} {url}".lower()
        # Avoid broad false positives; match explicit terms as tokens.
        return bool(
            re.search(
                r"(?:^|[^a-z0-9])(xxx|porn|nsfw|hentai|sex-video|adult-video|erotic|nude)(?:[^a-z0-9]|$)",
                value,
            )
        )

    def _is_bad_domain(self, host: str) -> bool:
        for d in self.bad_domains:
            if host == d or host.endswith("." + d):
                return True
        return False

    def _filter_bad_domains(self, rows: List[Dict]) -> List[Dict]:
        """Drop rows whose URL matches blocked domains."""
        if not self.bad_domains:
            # still allow wordpress heuristic block
            if not self.block_wordpress:
                return rows
        out: List[Dict] = []
        for row in rows:
            url = str(row.get("url") or "").strip()
            if not url:
                out.append(row)
                continue
            try:
                host = (urlparse(url).netloc or "").lower().split(":")[0].strip(".")
            except Exception:
                host = ""
            if self.block_wordpress:
                url_l = url.lower()
                if ("wordpress" in host) or ("/wp-content/" in url_l) or ("/wp-json/" in url_l):
                    self._emit(f"domain_blocked_wordpress host={host or 'unknown'}")
                    continue
            url_l = url.lower()
            if self._is_explicit_url(url_l, host):
                self._emit(f"url_blocked_explicit host={host or 'unknown'}")
                continue
            if any(tok in url_l for tok in self.blocked_url_tokens):
                self._emit(f"url_blocked_pattern host={host or 'unknown'}")
                continue
            if host and self._is_bad_domain(host):
                self._emit(f"domain_blocked host={host}")
                continue
            out.append(row)
        return out

    def _is_news_subtype(self, subtype: str) -> bool:
        """Return True when route subtype indicates news-style retrieval."""
        s = (subtype or "").strip().lower()
        if not s:
            return False
        news_tokens = {"news", "general_news", "politics_news", "breaking_news"}
        return s in news_tokens or "news" in s

    def _phase2_rank_candidates(self, claim: str, queries: List[str], rows: List[Dict], cap: int) -> List[Dict]:
        """Phase2 candidate scoring before downstream scrape enrichment (multi-only)."""
        if not rows:
            return rows
        query_terms = self._tokenize(" ".join([claim] + list(queries or [])))
        for row in rows:
            title = str(row.get("title") or "")
            text = str(row.get("text") or row.get("snippet") or row.get("content") or "")
            blob = f"{title} {text}"
            terms = self._tokenize(blob)
            overlap = 0.0
            if query_terms and terms:
                overlap = len(query_terms.intersection(terms)) / max(1, len(query_terms))
            lexical = min(1.0, overlap * 1.4)
            host = ""
            try:
                host = (urlparse(str(row.get("url") or "")).netloc or "").lower().split(":")[0].strip(".")
            except Exception:
                host = ""
            credibility = self._phase2_domain_score(host)
            domain_bonus = 0.0
            if any(tok in host for tok in ("gov", "who.int", "worldbank", "wikipedia", "reuters", "bbc")):
                domain_bonus = 0.06
            base = float(row.get("score", 0.5) or 0.5)
            phase2_score = (0.45 * lexical) + (0.35 * credibility) + (0.20 * base) + domain_bonus
            row["phase2_candidate_score"] = max(0.0, min(1.0, phase2_score))

        ranked = sorted(rows, key=lambda x: float(x.get("phase2_candidate_score", 0.0) or 0.0), reverse=True)
        return ranked[: max(1, cap)]

    def _tokenize(self, text: str) -> set[str]:
        toks = re.findall(r"[A-Za-z\u0900-\u0D7F]{3,}", str(text or "").lower())
        return set(toks)

    def _phase2_domain_score(self, host: str) -> float:
        h = str(host or "").lower()
        if not h:
            return 0.5
        if any(k in h for k in ("gov", "nic.in", "who.int", "worldbank.org", "un.org", "nasa.gov", "wikipedia.org")):
            return 0.95
        if any(k in h for k in ("reuters.com", "bbc.com", "apnews.com")):
            return 0.90
        if any(k in h for k in ("arxiv.org", "nature.com", "science.org")):
            return 0.88
        if any(k in h for k in ("medium.com", "blog", "wordpress")):
            return 0.55
        return 0.70
