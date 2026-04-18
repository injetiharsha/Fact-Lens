"""Tavily search adapter."""

import logging
import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TavilySearchAdapter:
    def __init__(self, api_keys: List[str]):
        self.clients = []
        self._idx = 0
        self._blocked_slots = set()
        self.timeout_s = max(1, int(os.getenv("WEB_SEARCH_TAVILY_TIMEOUT_SECONDS", "15")))
        self.enabled = False
        if not api_keys:
            return
        try:
            from tavily import TavilyClient

            for key in api_keys:
                self.clients.append(TavilyClient(api_key=key))
            self.enabled = len(self.clients) > 0
            if self.enabled:
                logger.info("Tavily adapter initialized with %d keys", len(self.clients))
        except ImportError:
            logger.warning("tavily-python not installed. pip install tavily-python")
        except Exception as exc:
            logger.error("Failed to initialize Tavily adapter: %s", exc)

    def _next_client(self) -> Optional[object]:
        if not self.clients:
            return None
        total = len(self.clients)
        for _ in range(total):
            slot = self._idx % total
            self._idx += 1
            if slot in self._blocked_slots:
                continue
            return slot, self.clients[slot]
        return None

    def _is_usage_limit_error(self, exc: Exception) -> bool:
        msg = str(exc).lower()
        return "exceeds your plan" in msg or "usage limit" in msg

    def _call_search(self, client: object, query: str, max_results: int, depth: str) -> List[Dict]:
        def _invoke() -> Dict:
            return client.search(query=query, max_results=max_results, search_depth=depth)

        pool = ThreadPoolExecutor(max_workers=1)
        fut = pool.submit(_invoke)
        try:
            response = fut.result(timeout=self.timeout_s)
        except FutureTimeout:
            fut.cancel()
            raise TimeoutError(
                f"Tavily search timed out after {self.timeout_s}s (depth={depth})"
            )
        finally:
            # Never block caller waiting for a timed-out worker thread.
            pool.shutdown(wait=False, cancel_futures=True)
        out: List[Dict] = []
        for result in response.get("results", []):
            out.append(
                {
                    "text": result.get("content", ""),
                    "title": result.get("title", "Tavily"),
                    "source": result.get("title", "Tavily"),
                    "url": result.get("url", ""),
                    "score": result.get("score", 0.5),
                    "type": "web_search",
                }
            )
        return out

    def search(self, query: str, max_results: int) -> List[Dict]:
        if not self.clients:
            return []
        last_exc: Exception | None = None
        attempted = 0
        total_active = len(self.clients) - len(self._blocked_slots)
        if total_active <= 0:
            logger.warning("All Tavily keys are blocked/exhausted for this run.")
            return []

        # Rotate keys: if one key errors/rate-limits, move to next key.
        for _ in range(len(self.clients)):
            nxt = self._next_client()
            if not nxt:
                break
            slot, client = nxt
            attempted += 1
            try:
                out = self._call_search(client=client, query=query, max_results=max_results, depth="advanced")
                if attempted > 1:
                    logger.info(
                        "Tavily failover succeeded with key slot %d after %d attempt(s)",
                        slot + 1,
                        attempted,
                    )
                return out
            except Exception as exc:
                last_exc = exc
                # Some Tavily plans reject "advanced" but still allow "basic".
                if self._is_usage_limit_error(exc):
                    try:
                        out = self._call_search(
                            client=client,
                            query=query,
                            max_results=max_results,
                            depth="basic",
                        )
                        logger.info(
                            "Tavily key slot %d fallback to basic depth succeeded.",
                            slot + 1,
                        )
                        return out
                    except Exception as basic_exc:
                        last_exc = basic_exc
                        if self._is_usage_limit_error(basic_exc):
                            self._blocked_slots.add(slot)
                            logger.warning(
                                "Tavily key slot %d exhausted; blocking for this run (%d blocked of %d).",
                                slot + 1,
                                len(self._blocked_slots),
                                len(self.clients),
                            )
                        else:
                            logger.warning(
                                "Tavily key slot %d advanced-limit, but basic call failed (attempt %d/%d): %s",
                                slot + 1,
                                attempted,
                                total_active,
                                basic_exc,
                            )
                else:
                    logger.warning(
                        "Tavily key slot %d failed (attempt %d/%d): %s",
                        slot + 1,
                        attempted,
                        total_active,
                        exc,
                    )
                continue

        if last_exc:
            raise last_exc
        return []
