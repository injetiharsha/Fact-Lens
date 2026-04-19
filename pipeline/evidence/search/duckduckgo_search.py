"""DuckDuckGo fallback search adapter."""

import logging
import os
from typing import Dict, List

logger = logging.getLogger(__name__)


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else int(default)
    except Exception:
        value = int(default)
    return max(minimum, value)


class DuckDuckGoSearchAdapter:
    def __init__(self):
        self.enabled = True
        self.timeout_s = _env_int("WEB_SEARCH_DDG_TIMEOUT_SECONDS", 10, minimum=1)

    def search(self, query: str, max_results: int) -> List[Dict]:
        try:
            try:
                from duckduckgo_search import DDGS
            except ImportError:
                from ddgs import DDGS

            try:
                with DDGS(timeout=self.timeout_s) as ddgs:
                    rows = list(ddgs.text(query, max_results=max_results))
            except TypeError:
                # Backward compatibility with DDGS versions without timeout arg.
                with DDGS() as ddgs:
                    rows = list(ddgs.text(query, max_results=max_results))

            out: List[Dict] = []
            for row in rows:
                link = row.get("href") or row.get("url") or row.get("link") or ""
                out.append(
                    {
                        "text": row.get("body", ""),
                        "title": row.get("title", "DuckDuckGo"),
                        "source": row.get("title", "DuckDuckGo"),
                        "url": link,
                        "score": 0.4,
                        "type": "web_search",
                    }
                )
            return out
        except ImportError:
            logger.warning("duckduckgo-search/ddgs not installed")
            return []
        except Exception as exc:
            logger.error("DDG search error: %s", exc)
            return []
