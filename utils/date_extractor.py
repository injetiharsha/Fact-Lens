"""Extract publication dates from HTML metadata and page text."""

from __future__ import annotations

import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Optional

from bs4 import BeautifulSoup


_META_DATE_KEYS = [
    "article:published_time",
    "article:modified_time",
    "og:published_time",
    "og:updated_time",
    "publish-date",
    "published_date",
    "pubdate",
    "date",
]


def _normalize_iso(dt: datetime) -> str:
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc).isoformat()


def _parse_date_value(raw: str) -> Optional[str]:
    value = (raw or "").strip()
    if not value:
        return None

    # ISO-8601 / close variants.
    try:
        v = value.replace("Z", "+00:00")
        return _normalize_iso(datetime.fromisoformat(v))
    except Exception:
        pass

    # RFC-like date strings.
    try:
        dt = parsedate_to_datetime(value)
        return _normalize_iso(dt)
    except Exception:
        pass

    # Loose "YYYY-MM-DD" in text.
    m = re.search(r"\b(20\d{2}|19\d{2})-(\d{1,2})-(\d{1,2})\b", value)
    if m:
        try:
            dt = datetime(int(m.group(1)), int(m.group(2)), int(m.group(3)), tzinfo=timezone.utc)
            return _normalize_iso(dt)
        except Exception:
            return None
    return None


def extract_publication_date(html: str) -> Optional[str]:
    """Return best-effort published/updated UTC timestamp from HTML."""
    if not html:
        return None

    soup = BeautifulSoup(html, "html.parser")

    # 1) Standard meta tags.
    for key in _META_DATE_KEYS:
        tag = soup.find("meta", attrs={"property": key}) or soup.find("meta", attrs={"name": key})
        if not tag:
            continue
        val = tag.get("content") or tag.get("value")
        parsed = _parse_date_value(str(val or ""))
        if parsed:
            return parsed

    # 2) <time datetime=...>.
    time_tag = soup.find("time")
    if time_tag:
        parsed = _parse_date_value(str(time_tag.get("datetime") or time_tag.get_text(" ", strip=True)))
        if parsed:
            return parsed

    # 3) JSON-LD datePublished / dateModified.
    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        content = script.get_text(" ", strip=True)
        for field in ("datePublished", "dateModified", "uploadDate"):
            m = re.search(rf'"{field}"\s*:\s*"([^"]+)"', content)
            if not m:
                continue
            parsed = _parse_date_value(m.group(1))
            if parsed:
                return parsed

    # 4) Last resort: scan visible text head.
    head_text = soup.get_text(" ", strip=True)[:2000]
    return _parse_date_value(head_text)

