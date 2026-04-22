"""Centralized URL/domain filtering and trust policy for retrieval layers."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, Set
from urllib.parse import urlparse


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _load_set(raw: str) -> Set[str]:
    out: Set[str] = set()
    for tok in str(raw or "").split(","):
        v = tok.strip().lower().strip(".")
        if v:
            out.add(v)
    return out


@dataclass
class RetrievalPolicy:
    """Shared policy object used by both search and scraper layers."""

    bad_domains: Set[str]
    blocked_url_tokens: Set[str]
    trusted_domains: Set[str]
    block_wordpress: bool
    block_explicit: bool

    @classmethod
    def from_env(
        cls,
        *,
        domain_env: str,
        token_env: str,
        block_wordpress_env: str,
        block_explicit_env: str,
        default_bad_domains: Iterable[str],
        default_tokens: Iterable[str],
    ) -> "RetrievalPolicy":
        raw_domains = os.getenv(domain_env, "")
        bad_domains = _load_set(raw_domains) if raw_domains.strip() else set(default_bad_domains)

        raw_tokens = os.getenv(token_env, "")
        blocked_tokens = _load_set(raw_tokens) if raw_tokens.strip() else set(default_tokens)

        trusted_raw = os.getenv(
            "RETRIEVAL_TRUSTED_DOMAINS",
            "gov.in,nic.in,who.int,worldbank.org,wikipedia.org,wikidata.org,reuters.com,bbc.com,apnews.com,nasa.gov",
        )
        trusted_domains = _load_set(trusted_raw)

        return cls(
            bad_domains=bad_domains,
            blocked_url_tokens=blocked_tokens,
            trusted_domains=trusted_domains,
            block_wordpress=_env_bool(block_wordpress_env, True),
            block_explicit=_env_bool(block_explicit_env, True),
        )

    @staticmethod
    def host_from_url(url: str) -> str:
        try:
            return (urlparse(str(url or "")).netloc or "").lower().split(":")[0].strip(".")
        except Exception:
            return ""

    def is_trusted_host(self, host: str) -> bool:
        h = str(host or "").lower().strip(".")
        if not h:
            return False
        for d in self.trusted_domains:
            if h == d or h.endswith("." + d):
                return True
        return False

    def is_bad_domain(self, host: str) -> bool:
        h = str(host or "").lower().strip(".")
        if not h:
            return False
        for d in self.bad_domains:
            if h == d or h.endswith("." + d):
                return True
        return False

    def is_explicit_url(self, url: str, host: str = "") -> bool:
        if not self.block_explicit:
            return False
        value = f"{host} {url}".lower()
        return bool(
            re.search(
                r"(?:^|[^a-z0-9])(xxx|porn|nsfw|hentai|sex-video|adult-video|erotic|nude)(?:[^a-z0-9]|$)",
                value,
            )
        )

    def is_wordpress_like(self, url: str, host: str = "") -> bool:
        if not self.block_wordpress:
            return False
        url_l = str(url or "").lower()
        host_l = str(host or "").lower()
        return ("wordpress" in host_l) or ("/wp-content/" in url_l) or ("/wp-json/" in url_l)

    def is_blocked_url_pattern(self, url: str) -> bool:
        u = str(url or "").lower()
        return any(tok in u for tok in self.blocked_url_tokens)

