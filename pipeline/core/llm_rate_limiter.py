"""Shared LLM request limiter used across verifier and helper call-sites."""

from __future__ import annotations

import os
import sqlite3
import time
import hashlib
import json
from datetime import datetime, timezone
from collections import deque
from threading import Lock, Semaphore
from typing import Any, Dict

import requests


def _env_truthy(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


class SharedLLMRateLimiter:
    """Cross-process request limiter + in-process concurrency guard for LLM calls."""

    _cfg_lock = Lock()
    _rl_lock = Lock()
    _rl_times_by_bucket: Dict[str, deque] = {}
    _concurrency_semaphore: Semaphore | None = None
    _concurrency_size: int | None = None
    _global_rate_db_init_lock = Lock()
    _global_rate_db_initialized = set()
    _cooldown_lock = Lock()
    _cooldown_until = 0.0

    def __init__(
        self,
        requests_per_minute: int,
        max_concurrent: int,
        global_rate_limit: bool,
        global_rate_db: str,
        cooldown_seconds: float = 3.0,
        provider: str = "",
    ) -> None:
        self.requests_per_minute = max(0, int(requests_per_minute))
        self.max_concurrent = max(1, int(max_concurrent))
        self.global_rate_limit = bool(global_rate_limit)
        self.global_rate_db = self._resolve_global_rate_db_path(global_rate_db)
        self.cooldown_seconds = max(0.0, float(cooldown_seconds))
        self.provider = str(provider or "").strip().lower()
        self.audit_enabled = _env_truthy("LLM_SHARED_AUDIT_ENABLE", True)
        self.run_id = self._resolve_run_id()
        self._ensure_global_rate_db()
        self._ensure_global_concurrency_semaphore()

    @classmethod
    def from_env(
        cls,
        provider: str = "",
        requests_per_minute: int | None = None,
        max_concurrent: int | None = None,
        global_rate_limit: bool | None = None,
        global_rate_db: str | None = None,
    ) -> "SharedLLMRateLimiter":
        provider_l = str(provider or "").strip().lower()

        rpm_fallback = os.getenv(
            "GLOBAL_LLM_REQUESTS_PER_MINUTE",
            os.getenv("LLM_VERIFIER_REQUESTS_PER_MINUTE", os.getenv("GLOBAL_REQUESTS_PER_MINUTE", "25")),
        )
        if provider_l == "sarvam":
            rpm_fallback = os.getenv("SARVAM_REQUESTS_PER_MINUTE", rpm_fallback)
        rpm = (
            int(requests_per_minute)
            if requests_per_minute is not None
            else int(os.getenv("LLM_SHARED_REQUESTS_PER_MINUTE", os.getenv("LLM_SHARED_RPM_PER_KEY", rpm_fallback)))
        )

        conc_fallback = os.getenv(
            "LLM_VERIFIER_MAX_CONCURRENT",
            os.getenv("MAX_CONCURRENT_REQUESTS", "4"),
        )
        conc = (
            int(max_concurrent)
            if max_concurrent is not None
            else int(os.getenv("LLM_SHARED_MAX_CONCURRENT", conc_fallback))
        )

        global_limit = (
            bool(global_rate_limit)
            if global_rate_limit is not None
            else _env_truthy(
                "LLM_SHARED_GLOBAL_RATE_LIMIT",
                _env_truthy("LLM_VERIFIER_GLOBAL_RATE_LIMIT", True),
            )
        )
        db_path = (
            str(global_rate_db).strip()
            if global_rate_db is not None and str(global_rate_db).strip()
            else os.getenv(
                "LLM_SHARED_RATE_DB",
                os.getenv("LLM_VERIFIER_GLOBAL_RATE_DB", ".llm_verifier_rate_limit.sqlite"),
            ).strip()
        )
        cooldown = float(os.getenv("LLM_SHARED_429_COOLDOWN_SECONDS", "3"))
        return cls(
            requests_per_minute=rpm,
            max_concurrent=conc,
            global_rate_limit=global_limit,
            global_rate_db=db_path,
            cooldown_seconds=cooldown,
            provider=provider_l,
        )

    def _resolve_global_rate_db_path(self, raw_path: str) -> str:
        path = (raw_path or ".llm_verifier_rate_limit.sqlite").strip()
        if os.path.isabs(path):
            return path
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return os.path.abspath(os.path.join(project_root, path))

    def _wait_for_cooldown(self) -> None:
        while True:
            with self._cooldown_lock:
                remaining = self.__class__._cooldown_until - time.time()
            if remaining <= 0:
                return
            time.sleep(min(max(remaining, 0.05), 1.0))

    def _note_rate_limited(self) -> None:
        if self.cooldown_seconds <= 0:
            return
        until = time.time() + self.cooldown_seconds
        with self._cooldown_lock:
            if until > self.__class__._cooldown_until:
                self.__class__._cooldown_until = until

    def _acquire_rate_token(self, bucket: str) -> None:
        rpm = self.requests_per_minute
        if rpm <= 0:
            return
        if self.global_rate_limit:
            self._acquire_rate_token_global(rpm, bucket=bucket)
            return
        while True:
            with self._rl_lock:
                q = self._rl_times_by_bucket.setdefault(bucket, deque())
                now = time.monotonic()
                while q and (now - q[0]) >= 60.0:
                    q.popleft()
                if len(q) < rpm:
                    q.append(now)
                    return
                wait_s = 60.0 - (now - q[0])
            time.sleep(max(wait_s, 0.05))

    def _ensure_global_rate_db(self) -> None:
        if (not self.global_rate_limit) and (not self.audit_enabled):
            return
        db_path = self.global_rate_db
        with self._global_rate_db_init_lock:
            if db_path in self._global_rate_db_initialized:
                return
            parent = os.path.dirname(db_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
            try:
                conn.execute("PRAGMA journal_mode=WAL;")
                conn.execute("CREATE TABLE IF NOT EXISTS llm_requests (ts REAL NOT NULL)")
                conn.execute("CREATE INDEX IF NOT EXISTS idx_llm_requests_ts ON llm_requests(ts)")
                conn.execute(
                    "CREATE TABLE IF NOT EXISTS llm_requests_keyed (bucket TEXT NOT NULL, ts REAL NOT NULL)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_llm_requests_keyed_bucket_ts ON llm_requests_keyed(bucket, ts)"
                )
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS llm_requests_audit (
                        ts REAL NOT NULL,
                        ts_utc TEXT NOT NULL,
                        run_id TEXT NOT NULL,
                        pid INTEGER NOT NULL,
                        provider TEXT NOT NULL,
                        bucket TEXT NOT NULL,
                        event TEXT NOT NULL,
                        status TEXT NOT NULL,
                        endpoint TEXT NOT NULL,
                        detail TEXT NOT NULL
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_llm_audit_run_ts ON llm_requests_audit(run_id, ts)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_llm_audit_ts ON llm_requests_audit(ts)"
                )
            finally:
                conn.close()
            self._global_rate_db_initialized.add(db_path)

    def _acquire_rate_token_global(self, rpm: int, bucket: str) -> None:
        db_path = self.global_rate_db
        while True:
            now = time.time()
            try:
                conn = sqlite3.connect(db_path, timeout=30.0, isolation_level=None)
                try:
                    conn.execute("BEGIN IMMEDIATE")
                    cutoff = now - 60.0
                    conn.execute("DELETE FROM llm_requests_keyed WHERE ts < ?", (cutoff,))
                    count = int(
                        conn.execute(
                            "SELECT COUNT(*) FROM llm_requests_keyed WHERE bucket = ?",
                            (bucket,),
                        ).fetchone()[0]
                    )
                    if count < rpm:
                        conn.execute(
                            "INSERT INTO llm_requests_keyed(bucket, ts) VALUES (?, ?)",
                            (bucket, now),
                        )
                        conn.execute("COMMIT")
                        self._append_audit(
                            bucket=bucket,
                            event="rate_token",
                            status="granted",
                            endpoint="",
                            detail=f"global=1 rpm={rpm}",
                        )
                        return
                    oldest = conn.execute(
                        "SELECT MIN(ts) FROM llm_requests_keyed WHERE bucket = ?",
                        (bucket,),
                    ).fetchone()[0]
                    conn.execute("COMMIT")
                except Exception:
                    try:
                        conn.execute("ROLLBACK")
                    except Exception:
                        pass
                    raise
                finally:
                    conn.close()
            except Exception:
                with self._rl_lock:
                    q = self._rl_times_by_bucket.setdefault(bucket, deque())
                    mono_now = time.monotonic()
                    while q and (mono_now - q[0]) >= 60.0:
                        q.popleft()
                    if len(q) < rpm:
                        q.append(mono_now)
                        return
                    wait_s = 60.0 - (mono_now - q[0])
                time.sleep(max(wait_s, 0.05))
                continue

            wait_s = 0.2
            if oldest:
                wait_s = max(0.05, 60.0 - (now - float(oldest)))
            self._append_audit(
                bucket=bucket,
                event="rate_token",
                status="waiting",
                endpoint="",
                detail=f"global=1 rpm={rpm} wait_s={round(wait_s,3)}",
            )
            time.sleep(wait_s)

    def _ensure_global_concurrency_semaphore(self) -> None:
        with self._cfg_lock:
            size = int(self.max_concurrent)
            if self._concurrency_semaphore is None or self._concurrency_size != size:
                self.__class__._concurrency_semaphore = Semaphore(size)
                self.__class__._concurrency_size = size

    def post_json(
        self,
        endpoint: str,
        headers: Dict[str, str],
        payload: Dict[str, Any],
        timeout: float,
    ) -> Dict[str, Any]:
        self._wait_for_cooldown()
        bucket = self._bucket_from_headers(headers)
        self._acquire_rate_token(bucket=bucket)
        sem = self.__class__._concurrency_semaphore
        if sem is None:
            self._ensure_global_concurrency_semaphore()
            sem = self.__class__._concurrency_semaphore
        assert sem is not None

        sem.acquire()
        try:
            self._append_audit(
                bucket=bucket,
                event="request",
                status="start",
                endpoint=endpoint,
                detail="post_json",
            )
            response = requests.post(endpoint, headers=headers, json=payload, timeout=timeout)
        except Exception as exc:
            msg = str(exc)
            self._append_audit(
                bucket=bucket,
                event="request",
                status="error",
                endpoint=endpoint,
                detail=msg[:512],
            )
            if "429" in msg or "Too Many Requests" in msg:
                self._note_rate_limited()
                self._append_audit(
                    bucket=bucket,
                    event="request",
                    status="rate_limited",
                    endpoint=endpoint,
                    detail=msg[:512],
                )
            raise
        finally:
            sem.release()

        if response.status_code == 429:
            self._note_rate_limited()
            self._append_audit(
                bucket=bucket,
                event="request",
                status="rate_limited",
                endpoint=endpoint,
                detail="http_429",
            )
        else:
            self._append_audit(
                bucket=bucket,
                event="request",
                status="ok",
                endpoint=endpoint,
                detail=f"http_{response.status_code}",
            )
        response.raise_for_status()
        return response.json() if response.content else {}

    def run_with_limits(self, headers: Dict[str, str], fn):
        """
        Execute an arbitrary outbound call under shared rate/concurrency limits.
        Useful for SDK clients that don't go through requests.post directly.
        """
        self._wait_for_cooldown()
        bucket = self._bucket_from_headers(headers or {})
        self._acquire_rate_token(bucket=bucket)
        sem = self.__class__._concurrency_semaphore
        if sem is None:
            self._ensure_global_concurrency_semaphore()
            sem = self.__class__._concurrency_semaphore
        assert sem is not None

        sem.acquire()
        try:
            self._append_audit(
                bucket=bucket,
                event="request",
                status="start",
                endpoint="run_with_limits",
                detail=getattr(fn, "__name__", "callable"),
            )
            return fn()
        except Exception as exc:
            msg = str(exc)
            self._append_audit(
                bucket=bucket,
                event="request",
                status="error",
                endpoint="run_with_limits",
                detail=msg[:512],
            )
            if "429" in msg or "Too Many Requests" in msg:
                self._note_rate_limited()
                self._append_audit(
                    bucket=bucket,
                    event="request",
                    status="rate_limited",
                    endpoint="run_with_limits",
                    detail=msg[:512],
                )
            raise
        finally:
            sem.release()

    def _bucket_from_headers(self, headers: Dict[str, str]) -> str:
        key_raw = ""
        sub_key = str(headers.get("API-Subscription-Key", "") or "").strip()
        auth = str(headers.get("Authorization", "") or "").strip()
        if sub_key:
            key_raw = sub_key
        elif auth.lower().startswith("bearer "):
            key_raw = auth[7:].strip()
        if not key_raw:
            return "no_key"
        digest = hashlib.sha256(key_raw.encode("utf-8")).hexdigest()[:16]
        return f"key:{digest}"

    def _resolve_run_id(self) -> str:
        for key in ("THESIS_RUN_ID", "RUN_ID", "BENCHMARK_RUN_ID", "EXPERIMENT_RUN_ID"):
            value = str(os.getenv(key, "") or "").strip()
            if value:
                return value
        ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        return f"adhoc_{ts}_pid{os.getpid()}"

    def _append_audit(self, bucket: str, event: str, status: str, endpoint: str, detail: str) -> None:
        if not self.audit_enabled:
            return
        try:
            self._ensure_global_rate_db()
            now = time.time()
            ts_utc = datetime.fromtimestamp(now, tz=timezone.utc).isoformat()
            conn = sqlite3.connect(self.global_rate_db, timeout=30.0, isolation_level=None)
            try:
                conn.execute(
                    """
                    INSERT INTO llm_requests_audit
                    (ts, ts_utc, run_id, pid, provider, bucket, event, status, endpoint, detail)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        float(now),
                        str(ts_utc),
                        str(self.run_id),
                        int(os.getpid()),
                        str(self.provider),
                        str(bucket or ""),
                        str(event or ""),
                        str(status or ""),
                        str(endpoint or ""),
                        str(detail or ""),
                    ),
                )
            finally:
                conn.close()
        except Exception:
            # Best-effort audit logging; never block pipeline execution.
            return

    def note_usage(
        self,
        headers: Dict[str, str],
        model: str,
        usage: Dict[str, Any],
    ) -> None:
        """Persist provider usage payload (tokens) into audit history."""
        if not self.audit_enabled:
            return
        if not isinstance(usage, dict) or not usage:
            return
        bucket = self._bucket_from_headers(headers or {})
        try:
            detail = json.dumps(usage, ensure_ascii=False)
        except Exception:
            detail = str(usage)
        self._append_audit(
            bucket=bucket,
            event="usage",
            status="ok",
            endpoint=str(model or ""),
            detail=detail[:2048],
        )
