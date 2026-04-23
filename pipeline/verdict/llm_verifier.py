import os
import json
import re
from threading import Lock
from typing import Any, Dict, List

from pipeline.core.llm_rate_limiter import SharedLLMRateLimiter


class LLMVerifier:
    """Provider-aware verifier adapter for neutral or low-confidence cases."""
    _key_rr_lock = Lock()
    _key_rr_counters: Dict[str, int] = {}

    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini"):
        self.provider = (provider or "openai").strip().lower()
        self.model = model
        self.base_url = self._resolve_base_url()
        self.api_keys = self._resolve_api_keys()
        self.api_key = self.api_keys[0] if self.api_keys else None
        self.timeout_s = float(os.getenv("LLM_VERIFIER_TIMEOUT_SECONDS", "25"))
        self.max_tokens = int(os.getenv("LLM_VERIFIER_MAX_TOKENS", "512"))
        rpm_default = os.getenv(
            "LLM_SHARED_REQUESTS_PER_MINUTE",
            os.getenv("LLM_SHARED_RPM_PER_KEY", os.getenv("GLOBAL_REQUESTS_PER_MINUTE", "25")),
        )
        if self.provider == "sarvam":
            rpm_value = os.getenv(
                "LLM_SHARED_REQUESTS_PER_MINUTE",
                os.getenv("LLM_SHARED_RPM_PER_KEY", os.getenv("SARVAM_REQUESTS_PER_MINUTE", rpm_default)),
            )
        else:
            rpm_value = os.getenv(
                "LLM_SHARED_REQUESTS_PER_MINUTE",
                os.getenv("LLM_SHARED_RPM_PER_KEY", os.getenv("LLM_VERIFIER_REQUESTS_PER_MINUTE", rpm_default)),
            )
        self.requests_per_minute = int(rpm_value)  # 0 => unlimited
        self.global_rate_limit = os.getenv("LLM_VERIFIER_GLOBAL_RATE_LIMIT", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        self.global_rate_db = self._resolve_global_rate_db_path(
            os.getenv("LLM_VERIFIER_GLOBAL_RATE_DB", ".llm_verifier_rate_limit.sqlite").strip()
        )
        conc_default = os.getenv("MAX_CONCURRENT_REQUESTS", "4")
        self.max_concurrent = max(1, int(os.getenv("LLM_VERIFIER_MAX_CONCURRENT", conc_default)))
        self._shared_limiter = SharedLLMRateLimiter.from_env(
            provider=self.provider,
            requests_per_minute=self.requests_per_minute,
            max_concurrent=self.max_concurrent,
            global_rate_limit=self.global_rate_limit,
            global_rate_db=self.global_rate_db,
        )

    def _resolve_global_rate_db_path(self, raw_path: str) -> str:
        """Resolve limiter DB path to a stable absolute file shared across processes."""
        path = (raw_path or ".llm_verifier_rate_limit.sqlite").strip()
        if os.path.isabs(path):
            return path
        project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
        return os.path.abspath(os.path.join(project_root, path))

    def _split_keys(self, list_env: str, single_env: str) -> List[str]:
        out: List[str] = []
        raw = str(os.getenv(list_env, "") or "").strip()
        if raw:
            for tok in raw.split(","):
                v = tok.strip()
                if v and v not in out:
                    out.append(v)
        single = str(os.getenv(single_env, "") or "").strip()
        if single:
            # Be tolerant: allow comma-separated fallback even in single-key env.
            for tok in single.split(","):
                v = tok.strip()
                if v and v not in out:
                    out.append(v)
        return out

    def _resolve_api_keys(self) -> List[str]:
        if self.provider == "openai":
            return self._split_keys("OPENAI_API_KEYS", "OPENAI_API_KEY")
        if self.provider == "groq":
            return self._split_keys("GROQ_API_KEYS", "GROQ_API_KEY")
        if self.provider == "openrouter":
            return self._split_keys("OPENROUTER_API_KEYS", "OPENROUTER_API_KEY")
        if self.provider == "sarvam":
            sub = str(os.getenv("SARVAM_API_SUBSCRIPTION_KEY", "") or "").strip()
            key = str(os.getenv("SARVAM_API_KEY", "") or "").strip()
            out = []
            if sub:
                out.append(sub)
            if key and key not in out:
                out.append(key)
            return out
        return []

    def _rotated_api_keys(self) -> List[str]:
        keys = list(self.api_keys or [])
        if len(keys) <= 1:
            return keys
        bucket = f"{self.provider}:{self.model}"
        with self._key_rr_lock:
            idx = self._key_rr_counters.get(bucket, 0) % len(keys)
            self._key_rr_counters[bucket] = idx + 1
        return keys[idx:] + keys[:idx]

    def _default_base_url(self) -> str:
        if self.provider == "groq":
            return "https://api.groq.com/openai/v1"
        if self.provider == "openrouter":
            return "https://openrouter.ai/api/v1"
        if self.provider == "sarvam":
            return "https://api.sarvam.ai/v1"
        return "https://api.openai.com/v1"

    def _resolve_base_url(self) -> str:
        """Resolve provider-specific base URL so EN and multi providers do not conflict."""
        if self.provider == "sarvam":
            return (
                os.getenv("SARVAM_API_BASE_URL", "").strip()
                or os.getenv("LLM_VERIFIER_BASE_URL_MULTI", "").strip()
                or self._default_base_url()
            )
        if self.provider == "groq":
            return (
                os.getenv("GROQ_API_BASE_URL", "").strip()
                or os.getenv("LLM_VERIFIER_BASE_URL_EN", "").strip()
                or os.getenv("LLM_VERIFIER_BASE_URL", "").strip()
                or self._default_base_url()
            )
        if self.provider == "openrouter":
            return (
                os.getenv("OPENROUTER_API_BASE_URL", "").strip()
                or os.getenv("LLM_VERIFIER_BASE_URL", "").strip()
                or self._default_base_url()
            )
        if self.provider == "openai":
            return (
                os.getenv("OPENAI_API_BASE_URL", "").strip()
                or os.getenv("LLM_VERIFIER_BASE_URL", "").strip()
                or self._default_base_url()
            )
        return os.getenv("LLM_VERIFIER_BASE_URL", "").strip() or self._default_base_url()

    def _build_evidence_snippets(self, evidence: List[Dict[str, Any]]) -> str:
        rows = evidence if isinstance(evidence, list) else []
        snippets = []
        for idx, ev in enumerate(rows, start=1):
            text = (ev.get("text") or "").strip().replace("\n", " ")
            source = ev.get("source") or "unknown"
            rel = ev.get("relevance")
            snippets.append(
                f"[{idx}] source={source}; relevance={rel}; text={text[:320]}"
            )
        return "\n".join(snippets)

    def _normalize_evidence_updates(self, raw_updates: Any, evidence: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and normalize LLM-provided evidence updates."""
        if not isinstance(raw_updates, list):
            return []
        max_idx = len(evidence if isinstance(evidence, list) else [])
        out: List[Dict[str, Any]] = []
        seen = set()
        for row in raw_updates:
            if not isinstance(row, dict):
                continue
            try:
                idx = int(row.get("index"))
            except Exception:
                continue
            if idx < 1 or idx > max_idx:
                continue
            if idx in seen:
                continue
            seen.add(idx)
            stance = self._normalize_verdict(str(row.get("stance", "neutral")))
            rel_raw = row.get("relevance")
            relevance = None
            if rel_raw is not None:
                try:
                    relevance = max(0.0, min(1.0, float(rel_raw)))
                except Exception:
                    relevance = None
            out.append({"index": idx, "stance": stance, "relevance": relevance})
        return out

    def _normalize_verdict(self, value: str) -> str:
        v = (value or "").strip().lower()
        if v in {
            "support",
            "supported",
            "supported_by_evidence",
            "true",
            "verified",
            "verifies",
            "entails",
            "entailment",
            "likely_true",
        }:
            return "support"
        if v in {
            "refute",
            "refuted",
            "refutes",
            "false",
            "misleading",
            "unsupported",
            "not_supported",
            "contradiction",
            "contradicted",
            "debunked",
            "likely_false",
        }:
            return "refute"
        return "neutral"

    def _parse_json_from_content(self, content: str) -> Dict[str, Any]:
        raw = self._strip_code_fences(content)
        if not raw:
            raise ValueError("Empty LLM response content")

        first_error = None
        try:
            return json.loads(raw)
        except Exception as exc:
            first_error = exc

        start = raw.find("{")
        end = raw.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidate = raw[start : end + 1]
            try:
                return json.loads(candidate)
            except Exception:
                pass

        # Attempt tolerant extraction from malformed JSON-like text.
        salvaged = self._salvage_json_like_object(raw)
        if salvaged:
            return salvaged

        if first_error:
            raise first_error
        raise ValueError("Unable to parse LLM response as JSON")

    def _salvage_json_like_object(self, raw: str) -> Dict[str, Any]:
        text = raw or ""
        verdict = None
        for key in ("verdict", "label", "classification", "decision", "result"):
            m = re.search(
                rf'["\']?{key}["\']?\s*[:=]\s*["\']?([a-zA-Z_ -]+)',
                text,
                flags=re.IGNORECASE,
            )
            if m:
                normalized = self._normalize_verdict(m.group(1))
                if normalized in {"support", "refute", "neutral"}:
                    verdict = normalized
                    break

        if verdict is None:
            # Fall back to plaintext cues if no keyed verdict was found.
            return self._extract_plaintext_verdict_with_mode(text, allow_lenient=True)

        confidence = 0.5
        cm = re.search(
            r'["\']?confidence["\']?\s*[:=]\s*["\']?([0-9]{1,3}(?:\.[0-9]+)?%?)',
            text,
            flags=re.IGNORECASE,
        )
        if cm:
            raw_conf = cm.group(1).strip()
            if raw_conf.endswith("%"):
                confidence = float(raw_conf[:-1]) / 100.0
            else:
                confidence = float(raw_conf)
                if confidence > 1.0 and confidence <= 100.0:
                    confidence = confidence / 100.0
            confidence = max(0.0, min(1.0, confidence))

        reason = "LLM verification complete"
        rm = re.search(
            r'["\']?(reason|rationale|explanation)["\']?\s*[:=]\s*["\']?(.+)',
            text,
            flags=re.IGNORECASE | re.DOTALL,
        )
        if rm:
            reason = rm.group(2).strip().strip('",} ')[:600]
        elif text:
            reason = text[:600]

        return {"verdict": verdict, "confidence": confidence, "reason": reason}

    def _strip_code_fences(self, content: str) -> str:
        raw = (content or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw)
        return raw.strip()

    def _extract_plaintext_verdict(self, content: str) -> Dict[str, Any]:
        return self._extract_plaintext_verdict_with_mode(content, allow_lenient=False)

    def _extract_plaintext_verdict_with_mode(self, content: str, allow_lenient: bool = False) -> Dict[str, Any]:
        raw = self._strip_code_fences(content)
        lower = raw.lower()

        verdict = None
        if re.search(r"\b(support|supported|entailment|true|verified)\b", lower):
            verdict = "support"
        if re.search(r"\b(refute|refuted|contradiction|contradicted|false|misleading|debunked|unsupported)\b", lower):
            verdict = "refute"
        if re.search(r"\b(neutral|uncertain|not enough evidence|insufficient|inconclusive|unknown|mixed)\b", lower):
            verdict = "neutral"
        if re.search(r"\b(not\s+supported|does\s+not\s+support|not\s+true|incorrect)\b", lower):
            verdict = "refute"
        if re.search(r"\b(cannot\s+verify|can['’]t\s+verify)\b", lower):
            verdict = "neutral"

        # Multilingual / looser cues for non-English outputs.
        if verdict is None and allow_lenient:
            cues = {
                "support": [
                    "correct", "yes", "supported by evidence", "claim is true", "सही", "सत्य",
                    "உண்மை", "நிரூபிக்கப்பட்டது", "నిజం", "సరైనది", "ಸತ್ಯ", "ಸರಿಯಾಗಿದೆ",
                    "സത്യം", "ശരി",
                ],
                "refute": [
                    "incorrect", "wrong", "no", "claim is false", "गलत", "असत्य",
                    "தவறு", "பொய்", "తప్పు", "అబద్ధం", "ತಪ್ಪು", "ಸುಳ್ಳು", "തെറ്റ്", "അസത്യ",
                ],
                "neutral": [
                    "cannot verify", "not enough", "insufficient", "unclear", "unknown",
                    "अनिश्चित", "पर्याप्त प्रमाण नहीं", "தெரியவில்லை", "போதுமான ஆதாரம் இல்லை",
                    "సరిపడ ఆధారాలు లేవు", "ಅಪರ್ಯಾಪ್ತ ಸಾಕ್ಷಿ", "മതിയായ തെളിവില്ല",
                ],
            }
            for token in cues["support"]:
                if token in lower:
                    verdict = "support"
                    break
            if verdict is None:
                for token in cues["refute"]:
                    if token in lower:
                        verdict = "refute"
                        break
            if verdict is None:
                for token in cues["neutral"]:
                    if token in lower:
                        verdict = "neutral"
                        break

        if verdict is None:
            raise ValueError("No recognizable verdict token in LLM response")

        confidence = 0.5
        m = re.search(r"confidence[^0-9]*([01](?:\.\d+)?)", lower)
        if m:
            confidence = float(m.group(1))
        else:
            pct = re.search(r"([0-9]{1,3}(?:\.\d+)?)\s*%", lower)
            if pct:
                confidence = float(pct.group(1)) / 100.0

        reason = "LLM verification complete"
        rm = re.search(r"reason[^:]*:\s*(.+)", raw, flags=re.IGNORECASE | re.DOTALL)
        if rm:
            reason = rm.group(1).strip()[:600]
        elif raw:
            reason = raw[:600]

        return {"verdict": verdict, "confidence": confidence, "reason": reason}

    def _looks_multilingual(self, text: str) -> bool:
        sample = text or ""
        script_ranges = [
            r"[\u0900-\u097F]",  # Devanagari
            r"[\u0B80-\u0BFF]",  # Tamil
            r"[\u0C00-\u0C7F]",  # Telugu
            r"[\u0C80-\u0CFF]",  # Kannada
            r"[\u0D00-\u0D7F]",  # Malayalam
        ]
        return any(re.search(p, sample) for p in script_ranges)

    def _post_completion(self, endpoint: str, headers: Dict[str, str], payload: Dict[str, Any]) -> Dict[str, Any]:
        # Single shared limiter path for all outbound chat-completion calls.
        return self._shared_limiter.post_json(
            endpoint=endpoint,
            headers=headers,
            payload=payload,
            timeout=self.timeout_s,
        )

    def verify(self, claim, evidence):
        if not self.api_keys:
            return {
                "verdict": "neutral",
                "confidence": 0.5,
                "reason": f"{self.provider} API key not set",
            }

        base = self.base_url.rstrip("/")
        endpoint = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        evidence_blob = self._build_evidence_snippets(evidence if isinstance(evidence, list) else [])
        is_multilingual_claim = self._looks_multilingual(str(claim))
        system_prompt = (
            "You are a strict fact-check verifier. "
            "Return JSON only with keys: verdict, confidence, reason, evidence_updates. "
            "verdict must be one of support/refute/neutral. "
            "confidence must be float between 0 and 1. "
            "reason must be one short sentence (max 160 chars). "
            "evidence_updates must be a JSON array (can be empty) with entries: "
            "{index: integer, stance: support|refute|neutral, relevance: 0..1}. "
            "index refers to Evidence [i] shown in prompt. "
            "IMPORTANT: Always output verdict exactly as one token: support or refute or neutral."
        )
        if is_multilingual_claim:
            system_prompt += (
                " The claim language may be non-English, but output verdict token in English "
                "(support/refute/neutral) exactly."
            )
        user_prompt = (
            f"Claim: {claim}\n"
            f"Evidence:\n{evidence_blob}\n\n"
            "Respond with JSON only. Keep reason concise."
        )

        payload = {
            "model": self.model,
            "temperature": 0,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        if self.provider == "sarvam":
            payload["stream"] = False
            payload["reasoning_effort"] = os.getenv("SARVAM_REASONING_EFFORT", "medium")

        last_exc: Exception | None = None
        for api_key in self._rotated_api_keys():
            headers = {"Content-Type": "application/json"}
            if self.provider == "sarvam":
                sub_key = os.getenv("SARVAM_API_SUBSCRIPTION_KEY", "").strip()
                if sub_key:
                    headers["API-Subscription-Key"] = sub_key
                else:
                    headers["Authorization"] = f"Bearer {api_key}"
            else:
                headers["Authorization"] = f"Bearer {api_key}"
            if self.provider == "openrouter":
                app_name = os.getenv("LLM_VERIFIER_APP_NAME", "rfcs")
                headers["X-Title"] = app_name

            try:
                data = self._post_completion(endpoint=endpoint, headers=headers, payload=payload)
                choice = data.get("choices", [{}])[0] if isinstance(data, dict) else {}
                finish_reason = str(choice.get("finish_reason", "") or "").strip().lower()
                content = str(choice.get("message", {}).get("content", "") or "").strip()

                # Keep finish_reason for diagnostics; old-style path does not re-issue requests.
                _ = finish_reason

                parsed = None
                try:
                    parsed = self._parse_json_from_content(content)
                except Exception:
                    parsed = self._extract_plaintext_verdict_with_mode(
                        content,
                        allow_lenient=is_multilingual_claim,
                    )
                verdict = self._normalize_verdict(str(parsed.get("verdict", "neutral")))
                confidence = float(parsed.get("confidence", 0.5))
                confidence = max(0.0, min(1.0, confidence))
                reason = str(parsed.get("reason", "LLM verification complete"))
                evidence_updates = self._normalize_evidence_updates(
                    parsed.get("evidence_updates"),
                    evidence if isinstance(evidence, list) else [],
                )
                return {
                    "verdict": verdict,
                    "confidence": confidence,
                    "reason": reason,
                    "evidence_updates": evidence_updates,
                }
            except Exception as exc:
                last_exc = exc
                msg = str(exc)
                retryable = ("429" in msg) or ("Too Many Requests" in msg) or ("401" in msg) or ("403" in msg)
                if retryable:
                    continue
                break

        try:
            raise last_exc if last_exc is not None else RuntimeError("No API key available")
        except Exception as exc:
            return {
                "verdict": "neutral",
                "confidence": 0.5,
                "reason": f"{self.provider} verify call failed: {exc}",
                "evidence_updates": [],
            }
