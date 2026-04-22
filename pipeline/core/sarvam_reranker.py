"""Sarvam-based final reranker for top-K evidence rows."""

from __future__ import annotations

import json
import os
import re
from typing import Any, Dict, List, Tuple

from pipeline.verdict.llm_verifier import LLMVerifier


def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _clamp(value: Any, low: float, high: float, fallback: float) -> float:
    try:
        num = float(value)
    except Exception:
        num = fallback
    return max(low, min(high, num))


def _normalize_stance(value: Any) -> str:
    v = str(value or "").strip().lower()
    if v in {"support", "supported", "entailment", "entails", "true"}:
        return "support"
    if v in {"refute", "refuted", "contradiction", "contradicts", "false"}:
        return "refute"
    return "neutral"


class SarvamReranker:
    """Reasoning-based reranking + stance tagging on top-K evidence rows."""

    def __init__(self) -> None:
        self.enabled = _env_bool("ENABLE_SARVAM_FINAL_RERANK", True)
        # Image pipeline can opt in independently without affecting text/PDF baseline.
        self.image_enable = _env_bool("IMAGE_ENABLE_SARVAM_RERANK", False)
        self.image_allow_en = _env_bool("IMAGE_SARVAM_ALLOW_EN", False)
        self.multi_only = _env_bool("SARVAM_RERANK_MULTI_ONLY", True)
        self.top_k = max(1, int(os.getenv("SARVAM_RERANK_TOP_K", "10")))
        self.max_tokens = max(256, int(os.getenv("SARVAM_RERANK_MAX_TOKENS", "1200")))
        self.relevance_blend = _clamp(
            os.getenv("SARVAM_RERANK_RELEVANCE_BLEND", "0.7"), 0.0, 1.0, 0.7
        )
        self.min_stance_conf = _clamp(
            os.getenv("SARVAM_RERANK_MIN_STANCE_CONF", "0.45"), 0.0, 1.0, 0.45
        )
        model = str(
            os.getenv(
                "SARVAM_RERANK_MODEL",
                os.getenv("LLM_VERIFIER_MODEL_MULTI", "sarvam-30b"),
            )
        ).strip()
        self.client = LLMVerifier(provider="sarvam", model=model)

    def _parse_json(self, content: str) -> Dict[str, Any]:
        raw = str(content or "").strip()
        if raw.startswith("```"):
            raw = re.sub(r"^```[a-zA-Z0-9_-]*\s*", "", raw)
            raw = re.sub(r"\s*```$", "", raw).strip()
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except Exception:
            start = raw.find("{")
            if start != -1:
                candidate = raw[start:]
                repaired = self._repair_truncated_json(candidate)
                if repaired:
                    try:
                        return json.loads(repaired)
                    except Exception:
                        pass
        return {}

    def _repair_truncated_json(self, text: str) -> str:
        s = str(text or "").strip()
        if not s:
            return ""
        # Trim trailing fence noise.
        s = re.sub(r"\s*```$", "", s).strip()
        # If cut while inside string, close quote.
        quote_count = 0
        escaped = False
        for ch in s:
            if escaped:
                escaped = False
                continue
            if ch == "\\":
                escaped = True
                continue
            if ch == '"':
                quote_count += 1
        if quote_count % 2 == 1:
            s += '"'

        open_curly = s.count("{")
        close_curly = s.count("}")
        open_square = s.count("[")
        close_square = s.count("]")
        if close_square < open_square:
            s += "]" * (open_square - close_square)
        if close_curly < open_curly:
            s += "}" * (open_curly - close_curly)
        return s

    def _extract_updates(self, parsed: Dict[str, Any]) -> List[Dict[str, Any]]:
        if not isinstance(parsed, dict):
            return []
        items = parsed.get("items")
        if isinstance(items, list):
            return items
        # Compatibility with existing verifier-style schema.
        updates = parsed.get("evidence_updates")
        if isinstance(updates, list):
            out: List[Dict[str, Any]] = []
            for row in updates:
                if not isinstance(row, dict):
                    continue
                out.append(
                    {
                        "index": row.get("index"),
                        "relevance": row.get("relevance"),
                        "stance": row.get("stance"),
                        "confidence": row.get("confidence", 0.6),
                        "reason": row.get("reason", ""),
                    }
                )
            return out
        return []

    def _extract_updates_from_text(self, content: str) -> List[Dict[str, Any]]:
        raw = str(content or "")
        if not raw:
            return []
        # Fallback: parse loose objects that mention index + stance/relevance.
        # Works even when JSON array is truncated.
        pattern = re.compile(
            r"(?:index|idx)\s*[:=]\s*(\d+).*?"
            r"(?:relevance|score)\s*[:=]\s*([0-9]*\.?[0-9]+).*?"
            r"(?:stance|label)\s*[:=]\s*['\"]?(support|refute|neutral)['\"]?"
            r"(?:.*?(?:confidence|conf)\s*[:=]\s*([0-9]*\.?[0-9]+))?"
            r"(?:.*?(?:reason|why)\s*[:=]\s*['\"]?([^,\n\r\}\]]{1,220}))?",
            flags=re.IGNORECASE | re.DOTALL,
        )
        out: List[Dict[str, Any]] = []
        seen = set()
        for m in pattern.finditer(raw):
            idx = int(m.group(1))
            if idx in seen:
                continue
            seen.add(idx)
            out.append(
                {
                    "index": idx,
                    "relevance": m.group(2),
                    "stance": m.group(3),
                    "confidence": m.group(4) if m.group(4) is not None else 0.6,
                    "reason": (m.group(5) or "").strip(),
                }
            )
        return out

    def _extract_updates_from_line_format(self, content: str) -> List[Dict[str, Any]]:
        raw = str(content or "")
        if not raw:
            return []
        out: List[Dict[str, Any]] = []
        seen = set()
        # Example:
        # EVIDENCE 1 | relevance=0.82 | stance=support | conf=0.77 | reason=...
        pat = re.compile(
            r"EVIDENCE\s+(\d+)\s*\|.*?"
            r"relevance\s*=\s*([0-9]*\.?[0-9]+)\s*\|.*?"
            r"stance\s*=\s*(support|refute|neutral)\s*\|.*?"
            r"(?:conf(?:idence)?\s*=\s*([0-9]*\.?[0-9]+)\s*\|)?\s*"
            r"reason\s*=\s*(.+)$",
            flags=re.IGNORECASE,
        )
        for line in raw.splitlines():
            line_s = line.strip()
            if not line_s:
                continue
            m = pat.search(line_s)
            if not m:
                continue
            idx = int(m.group(1))
            if idx in seen:
                continue
            seen.add(idx)
            out.append(
                {
                    "index": idx,
                    "relevance": m.group(2),
                    "stance": m.group(3),
                    "confidence": m.group(4) if m.group(4) is not None else 0.6,
                    "reason": (m.group(5) or "").strip()[:220],
                }
            )
        return out

    def _build_prompt(self, claim: str, rows: List[Dict], provisional_verdict: str) -> Tuple[str, str]:
        parts: List[str] = []
        for idx, ev in enumerate(rows, start=1):
            text = str(ev.get("text") or "").replace("\n", " ").strip()
            source = str(ev.get("source") or "unknown").strip()
            url = str(ev.get("url") or "").strip()
            base_rel = _clamp(ev.get("relevance", 0.0), 0.0, 1.0, 0.0)
            parts.append(
                f"[{idx}] source={source}; url={url}; base_relevance={base_rel:.3f}; text={text[:320]}"
            )
        blob = "\n".join(parts)
        system = (
            "You are a strict evidence reranker for fact-checking. "
            "Output JSON only. No markdown. "
            "Schema: {"
            "\"items\":[{\"index\":1,\"relevance\":0.0,\"stance\":\"support|refute|neutral\",\"confidence\":0.0,\"reason\":\"...\"}],"
            "\"verification\":{\"supports_provisional_verdict\":true,\"contradictions_exist\":false,"
            "\"confidence_adjustment\":0.0,\"explanation\":\"...\"}"
            "}. "
            "Rules: relevance/confidence in [0,1], reason one short line."
        )
        user = (
            f"Claim: {claim}\n"
            f"Provisional verdict: {provisional_verdict}\n"
            f"Evidence snippets:\n{blob}\n\n"
            "Task:\n"
            "1) For each evidence item assign relevance 0-1.\n"
            "2) Assign stance support/refute/neutral against claim.\n"
            "3) Add one-line reason.\n"
            "4) Verify provisional verdict against shortlisted evidence and report contradictions."
        )
        return system, user

    def _build_line_prompt(
        self,
        claim: str,
        rows: List[Dict],
        provisional_verdict: str,
    ) -> Tuple[str, str]:
        parts: List[str] = []
        for idx, ev in enumerate(rows, start=1):
            text = str(ev.get("text") or "").replace("\n", " ").strip()
            source = str(ev.get("source") or "unknown").strip()
            base_rel = _clamp(ev.get("relevance", 0.0), 0.0, 1.0, 0.0)
            parts.append(
                f"[{idx}] source={source}; base_relevance={base_rel:.3f}; text={text[:250]}"
            )
        blob = "\n".join(parts)
        system = (
            "You are evidence reranker. Output plain text lines only, no JSON, no markdown. "
            "Each evidence line MUST follow exact format: "
            "EVIDENCE <index> | relevance=<0-1> | stance=<support|refute|neutral> | "
            "conf=<0-1> | reason=<one short sentence>. "
            "Then add one final line: "
            "VERIFY | supports_provisional=<true|false> | contradictions=<true|false> | "
            "delta=<-1..1> | reason=<one short sentence>."
        )
        user = (
            f"Claim: {claim}\n"
            f"Provisional verdict: {provisional_verdict}\n"
            f"Evidence snippets:\n{blob}\n\n"
            "Return one EVIDENCE line for every index."
        )
        return system, user

    def _call_chat(self, system_prompt: str, user_prompt: str, max_tokens: int, reasoning_effort: str) -> str:
        base = self.client.base_url.rstrip("/")
        endpoint = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        sub_key = os.getenv("SARVAM_API_SUBSCRIPTION_KEY", "").strip()
        if sub_key:
            headers["API-Subscription-Key"] = sub_key
        else:
            headers["Authorization"] = f"Bearer {self.client.api_key}"
        payload: Dict[str, Any] = {
            "model": self.client.model,
            "temperature": 0,
            "max_tokens": max_tokens,
            "stream": False,
            "reasoning_effort": reasoning_effort,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        }
        data = self.client._post_completion(  # pylint: disable=protected-access
            endpoint=endpoint,
            headers=headers,
            payload=payload,
        )
        return str(
            ((data.get("choices") or [{}])[0].get("message") or {}).get("content") or ""
        ).strip()

    def rerank(
        self,
        claim: str,
        evidence_list: List[Dict],
        language: str = "en",
        provisional_verdict: str = "neutral",
        is_image_mode: bool = False,
    ) -> Tuple[List[Dict], Dict[str, Any]]:
        meta: Dict[str, Any] = {
            "enabled": bool(self.enabled or (is_image_mode and self.image_enable)),
            "applied": False,
            "reason": "disabled",
            "top_k": self.top_k,
            "updated_items": 0,
        }
        if not self.enabled and not (is_image_mode and self.image_enable):
            return evidence_list, meta
        lang_is_en = str(language or "en").lower().startswith("en")
        if is_image_mode and (not self.image_allow_en) and lang_is_en:
            meta["reason"] = "skipped_image_en_language"
            return evidence_list, meta
        if self.multi_only and lang_is_en:
            meta["reason"] = "skipped_en_language"
            return evidence_list, meta
        if not evidence_list:
            meta["reason"] = "no_evidence"
            return evidence_list, meta
        if not self.client.api_key:
            meta["reason"] = "sarvam_api_key_missing"
            return evidence_list, meta

        rows = list(evidence_list)
        top_n = min(self.top_k, len(rows))
        head = [dict(x) for x in rows[:top_n]]
        system_prompt, user_prompt = self._build_prompt(claim, head, provisional_verdict)

        try:
            content = self._call_chat(
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                max_tokens=self.max_tokens,
                reasoning_effort=os.getenv("SARVAM_REASONING_EFFORT", "medium"),
            )
            parsed = self._parse_json(content)
            updates = self._extract_updates(parsed)
            if not updates:
                updates = self._extract_updates_from_text(content)
            if not updates:
                # Fallback pass: deterministic line format, shorter output, low reasoning.
                line_system, line_user = self._build_line_prompt(
                    claim=claim,
                    rows=head,
                    provisional_verdict=provisional_verdict,
                )
                content2 = self._call_chat(
                    system_prompt=line_system,
                    user_prompt=line_user,
                    max_tokens=max(320, min(self.max_tokens, 900)),
                    reasoning_effort="low",
                )
                updates = self._extract_updates_from_line_format(content2)
                if not updates:
                    updates = self._extract_updates_from_text(content2)
                if not updates:
                    # Last fallback: reuse verifier-style structured updates.
                    try:
                        vres = self.client.verify(claim=claim, evidence=head)
                    except Exception:
                        vres = {}
                    v_updates = vres.get("evidence_updates") if isinstance(vres, dict) else None
                    if isinstance(v_updates, list) and v_updates:
                        updates = [
                            {
                                "index": row.get("index"),
                                "relevance": row.get("relevance"),
                                "stance": row.get("stance"),
                                "confidence": row.get("confidence", 0.6),
                                "reason": vres.get("reason", ""),
                            }
                            for row in v_updates
                            if isinstance(row, dict)
                        ]
                if not updates:
                    meta["reason"] = "invalid_rerank_response"
                    meta["raw_excerpt"] = (content or content2)[:220]
                    return evidence_list, meta

            changed = 0
            for row in updates:
                if not isinstance(row, dict):
                    continue
                try:
                    idx = int(row.get("index", 0))
                except Exception:
                    continue
                if idx < 1 or idx > top_n:
                    continue
                ev = head[idx - 1]
                old_rel = _clamp(ev.get("relevance", 0.0), 0.0, 1.0, 0.0)
                llm_rel = _clamp(row.get("relevance", old_rel), 0.0, 1.0, old_rel)
                blended = ((1.0 - self.relevance_blend) * old_rel) + (self.relevance_blend * llm_rel)
                stance = _normalize_stance(row.get("stance", "neutral"))
                conf = _clamp(row.get("confidence", 0.5), 0.0, 1.0, 0.5)
                reason = str(row.get("reason") or "").strip()[:220]

                ev["relevance"] = blended
                ev["sarvam_relevance"] = llm_rel
                ev["sarvam_stance"] = stance
                ev["sarvam_stance_confidence"] = conf
                ev["sarvam_reason"] = reason
                ev["_sarvam_enhanced"] = True
                changed += 1

            rows[:top_n] = head
            rows.sort(key=lambda x: float(x.get("relevance", 0.0) or 0.0), reverse=True)
            meta["applied"] = changed > 0
            meta["reason"] = "ok" if changed > 0 else "no_valid_updates"
            meta["updated_items"] = changed
            if isinstance(parsed, dict) and isinstance(parsed.get("verification"), dict):
                meta["verification"] = parsed["verification"]
            return rows, meta
        except Exception as exc:
            meta["reason"] = f"sarvam_rerank_failed: {exc}"
            return evidence_list, meta
