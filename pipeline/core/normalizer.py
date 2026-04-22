"""Claim normalization and query rephrasing."""

import logging
import os
import re
import string
import threading
import requests
from typing import List

logger = logging.getLogger(__name__)


class ClaimNormalizer:
    """Clean and normalize claims for processing."""
    _i2e_lock = threading.Lock()
    _i2e_initialized = False
    _i2e_enabled = False
    _i2e_error = ""
    _i2e_tokenizer = None
    _i2e_model = None
    _i2e_ip = None
    _i2e_device = "cpu"
    _i2e_llm_cooldown_until = 0.0
    _query_rephrase_cache = {}
    _query_rephrase_cache_order = []
    _query_rephrase_cache_max = 512

    def normalize(self, claim: str) -> str:
        """Normalize claim text - clean whitespace, punctuation, encoding."""
        # Strip whitespace
        claim = claim.strip()
        # Normalize whitespace
        claim = re.sub(r'\s+', ' ', claim)
        # Remove zero-width characters
        claim = re.sub(r'[\u200b\u200c\u200d\ufeff]', '', claim)
        # Normalize quotes
        claim = claim.replace('"', '"').replace('"', '"').replace(''', "'").replace(''', "'")
        return claim
    
    def rephrase_for_search(self, claim: str, language: str = "en") -> list:
        """Generate search queries; keep EN and MULTI logic separate."""
        key = (str(claim or "").strip(), str(language or "en").strip().lower())
        cached = self.__class__._query_rephrase_cache.get(key)
        if cached is not None:
            return list(cached)

        lang = (language or "en").strip().lower()
        if lang.startswith("en"):
            out = self._rephrase_for_search_en(claim)
        else:
            out = self._rephrase_for_search_multi(claim, language=lang)

        self.__class__._query_rephrase_cache[key] = list(out)
        self.__class__._query_rephrase_cache_order.append(key)
        while len(self.__class__._query_rephrase_cache_order) > self.__class__._query_rephrase_cache_max:
            drop = self.__class__._query_rephrase_cache_order.pop(0)
            self.__class__._query_rephrase_cache.pop(drop, None)
        return out

    def _rephrase_for_search_en(self, claim: str) -> list:
        """English query formulation (old-style oriented)."""
        queries: List[str] = [claim]

        entities = self._extract_keywords(claim)
        if entities:
            queries.append(entities)

        question_form = self._to_question(claim)
        if question_form and question_form != claim:
            queries.append(question_form)

        return self._dedupe_queries(queries, cap=3)

    def _rephrase_for_search_multi(self, claim: str, language: str) -> list:
        """Multilingual query formulation with ASCII anchors."""
        queries: List[str] = [claim]

        entities = self._extract_keywords(claim)
        if entities:
            queries.append(entities)

        # Multi-only fallback: add translated English query if translator is available.
        # Controlled by env to avoid changing EN behavior and to keep rollout safe.
        if os.getenv("MULTI_ENABLE_EN_QUERY_TRANSLATION", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }:
            translated = self._translate_indic_to_english(claim, language=language)
            if translated:
                queries.append(translated)

        # Indic claims skip question wrapper to avoid noisy translations.
        return self._dedupe_queries(queries, cap=3)

    def _dedupe_queries(self, queries: List[str], cap: int = 4) -> List[str]:
        out: List[str] = []
        seen = set()
        for q in queries:
            k = q.strip().lower()
            if not k or k in seen:
                continue
            seen.add(k)
            out.append(q.strip())
        return out[:cap]
    
    def _extract_keywords(self, claim: str) -> str:
        """Extract keywords/entities from claim."""
        # Remove stopwords
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
                     'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
                     'should', 'may', 'might', 'shall', 'can', 'need', 'dare', 'ought',
                     'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by', 'from',
                     'as', 'into', 'through', 'during', 'before', 'after', 'above', 'below',
                     'between', 'out', 'off', 'over', 'under', 'again', 'further', 'then',
                     'once', 'here', 'there', 'when', 'where', 'why', 'how', 'all', 'both',
                     'each', 'few', 'more', 'most', 'other', 'some', 'such', 'no', 'nor',
                     'not', 'only', 'own', 'same', 'so', 'than', 'too', 'very', 'just',
                     'because', 'but', 'and', 'or', 'if', 'while', 'that', 'this', 'these',
                     'those', 'it', 'its'}
        
        words = claim.split()
        keywords = [w for w in words if w.lower() not in stopwords and w[0].isalnum()]
        
        # Keep only significant words (avoid single chars, pure punctuation)
        keywords = [w for w in keywords if len(w) > 1 and not all(c in string.punctuation for c in w)]
        
        return ' '.join(keywords[:10])  # keep more specificity for numeric/legal claims

    def _extract_ascii_anchors(self, claim: str) -> str:
        """Extract Latin tokens and numbers as cross-language search anchors."""
        tokens = re.findall(r"[A-Za-z][A-Za-z0-9._/-]*|\d+(?:[.,]\d+)?(?:%|[A-Za-z]+)?", claim)
        if not tokens:
            return ""
        # Keep order and uniqueness.
        out: List[str] = []
        seen = set()
        for tok in tokens:
            t = tok.strip()
            if len(t) < 2:
                continue
            key = t.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(t)
        return " ".join(out[:10])

    def _is_english_like(self, claim: str) -> bool:
        """Heuristic: mostly ASCII letters/spaces means English-like text."""
        if not claim:
            return True
        letters = [c for c in claim if c.isalpha()]
        if not letters:
            return True
        ascii_letters = sum(1 for c in letters if ord(c) < 128)
        return (ascii_letters / max(1, len(letters))) >= 0.8
    
    def _to_question(self, claim: str) -> str:
        """Convert claim to question form for search."""
        # If already a question, return as-is
        if claim.endswith('?'):
            return claim

        base = claim.rstrip().rstrip(".!;:")
        if not base:
            return claim

        # Add "is it true that" prefix
        return f"Is it true that {base}?"

    def _translate_indic_to_english(self, text: str, language: str) -> str:
        """Best-effort Indic->English translation for retrieval fallback."""
        lang = (language or "").lower()
        if lang in {"", "en"}:
            return ""
        prefer_llm = os.getenv("MULTI_QUERY_TRANSLATION_PREFER_LLM", "1").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not self._ensure_indic_to_en_translator():
            if prefer_llm:
                # Requested order: web -> chat-completions -> Sarvam Translate API.
                llm_provider = str(
                    os.getenv(
                        "TRANSLATION_LLM_PROVIDER",
                        os.getenv("LLM_VERIFIER_PROVIDER_EN", os.getenv("LLM_VERIFIER_PROVIDER", "openai")),
                    )
                ).strip().lower()
                web = self._translate_indic_to_english_web(text)
                if web:
                    return web
                llm = self._translate_indic_to_english_llm(text)
                if llm:
                    return llm
                if llm_provider == "sarvam":
                    api_out = self._translate_indic_to_english_sarvam_api(text, lang)
                    if api_out:
                        return api_out
                return ""
            web = self._translate_indic_to_english_web(text)
            if web:
                return web
            return self._translate_indic_to_english_llm(text)

        try:
            pre = self.__class__._i2e_ip.preprocess_batch(
                [text],
                src_lang=self._lang_to_indictrans_tag(lang),
                tgt_lang="eng_Latn",
            )
            toks = self.__class__._i2e_tokenizer(
                pre,
                truncation=True,
                padding="longest",
                return_tensors="pt",
                max_length=int(os.getenv("MULTI_QUERY_TRANSLATION_MAX_LENGTH", "256")),
            )
            toks = {k: v.to(self.__class__._i2e_device) for k, v in toks.items()}

            import torch  # local import keeps base path light if feature disabled

            with torch.no_grad():
                gen = self.__class__._i2e_model.generate(
                    **toks,
                    num_beams=3,
                    max_length=int(os.getenv("MULTI_QUERY_TRANSLATION_MAX_LENGTH", "256")),
                    min_length=0,
                )
            dec = self.__class__._i2e_tokenizer.batch_decode(
                gen, skip_special_tokens=True, clean_up_tokenization_spaces=True
            )
            out = self.__class__._i2e_ip.postprocess_batch(dec, lang="eng_Latn")
            translated = (out[0] if out else "").strip()
            if self._is_english_like(translated):
                return translated
            web = self._translate_indic_to_english_web(text)
            if web:
                return web
            return translated
        except Exception as exc:
            logger.warning("Indic->EN query translation failed; skipping fallback: %s", exc)
            if prefer_llm:
                llm_provider = str(
                    os.getenv(
                        "TRANSLATION_LLM_PROVIDER",
                        os.getenv("LLM_VERIFIER_PROVIDER_EN", os.getenv("LLM_VERIFIER_PROVIDER", "openai")),
                    )
                ).strip().lower()
                web = self._translate_indic_to_english_web(text)
                if web:
                    return web
                llm = self._translate_indic_to_english_llm(text)
                if llm:
                    return llm
                if llm_provider == "sarvam":
                    api_out = self._translate_indic_to_english_sarvam_api(text, lang)
                    if api_out:
                        return api_out
                return ""
            web = self._translate_indic_to_english_web(text)
            if web:
                return web
            return self._translate_indic_to_english_llm(text)

    def _lang_to_sarvam_code(self, lang: str) -> str:
        mapping = {
            "hi": "hi-IN",
            "te": "te-IN",
            "ta": "ta-IN",
            "ml": "ml-IN",
            "kn": "kn-IN",
            "en": "en-IN",
        }
        return mapping.get((lang or "").lower(), "hi-IN")

    def _translate_indic_to_english_sarvam_api(self, text: str, language: str) -> str:
        """
        Dedicated Sarvam text-translate API path.
        Uses chat-LLM translation only as fallback when this path fails.
        """
        if os.getenv("SARVAM_TRANSLATE_ENABLE", "1").strip().lower() not in {"1", "true", "yes", "on"}:
            return ""
        api_key = (
            os.getenv("SARVAM_API_SUBSCRIPTION_KEY", "").strip()
            or os.getenv("SARVAM_API_KEY", "").strip()
        )
        if not api_key:
            return ""

        source_code = self._lang_to_sarvam_code(language)
        endpoint = os.getenv("SARVAM_TRANSLATE_API_URL", "https://api.sarvam.ai/translate").strip()
        payload = {
            "input": text,
            "source_language_code": source_code,
            "target_language_code": os.getenv("SARVAM_TRANSLATE_TARGET_LANGUAGE", "en-IN"),
            "speaker_gender": os.getenv("SARVAM_TRANSLATE_SPEAKER_GENDER", "Male"),
            "mode": os.getenv("SARVAM_TRANSLATE_MODE", "modern-colloquial"),
            "model": os.getenv("SARVAM_TRANSLATE_MODEL", "mayura:v1"),
            "enable_preprocessing": os.getenv("SARVAM_TRANSLATE_ENABLE_PREPROCESSING", "0").strip().lower()
            in {"1", "true", "yes", "on"},
            "numerals_format": os.getenv("SARVAM_TRANSLATE_NUMERALS_FORMAT", "native"),
        }
        headers = {"API-Subscription-Key": api_key, "Content-Type": "application/json"}
        try:
            timeout_s = max(5, int(os.getenv("SARVAM_TRANSLATE_TIMEOUT_SECONDS", "20")))
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json() if resp.content else {}
            # Try common response shapes defensively.
            out = ""
            if isinstance(data, dict):
                out = str(
                    data.get("translated_text")
                    or data.get("translation")
                    or data.get("output")
                    or data.get("text")
                    or ""
                ).strip()
            if out.lower() in {"none", "null"}:
                out = ""
            return out if self._is_english_like(out) else ""
        except Exception as exc:
            logger.warning("Sarvam translate API failed: %s", exc)
            return ""

    def _translate_indic_to_english_web(self, text: str) -> str:
        """
        Non-LLM fallback using public Google translate endpoint.
        Avoids LLM 429 dependency for query translation.
        """
        if os.getenv("MULTI_QUERY_TRANSLATION_WEB_ENABLE", "1").strip().lower() not in {
            "1",
            "true",
            "yes",
            "on",
        }:
            return ""
        try:
            url = "https://translate.googleapis.com/translate_a/single"
            params = {
                "client": "gtx",
                "sl": "auto",
                "tl": "en",
                "dt": "t",
                "q": text,
            }
            timeout_s = max(2, int(os.getenv("MULTI_QUERY_TRANSLATION_WEB_TIMEOUT_SECONDS", "8")))
            resp = requests.get(url, params=params, timeout=timeout_s)
            resp.raise_for_status()
            data = resp.json()
            # Expected: [[["translated","original",...],...], ...]
            chunks = data[0] if isinstance(data, list) and data else []
            out = "".join(
                str(c[0]) for c in chunks
                if isinstance(c, list) and len(c) > 0 and c[0] is not None
            ).strip()
            return out if self._is_english_like(out) else ""
        except Exception as exc:
            logger.warning("Web query translation fallback failed: %s", exc)
            return ""

    def _translate_indic_to_english_llm(self, text: str) -> str:
        """LLM translation fallback for query generation when local model is unavailable."""
        import time
        if time.time() < float(getattr(self.__class__, "_i2e_llm_cooldown_until", 0.0)):
            return ""
        provider = str(
            os.getenv(
                "TRANSLATION_LLM_PROVIDER",
                os.getenv("LLM_VERIFIER_PROVIDER_EN", os.getenv("LLM_VERIFIER_PROVIDER", "openai")),
            )
        ).strip().lower()
        model = str(
            os.getenv(
                "TRANSLATION_LLM_MODEL",
                os.getenv("LLM_VERIFIER_MODEL_EN", os.getenv("LLM_VERIFIER_MODEL", "gpt-4o-mini")),
            )
        ).strip()
        base_url = str(os.getenv("TRANSLATION_LLM_BASE_URL", "")).strip()
        if not base_url:
            if provider == "groq":
                base_url = "https://api.groq.com/openai/v1"
            elif provider == "openrouter":
                base_url = "https://openrouter.ai/api/v1"
            elif provider == "sarvam":
                base_url = "https://api.sarvam.ai/v1"
            else:
                base_url = "https://api.openai.com/v1"

        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY", "").strip()
        elif provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        elif provider == "sarvam":
            api_key = (
                os.getenv("SARVAM_API_SUBSCRIPTION_KEY", "").strip()
                or os.getenv("SARVAM_API_KEY", "").strip()
            )
        else:
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return ""

        base = base_url.rstrip("/")
        endpoint = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
        headers = {"Content-Type": "application/json"}
        if provider == "sarvam":
            sub_key = os.getenv("SARVAM_API_SUBSCRIPTION_KEY", "").strip()
            if sub_key:
                headers["API-Subscription-Key"] = sub_key
            else:
                headers["Authorization"] = f"Bearer {api_key}"
        else:
            headers["Authorization"] = f"Bearer {api_key}"
        base_max_tokens = max(64, int(os.getenv("MULTI_QUERY_TRANSLATION_LLM_MAX_TOKENS", "256")))
        payload = {
            "model": model,
            "temperature": 0,
            "max_tokens": base_max_tokens,
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "Translate to concise English for web search. "
                        "Output ONLY the translated sentence. No reasoning."
                    ),
                },
                {"role": "user", "content": text},
            ],
        }
        if provider == "sarvam":
            payload["stream"] = False
            payload["reasoning_effort"] = os.getenv(
                "TRANSLATION_SARVAM_REASONING_EFFORT",
                os.getenv("SARVAM_REASONING_EFFORT", "medium"),
            )
        retries = max(0, int(os.getenv("MULTI_QUERY_TRANSLATION_LLM_RETRIES", "2")))
        backoff = max(0.1, float(os.getenv("MULTI_QUERY_TRANSLATION_LLM_BACKOFF_SEC", "0.6")))
        for attempt in range(retries + 1):
            try:
                attempt_payload = dict(payload)
                if provider == "sarvam" and attempt > 0:
                    # Sarvam can return reasoning_content with null content.
                    # Retry with lower reasoning effort and tighter completion budget.
                    attempt_payload["reasoning_effort"] = "low"
                    attempt_payload["max_tokens"] = max(64, min(base_max_tokens, 192))
                resp = requests.post(endpoint, headers=headers, json=attempt_payload, timeout=20)
                resp.raise_for_status()
                data = resp.json() if resp.content else {}
                msg = (data.get("choices", [{}])[0].get("message", {}) or {})
                raw_content = msg.get("content", "")
                out = raw_content.strip() if isinstance(raw_content, str) else ""
                if out.lower() in {"none", "null"}:
                    out = ""
                finish_reason = str((data.get("choices", [{}])[0] or {}).get("finish_reason", "")).strip().lower()
                if provider == "sarvam" and not out and finish_reason == "length":
                    if attempt < retries:
                        time.sleep(backoff * (attempt + 1))
                        continue
                    logger.warning("Sarvam translation returned empty content due to finish_reason=length")
                    return ""
                return out if self._is_english_like(out) else ""
            except Exception as exc:
                is_last = attempt >= retries
                if not is_last:
                    time.sleep(backoff * (attempt + 1))
                    continue
                logger.warning("LLM query translation fallback failed: %s", exc)
                # Cool down retries on rate-limit bursts.
                try:
                    msg = str(exc)
                    if "429" in msg or "Too Many Requests" in msg:
                        self.__class__._i2e_llm_cooldown_until = time.time() + float(
                            os.getenv("MULTI_QUERY_TRANSLATION_LLM_COOLDOWN_SEC", "300")
                        )
                except Exception:
                    pass
                return ""

    def _ensure_indic_to_en_translator(self) -> bool:
        cls = self.__class__
        local_enable = os.getenv("MULTI_QUERY_TRANSLATION_LOCAL_ENABLE", "0").strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        if not local_enable:
            cls._i2e_initialized = True
            cls._i2e_enabled = False
            cls._i2e_error = "local translator disabled by env"
            return False
        if cls._i2e_initialized:
            return cls._i2e_enabled

        with cls._i2e_lock:
            if cls._i2e_initialized:
                return cls._i2e_enabled
            try:
                import torch
                from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
                from IndicTransToolkit import IndicProcessor

                model_name = os.getenv(
                    "MULTI_QUERY_TRANSLATION_MODEL",
                    "ai4bharat/indictrans2-indic-en-dist-200M",
                ).strip()
                force_cpu = os.getenv("MULTI_QUERY_TRANSLATION_CPU", "0").strip().lower() in {
                    "1",
                    "true",
                    "yes",
                    "on",
                }
                device = "cpu" if force_cpu else ("cuda" if torch.cuda.is_available() else "cpu")
                dtype = torch.float16 if device == "cuda" else torch.float32

                tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
                model = AutoModelForSeq2SeqLM.from_pretrained(
                    model_name,
                    trust_remote_code=True,
                    torch_dtype=dtype,
                ).to(device)
                model.eval()

                cls._i2e_tokenizer = tokenizer
                cls._i2e_model = model
                cls._i2e_ip = IndicProcessor(inference=True)
                cls._i2e_device = device
                cls._i2e_enabled = True
                cls._i2e_error = ""
                logger.info("Enabled multi query translation fallback with model=%s device=%s", model_name, device)
            except Exception as exc:
                cls._i2e_enabled = False
                cls._i2e_error = str(exc)
                logger.warning("Multi query translation disabled: %s", exc)
            finally:
                cls._i2e_initialized = True

        return cls._i2e_enabled

    def _lang_to_indictrans_tag(self, lang: str) -> str:
        mapping = {
            "hi": "hin_Deva",
            "te": "tel_Telu",
            "ta": "tam_Taml",
            "ml": "mal_Mlym",
            "kn": "kan_Knda",
        }
        return mapping.get((lang or "").lower(), "hin_Deva")
