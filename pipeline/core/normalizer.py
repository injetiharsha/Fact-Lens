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
        lang = (language or "en").strip().lower()
        if lang.startswith("en"):
            return self._rephrase_for_search_en(claim)
        return self._rephrase_for_search_multi(claim, language=lang)

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

        # Cross-script anchor query: keep ASCII entities/numbers for EN-indexed pages.
        anchors = self._extract_ascii_anchors(claim)
        if anchors:
            queries.append(anchors)

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
        return self._dedupe_queries(queries, cap=4)

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
        if not self._ensure_indic_to_en_translator():
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
            web = self._translate_indic_to_english_web(text)
            if web:
                return web
            return self._translate_indic_to_english_llm(text)

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
        provider = str(os.getenv("LLM_VERIFIER_PROVIDER", "openai")).strip().lower()
        model = str(os.getenv("LLM_VERIFIER_MODEL", "gpt-4o-mini")).strip()
        base_url = str(os.getenv("LLM_VERIFIER_BASE_URL", "")).strip()
        if not base_url:
            if provider == "groq":
                base_url = "https://api.groq.com/openai/v1"
            elif provider == "openrouter":
                base_url = "https://openrouter.ai/api/v1"
            else:
                base_url = "https://api.openai.com/v1"

        if provider == "groq":
            api_key = os.getenv("GROQ_API_KEY", "").strip()
        elif provider == "openrouter":
            api_key = os.getenv("OPENROUTER_API_KEY", "").strip()
        else:
            api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            return ""

        endpoint = f"{base_url.rstrip('/')}/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": model,
            "temperature": 0,
            "max_tokens": 256,
            "messages": [
                {
                    "role": "system",
                    "content": "Translate to concise English for web search. Return only translated text.",
                },
                {"role": "user", "content": text},
            ],
        }
        try:
            resp = requests.post(endpoint, headers=headers, json=payload, timeout=20)
            resp.raise_for_status()
            data = resp.json() if resp.content else {}
            out = str(
                (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
            ).strip()
            return out if self._is_english_like(out) else ""
        except Exception as exc:
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
