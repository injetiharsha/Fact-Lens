"""Claim analysis endpoint."""

import logging
import os
import re
from threading import Lock
import requests
from fastapi import APIRouter, UploadFile, File, Form, Body
from api.schemas import (
    ClaimRequest,
    VerdictResponse,
    EvidenceItem,
    AnalysisDetails,
    ImageAnalysisResponse,
    PDFAnalysisResponse,
)
from api.runtime_limits import GlobalRequestLimiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["claims"])
_CLAIM_PIPELINE_CACHE = {}
_CLAIM_PIPELINE_LOCK = Lock()
_ENV_FILE_CACHE: dict | None = None
_SCRIPT_LANG_PATTERNS = [
    ("te", re.compile(r"[\u0C00-\u0C7F]")),  # Telugu
    ("ta", re.compile(r"[\u0B80-\u0BFF]")),  # Tamil
    ("kn", re.compile(r"[\u0C80-\u0CFF]")),  # Kannada
    ("ml", re.compile(r"[\u0D00-\u0D7F]")),  # Malayalam
    ("hi", re.compile(r"[\u0900-\u097F]")),  # Devanagari/Hindi
]

def _env_bool(name: str, default: bool) -> bool:
    val = os.getenv(name)
    if val is None:
        return default
    return val.strip().lower() in {"1", "true", "yes", "on"}


def _load_env_file_cache() -> dict:
    """Best-effort .env parser fallback when process env is not populated."""
    global _ENV_FILE_CACHE
    if _ENV_FILE_CACHE is not None:
        return _ENV_FILE_CACHE

    out: dict = {}
    env_path = os.path.join(os.getcwd(), ".env")
    try:
        if os.path.exists(env_path):
            with open(env_path, "r", encoding="utf-8") as f:
                for raw in f:
                    line = raw.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    k, v = line.split("=", 1)
                    out[k.strip()] = v.strip()
    except Exception:
        out = {}
    _ENV_FILE_CACHE = out
    return out


def _env_or_file(name: str, default: str | None = None) -> str | None:
    val = os.getenv(name)
    if val is not None and str(val).strip() != "":
        return val
    return _load_env_file_cache().get(name, default)


def _env_int(name: str, default: int, minimum: int = 0) -> int:
    raw = os.getenv(name)
    try:
        value = int(raw) if raw is not None else int(default)
    except Exception:
        value = int(default)
    return max(minimum, value)


def _select_checkpoint(component: str, language: str) -> str | None:
    lang_is_en = language.lower().startswith("en")
    upper = component.upper()

    en_enabled = _env_bool(f"ENABLE_{upper}_EN", True)
    multi_enabled = _env_bool(f"ENABLE_{upper}_MULTI", True)
    en_path = os.getenv(f"{upper}_EN_PATH", f"checkpoints/{component}/en")
    multi_path = os.getenv(f"{upper}_MULTI_PATH", f"checkpoints/{component}/multi")

    if lang_is_en and en_enabled:
        return en_path
    if (not lang_is_en) and multi_enabled:
        return multi_path
    if en_enabled:
        return en_path
    if multi_enabled:
        return multi_path
    return None


def _llm_provider_for_language(language: str) -> str:
    lang_is_en = str(language or "en").lower().startswith("en")
    if lang_is_en:
        return str(
            _env_or_file(
                "LLM_VERIFIER_PROVIDER_EN",
                _env_or_file("LLM_VERIFIER_PROVIDER", "openai"),
            )
        ).strip().lower()
    return str(
        _env_or_file(
            "LLM_VERIFIER_PROVIDER_MULTI",
            _env_or_file("LLM_VERIFIER_PROVIDER", "openai"),
        )
    ).strip().lower()


def _llm_model_for_language(language: str) -> str:
    lang_is_en = str(language or "en").lower().startswith("en")
    if lang_is_en:
        return str(
            _env_or_file(
                "LLM_VERIFIER_MODEL_EN",
                _env_or_file("LLM_VERIFIER_MODEL", "gpt-4o-mini"),
            )
        ).strip()
    return str(
        _env_or_file(
            "LLM_VERIFIER_MODEL_MULTI",
            _env_or_file("LLM_VERIFIER_MODEL", "gpt-4o-mini"),
        )
    ).strip()


def _pipeline_config(language: str, mode: str = "claim") -> dict:
    lang_is_en = str(language or "en").lower().startswith("en")
    mode_l = str(mode or "claim").strip().lower()
    doc_min_words_default = 15
    doc_max_words_default = 80
    doc_checkability_default = True
    min_words_key = "DOCUMENT_MIN_WORDS"
    max_words_key = "DOCUMENT_MAX_WORDS"
    checkability_key = "ENABLE_DOCUMENT_CHECKABILITY_FILTER"
    if mode_l == "image":
        # Keep image OCR claim extraction looser than PDF/doc text extraction.
        doc_min_words_default = 8
        doc_max_words_default = 100
        # Image OCR is noisy; keep sentence-level checkability on by default
        # to drop background/context-only fragments before verification.
        doc_checkability_default = True
        min_words_key = "IMAGE_DOCUMENT_MIN_WORDS"
        max_words_key = "IMAGE_DOCUMENT_MAX_WORDS"
        checkability_key = "IMAGE_ENABLE_DOCUMENT_CHECKABILITY_FILTER"
    elif mode_l == "pdf":
        min_words_key = "PDF_DOCUMENT_MIN_WORDS"
        max_words_key = "PDF_DOCUMENT_MAX_WORDS"
        checkability_key = "PDF_ENABLE_DOCUMENT_CHECKABILITY_FILTER"

    return {
        "claim_checkability_checkpoint": _select_checkpoint("checkability", language),
        "context_checkpoint": _select_checkpoint("context", language),
        "relevance_checkpoint": _select_checkpoint("relevance", language),
        "max_evidence": (
            _env_int("PIPELINE_MAX_EVIDENCE_EN", _env_int("PIPELINE_MAX_EVIDENCE", 5, minimum=1), minimum=1)
            if lang_is_en
            else _env_int("PIPELINE_MAX_EVIDENCE_MULTI", _env_int("PIPELINE_MAX_EVIDENCE", 5, minimum=1), minimum=1)
        ),
        "enable_two_stage_relevance": _env_bool("ENABLE_TWO_STAGE_RELEVANCE", True),
        "relevance_bi_encoder_model": os.getenv(
            "RELEVANCE_BI_ENCODER_MODEL", "intfloat/multilingual-e5-small"
        ),
        "relevance_shortlist_k": int(os.getenv("RELEVANCE_SHORTLIST_K", "20")),
        "relevance_top_k": (
            int(os.getenv("RELEVANCE_TOP_K", "0")) if int(os.getenv("RELEVANCE_TOP_K", "0")) > 0 else None
        ),
        "relevance_drop_threshold": float(
            os.getenv(
                "RELEVANCE_KEEP_THRESHOLD",
                os.getenv("RELEVANCE_DROP_THRESHOLD", "0.30"),
            )
        ),
        "stance_checkpoint": _select_checkpoint("stance", language),
        "enable_checkability_stage": _env_bool("ENABLE_CHECKABILITY_STAGE", True),
        "enable_document_checkability_filter": _env_bool(
            checkability_key,
            _env_bool("ENABLE_DOCUMENT_CHECKABILITY_FILTER", doc_checkability_default),
        ),
        "document_min_words": _env_int(
            min_words_key,
            _env_int("DOCUMENT_MIN_WORDS", doc_min_words_default, minimum=3),
            minimum=3,
        ),
        "document_max_words": _env_int(
            max_words_key,
            _env_int("DOCUMENT_MAX_WORDS", doc_max_words_default, minimum=10),
            minimum=10,
        ),
        "enable_llm_verifier": _env_bool("ENABLE_LLM_VERIFIER", True),
        "llm_provider": _llm_provider_for_language(language),
        "llm_model": _llm_model_for_language(language),
        "llm_neutral_only": _env_bool("LLM_VERIFIER_NEUTRAL_ONLY", True),
        "llm_conf_threshold": float(os.getenv("LLM_VERIFIER_CONF_THRESHOLD", "0.55")),
    }


REQUEST_LIMITER = GlobalRequestLimiter(
    requests_per_minute=int(os.getenv("GLOBAL_REQUESTS_PER_MINUTE", "30")),
    max_concurrent=int(os.getenv("MAX_CONCURRENT_REQUESTS", "4")),
)


def _pipeline_bucket(language: str) -> str:
    return "en" if language.lower().startswith("en") else "multi"


def _auto_detect_language(text: str, requested_language: str | None) -> str:
    req = (requested_language or "").strip().lower()
    if req and req != "auto" and req != "en":
        return req

    snippet = text or ""
    for lang_code, pattern in _SCRIPT_LANG_PATTERNS:
        if pattern.search(snippet):
            return lang_code
    return "en"


def _looks_multi_claim_text(text: str) -> bool:
    """Heuristic: detect if OCR text likely contains multiple independent claims."""
    sample = (text or "").strip()
    if not sample:
        return False

    # Common separators from OCR/social cards.
    chunks = re.split(r"[\n\r]+|[.!?।]+|\u2022|-{2,}", sample)
    candidates = []
    for ch in chunks:
        s = ch.strip()
        if len(s) < 20:
            continue
        if len(s.split()) < 5:
            continue
        candidates.append(s)
    return len(candidates) >= 2


def _llm_translate_to_english(text: str) -> str:
    """Fallback translator via configured LLM verifier provider."""
    provider = str(
        _env_or_file(
            "TRANSLATION_LLM_PROVIDER",
            _env_or_file("LLM_VERIFIER_PROVIDER_EN", _env_or_file("LLM_VERIFIER_PROVIDER", "openai")),
        )
    ).strip().lower()
    model = str(
        _env_or_file(
            "TRANSLATION_LLM_MODEL",
            _env_or_file("LLM_VERIFIER_MODEL_EN", _env_or_file("LLM_VERIFIER_MODEL", "gpt-4o-mini")),
        )
    ).strip()
    base_url = str(_env_or_file("TRANSLATION_LLM_BASE_URL", "")).strip()
    if not base_url:
        if provider == "groq":
            base_url = "https://api.groq.com/openai/v1"
        elif provider == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
        elif provider == "sarvam":
            base_url = "https://api.sarvam.ai/v1"
        else:
            base_url = "https://api.openai.com/v1"

    api_key = None
    if provider == "groq":
        api_key = _env_or_file("GROQ_API_KEY")
    elif provider == "openrouter":
        api_key = _env_or_file("OPENROUTER_API_KEY")
    elif provider == "sarvam":
        api_key = _env_or_file("SARVAM_API_SUBSCRIPTION_KEY") or _env_or_file("SARVAM_API_KEY")
    else:
        api_key = _env_or_file("OPENAI_API_KEY")

    if not api_key:
        return ""

    base = base_url.rstrip("/")
    endpoint = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if provider == "sarvam":
        sub_key = str(_env_or_file("SARVAM_API_SUBSCRIPTION_KEY", "") or "").strip()
        if sub_key:
            headers["API-Subscription-Key"] = sub_key
        else:
            headers["Authorization"] = f"Bearer {api_key}"
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 512,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Translate the user text to clear English. "
                    "Return only translated text, no labels, no quotes."
                ),
            },
            {"role": "user", "content": text},
        ],
    }
    if provider == "sarvam":
        payload["stream"] = False
        payload["reasoning_effort"] = _env_or_file("SARVAM_REASONING_EFFORT", "medium")
    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=25)
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
        out = str(
            (data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")
        ).strip()
        return out
    except Exception as exc:
        logger.warning("LLM translation fallback failed: %s", exc)
        return ""


def _translate_text_to_en(text: str, source_language: str | None = None) -> tuple[str, bool]:
    """Translate arbitrary text to English with model + LLM fallback."""
    value = str(text or "").strip()
    if not value:
        return "", False
    src = source_language or _auto_detect_language(value, "auto")
    if src == "en":
        return value, True

    translated = value
    try:
        from pipeline.core.normalizer import ClaimNormalizer

        normalizer = ClaimNormalizer()
        translated_out = normalizer._translate_indic_to_english(  # pylint: disable=protected-access
            text=value,
            language=src,
        )
        if translated_out:
            translated = translated_out
    except Exception as exc:
        logger.warning("Preview translation failed; returning original text: %s", exc)

    if translated == value:
        llm_translated = _llm_translate_to_english(value)
        if llm_translated:
            translated = llm_translated

    return translated, bool(translated != value or src == "en")


def _llm_pick_best_claim(candidates: list[str], language: str = "en") -> str:
    """Pick the most verifiable claim from candidate list using LLM; fallback empty on any failure."""
    cands = [str(c or "").strip() for c in (candidates or []) if str(c or "").strip()]
    if len(cands) < 2:
        return cands[0] if cands else ""

    provider = _llm_provider_for_language(language)
    model = _llm_model_for_language(language)
    base_url = str(_env_or_file("LLM_VERIFIER_BASE_URL", "")).strip()
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
        api_key = _env_or_file("GROQ_API_KEY")
    elif provider == "openrouter":
        api_key = _env_or_file("OPENROUTER_API_KEY")
    elif provider == "sarvam":
        api_key = _env_or_file("SARVAM_API_SUBSCRIPTION_KEY") or _env_or_file("SARVAM_API_KEY")
    else:
        api_key = _env_or_file("OPENAI_API_KEY")
    if not api_key:
        return ""

    prompt_rows = "\n".join([f"{i+1}. {c}" for i, c in enumerate(cands[:8])])
    base = base_url.rstrip("/")
    endpoint = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if provider == "sarvam":
        sub_key = str(_env_or_file("SARVAM_API_SUBSCRIPTION_KEY", "") or "").strip()
        if sub_key:
            headers["API-Subscription-Key"] = sub_key
        else:
            headers["Authorization"] = f"Bearer {api_key}"
    else:
        headers["Authorization"] = f"Bearer {api_key}"
    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 16,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Pick one best verifiable factual claim from numbered candidates. "
                    "Return only the number."
                ),
            },
            {"role": "user", "content": f"Language={language}\nCandidates:\n{prompt_rows}"},
        ],
    }
    if provider == "sarvam":
        payload["stream"] = False
        payload["reasoning_effort"] = _env_or_file("SARVAM_REASONING_EFFORT", "medium")
    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=12)
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
        out = str((data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")).strip()
        m = re.search(r"\b(\d+)\b", out)
        if not m:
            return ""
        idx = int(m.group(1)) - 1
        if 0 <= idx < len(cands[:8]):
            return cands[idx]
    except Exception as exc:
        logger.warning("LLM claim selection fallback failed: %s", exc)
    return ""


def _llm_summarize_claim_text(text: str, language: str = "en") -> str:
    """Rewrite a long/noisy OCR claim into one concise factual claim."""
    src = str(text or "").strip()
    if not src:
        return ""

    provider = _llm_provider_for_language(language)
    model = _llm_model_for_language(language)
    base_url = str(_env_or_file("LLM_VERIFIER_BASE_URL", "")).strip()
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
        api_key = _env_or_file("GROQ_API_KEY")
    elif provider == "openrouter":
        api_key = _env_or_file("OPENROUTER_API_KEY")
    elif provider == "sarvam":
        api_key = _env_or_file("SARVAM_API_SUBSCRIPTION_KEY") or _env_or_file("SARVAM_API_KEY")
    else:
        api_key = _env_or_file("OPENAI_API_KEY")
    if not api_key:
        return ""

    base = base_url.rstrip("/")
    endpoint = base if base.endswith("/chat/completions") else f"{base}/chat/completions"
    headers = {"Content-Type": "application/json"}
    if provider == "sarvam":
        sub_key = str(_env_or_file("SARVAM_API_SUBSCRIPTION_KEY", "") or "").strip()
        if sub_key:
            headers["API-Subscription-Key"] = sub_key
        else:
            headers["Authorization"] = f"Bearer {api_key}"
    else:
        headers["Authorization"] = f"Bearer {api_key}"

    payload = {
        "model": model,
        "temperature": 0,
        "max_tokens": 120,
        "messages": [
            {
                "role": "system",
                "content": (
                    "Rewrite OCR text as one factual claim in the same language. "
                    "Remove duplicate or rhetorical lines. Keep entities, numbers, dates unchanged. "
                    "Return only one sentence claim text."
                ),
            },
            {"role": "user", "content": f"Language={language}\nText:\n{src}"},
        ],
    }
    if provider == "sarvam":
        payload["stream"] = False
        payload["reasoning_effort"] = _env_or_file("SARVAM_REASONING_EFFORT", "medium")
    try:
        resp = requests.post(endpoint, headers=headers, json=payload, timeout=16)
        resp.raise_for_status()
        data = resp.json() if resp.content else {}
        out = str((data.get("choices", [{}])[0].get("message", {}) or {}).get("content", "")).strip()
        out = re.sub(r"^\s*['\"`]+|['\"`]+\s*$", "", out).strip()
        out = re.sub(r"\s+", " ", out)
        # Safety: reject very short/empty rewrites.
        if len(out) < 16:
            return ""
        return out
    except Exception as exc:
        logger.warning("LLM claim summarization fallback failed: %s", exc)
        return ""


def _maybe_summarize_image_claim(claim: str, language: str = "en") -> tuple[str, bool]:
    """Summarize long OCR-derived claim when enabled and above threshold."""
    src = str(claim or "").strip()
    if not src:
        return src, False
    if not _env_bool("IMAGE_LLM_SUMMARIZATION_ENABLE", True):
        return src, False
    min_chars = _env_int("IMAGE_LLM_SUMMARIZE_MIN_CHARS", 180, minimum=80)
    if len(src) < min_chars:
        return src, False
    summarized = _llm_summarize_claim_text(src, language=language)
    if not summarized:
        return src, False
    return summarized, summarized != src


def _summarize_image_candidates(candidates: list[str], language: str = "en", max_items: int = 2) -> tuple[list[str], int]:
    """Optionally summarize long OCR candidates before downstream verification."""
    rows = [str(c or "").strip() for c in (candidates or []) if str(c or "").strip()]
    if not rows:
        return [], 0
    if not _env_bool("IMAGE_LLM_SUMMARIZE_CANDIDATES_ENABLE", True):
        return rows[: max(1, int(max_items))], 0

    min_chars = _env_int("IMAGE_LLM_SUMMARIZE_CANDIDATE_MIN_CHARS", 140, minimum=80)
    out: list[str] = []
    applied = 0
    for cand in rows[: max(1, int(max_items))]:
        item = cand
        if len(cand) >= min_chars:
            summarized, changed = _maybe_summarize_image_claim(cand, language=language)
            if changed and summarized:
                item = summarized
                applied += 1
        out.append(item)
    out = _dedupe_and_limit_claims(out, max_items=max_items)
    return out, applied


def _dedupe_and_limit_claims(candidates: list[str], max_items: int = 2) -> list[str]:
    """Keep a small set of distinct claim candidates.

    Dedup policy:
    - rank by informativeness first
    - drop a candidate if >=60% overlap with an already-kept claim
    - if nothing survives except overlap variants, keep exactly one best claim
    """
    from difflib import SequenceMatcher
    import re as _re

    rows = [str(c or "").strip() for c in (candidates or []) if str(c or "").strip()]
    if not rows:
        return []

    limit = max(1, int(max_items))
    overlap_threshold = 0.60

    def _norm(s: str) -> str:
        s = s.lower().strip()
        s = _re.sub(r"\s+", " ", s)
        return s

    def _tokens(s: str) -> set[str]:
        # Keep unicode word chars so Indic scripts are retained.
        return set(_re.findall(r"\w+", s.lower(), flags=_re.UNICODE))

    def _informativeness_score(s: str) -> tuple[int, int]:
        toks = _tokens(s)
        return (len(toks), len(s))

    def _overlap(a: str, b: str) -> float:
        an, bn = _norm(a), _norm(b)
        if not an or not bn:
            return 0.0

        at, bt = _tokens(an), _tokens(bn)
        tok_containment = 0.0
        if at and bt:
            tok_containment = max(
                len(at & bt) / max(1, len(at)),
                len(at & bt) / max(1, len(bt)),
            )

        # Sequence similarity catches OCR/window duplicates with small token drift.
        seq = SequenceMatcher(a=an, b=bn).ratio()

        # Substring containment in normalized strings.
        sub = 0.0
        if an in bn or bn in an:
            short = min(len(an), len(bn))
            long = max(len(an), len(bn))
            sub = (short / long) if long > 0 else 0.0

        return max(tok_containment, seq, sub)

    ranked = sorted(rows, key=_informativeness_score, reverse=True)

    kept: list[str] = []
    for cand in ranked:
        if any(_overlap(cand, k) >= overlap_threshold for k in kept):
            continue
        kept.append(cand)
        if len(kept) >= limit:
            break

    if kept:
        return kept
    return [ranked[0]]


def get_claim_pipeline(language: str):
    from pipeline.claim_pipeline import ClaimPipeline

    bucket = _pipeline_bucket(language)
    cached = _CLAIM_PIPELINE_CACHE.get(bucket)
    if cached is not None:
        return cached

    with _CLAIM_PIPELINE_LOCK:
        cached = _CLAIM_PIPELINE_CACHE.get(bucket)
        if cached is not None:
            return cached
        pipeline = ClaimPipeline(_pipeline_config(language))
        _CLAIM_PIPELINE_CACHE[bucket] = pipeline
        logger.info("Loaded claim pipeline cache bucket=%s", bucket)
        return pipeline


def preload_claim_pipelines() -> None:
    if not _env_bool("PRELOAD_PIPELINES_ON_STARTUP", True):
        logger.info("Pipeline preload skipped (PRELOAD_PIPELINES_ON_STARTUP=0).")
        return

    for language in ("en", "hi"):
        try:
            get_claim_pipeline(language)
        except Exception as exc:
            logger.exception("Failed to preload claim pipeline for %s: %s", language, exc)


@router.post("/analyze", response_model=VerdictResponse)
async def analyze_claim(request: ClaimRequest):
    """Analyze a text claim and return verdict."""
    await REQUEST_LIMITER.acquire()
    try:
        effective_language = _auto_detect_language(request.claim, request.language)
        pipeline = get_claim_pipeline(effective_language)

        # Run analysis
        result = pipeline.analyze(
            claim=request.claim,
            language=effective_language,
            recency_mode=str(request.recency_mode or "general"),
            recency_start=str(request.recency_start or ""),
            recency_end=str(request.recency_end or ""),
        )

        return VerdictResponse(
            verdict=result.verdict,
            confidence=result.confidence,
            evidence=[EvidenceItem(**ev) for ev in result.evidence],
            reasoning=result.reasoning,
            details=AnalysisDetails(**result.details)
        )
    finally:
        REQUEST_LIMITER.release()


@router.post("/analyze-image", response_model=ImageAnalysisResponse)
async def analyze_image(
    image: UploadFile = File(...),
    claim: str = Form(""),
    language: str = Form("auto"),
    recency_mode: str = Form("general"),
    recency_start: str = Form(""),
    recency_end: str = Form(""),
):
    """Analyze an image claim (quality check -> OCR -> pipeline)."""
    import tempfile
    from pipeline.ingestion.image import ImageInputPipeline
    from pipeline.document_pipeline import DocumentPipeline
    
    await REQUEST_LIMITER.acquire()
    try:
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            content = await image.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            image_pipeline = ImageInputPipeline()
            image_result = image_pipeline.process(
                image_path=tmp_path,
                claim_text=claim,
                language=language,
            )
            extracted_text = image_result.ocr_text
            claim_text = image_result.claim_text
            effective_language = _auto_detect_language(claim_text, language)
            explicit_claim_provided = bool(str(claim or "").strip())
            force_single_when_claim = _env_bool(
                "IMAGE_FORCE_SINGLE_CLAIM_WHEN_CLAIM_PROVIDED",
                True,
            )
            doc_pipeline = DocumentPipeline(_pipeline_config(effective_language, mode="image"))

            if not claim_text:
                return ImageAnalysisResponse(
                    mode="error",
                    verdict="neutral",
                    confidence=0.0,
                    error="No text found in image. Please provide clearer image or enter claim text.",
                    ocr_text=extracted_text,
                    ocr_engine=image_result.ocr_engine,
                    ocr_confidence=image_result.ocr_confidence,
                    image_quality=image_result.image_quality,
                    warnings=image_result.warnings,
                )

            def _run_single_claim_mode(selected_claim: str, extra_warning: str | None = None) -> ImageAnalysisResponse:
                claim_for_eval = (selected_claim or "").strip() or claim_text
                summarized_claim, summarized = _maybe_summarize_image_claim(
                    claim_for_eval,
                    language=effective_language,
                )
                if summarized:
                    claim_for_eval = summarized_claim
                    image_result.warnings.append("WARN: LLM claim summarization applied for long OCR text.")
                if extra_warning:
                    image_result.warnings.append(extra_warning)
                if claim_for_eval != claim_text:
                    image_result.warnings.append(
                        "WARN: Auto-selected a checkable mini-claim from OCR text."
                    )

                pipeline = get_claim_pipeline(effective_language)
                result = pipeline.analyze(
                    claim=claim_for_eval,
                    language=effective_language,
                    image_path=tmp_path,
                    recency_mode=str(recency_mode or "general"),
                    recency_start=str(recency_start or ""),
                    recency_end=str(recency_end or ""),
                )

                return ImageAnalysisResponse(
                    mode="single_claim",
                    verdict=result.verdict,
                    confidence=result.confidence,
                    evidence=result.evidence,
                    reasoning=result.reasoning,
                    details=result.details,
                    ocr_text=extracted_text,
                    ocr_engine=image_result.ocr_engine,
                    ocr_confidence=image_result.ocr_confidence,
                    image_quality=image_result.image_quality,
                    warnings=image_result.warnings,
                )

            if (not (explicit_claim_provided and force_single_when_claim)) and _looks_multi_claim_text(claim_text):
                ranked_candidates = doc_pipeline.rank_claim_candidates(
                    claim_text,
                    language=effective_language,
                )
                ranked_candidates = _dedupe_and_limit_claims(ranked_candidates, max_items=2)
                ranked_candidates, doc_sum_applied = _summarize_image_candidates(
                    ranked_candidates,
                    language=effective_language,
                    max_items=2,
                )
                if doc_sum_applied > 0:
                    image_result.warnings.append(
                        f"WARN: LLM claim summarization applied on {doc_sum_applied} image candidates."
                    )
                if not ranked_candidates:
                    return _run_single_claim_mode(
                        claim_text,
                        extra_warning="WARN: No claim picked from multi-claim OCR; forced single-claim fallback.",
                    )
                doc_claim_rows = []
                for cand in ranked_candidates:
                    out = get_claim_pipeline(effective_language).analyze(
                        claim=cand,
                        language=effective_language,
                        image_path=tmp_path,
                        recency_mode=str(recency_mode or "general"),
                        recency_start=str(recency_start or ""),
                        recency_end=str(recency_end or ""),
                    )
                    doc_claim_rows.append(
                        {
                            "claim": cand,
                            "verdict": out.verdict,
                            "confidence": out.confidence,
                            "evidence_count": len(out.evidence),
                        }
                    )
                if doc_claim_rows:
                    best_doc = max(doc_claim_rows, key=lambda row: float(row.get("confidence", 0.0)))
                    from pipeline.document_pipeline import DocumentResult

                    doc_result = DocumentResult(
                        claims=doc_claim_rows,
                        summary_verdict=str(best_doc.get("verdict", "neutral")),
                        summary_confidence=float(best_doc.get("confidence", 0.0) or 0.0),
                    )
                else:
                    doc_result = doc_pipeline.analyze_text(text=claim_text, language=effective_language)
                best_claim = None
                if doc_result.claims:
                    best_claim = max(
                        doc_result.claims,
                        key=lambda row: float(row.get("confidence", 0.0)),
                    )

                # Frontend-friendly payload:
                # provide summary + full single-claim style details from best claim.
                best_evidence = []
                best_reasoning = None
                best_details = None
                best_verdict = doc_result.summary_verdict
                best_confidence = doc_result.summary_confidence
                if best_claim and best_claim.get("claim"):
                    best_result = get_claim_pipeline(effective_language).analyze(
                        claim=str(best_claim.get("claim")),
                        language=effective_language,
                        image_path=tmp_path,
                        recency_mode=str(recency_mode or "general"),
                        recency_start=str(recency_start or ""),
                        recency_end=str(recency_end or ""),
                    )
                    best_evidence = best_result.evidence
                    best_reasoning = best_result.reasoning
                    best_details = best_result.details
                    best_verdict = best_result.verdict
                    best_confidence = best_result.confidence
                elif not best_claim:
                    return _run_single_claim_mode(
                        claim_text,
                        extra_warning=(
                            "WARN: No strong mini-claim extracted from OCR; forced single-claim fallback."
                        ),
                    )

                return ImageAnalysisResponse(
                    mode="document",
                    summary_verdict=doc_result.summary_verdict,
                    summary_confidence=doc_result.summary_confidence,
                    summary_claim=(best_claim or {}).get("claim"),
                    # Backward-compat aliases for existing UI flow.
                    verdict=best_verdict,
                    confidence=best_confidence,
                    evidence=best_evidence,
                    reasoning=best_reasoning,
                    details=best_details,
                    claims=doc_result.claims,
                    ocr_text=extracted_text,
                    ocr_engine=image_result.ocr_engine,
                    ocr_confidence=image_result.ocr_confidence,
                    image_quality=image_result.image_quality,
                    warnings=image_result.warnings,
                )

            # Single-claim path
            ranked_candidates = doc_pipeline.rank_claim_candidates(
                claim_text,
                language=effective_language,
            )
            ranked_candidates = _dedupe_and_limit_claims(ranked_candidates, max_items=2)
            selected_claim = (ranked_candidates[0] if ranked_candidates else "") or claim_text
            llm_claim_pick_enabled = _env_bool("IMAGE_LLM_CLAIM_SELECTION_ENABLE", False)
            if llm_claim_pick_enabled and len(ranked_candidates) > 1:
                llm_pick = _llm_pick_best_claim(
                    candidates=ranked_candidates[:6],
                    language=effective_language,
                )
                if llm_pick:
                    selected_claim = llm_pick
                    image_result.warnings.append("WARN: LLM-assisted claim selection applied for image OCR.")
            return _run_single_claim_mode(selected_claim)

        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    finally:
        REQUEST_LIMITER.release()


@router.post("/extract-ocr-preview")
async def extract_ocr_preview(
    image: UploadFile = File(...),
    claim: str = Form(""),
    language: str = Form("auto"),
):
    """Run OCR only and return extracted claim preview for image mode UI."""
    import tempfile
    from pipeline.ingestion.image import ImageInputPipeline

    await REQUEST_LIMITER.acquire()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            content = await image.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            image_pipeline = ImageInputPipeline()
            image_result = image_pipeline.process(
                image_path=tmp_path,
                claim_text=claim,
                language=language,
            )
            claim_text_out = image_result.claim_text or ""
            effective_language = _auto_detect_language(claim_text_out, language)
            return {
                "ocr_text": image_result.ocr_text,
                "claim_text": claim_text_out,
                "ocr_engine": image_result.ocr_engine,
                "ocr_confidence": image_result.ocr_confidence,
                "image_quality": image_result.image_quality,
                "warnings": image_result.warnings,
                "effective_language": effective_language,
                "multi_claim_candidate": _looks_multi_claim_text(claim_text_out),
            }
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    finally:
        REQUEST_LIMITER.release()


@router.post("/extract-pdf-preview")
async def extract_pdf_preview(
    pdf: UploadFile = File(...),
    claim: str = Form(""),
    language: str = Form("auto"),
    page_spec: str = Form(""),
):
    """Extract PDF text preview only for frontend claim preview."""
    import tempfile
    from pipeline.ingestion.pdf import PDFInputPipeline

    await REQUEST_LIMITER.acquire()
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            content = await pdf.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            pdf_pipeline = PDFInputPipeline()
            pdf_result = pdf_pipeline.process(pdf_path=tmp_path, claim_text=claim, page_spec=page_spec)
            claim_text_out = pdf_result.claim_text or ""
            effective_language = _auto_detect_language(claim_text_out, language)
            return {
                "pdf_text": pdf_result.extracted_text,
                "claim_text": claim_text_out,
                "page_count": pdf_result.page_count,
                "selected_pages": pdf_result.selected_pages,
                "selected_page_spec": pdf_result.selected_page_spec,
                "extraction_engine": pdf_result.extraction_engine,
                "warnings": pdf_result.warnings,
                "effective_language": effective_language,
                "multi_claim_candidate": _looks_multi_claim_text(claim_text_out),
            }
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    finally:
        REQUEST_LIMITER.release()


@router.post("/translate-preview")
async def translate_preview(payload: dict = Body(...)):
    """Translate preview text to English for frontend display."""
    text = str(payload.get("text") or "").strip()
    target_language = str(payload.get("target_language") or "en").strip().lower()
    if not text:
        return {
            "translated_text": "",
            "source_language": "unknown",
            "target_language": target_language,
            "translated": False,
            "ok": False,
        }

    source_language = _auto_detect_language(text, "auto")
    if target_language != "en":
        return {
            "translated_text": text,
            "source_language": source_language,
            "target_language": target_language,
            "translated": False,
            "ok": True,
        }

    translated, translated_ok = _translate_text_to_en(text=text, source_language=source_language)

    return {
        "translated_text": translated,
        "source_language": source_language,
        "target_language": target_language,
        "translated": translated_ok,
        "ok": True,
    }


@router.post("/translate-batch")
async def translate_batch(payload: dict = Body(...)):
    """Translate a list of text snippets to English."""
    rows = payload.get("texts") or []
    target_language = str(payload.get("target_language") or "en").strip().lower()
    if not isinstance(rows, list):
        rows = []
    if target_language != "en":
        return {
            "translated_texts": [str(x or "") for x in rows],
            "target_language": target_language,
            "translated": False,
            "ok": True,
        }

    out: list[str] = []
    any_translated = False
    for item in rows:
        src = str(item or "").strip()
        if not src:
            out.append("")
            continue
        translated, ok = _translate_text_to_en(src, source_language=_auto_detect_language(src, "auto"))
        out.append(translated)
        any_translated = any_translated or ok

    return {
        "translated_texts": out,
        "target_language": target_language,
        "translated": any_translated,
        "ok": True,
    }


@router.post("/analyze-document")
async def analyze_document(
    text: str = Form(...),
    language: str = Form("en"),
):
    """Analyze document text by extracting and evaluating multiple claim candidates."""
    from pipeline.document_pipeline import DocumentPipeline

    await REQUEST_LIMITER.acquire()
    try:
        pipeline = DocumentPipeline(_pipeline_config(language, mode="document"))
        result = pipeline.analyze_text(text=text, language=language)
        return {
            "summary_verdict": result.summary_verdict,
            "summary_confidence": result.summary_confidence,
            "claims": result.claims,
        }
    finally:
        REQUEST_LIMITER.release()


@router.post("/analyze-pdf", response_model=PDFAnalysisResponse)
async def analyze_pdf(
    pdf: UploadFile = File(...),
    claim: str = Form(""),
    language: str = Form("auto"),
    page_spec: str = Form(""),
    recency_mode: str = Form("general"),
    recency_start: str = Form(""),
    recency_end: str = Form(""),
):
    """Analyze a PDF by extracting text and running claim/document pipeline."""
    import tempfile
    from pipeline.ingestion.pdf import PDFInputPipeline
    from pipeline.document_pipeline import DocumentPipeline

    await REQUEST_LIMITER.acquire()
    try:
        suffix = ".pdf"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            content = await pdf.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            pdf_pipeline = PDFInputPipeline()
            pdf_result = pdf_pipeline.process(pdf_path=tmp_path, claim_text=claim, page_spec=page_spec)

            extracted_text = pdf_result.extracted_text
            claim_text = pdf_result.claim_text
            effective_language = _auto_detect_language(claim_text, language)
            explicit_claim_provided = bool(str(claim or "").strip())
            force_single_when_claim = _env_bool(
                "PDF_FORCE_SINGLE_CLAIM_WHEN_CLAIM_PROVIDED",
                True,
            )

            if not claim_text:
                return PDFAnalysisResponse(
                    mode="error",
                    verdict="neutral",
                    confidence=0.0,
                    error="No text found in PDF. If this is a scanned PDF, OCR pipeline is needed.",
                    selected_claim=None,
                    pdf_text=extracted_text,
                    page_count=pdf_result.page_count,
                    selected_pages=pdf_result.selected_pages,
                    selected_page_spec=pdf_result.selected_page_spec,
                    extraction_engine=pdf_result.extraction_engine,
                    warnings=pdf_result.warnings,
                )

            if (not (explicit_claim_provided and force_single_when_claim)) and _looks_multi_claim_text(claim_text):
                doc_pipeline = DocumentPipeline(_pipeline_config(effective_language, mode="pdf"))
                doc_result = doc_pipeline.analyze_text(text=claim_text, language=effective_language)
                best_claim = None
                if doc_result.claims:
                    best_claim = max(
                        doc_result.claims,
                        key=lambda row: float(row.get("confidence", 0.0)),
                    )

                best_evidence = []
                best_reasoning = None
                best_details = None
                best_verdict = doc_result.summary_verdict
                best_confidence = doc_result.summary_confidence
                if best_claim and best_claim.get("claim"):
                    # Try top candidates until one is not uncheckable.
                    ordered_claims = [str(best_claim.get("claim"))]
                    for row in doc_result.claims:
                        c = str(row.get("claim") or "")
                        if c and c not in ordered_claims:
                            ordered_claims.append(c)

                    best_result = None
                    for c in ordered_claims[:5]:
                        probe = get_claim_pipeline(effective_language).analyze(
                            claim=c,
                            language=effective_language,
                            recency_mode=str(recency_mode or "general"),
                            recency_start=str(recency_start or ""),
                            recency_end=str(recency_end or ""),
                        )
                        check_meta = str((probe.details or {}).get("checkability", ""))
                        if not check_meta.lower().startswith("uncheckable"):
                            best_result = probe
                            best_claim = {"claim": c}
                            break
                        if best_result is None:
                            best_result = probe

                    if best_result is None:
                        best_result = get_claim_pipeline(effective_language).analyze(
                            claim=str(best_claim.get("claim")),
                            language=effective_language,
                            recency_mode=str(recency_mode or "general"),
                            recency_start=str(recency_start or ""),
                            recency_end=str(recency_end or ""),
                        )
                    best_evidence = best_result.evidence
                    best_reasoning = best_result.reasoning
                    best_details = best_result.details
                    best_verdict = best_result.verdict
                    best_confidence = best_result.confidence

                return PDFAnalysisResponse(
                    mode="document",
                    summary_verdict=doc_result.summary_verdict,
                    summary_confidence=doc_result.summary_confidence,
                    summary_claim=(best_claim or {}).get("claim"),
                    selected_claim=(best_claim or {}).get("claim"),
                    verdict=best_verdict,
                    confidence=best_confidence,
                    evidence=best_evidence,
                    reasoning=best_reasoning,
                    details=best_details,
                    claims=doc_result.claims,
                    pdf_text=extracted_text,
                    page_count=pdf_result.page_count,
                    selected_pages=pdf_result.selected_pages,
                    selected_page_spec=pdf_result.selected_page_spec,
                    extraction_engine=pdf_result.extraction_engine,
                    warnings=pdf_result.warnings,
                )

            doc_pipeline = DocumentPipeline(_pipeline_config(effective_language, mode="pdf"))
            ranked_candidates = doc_pipeline.rank_claim_candidates(
                claim_text,
                language=effective_language,
            )
            selected_claim = (ranked_candidates[0] if ranked_candidates else "") or claim_text
            llm_claim_pick_enabled = _env_bool("PDF_LLM_CLAIM_SELECTION_ENABLE", True)
            if llm_claim_pick_enabled and len(ranked_candidates) > 1:
                llm_pick = _llm_pick_best_claim(
                    candidates=ranked_candidates[:6],
                    language=effective_language,
                )
                if llm_pick:
                    selected_claim = llm_pick
                    pdf_result.warnings.append("WARN: LLM-assisted claim selection applied for PDF.")
            if selected_claim != claim_text:
                pdf_result.warnings.append("WARN: Auto-selected a verifiable mini-claim from PDF text.")

            pipeline = get_claim_pipeline(effective_language)
            result = pipeline.analyze(
                claim=selected_claim,
                language=effective_language,
                recency_mode=str(recency_mode or "general"),
                recency_start=str(recency_start or ""),
                recency_end=str(recency_end or ""),
            )

            return PDFAnalysisResponse(
                mode="single_claim",
                verdict=result.verdict,
                confidence=result.confidence,
                selected_claim=selected_claim,
                evidence=result.evidence,
                reasoning=result.reasoning,
                details=result.details,
                pdf_text=extracted_text,
                page_count=pdf_result.page_count,
                selected_pages=pdf_result.selected_pages,
                selected_page_spec=pdf_result.selected_page_spec,
                extraction_engine=pdf_result.extraction_engine,
                warnings=pdf_result.warnings,
            )
        finally:
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    finally:
        REQUEST_LIMITER.release()
