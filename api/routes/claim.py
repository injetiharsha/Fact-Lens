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


def _pipeline_config(language: str) -> dict:
    return {
        "claim_checkability_checkpoint": _select_checkpoint("checkability", language),
        "context_checkpoint": _select_checkpoint("context", language),
        "relevance_checkpoint": _select_checkpoint("relevance", language),
        "max_evidence": _env_int("PIPELINE_MAX_EVIDENCE", 5, minimum=1),
        "enable_two_stage_relevance": _env_bool("ENABLE_TWO_STAGE_RELEVANCE", True),
        "relevance_bi_encoder_model": os.getenv(
            "RELEVANCE_BI_ENCODER_MODEL", "intfloat/multilingual-e5-small"
        ),
        "relevance_shortlist_k": int(os.getenv("RELEVANCE_SHORTLIST_K", "20")),
        "relevance_top_k": (
            int(os.getenv("RELEVANCE_TOP_K", "0")) if int(os.getenv("RELEVANCE_TOP_K", "0")) > 0 else None
        ),
        "relevance_drop_threshold": max(
            0.60,
            float(os.getenv("RELEVANCE_DROP_THRESHOLD", "0.60")),
        ),
        "stance_checkpoint": _select_checkpoint("stance", language),
        "enable_llm_verifier": _env_bool("ENABLE_LLM_VERIFIER", True),
        "llm_provider": os.getenv("LLM_VERIFIER_PROVIDER", "openai"),
        "llm_model": os.getenv("LLM_VERIFIER_MODEL", "gpt-4o-mini"),
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
    provider = str(_env_or_file("LLM_VERIFIER_PROVIDER", "openai")).strip().lower()
    model = str(_env_or_file("LLM_VERIFIER_MODEL", "gpt-4o-mini")).strip()
    base_url = str(_env_or_file("LLM_VERIFIER_BASE_URL", "")).strip()
    if not base_url:
        if provider == "groq":
            base_url = "https://api.groq.com/openai/v1"
        elif provider == "openrouter":
            base_url = "https://openrouter.ai/api/v1"
        else:
            base_url = "https://api.openai.com/v1"

    api_key = None
    if provider == "groq":
        api_key = _env_or_file("GROQ_API_KEY")
    elif provider == "openrouter":
        api_key = _env_or_file("OPENROUTER_API_KEY")
    else:
        api_key = _env_or_file("OPENAI_API_KEY")

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

            if (not (explicit_claim_provided and force_single_when_claim)) and _looks_multi_claim_text(claim_text):
                from pipeline.document_pipeline import DocumentPipeline

                doc_pipeline = DocumentPipeline(_pipeline_config(effective_language))
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
            pipeline = get_claim_pipeline(effective_language)
            result = pipeline.analyze(
                claim=claim_text,
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
        pipeline = DocumentPipeline(_pipeline_config(language))
        result = pipeline.analyze_text(text=text, language=language)
        return {
            "summary_verdict": result.summary_verdict,
            "summary_confidence": result.summary_confidence,
            "claims": result.claims,
        }
    finally:
        REQUEST_LIMITER.release()
