"""Claim analysis endpoint."""

import logging
import os
import re
from threading import Lock
from fastapi import APIRouter, UploadFile, File, Form
from api.schemas import ClaimRequest, VerdictResponse, EvidenceItem, AnalysisDetails
from api.runtime_limits import GlobalRequestLimiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api", tags=["claims"])
_CLAIM_PIPELINE_CACHE = {}
_CLAIM_PIPELINE_LOCK = Lock()
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
        "relevance_drop_threshold": float(os.getenv("RELEVANCE_DROP_THRESHOLD", "0.30")),
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
            language=effective_language
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


@router.post("/analyze-image")
async def analyze_image(
    image: UploadFile = File(...),
    claim: str = Form(""),
    language: str = Form("en")
):
    """Analyze an image claim (OCR → pipeline)."""
    import tempfile
    from pipeline.core.ocr import OCRProcessor
    
    await REQUEST_LIMITER.acquire()
    try:
        # Save uploaded image temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
            content = await image.read()
            tmp.write(content)
            tmp_path = tmp.name

        try:
            # Extract text via OCR
            ocr_processor = OCRProcessor()
            extracted_text = ocr_processor.extract(tmp_path)

            # Use extracted text or provided claim
            claim_text = claim if claim else extracted_text
            effective_language = _auto_detect_language(claim_text, language)

            if not claim_text:
                return {
                    "error": "No text found in image. Please provide clearer image or enter claim text."
                }

            # Run through pipeline
            pipeline = get_claim_pipeline(effective_language)
            result = pipeline.analyze(
                claim=claim_text,
                language=effective_language,
                image_path=tmp_path
            )

            return {
                "verdict": result.verdict,
                "confidence": result.confidence,
                "evidence": result.evidence,
                "reasoning": result.reasoning,
                "details": result.details,
                "ocr_text": extracted_text
            }

        finally:
            # Cleanup temp file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    finally:
        REQUEST_LIMITER.release()


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
