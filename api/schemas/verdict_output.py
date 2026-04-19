from typing import List, Optional
from pydantic import BaseModel


class EvidenceItem(BaseModel):
    text: str
    source: str
    url: Optional[str] = None
    relevance: float
    credibility: float
    stance: str
    score: float


class AnalysisDetails(BaseModel):
    checkability: str
    context: str
    sources_checked: int
    evidence_count: int
    llm_verifier: Optional[dict] = None


class VerdictResponse(BaseModel):
    verdict: str
    confidence: float
    evidence: List[EvidenceItem]
    reasoning: str
    details: AnalysisDetails


class DocumentClaimItem(BaseModel):
    claim: str
    verdict: str
    confidence: float
    evidence_count: int


class ImageAnalysisResponse(BaseModel):
    """Stable frontend contract for image analysis endpoint."""

    # mode: single_claim | document | error
    mode: str
    verdict: str
    confidence: float

    # present for single-claim mode
    evidence: List[EvidenceItem] = []
    reasoning: Optional[str] = None
    details: Optional[AnalysisDetails] = None

    # present for document mode
    claims: List[DocumentClaimItem] = []
    summary_verdict: Optional[str] = None
    summary_confidence: Optional[float] = None
    summary_claim: Optional[str] = None

    # OCR/meta common fields
    ocr_text: str = ""
    ocr_engine: str = ""
    ocr_confidence: float = 0.0
    image_quality: dict = {}
    warnings: List[str] = []

    # present for error mode
    error: Optional[str] = None


class PDFAnalysisResponse(BaseModel):
    """Stable frontend contract for PDF analysis endpoint."""

    # mode: single_claim | document | error
    mode: str
    verdict: str
    confidence: float

    # present for single-claim mode
    evidence: List[EvidenceItem] = []
    reasoning: Optional[str] = None
    details: Optional[AnalysisDetails] = None

    # present for document mode
    claims: List[DocumentClaimItem] = []
    summary_verdict: Optional[str] = None
    summary_confidence: Optional[float] = None
    summary_claim: Optional[str] = None

    # PDF/meta common fields
    pdf_text: str = ""
    page_count: int = 0
    extraction_engine: str = ""
    warnings: List[str] = []

    # present for error mode
    error: Optional[str] = None
