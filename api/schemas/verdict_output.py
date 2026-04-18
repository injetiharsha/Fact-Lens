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
