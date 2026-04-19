"""Compatibility shim; canonical schemas live in api/schemas/."""

from api.schemas import (
    AnalysisDetails,
    ClaimRequest,
    DocumentClaimItem,
    EvidenceItem,
    ImageAnalysisResponse,
    PDFAnalysisResponse,
    ImageUploadRequest,
    VerdictResponse,
)

__all__ = [
    "ClaimRequest",
    "ImageUploadRequest",
    "EvidenceItem",
    "AnalysisDetails",
    "DocumentClaimItem",
    "ImageAnalysisResponse",
    "PDFAnalysisResponse",
    "VerdictResponse",
]
