"""Compatibility shim; canonical schemas live in api/schemas/."""

from api.schemas import (
    AnalysisDetails,
    ClaimRequest,
    EvidenceItem,
    ImageUploadRequest,
    VerdictResponse,
)

__all__ = [
    "ClaimRequest",
    "ImageUploadRequest",
    "EvidenceItem",
    "AnalysisDetails",
    "VerdictResponse",
]
