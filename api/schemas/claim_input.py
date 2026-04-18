from pydantic import BaseModel, Field


class ClaimRequest(BaseModel):
    claim: str = Field(..., min_length=1, description="Claim text to verify")
    language: str = Field("en", description="Language code (en, te, hi, ta, kn, ml)")


class ImageUploadRequest(BaseModel):
    claim: str = Field("", description="Optional claim text")
    language: str = Field("en", description="Language code")
