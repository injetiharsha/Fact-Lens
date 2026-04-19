from pydantic import BaseModel, Field


class ClaimRequest(BaseModel):
    claim: str = Field(..., min_length=1, description="Claim text to verify")
    language: str = Field("en", description="Language code (en, te, hi, ta, kn, ml)")
    recency_mode: str = Field(
        "general",
        description="general | last_1_day | last_7_days | last_30_days | last_1_year | custom",
    )
    recency_start: str = Field("", description="ISO date/datetime start for custom recency mode")
    recency_end: str = Field("", description="ISO date/datetime end for custom recency mode")


class ImageUploadRequest(BaseModel):
    claim: str = Field("", description="Optional claim text")
    language: str = Field("en", description="Language code")
    recency_mode: str = Field(
        "general",
        description="general | last_1_day | last_7_days | last_30_days | last_1_year | custom",
    )
    recency_start: str = Field("", description="ISO date/datetime start for custom recency mode")
    recency_end: str = Field("", description="ISO date/datetime end for custom recency mode")
