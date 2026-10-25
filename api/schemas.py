"""Public API response schemas."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel


class ClaimAccepted(BaseModel):
    claim_id: str
    status: str
    status_url: str


class ClaimResponse(BaseModel):
    claim_id: str
    status: str
    created_at: datetime
    updated_at: datetime
    detections: Optional[dict[str, Any]] = None
    report: Optional[str] = None
    rationale: Optional[str] = None
    reasoning_tokens: Optional[int] = None
    report_pdf_url: Optional[str] = None
    annotated_image_url: Optional[str] = None
    error: Optional[str] = None


class ClaimListResponse(BaseModel):
    claims: list[ClaimResponse]
