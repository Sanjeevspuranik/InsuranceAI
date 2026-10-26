"""Public API response schemas."""

from datetime import datetime
from typing import Any, Optional

from pydantic import BaseModel, Field


class ClaimAccepted(BaseModel):
    claim_id: str
    status: str
    status_url: str


class ClaimResponse(BaseModel):
    claim_id: str
    status: str
    stage: Optional[str] = None
    progress_pct: Optional[int] = None
    created_at: datetime
    updated_at: datetime
    detections: Optional[dict[str, Any]] = None
    report: Optional[str] = None
    rationale: Optional[str] = None
    reasoning_tokens: Optional[int] = None
    cost_estimate: Optional[dict[str, Any]] = None
    internal_damage_estimate: Optional[dict[str, Any]] = None
    policy_input: Optional[dict[str, Any]] = None
    claim_breakdown: Optional[dict[str, Any]] = None
    report_pdf_url: Optional[str] = None
    annotated_image_url: Optional[str] = None
    error: Optional[str] = None


class ClaimListResponse(BaseModel):
    claims: list[ClaimResponse]


class PolicyInputRequest(BaseModel):
    template: str = "standard_comprehensive"
    idv_inr: Optional[float] = Field(default=None, gt=0, le=1_000_000_000)
    vehicle_age_months: int = Field(default=0, ge=0, le=600)
    engine_cc: int = Field(default=1200, gt=0, le=10_000)
    voluntary_deductible_inr: float = Field(default=0, ge=0, le=10_000_000)
    copay_pct: Optional[float] = Field(default=None, ge=0, le=100)
    consumables_addon: bool = False


class PolicyTemplate(BaseModel):
    id: str
    label: str
    description: str
    own_damage: bool
    zero_depreciation: bool
