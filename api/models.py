"""Persistence models for claims and database-backed worker jobs."""

from datetime import datetime, timezone

from sqlalchemy import DateTime, Index, Integer, JSON, String, Text
from sqlalchemy.orm import Mapped, mapped_column

from api.db import Base


def utc_now() -> datetime:
    return datetime.now(timezone.utc)


class Claim(Base):
    __tablename__ = "claims"
    __table_args__ = (
        Index("ix_claims_tenant_created", "tenant_id", "created_at"),
        Index("ix_claims_tenant_status_created", "tenant_id", "status", "created_at"),
    )

    id: Mapped[str] = mapped_column(String(32), primary_key=True)
    tenant_id: Mapped[str] = mapped_column(String(128), nullable=False)
    owner_subject: Mapped[str] = mapped_column(String(256), nullable=False)
    source_image_key: Mapped[str] = mapped_column(String(512), nullable=False)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="queued")
    stage: Mapped[str | None] = mapped_column(String(200), nullable=True)
    progress_pct: Mapped[int | None] = mapped_column(Integer, nullable=True)
    detections: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    report_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    rationale: Mapped[str | None] = mapped_column(Text, nullable=True)
    reasoning_tokens: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cost_estimate: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    internal_damage_estimate: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    policy_input: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    claim_breakdown: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    pdf_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    annotated_image_key: Mapped[str | None] = mapped_column(String(512), nullable=True)
    error_message: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, nullable=False
    )
    updated_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), default=utc_now, onupdate=utc_now, nullable=False
    )
