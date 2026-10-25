"""FastAPI application for authenticated vehicle-damage claims."""

import logging
import time
import uuid
from contextlib import asynccontextmanager
from io import BytesIO

from botocore.exceptions import BotoCoreError
from fastapi import Depends, FastAPI, File, HTTPException, Request, UploadFile, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, StreamingResponse
from PIL import Image, UnidentifiedImageError
from sqlalchemy import desc, select, text
from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.orm import Session

from api.auth import Principal, can_access_claim, get_current_principal
from api.config import settings
from api.db import SessionLocal, get_db, verify_database_schema
from api.logging_config import (
    claim_id_context,
    configure_logging,
    request_id_context,
)
from api.models import Claim, utc_now
from api.schemas import ClaimAccepted, ClaimListResponse, ClaimResponse
from api.storage import storage

configure_logging()
logger = logging.getLogger(__name__)
IMAGE_FORMATS = {
    "JPEG": ("image/jpeg", ".jpg"),
    "PNG": ("image/png", ".png"),
    "WEBP": ("image/webp", ".webp"),
}
MAX_IMAGE_PIXELS = 40_000_000


@asynccontextmanager
async def lifespan(_app: FastAPI):
    settings.validate()
    verify_database_schema()
    logger.info("api.started auth_mode=%s", settings.auth_mode)
    yield
    logger.info("api.stopped")


app = FastAPI(
    title="InsuranceAI Claims API",
    version="1.0.0",
    description="Authenticated vehicle-damage claim processing API.",
    lifespan=lifespan,
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.get_cors_origins(),
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Authorization", "Content-Type"],
    expose_headers=["X-Request-ID"],
)


@app.middleware("http")
async def limit_upload_request_size(request: Request, call_next):
    if request.method == "POST" and request.url.path == "/api/v1/claims":
        content_length = request.headers.get("content-length")
        if content_length:
            try:
                request_size = int(content_length)
            except ValueError:
                logger.warning("upload.request.rejected reason=invalid_content_length")
                return JSONResponse(
                    status_code=400,
                    content={"detail": "Invalid Content-Length header."},
                )
            if request_size > settings.max_upload_bytes + 64 * 1024:
                logger.warning(
                    "upload.request.rejected reason=request_too_large "
                    "request_size_bytes=%d max_image_bytes=%d",
                    request_size,
                    settings.max_upload_bytes,
                )
                return JSONResponse(
                    status_code=413,
                    content={"detail": "Uploaded request is too large."},
                )
    return await call_next(request)


@app.middleware("http")
async def log_http_request(request: Request, call_next):
    request_id = uuid.uuid4().hex
    path_parts = request.url.path.strip("/").split("/")
    claim_id = (
        path_parts[3]
        if len(path_parts) >= 4 and path_parts[:3] == ["api", "v1", "claims"]
        else "-"
    )
    request_token = request_id_context.set(request_id)
    claim_token = claim_id_context.set(claim_id)
    request.state.request_id = request_id
    started_at = time.perf_counter()
    logger.debug(
        "http.request.started method=%s path=%s",
        request.method,
        request.url.path,
    )

    try:
        response = await call_next(request)
    except Exception:
        duration_ms = (time.perf_counter() - started_at) * 1000
        logger.exception(
            "http.request.failed method=%s path=%s duration_ms=%.1f",
            request.method,
            request.url.path,
            duration_ms,
        )
        raise
    else:
        duration_ms = (time.perf_counter() - started_at) * 1000
        if request.method == "OPTIONS":
            logger.info(
                "http.preflight.completed origin=%r requested_method=%s "
                "requested_headers=%r status=%d duration_ms=%.1f",
                request.headers.get("origin", "-")[:256],
                request.headers.get("access-control-request-method", "-"),
                request.headers.get("access-control-request-headers", "-")[:256],
                response.status_code,
                duration_ms,
            )
        else:
            logger.info(
                "http.request.completed method=%s path=%s status=%d duration_ms=%.1f",
                request.method,
                request.url.path,
                response.status_code,
                duration_ms,
            )
        response.headers["X-Request-ID"] = request_id
        return response
    finally:
        claim_id_context.reset(claim_token)
        request_id_context.reset(request_token)


def _accessible_claim(
    claim_id: str, principal: Principal, db: Session
) -> Claim:
    claim = db.get(Claim, claim_id)
    if claim is None or not can_access_claim(principal, claim):
        logger.warning(
            "claim.access.denied reason=missing_or_unauthorized",
            extra={"claim_id": claim_id},
        )
        raise HTTPException(status_code=404, detail="Claim not found.")
    return claim


def _claim_response(claim: Claim) -> ClaimResponse:
    return ClaimResponse(
        claim_id=claim.id,
        status=claim.status,
        created_at=claim.created_at,
        updated_at=claim.updated_at,
        detections=claim.detections,
        report=claim.report_text,
        rationale=claim.rationale,
        reasoning_tokens=claim.reasoning_tokens,
        report_pdf_url=(
            f"/api/v1/claims/{claim.id}/report.pdf" if claim.pdf_key else None
        ),
        annotated_image_url=(
            f"/api/v1/claims/{claim.id}/annotated-image"
            if claim.annotated_image_key
            else None
        ),
        error=claim.error_message,
    )


@app.get("/health/live")
def live():
    return {"status": "ok"}


@app.get("/health/ready")
def ready(db: Session = Depends(get_db)):
    db.execute(text("SELECT 1"))
    return {"status": "ready"}


@app.post(
    "/api/v1/claims",
    status_code=status.HTTP_202_ACCEPTED,
    response_model=ClaimAccepted,
)
def create_claim(
    file: UploadFile = File(...),
    principal: Principal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    content = file.file.read(settings.max_upload_bytes + 1)
    if not content:
        logger.warning("claim.upload.rejected reason=empty_image")
        raise HTTPException(status_code=400, detail="Uploaded image is empty.")
    if len(content) > settings.max_upload_bytes:
        logger.warning(
            "claim.upload.rejected reason=image_too_large size_bytes=%d max_bytes=%d",
            len(content),
            settings.max_upload_bytes,
        )
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail=f"Image must be {settings.max_upload_bytes // (1024 * 1024)} MB or smaller.",
        )

    try:
        with Image.open(BytesIO(content)) as image:
            image_format = image.format
            if image.width * image.height > MAX_IMAGE_PIXELS:
                logger.warning(
                    "claim.upload.rejected reason=image_dimensions_too_large "
                    "width=%d height=%d",
                    image.width,
                    image.height,
                )
                raise HTTPException(
                    status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    detail="Image dimensions are too large.",
                )
            image.verify()
    except HTTPException:
        raise
    except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
        logger.warning("claim.upload.rejected reason=invalid_image")
        raise HTTPException(status_code=400, detail="Uploaded file is not a valid image.") from exc

    image_metadata = IMAGE_FORMATS.get(image_format)
    if image_metadata is None:
        logger.warning("claim.upload.rejected reason=unsupported_image_format")
        raise HTTPException(
            status_code=415,
            detail="Supported image formats are JPEG, PNG, and WEBP.",
        )

    content_type, extension = image_metadata
    claim_id = uuid.uuid4().hex
    image_key = f"claims/{claim_id}/original{extension}"
    try:
        storage.put_bytes(image_key, content, content_type)
        claim = Claim(
            id=claim_id,
            tenant_id=principal.tenant_id,
            owner_subject=principal.subject,
            source_image_key=image_key,
            status="queued",
        )
        db.add(claim)
        db.commit()
    except (SQLAlchemyError, OSError, BotoCoreError, ValueError):
        db.rollback()
        logger.exception(
            "claim.queue.failed", extra={"claim_id": claim_id}
        )
        try:
            storage.delete(image_key)
        except (OSError, BotoCoreError, ValueError):
            logger.exception(
                "claim.upload.cleanup_failed", extra={"claim_id": claim_id}
            )
        raise HTTPException(status_code=503, detail="Could not queue the claim.")

    logger.info(
        "claim.queued image_format=%s image_size_bytes=%d",
        image_format,
        len(content),
        extra={"claim_id": claim_id},
    )
    return ClaimAccepted(
        claim_id=claim_id,
        status="queued",
        status_url=f"/api/v1/claims/{claim_id}",
    )


@app.get("/api/v1/claims", response_model=ClaimListResponse)
def list_claims(
    principal: Principal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    query = select(Claim).where(Claim.tenant_id == principal.tenant_id)
    if not {"adjuster", "admin"}.intersection(principal.roles):
        query = query.where(Claim.owner_subject == principal.subject)
    claims = db.scalars(
        query.order_by(desc(Claim.created_at)).limit(50)
    ).all()
    return ClaimListResponse(claims=[_claim_response(claim) for claim in claims])


@app.get("/api/v1/claims/{claim_id}", response_model=ClaimResponse)
def get_claim(
    claim_id: str,
    principal: Principal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    claim = _accessible_claim(claim_id, principal, db)
    return _claim_response(claim)


@app.get("/api/v1/claims/{claim_id}/report.pdf")
def download_report(
    claim_id: str,
    principal: Principal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    claim = _accessible_claim(claim_id, principal, db)
    if claim.status != "completed" or not claim.pdf_key:
        raise HTTPException(status_code=409, detail="The PDF is not ready yet.")
    content = storage.get_bytes(claim.pdf_key)
    return StreamingResponse(
        iter([content]),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="claim-{claim.id}.pdf"'},
    )


@app.get("/api/v1/claims/{claim_id}/annotated-image")
def get_annotated_image(
    claim_id: str,
    principal: Principal = Depends(get_current_principal),
    db: Session = Depends(get_db),
):
    claim = _accessible_claim(claim_id, principal, db)
    if claim.status != "completed" or not claim.annotated_image_key:
        raise HTTPException(status_code=409, detail="The annotated image is not ready yet.")
    content = storage.get_bytes(claim.annotated_image_key)
    return StreamingResponse(iter([content]), media_type="image/jpeg")
