"""Database-backed worker for claim analysis jobs."""

import logging
import shutil
import tempfile
import time
from pathlib import Path

from sqlalchemy import select, update

from api.config import PROJECT_ROOT, settings
from api.db import SessionLocal, verify_database_schema
from api.logging_config import claim_id_context, configure_logging
from api.models import Claim, utc_now
from api.storage import storage

configure_logging()
logger = logging.getLogger(__name__)


def claim_next_job() -> str | None:
    with SessionLocal() as db:
        claim_id = db.scalar(
            select(Claim.id)
            .where(Claim.status == "queued")
            .order_by(Claim.created_at)
            .limit(1)
        )
        if claim_id is None:
            return None
        result = db.execute(
            update(Claim)
            .where(Claim.id == claim_id, Claim.status == "queued")
            .values(status="processing", updated_at=utc_now())
        )
        db.commit()
        if result.rowcount != 1:
            return None
        logger.info("claim.job.claimed", extra={"claim_id": claim_id})
        return claim_id


def process_claim(claim_id: str) -> None:
    claim_context_token = claim_id_context.set(claim_id)
    started_at = time.perf_counter()
    output_directory = None
    try:
        logger.info("claim.job.started")
        with SessionLocal() as db:
            claim = db.get(Claim, claim_id)
            if claim is None:
                raise LookupError("Queued claim record no longer exists.")
            source_image_key = claim.source_image_key

        source_image_bytes = storage.get_bytes(source_image_key)
        logger.debug("claim.source.loaded size_bytes=%d", len(source_image_bytes))
        suffix = Path(source_image_key).suffix or ".jpg"
        with tempfile.TemporaryDirectory(prefix="insuranceai-claim-") as temp_dir:
            source_path = Path(temp_dir) / f"source{suffix}"
            source_path.write_bytes(source_image_bytes)

            from insuranceai.workflow import run_claim_pipeline_with_usage

            def log_pipeline_progress(progress: float, stage: str) -> None:
                logger.info(
                    "claim.pipeline.progress stage=%r progress_pct=%.0f",
                    stage,
                    progress * 100,
                )

            result = run_claim_pipeline_with_usage(
                str(source_path),
                progress_callback=log_pipeline_progress,
            )
            logger.info(
                "claim.pipeline.completed reasoning_tokens=%s",
                result.reasoning_tokens,
            )
            pdf_file = Path(result.pdf_path)
            output_directory = pdf_file.parent.resolve()
            annotated_file = output_directory / "segmented_damage.jpg"
            if not pdf_file.is_file():
                raise FileNotFoundError("Claim workflow did not create a PDF.")
            if not annotated_file.is_file():
                raise FileNotFoundError("Claim workflow did not create an annotated image.")

            pdf_key = f"claims/{claim_id}/report.pdf"
            image_key = f"claims/{claim_id}/annotated.jpg"
            pdf_bytes = pdf_file.read_bytes()
            annotated_image_bytes = annotated_file.read_bytes()
            storage.put_bytes(pdf_key, pdf_bytes, "application/pdf")
            storage.put_bytes(image_key, annotated_image_bytes, "image/jpeg")
            logger.info(
                "claim.artifacts.saved pdf_size_bytes=%d annotated_image_size_bytes=%d",
                len(pdf_bytes),
                len(annotated_image_bytes),
            )

        with SessionLocal() as db:
            claim = db.get(Claim, claim_id)
            if claim is None:
                raise RuntimeError("Claim record disappeared during processing.")
            claim.status = "completed"
            claim.report_text = result.report_text
            claim.rationale = result.rationale
            claim.reasoning_tokens = result.reasoning_tokens
            claim.pdf_key = pdf_key
            claim.annotated_image_key = image_key
            claim.error_message = None
            claim.updated_at = utc_now()
            db.commit()
        logger.info(
            "claim.job.completed duration_ms=%.1f",
            (time.perf_counter() - started_at) * 1000,
        )
    except Exception:
        # Isolate failures to this claim so one bad job does not stop the worker.
        logger.exception(
            "claim.job.failed duration_ms=%.1f",
            (time.perf_counter() - started_at) * 1000,
        )
        with SessionLocal() as db:
            claim = db.get(Claim, claim_id)
            if claim is not None:
                claim.status = "failed"
                claim.error_message = "Analysis failed. Please retry or contact support."
                claim.updated_at = utc_now()
                db.commit()
                logger.info("claim.status.updated status=failed")
    finally:
        try:
            if output_directory is not None:
                artifacts_root = (PROJECT_ROOT / "segmented_outputs").resolve()
                if output_directory.parent == artifacts_root and output_directory.is_dir():
                    shutil.rmtree(output_directory)
                    logger.debug("claim.temporary_artifacts.cleaned")
        except OSError:
            logger.exception("claim.temporary_artifacts.cleanup_failed")
        finally:
            claim_id_context.reset(claim_context_token)


def run_worker(poll_seconds: float = settings.worker_poll_seconds) -> None:
    settings.validate()
    verify_database_schema()
    logger.info("worker.started poll_seconds=%s", poll_seconds)
    while True:
        claim_id = claim_next_job()
        if claim_id is None:
            time.sleep(poll_seconds)
            continue
        process_claim(claim_id)


if __name__ == "__main__":
    run_worker()
