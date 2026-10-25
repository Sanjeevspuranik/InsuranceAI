import os
import shutil
import sys
import unittest
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4
from unittest.mock import patch

from PIL import Image

TEST_DATA = Path("data") / f"api-tests-{uuid4().hex}"
TEST_DATA.mkdir(parents=True, exist_ok=True)
os.environ.update(
    {
        "APP_ENV": "development",
        "API_AUTH_MODE": "development",
        "API_DEV_TOKEN": "test-only-token",
        "DATABASE_URL": f"sqlite:///./{TEST_DATA.as_posix()}/test.db",
        "STORAGE_URL": f"file:./{TEST_DATA.as_posix()}/storage",
        "DEFAULT_TENANT_ID": "test-tenant",
    }
)

from fastapi.testclient import TestClient

from api.db import SessionLocal, engine, initialize_database
from api.main import app
from api.models import Claim
from api.worker import claim_next_job, process_claim


class ClaimsApiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        initialize_database()
        cls.client_context = TestClient(app)
        cls.client = cls.client_context.__enter__()
        cls.auth_headers = {"Authorization": "Bearer test-only-token"}

    @classmethod
    def tearDownClass(cls):
        cls.client_context.__exit__(None, None, None)
        engine.dispose()
        shutil.rmtree(TEST_DATA, ignore_errors=True)

    def _jpeg_bytes(self):
        image_buffer = BytesIO()
        Image.new("RGB", (24, 24), color="blue").save(image_buffer, format="JPEG")
        return image_buffer.getvalue()

    def setUp(self):
        with SessionLocal() as db:
            db.query(Claim).delete()
            db.commit()

    def test_upload_auth_status_and_worker_queue(self):
        unauthorized = self.client.post(
            "/api/v1/claims",
            files={"file": ("vehicle.jpg", self._jpeg_bytes(), "image/jpeg")},
        )
        self.assertEqual(unauthorized.status_code, 401)

        response = self.client.post(
            "/api/v1/claims",
            headers=self.auth_headers,
            files={"file": ("vehicle.jpg", self._jpeg_bytes(), "image/jpeg")},
        )
        self.assertEqual(response.status_code, 202, response.text)
        claim_id = response.json()["claim_id"]

        claim = self.client.get(
            f"/api/v1/claims/{claim_id}", headers=self.auth_headers
        )
        self.assertEqual(claim.status_code, 200)
        self.assertEqual(claim.json()["status"], "queued")

        self.assertEqual(claim_next_job(), claim_id)
        processing = self.client.get(
            f"/api/v1/claims/{claim_id}", headers=self.auth_headers
        )
        self.assertEqual(processing.json()["status"], "processing")

        def fake_pipeline(_image_path, progress_callback=None):
            if progress_callback is not None:
                progress_callback(0.5, "Test pipeline stage")
            output_dir = Path("segmented_outputs") / f"worker-test-{claim_id}"
            output_dir.mkdir(parents=True, exist_ok=True)
            pdf_path = output_dir / "damage_report.pdf"
            pdf_path.write_bytes(b"test-pdf")
            (output_dir / "segmented_damage.jpg").write_bytes(b"test-image")
            return SimpleNamespace(
                report_text="## Summary\nTest report.",
                rationale="Visible test evidence.",
                pdf_path=str(pdf_path),
                reasoning_tokens=19,
            )

        fake_workflow = SimpleNamespace(run_claim_pipeline_with_usage=fake_pipeline)
        with patch.dict(sys.modules, {"insuranceai.workflow": fake_workflow}):
            process_claim(claim_id)

        completed = self.client.get(
            f"/api/v1/claims/{claim_id}", headers=self.auth_headers
        )
        self.assertEqual(completed.json()["status"], "completed")
        self.assertEqual(completed.json()["rationale"], "Visible test evidence.")
        self.assertEqual(completed.json()["reasoning_tokens"], 19)
        pdf = self.client.get(
            f"/api/v1/claims/{claim_id}/report.pdf", headers=self.auth_headers
        )
        self.assertEqual(pdf.status_code, 200)
        self.assertEqual(pdf.content, b"test-pdf")

    def test_invalid_image_and_cross_tenant_access(self):
        invalid = self.client.post(
            "/api/v1/claims",
            headers=self.auth_headers,
            files={"file": ("not-image.jpg", b"not an image", "image/jpeg")},
        )
        self.assertEqual(invalid.status_code, 400)

        created = self.client.post(
            "/api/v1/claims",
            headers=self.auth_headers,
            files={"file": ("vehicle.jpg", self._jpeg_bytes(), "image/jpeg")},
        )
        claim_id = created.json()["claim_id"]
        with SessionLocal() as db:
            claim = db.get(Claim, claim_id)
            claim.tenant_id = "another-tenant"
            db.commit()

        response = self.client.get(
            f"/api/v1/claims/{claim_id}", headers=self.auth_headers
        )
        self.assertEqual(response.status_code, 404)

    def test_health_endpoints(self):
        live = self.client.get("/health/live")
        ready = self.client.get("/health/ready")
        self.assertEqual(live.status_code, 200)
        self.assertEqual(ready.status_code, 200)
        self.assertTrue(live.headers.get("x-request-id"))
        self.assertNotEqual(live.headers["x-request-id"], ready.headers["x-request-id"])


if __name__ == "__main__":
    unittest.main()
