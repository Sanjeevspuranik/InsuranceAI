"""Environment-backed application configuration."""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import List
from urllib.parse import unquote, urlsplit

from dotenv import load_dotenv

load_dotenv()

PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    app_env: str = os.getenv("APP_ENV", "development")
    database_url: str = os.getenv(
        "DATABASE_URL", "sqlite:///./data/insuranceai.db"
    )
    storage_url: str = os.getenv("STORAGE_URL", "file:./data/storage")
    auth_mode: str = os.getenv("API_AUTH_MODE", "development")
    dev_token: str = os.getenv("API_DEV_TOKEN", "")
    oidc_issuer_url: str = os.getenv("OIDC_ISSUER_URL", "")
    oidc_jwks_url: str = os.getenv("OIDC_JWKS_URL", "")
    oidc_audience: str = os.getenv("OIDC_AUDIENCE", "")
    oidc_tenant_claim: str = os.getenv("OIDC_TENANT_CLAIM", "tenant_id")
    default_tenant_id: str = os.getenv("DEFAULT_TENANT_ID", "local")
    cors_origins: str = os.getenv(
        "CORS_ORIGINS", "http://localhost:3000"
    )
    max_upload_bytes: int = int(os.getenv("MAX_UPLOAD_BYTES", str(20 * 1024 * 1024)))
    worker_poll_seconds: float = float(os.getenv("WORKER_POLL_SECONDS", "1"))
    s3_endpoint_url: str = os.getenv("S3_ENDPOINT_URL", "")

    def validate(self) -> None:
        if self.auth_mode not in {"development", "oidc"}:
            raise ValueError("API_AUTH_MODE must be 'development' or 'oidc'.")
        if self.auth_mode == "development":
            if self.app_env.lower() == "production":
                raise ValueError("Development-token authentication is disabled in production.")
            if not self.dev_token:
                raise ValueError("Set API_DEV_TOKEN for development authentication.")
        if self.auth_mode == "oidc" and not all(
            (self.oidc_issuer_url, self.oidc_jwks_url, self.oidc_audience)
        ):
            raise ValueError(
                "OIDC authentication requires OIDC_ISSUER_URL, "
                "OIDC_JWKS_URL, and OIDC_AUDIENCE."
            )
        if self.max_upload_bytes <= 0:
            raise ValueError("MAX_UPLOAD_BYTES must be greater than zero.")

    def get_cors_origins(self) -> List[str]:
        return [origin.strip() for origin in self.cors_origins.split(",") if origin.strip()]

    def local_storage_root(self) -> Path:
        parsed = urlsplit(self.storage_url)
        if parsed.scheme != "file":
            raise ValueError("STORAGE_URL is not configured for local file storage.")

        raw_path = unquote(parsed.path or parsed.netloc)
        if os.name == "nt" and len(raw_path) > 2 and raw_path[0] == "/" and raw_path[2] == ":":
            raw_path = raw_path[1:]
        path = Path(raw_path)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path.resolve()


settings = Settings()
