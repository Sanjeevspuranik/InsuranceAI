"""Development-token and OIDC JWT authentication."""

import hmac
import logging
from dataclasses import dataclass
from typing import Any

import jwt
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from jwt import PyJWKClient
from jwt.exceptions import PyJWKClientError

from api.config import settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class Principal:
    subject: str
    tenant_id: str
    roles: tuple[str, ...]


bearer_scheme = HTTPBearer(auto_error=False)
_jwks_client = (
    PyJWKClient(settings.oidc_jwks_url)
    if settings.auth_mode == "oidc" and settings.oidc_jwks_url
    else None
)


def _roles_from_claims(claims: dict[str, Any]) -> tuple[str, ...]:
    roles = claims.get("roles", [])
    if not isinstance(roles, list):
        roles = []
    if isinstance(claims.get("realm_access"), dict):
        roles = [*roles, *claims["realm_access"].get("roles", [])]
    if not isinstance(roles, list):
        return ()
    return tuple(sorted({role for role in roles if isinstance(role, str)}))


def get_current_principal(
    credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme),
) -> Principal:
    if credentials is None or credentials.scheme.lower() != "bearer":
        logger.warning("auth.failed reason=missing_bearer_token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Bearer authentication is required.",
            headers={"WWW-Authenticate": "Bearer"},
        )

    token = credentials.credentials
    if settings.auth_mode == "development":
        if not hmac.compare_digest(token, settings.dev_token):
            logger.warning("auth.failed reason=invalid_development_token")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid access token.",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return Principal("local-dev-user", settings.default_tenant_id, ("admin",))

    if _jwks_client is None:
        logger.error("auth.unavailable reason=oidc_not_configured")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="OIDC authentication is not configured.",
        )

    try:
        signing_key = _jwks_client.get_signing_key_from_jwt(token)
        claims = jwt.decode(
            token,
            signing_key.key,
            algorithms=["RS256", "ES256"],
            audience=settings.oidc_audience,
            issuer=settings.oidc_issuer_url,
            options={"require": ["exp", "sub", "iss", "aud"]},
        )
    except PyJWKClientError as exc:
        logger.exception("auth.unavailable reason=oidc_signing_key_lookup_failed")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Could not verify the identity provider signing key.",
        ) from exc
    except jwt.PyJWTError as exc:
        logger.warning("auth.failed reason=invalid_oidc_token")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired access token.",
            headers={"WWW-Authenticate": "Bearer"},
        ) from exc

    tenant_id = claims.get(settings.oidc_tenant_claim, settings.default_tenant_id)
    return Principal(
        subject=str(claims["sub"]),
        tenant_id=str(tenant_id),
        roles=_roles_from_claims(claims),
    )


def can_access_claim(principal: Principal, claim) -> bool:
    if principal.tenant_id != claim.tenant_id:
        return False
    return (
        principal.subject == claim.owner_subject
        or bool({"adjuster", "admin"}.intersection(principal.roles))
    )
