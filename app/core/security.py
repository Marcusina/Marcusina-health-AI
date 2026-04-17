"""
Security
========
Validates requests from the Fastify backend using a shared secret key.
Uses HMAC comparison to prevent timing attacks.
"""

import hmac
from fastapi import HTTPException, Security, status
from fastapi.security import APIKeyHeader
from app.core.config import get_settings

settings = get_settings()
_api_key_header = APIKeyHeader(name="X-AI-Secret", auto_error=False)


async def verify_internal_secret(api_key: str = Security(_api_key_header)) -> str:
    if not api_key:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing X-AI-Secret header",
        )
    if not hmac.compare_digest(api_key.encode(), settings.API_SECRET_KEY.encode()):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid secret",
        )
    return api_key
