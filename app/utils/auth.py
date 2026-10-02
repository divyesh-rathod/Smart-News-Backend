import asyncio
import secrets
from datetime import datetime, timedelta, timezone
from fastapi import Depends, HTTPException, status
from fastapi.security import APIKeyHeader, OAuth2PasswordBearer
import jwt

from app.config import settings



async def create_access_token(
    data: dict,
    expires_delta: timedelta | None = None
) -> str:
    to_encode = data.copy()
    now = datetime.now(timezone.utc)
    expire = now + (expires_delta or timedelta(days=settings.ACCESS_TOKEN_EXPIRE_DAYS))
    to_encode.update({"iat": now, "exp": expire})

    loop = asyncio.get_running_loop()
    token = await loop.run_in_executor(
        None,
        lambda: jwt.encode(to_encode, settings.SECRET_KEY, algorithm=settings.ALGORITHM)
    )
    return token




oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")


async def get_current_user(token: str = Depends(oauth2_scheme)):
    # 1) Decode & validate
    loop = asyncio.get_running_loop()
    try:
        payload = await loop.run_in_executor(
            None,
            lambda: jwt.decode(
                token,
                settings.SECRET_KEY,
                algorithms=[settings.ALGORITHM],
            ),
        )
    except jwt.PyJWTError:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or expired token",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # 2) Look up the user (by sub/ID or email in payload)
    from app.services.user_services import get_user_by_id
    user = await get_user_by_id(payload["user_id"])
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="User not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    return user


admin_token_header = APIKeyHeader(name="X-Admin-Token", auto_error=False)


async def require_admin_token(token: str | None = Depends(admin_token_header)) -> None:
    if not settings.ADMIN_API_KEY:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Admin endpoints are disabled: ADMIN_API_KEY is not set",
        )
    # Compare bytes: compare_digest rejects non-ASCII str, and the header value comes from the client.
    if token is None or not secrets.compare_digest(token.encode(), settings.ADMIN_API_KEY.encode()):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing admin token",
        )
   