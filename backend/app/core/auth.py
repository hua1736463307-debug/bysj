"""密码哈希、JWT 与会话撤销。

角色模型收敛为 admin / researcher / reviewer；历史 ``users.role='user'``
映射为 researcher（CONTRACTS C02）。
"""
from __future__ import annotations

import uuid
from datetime import datetime, timedelta, timezone

from jose import JWTError, jwt
from passlib.context import CryptContext

from app.core.config import Settings
from app.core.errors import ApiError

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

PRODUCT_ROLES = {"admin", "researcher", "reviewer"}
LEGACY_ROLE_MAP = {"user": "researcher", "admin": "admin", "researcher": "researcher", "reviewer": "reviewer"}


def hash_password(password: str) -> str:
    if not password or len(password) < 8:
        raise ApiError(422, "weak_password", "密码至少 8 位")
    return pwd_context.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    try:
        return pwd_context.verify(plain, hashed)
    except Exception:
        return False


def product_role(raw: str | None) -> str:
    return LEGACY_ROLE_MAP.get(str(raw or "").strip().lower(), "researcher")


def create_access_token(
    settings: Settings,
    *,
    user_id: int,
    username: str,
    role: str,
) -> tuple[str, str, datetime]:
    now = datetime.now(timezone.utc)
    expires = now + timedelta(minutes=settings.jwt_expire_minutes)
    jti = str(uuid.uuid4())
    payload = {
        "sub": str(user_id),
        "username": username,
        "role": product_role(role),
        "jti": jti,
        "iat": int(now.timestamp()),
        "exp": int(expires.timestamp()),
        "iss": settings.jwt_issuer,
    }
    token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    return token, jti, expires


def decode_access_token(settings: Settings, token: str) -> dict:
    try:
        payload = jwt.decode(
            token,
            settings.jwt_secret_key,
            algorithms=[settings.jwt_algorithm],
            issuer=settings.jwt_issuer,
        )
    except jwt.ExpiredSignatureError as exc:
        raise ApiError(401, "token_expired", "登录已过期，请重新登录") from exc
    except JWTError as exc:
        raise ApiError(401, "token_invalid", "无效的登录凭证") from exc
    for claim in ("sub", "jti", "exp"):
        if not payload.get(claim):
            raise ApiError(401, "token_invalid", "无效的登录凭证")
    return payload
