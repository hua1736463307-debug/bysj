"""请求依赖：当前用户、角色校验、分页参数。"""
from __future__ import annotations

from dataclasses import dataclass

from fastapi import Depends, Request
from sqlalchemy.orm import Session

from app.core.auth import decode_access_token, product_role
from app.core.config import Settings, get_settings
from app.core.errors import ApiError
from app.db.models import AuthSession, User
from app.db.session import get_db


@dataclass(frozen=True)
class CurrentUser:
    id: int
    username: str
    role: str
    raw_role: str

    @property
    def is_admin(self) -> bool:
        return self.role == "admin"

    @property
    def can_review(self) -> bool:
        return self.role in {"admin", "reviewer"}

    def to_public(self) -> dict:
        return {"id": self.id, "username": self.username, "role": self.role}


def settings_dep() -> Settings:
    return get_settings()


def _bearer_token(request: Request) -> str:
    header = request.headers.get("Authorization", "")
    scheme, _, token = header.partition(" ")
    if scheme.lower() != "bearer" or not token.strip():
        raise ApiError(401, "unauthenticated", "需要登录后访问", headers={"WWW-Authenticate": "Bearer"})
    return token.strip()


def get_current_user(
    request: Request,
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
) -> CurrentUser:
    payload = decode_access_token(settings, _bearer_token(request))
    jti = payload["jti"]
    session = db.query(AuthSession).filter(AuthSession.jti == jti).one_or_none()
    if session is None or session.revoked_at is not None:
        raise ApiError(401, "session_revoked", "登录状态已失效，请重新登录")
    try:
        user_id = int(payload["sub"])
    except (TypeError, ValueError) as exc:
        raise ApiError(401, "token_invalid", "无效的登录凭证") from exc
    user = db.get(User, user_id)
    if user is None or user.status != "active":
        raise ApiError(401, "user_inactive", "账号不可用")
    return CurrentUser(
        id=user.id,
        username=user.username,
        role=product_role(user.role),
        raw_role=user.role,
    )


def require_roles(*roles: str):
    allowed = set(roles)

    def _dependency(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
        if user.role not in allowed and not (user.is_admin and "admin" not in allowed):
            raise ApiError(403, "forbidden", "没有执行该操作的权限")
        return user

    return _dependency


require_admin = require_roles("admin")


def require_reviewer(user: CurrentUser = Depends(get_current_user)) -> CurrentUser:
    if not user.can_review:
        raise ApiError(403, "forbidden", "需要复核权限")
    return user
