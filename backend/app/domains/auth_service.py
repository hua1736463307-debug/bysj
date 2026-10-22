"""登录、会话与基础用户初始化。"""
from __future__ import annotations

import logging
from datetime import datetime, timedelta

from sqlalchemy import func
from sqlalchemy.orm import Session

from app.core.auth import create_access_token, hash_password, product_role, verify_password
from app.core.config import Settings
from app.core.errors import ApiError
from app.db.base import utcnow
from app.db.models import AuthSession, User

logger = logging.getLogger(__name__)

MAX_FAILED_ATTEMPTS = 5
LOCK_MINUTES = 30


def _normalize_username(username: str) -> str:
    value = str(username or "").strip()
    if not value or len(value) > 64:
        raise ApiError(422, "invalid_username", "用户名不能为空且不超过 64 字符")
    return value


def authenticate(
    db: Session,
    settings: Settings,
    *,
    username: str,
    password: str,
    user_agent: str | None = None,
    client_ip: str | None = None,
) -> dict:
    username = _normalize_username(username)
    user = db.query(User).filter(func.lower(User.username) == username.lower()).one_or_none()
    now = utcnow()
    if user is None:
        # 统一错误信息，避免用户名枚举。
        raise ApiError(401, "invalid_credentials", "用户名或密码错误")
    if user.locked_until and user.locked_until > now:
        raise ApiError(423, "account_locked", "登录失败次数过多，账号已临时锁定")
    if user.status != "active":
        raise ApiError(403, "user_disabled", "账号已停用")

    if not verify_password(password, user.password_hash):
        user.failed_login_attempts = int(user.failed_login_attempts or 0) + 1
        if user.failed_login_attempts >= MAX_FAILED_ATTEMPTS:
            user.locked_until = now + timedelta(minutes=LOCK_MINUTES)
            user.failed_login_attempts = 0
        db.commit()
        raise ApiError(401, "invalid_credentials", "用户名或密码错误")

    token, jti, expires_at = create_access_token(
        settings, user_id=user.id, username=user.username, role=user.role
    )
    user.failed_login_attempts = 0
    user.locked_until = None
    user.last_login_at = now
    db.add(
        AuthSession(
            user_id=user.id,
            jti=jti,
            expires_at=expires_at.replace(tzinfo=None),
            user_agent=(user_agent or "")[:255] or None,
            client_ip=(client_ip or "")[:64] or None,
        )
    )
    db.commit()
    return {
        "token": token,
        "token_type": "bearer",
        "expires_at": expires_at.isoformat(),
        "user": {**user.to_public(), "role": product_role(user.role)},
    }


def revoke_session(db: Session, jti: str) -> None:
    session = db.query(AuthSession).filter(AuthSession.jti == jti).one_or_none()
    if session and session.revoked_at is None:
        session.revoked_at = utcnow()
        db.commit()


def ensure_bootstrap_admin(db: Session, settings: Settings) -> None:
    """初始化首个管理员。

    未配置 BOOTSTRAP_ADMIN_PASSWORD 时只记录日志，不生成默认弱口令。
    """
    existing = db.query(User).count()
    if existing:
        return
    if not settings.bootstrap_admin_password:
        logger.warning(
            "bootstrap_admin_skipped reason=missing BOOTSTRAP_ADMIN_PASSWORD; no default password is created"
        )
        return
    username = (settings.bootstrap_admin_username or "admin").strip()
    admin = User(
        username=username,
        password_hash=hash_password(settings.bootstrap_admin_password),
        role="admin",
        status="active",
        nickname="管理员",
    )
    db.add(admin)
    db.commit()
    logger.info("bootstrap_admin_created username=%s", username)


def create_user(
    db: Session,
    *,
    username: str,
    password: str,
    role: str = "researcher",
    nickname: str | None = None,
) -> User:
    role = product_role(role)
    if role == "admin":
        raise ApiError(422, "invalid_role", "该接口不能创建管理员")
    username = _normalize_username(username)
    if db.query(User).filter(func.lower(User.username) == username.lower()).one_or_none():
        raise ApiError(409, "username_taken", "用户名已存在")
    user = User(
        username=username,
        password_hash=hash_password(password),
        role=role,
        status="active",
        nickname=nickname,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    return user
