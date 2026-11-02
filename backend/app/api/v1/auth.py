"""v1 鉴权接口。"""
from __future__ import annotations

from typing import Literal

from fastapi import APIRouter, Depends, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, get_current_user, require_admin, settings_dep
from app.core.config import Settings
from app.core.errors import ApiError, ok
from app.domains import auth_service
from app.db.session import get_db

router = APIRouter(prefix="/auth", tags=["auth"])


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=1, max_length=256)


class LoginResponse(BaseModel):
    token: str
    token_type: str
    expires_at: str
    user: dict


@router.post("/login", response_model=None)
def login(
    payload: LoginRequest,
    request: Request,
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    request.state.username = payload.username.strip()[:64]
    result = auth_service.authenticate(
        db,
        settings,
        username=payload.username,
        password=payload.password,
        user_agent=request.headers.get("User-Agent"),
        client_ip=request.client.host if request.client else None,
    )
    return ok(result, message="登录成功")


@router.get("/me", response_model=None)
def me(user: CurrentUser = Depends(get_current_user)):
    return ok(user.to_public())


@router.get("/navigation", response_model=None)
def navigation(_: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db)):
    from app.domains.navigation import list_navigation
    return ok({"items": list_navigation(db)})


@router.post("/logout", response_model=None)
def logout(
    request: Request,
    user: CurrentUser = Depends(get_current_user),
    settings: Settings = Depends(settings_dep),
    db: Session = Depends(get_db),
):
    token = request.headers.get("Authorization", "").partition(" ")[2].strip()
    from app.core.auth import decode_access_token

    payload = decode_access_token(settings, token)
    auth_service.revoke_session(db, payload["jti"])
    return ok({"logged_out": True}, message="已退出登录")


class UserCreateRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    password: str = Field(min_length=8, max_length=256)
    role: str = "researcher"
    nickname: str | None = Field(default=None, max_length=64)


@router.post("/users", response_model=None)
def create_user(
    payload: UserCreateRequest,
    _: CurrentUser = Depends(require_admin),
    db: Session = Depends(get_db),
):
    user = auth_service.create_user(
        db,
        username=payload.username,
        password=payload.password,
        role=payload.role,
        nickname=payload.nickname,
    )
    return ok(user.to_public(), message="用户已创建")


@router.get("/users", response_model=None)
def list_users(
    _: CurrentUser = Depends(require_admin),
    db: Session = Depends(get_db),
    page: int | None = Query(default=None, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    search: str = Query(default="", max_length=64),
):
    from app.db.models import User
    from sqlalchemy import or_

    query = db.query(User).filter(User.status != "deleted")
    if search.strip():
        query = query.filter(or_(
            User.username.contains(search.strip(), autoescape=True),
            User.nickname.contains(search.strip(), autoescape=True),
        ))
    total = query.count()
    query = query.order_by(User.id.asc())
    # Keep the unpaged response used by existing callers.
    if page is not None:
        query = query.offset((page - 1) * page_size).limit(page_size)
    return ok({"items": [user.to_public() for user in query.all()], "total": total})


class UserUpdateRequest(BaseModel):
    nickname: str = Field(max_length=64)
    role: Literal["researcher", "reviewer"]
    status: Literal["active", "disabled"]


class PasswordResetRequest(BaseModel):
    password: str = Field(min_length=8, max_length=72)


class PasswordChangeRequest(BaseModel):
    current_password: str = Field(min_length=1, max_length=256)
    new_password: str = Field(min_length=8, max_length=72)


@router.post("/password", response_model=None)
def change_password(
    payload: PasswordChangeRequest, user: CurrentUser = Depends(get_current_user), db: Session = Depends(get_db),
):
    from app.db.models import User
    from app.core.auth import verify_password, hash_password
    record = db.query(User).filter_by(id=user.id).with_for_update().one()
    if not verify_password(payload.current_password, record.password_hash):
        raise ApiError(403, "password_mismatch", "当前密码不正确")
    if len(payload.new_password.encode("utf-8")) > 72:
        raise ApiError(422, "password_too_long", "新密码的 UTF-8 长度须不超过 72 字节")
    record.password_hash = hash_password(payload.new_password)
    auth_service._revoke_user_sessions(db, user.id)
    db.commit()
    return ok({"changed": True}, message="密码已修改，请重新登录")


@router.patch("/users/{user_id}", response_model=None)
def update_user(
    user_id: int, payload: UserUpdateRequest,
    _: CurrentUser = Depends(require_admin), db: Session = Depends(get_db),
):
    user = auth_service.update_user(db, user_id, payload.model_dump())
    return ok(user.to_public(), message="用户已更新")


@router.post("/users/{user_id}/reset-password", response_model=None)
def reset_password(
    user_id: int, payload: PasswordResetRequest,
    _: CurrentUser = Depends(require_admin), db: Session = Depends(get_db),
):
    auth_service.reset_user_password(db, user_id, payload.password)
    return ok({"reset": True}, message="密码已重置，原会话已失效")


@router.delete("/users/{user_id}", response_model=None)
def delete_user(
    user_id: int, _: CurrentUser = Depends(require_admin), db: Session = Depends(get_db),
):
    auth_service.delete_user(db, user_id)
    return ok({"deleted": True}, message="账号已删除，历史研究记录保留")


@router.get("/sessions", response_model=None)
def list_sessions(
    page: int = Query(default=1, ge=1), page_size: int = Query(default=20, ge=1, le=100),
    _: CurrentUser = Depends(require_admin), db: Session = Depends(get_db),
):
    from app.db.models import AuthSession, User
    from app.db.base import utcnow

    query = db.query(AuthSession, User).join(User, User.id == AuthSession.user_id).filter(
        AuthSession.revoked_at.is_(None), AuthSession.expires_at > utcnow(), User.status == "active",
    )
    total = query.count()
    rows = query.order_by(AuthSession.issued_at.desc(), AuthSession.id).offset((page - 1) * page_size).limit(page_size).all()
    return ok({"items": [{
        "session_id": session.id, "username": user.username,
        "client_ip": session.client_ip, "user_agent": session.user_agent,
        "issued_at": session.issued_at.isoformat(), "expires_at": session.expires_at.isoformat(),
    } for session, user in rows], "total": total})


@router.delete("/sessions/{session_id}", response_model=None)
def revoke_session(
    session_id: str, _: CurrentUser = Depends(require_admin), db: Session = Depends(get_db),
):
    from app.db.models import AuthSession
    session = db.get(AuthSession, session_id)
    if session is None:
        raise ApiError(404, "session_not_found", "会话不存在")
    auth_service.revoke_session(db, session.jti)
    return ok({"revoked": True}, message="会话已注销")
