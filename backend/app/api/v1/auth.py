"""v1 鉴权接口。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Request
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
    nickname: str | None = None


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
):
    from app.db.models import User

    users = db.query(User).order_by(User.id.asc()).all()
    return ok({"items": [user.to_public() for user in users], "total": len(users)})
