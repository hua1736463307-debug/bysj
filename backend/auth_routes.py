"""登录认证接口。"""
import uuid
from datetime import datetime, timedelta

from fastapi import APIRouter, Depends, Header, HTTPException, Request
from pydantic import BaseModel, Field
from sqlalchemy.orm import Session

from auth_utils import (
    ACCESS_TOKEN_EXPIRE_MINUTES,
    authenticate_user,
    build_user_payload,
    create_access_token,
    get_current_user,
    revoke_session_from_authorization,
)
from database import OnlineSession, get_db
from log_utils import (
    commit_auth_transition,
    get_client_ip,
    parse_user_agent,
    record_auth_audit,
    record_login_log,
)


router = APIRouter(tags=["认证"])


class LoginRequest(BaseModel):
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=1, max_length=128)


@router.post("/auth/login")
@router.post("/api/auth/login")
def login(payload: LoginRequest, http_request: Request, db: Session = Depends(get_db)):
    try:
        user = authenticate_user(db, payload.username, payload.password)
    except HTTPException as exc:
        record_login_log(db, payload.username, "失败", str(exc.detail), http_request)
        raise

    if not user:
        record_login_log(db, payload.username, "失败", "用户名或密码错误", http_request)
        raise HTTPException(status_code=401, detail="用户名或密码错误")

    record_login_log(db, user.username, "成功", "登录成功", http_request)

    session_id = uuid.uuid4().hex
    expires_at = datetime.now() + timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    token = create_access_token(
        data={
            "sub": str(user.id),
            "username": user.username,
            "role": user.role,
            "jti": session_id,
        },
        expires_delta=timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES),
    )

    browser, os_name = parse_user_agent(http_request.headers.get("user-agent", ""))
    user.last_login_ip = get_client_ip(http_request)
    user.last_login_at = datetime.now()
    db.add(OnlineSession(
        session_id=session_id,
        user_id=user.id,
        username=user.username,
        ip_address=user.last_login_ip,
        location="内网IP",
        browser=browser,
        os=os_name,
        status="active",
        login_time=datetime.now(),
        last_active_at=datetime.now(),
        expires_at=expires_at,
    ))
    db.commit()

    return {
        "token": token,
        "access_token": token,
        "token_type": "bearer",
        "user": build_user_payload(user, db),
    }


@router.get("/auth/me")
@router.get("/api/auth/me")
def me(current_user=Depends(get_current_user)):
    return {"user": current_user}


@router.post("/auth/logout")
@router.post("/api/auth/logout")
def logout(
    request: Request,
    authorization: str = Header(default=None),
    current_user=Depends(get_current_user),
    db: Session = Depends(get_db),
):
    try:
        session = revoke_session_from_authorization(db, authorization, commit=False)
        if session:
            record_auth_audit(
                db,
                operator=str(current_user.get("username") or session.username),
                session_id=session.session_id,
                operation_type="退出",
                reason="manual_logout",
                detail="用户主动退出登录",
                request=request,
                event_key=f"{session.session_id}:manual_logout",
                commit=False,
            )
        commit_auth_transition(db, "manual_logout")
    except Exception:
        db.rollback()
        raise HTTPException(status_code=503, detail="退出审计暂不可用，请稍后重试")
    return {"message": "已退出登录"}
