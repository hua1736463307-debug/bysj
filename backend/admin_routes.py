"""管理员用户管理接口。"""
import time
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import or_
from sqlalchemy.orm import Session

from auth_utils import (
    count_active_admins,
    get_password_hash,
    lock_admin_membership_change,
    normalize_role,
    normalize_status,
    require_admin,
    require_any_menu,
    revoke_active_sessions_for_user,
    synchronize_user_compat_role,
)
from database import Department, OnlineSession, Role, User, UserRole, get_db
from log_utils import record_operation_log


require_user_menu = require_any_menu("/system/user")


router = APIRouter(tags=["管理员用户管理"])


class CreateUserRequest(BaseModel):
    user_no: str = Field(..., min_length=1, alias="userNo")
    username: str = Field(..., min_length=1, max_length=64)
    password: str = Field(..., min_length=6, max_length=128)
    nickname: str = Field("", alias="nickName")
    dept_id: Optional[int] = Field(None, alias="deptId")
    phone: str = ""
    email: str = ""
    remark: str = ""
    role: str = "user"
    role_ids: List[int] = Field(default_factory=list, alias="roleIds")
    status: str = "active"

    model_config = {"populate_by_name": True}


class UpdateUserRequest(BaseModel):
    user_no: Optional[str] = Field(None, min_length=1, alias="userNo")
    password: Optional[str] = Field(None, min_length=6, max_length=128)
    nickname: Optional[str] = Field(None, alias="nickName")
    dept_id: Optional[int] = Field(None, alias="deptId")
    phone: Optional[str] = None
    email: Optional[str] = None
    remark: Optional[str] = None
    role: Optional[str] = None
    role_ids: Optional[List[int]] = Field(None, alias="roleIds")
    status: Optional[str] = None

    model_config = {"populate_by_name": True}


class ResetPasswordRequest(BaseModel):
    password: str = Field(..., min_length=6, max_length=128)


def _get_user_or_404(db: Session, user_id: int) -> User:
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="用户不存在")
    return user


def _admin_name(admin) -> str:
    return str((admin or {}).get("username") or "admin")


def _normalize_user_no(value: Optional[str], required: bool = False) -> Optional[str]:
    user_no = str(value or "").strip()
    if required and not user_no:
        raise HTTPException(status_code=400, detail="用户编号不能为空")
    return user_no or None


def _ensure_user_no_available(db: Session, user_no: str, excluded_user_id: Optional[int] = None) -> None:
    query = db.query(User).filter(User.user_no == user_no)
    if excluded_user_id is not None:
        query = query.filter(User.id != excluded_user_id)
    if query.first():
        raise HTTPException(status_code=400, detail="用户编号已存在")


def _parse_time(value: str, end_of_day=False):
    if not value:
        return None
    text = str(value).strip().replace("/", "-")
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if end_of_day and parsed.hour == parsed.minute == parsed.second == 0:
        return parsed.replace(hour=23, minute=59, second=59, microsecond=999999)
    return parsed


def _user_dict(db: Session, user: User):
    data = user.to_dict()
    department = db.query(Department).filter(Department.id == user.dept_id).first() if user.dept_id else None
    links = db.query(UserRole).filter(UserRole.user_id == user.id).all()
    role_ids = [item.role_id for item in links]
    roles = db.query(Role).filter(Role.id.in_(role_ids)).order_by(Role.sort_order.asc()).all() if role_ids else []
    data.update({
        "deptName": department.name if department else "",
        "roleIds": role_ids,
        "roleNames": [item.name for item in roles],
        "roleKeys": [item.role_key for item in roles],
    })
    return data


def _set_user_roles(db: Session, user: User, role_ids: Optional[List[int]], fallback_role: Optional[str] = None) -> bool:
    if role_ids is None:
        if fallback_role is None:
            return False
        role = db.query(Role).filter(Role.role_key == normalize_role(fallback_role)).first()
        role_ids = [role.id] if role else []

    normalized_ids = []
    for value in role_ids:
        try:
            role_id = int(value)
        except (TypeError, ValueError):
            continue
        if role_id not in normalized_ids and db.query(Role).filter(Role.id == role_id).first():
            normalized_ids.append(role_id)

    db.query(UserRole).filter(UserRole.user_id == user.id).delete(synchronize_session=False)
    for role_id in normalized_ids:
        db.add(UserRole(user_id=user.id, role_id=role_id))
    db.flush()
    return synchronize_user_compat_role(db, user)


@router.get("/admin/users")
@router.get("/api/admin/users")
def list_users(
    page: int = Query(1, ge=1),
    page_size: int = Query(20, ge=1, le=200),
    search: str = "",
    user_name: str = "",
    phone: str = "",
    status: str = "",
    dept_id: Optional[int] = None,
    start_time: str = "",
    end_time: str = "",
    admin=Depends(require_user_menu),
    db: Session = Depends(get_db),
):
    query = db.query(User)
    search_text = str(search or user_name or "").strip()
    if search_text:
        query = query.filter(or_(User.user_no.contains(search_text), User.username.contains(search_text), User.nickname.contains(search_text), User.role.contains(search_text)))
    if phone.strip():
        query = query.filter(User.phone.contains(phone.strip()))
    if status.strip():
        query = query.filter(User.status == status.strip())
    if dept_id:
        department_ids = {dept_id}
        while True:
            child_ids = {row[0] for row in db.query(Department.id).filter(Department.parent_id.in_(department_ids)).all()}
            new_ids = child_ids - department_ids
            if not new_ids:
                break
            department_ids.update(new_ids)
        query = query.filter(User.dept_id.in_(department_ids))
    start_dt = _parse_time(start_time)
    end_dt = _parse_time(end_time, True)
    if start_dt:
        query = query.filter(User.created_at >= start_dt)
    if end_dt:
        query = query.filter(User.created_at <= end_dt)

    total = query.count()
    items = query.order_by(User.user_no.asc(), User.id.asc()).offset((page - 1) * page_size).limit(page_size).all()
    return {"total": total, "page": page, "page_size": page_size, "users": [_user_dict(db, item) for item in items]}


@router.post("/admin/users")
@router.post("/api/admin/users")
def create_user(payload: CreateUserRequest, http_request: Request, admin=Depends(require_admin), db: Session = Depends(get_db)):
    started_at = time.perf_counter()
    username = payload.username.strip()
    user_no = _normalize_user_no(payload.user_no, required=True)
    try:
        _ensure_user_no_available(db, user_no)
        if db.query(User).filter(User.username == username).first():
            raise HTTPException(status_code=400, detail="用户名已存在")
        if payload.dept_id and not db.query(Department).filter(Department.id == payload.dept_id).first():
            raise HTTPException(status_code=400, detail="部门不存在")
        user = User(
            user_no=user_no,
            username=username,
            password_hash=get_password_hash(payload.password),
            password_ciphertext=None,
            nickname=payload.nickname.strip() or username,
            dept_id=payload.dept_id,
            phone=payload.phone.strip(),
            email=payload.email.strip(),
            remark=payload.remark.strip(),
            role=normalize_role(payload.role),
            status=normalize_status(payload.status),
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        db.add(user)
        db.flush()
        _set_user_roles(db, user, payload.role_ids or None, payload.role)
        db.commit()
        db.refresh(user)
        record_operation_log(db, title="用户管理", operation_type="新增", operator=_admin_name(admin), status="成功", detail=f"新增用户：{user_no}/{username}", request=http_request, started_at=started_at)
        return {"message": "用户创建成功", "user": _user_dict(db, user)}
    except HTTPException as exc:
        record_operation_log(db, title="用户管理", operation_type="新增", operator=_admin_name(admin), status="失败", detail=f"新增用户失败：{username or '-'}", request=http_request, started_at=started_at, error_message=str(exc.detail))
        raise


@router.put("/admin/users/{user_id}")
@router.put("/api/admin/users/{user_id}")
def update_user(user_id: int, payload: UpdateUserRequest, http_request: Request, admin=Depends(require_admin), db: Session = Depends(get_db)):
    started_at = time.perf_counter()
    lock_admin_membership_change(db)
    admins_before = count_active_admins(db)
    user = _get_user_or_404(db, user_id)
    old_role, old_status = user.role, user.status
    new_status = normalize_status(payload.status) if payload.status is not None else user.status
    requested_role = normalize_role(payload.role) if payload.role is not None else None
    if payload.dept_id and not db.query(Department).filter(Department.id == payload.dept_id).first():
        raise HTTPException(status_code=400, detail="部门不存在")

    if payload.user_no is not None:
        user_no = _normalize_user_no(payload.user_no, required=True)
        _ensure_user_no_available(db, user_no, excluded_user_id=user.id)
        user.user_no = user_no
    if payload.nickname is not None:
        user.nickname = payload.nickname.strip()
    if payload.dept_id is not None:
        user.dept_id = payload.dept_id
    if payload.phone is not None:
        user.phone = payload.phone.strip()
    if payload.email is not None:
        user.email = payload.email.strip()
    if payload.remark is not None:
        user.remark = payload.remark.strip()
    if payload.password is not None:
        user.password_hash = get_password_hash(payload.password)
        user.password_ciphertext = None
        user.failed_login_attempts = 0
        user.locked_until = None
        revoke_active_sessions_for_user(db, user.id)
    user.status = new_status
    role_changed = _set_user_roles(db, user, payload.role_ids, requested_role)
    user.updated_at = datetime.now()
    if role_changed or user.status != "active":
        revoke_active_sessions_for_user(db, user.id)
    db.flush()
    if admins_before > 0 and count_active_admins(db) < 1:
        db.rollback()
        raise HTTPException(status_code=400, detail="不能移除或禁用最后一个可用管理员")
    db.commit()
    db.refresh(user)
    record_operation_log(db, title="用户管理", operation_type="修改", operator=_admin_name(admin), status="成功", detail=f"修改用户：{user.user_no or user.id}/{user.username}，角色 {old_role}->{user.role}，状态 {old_status}->{new_status}", request=http_request, started_at=started_at)
    return {"message": "用户更新成功", "user": _user_dict(db, user)}


@router.delete("/admin/users/{user_id}")
@router.delete("/api/admin/users/{user_id}")
def delete_user(user_id: int, http_request: Request, admin=Depends(require_admin), db: Session = Depends(get_db)):
    started_at = time.perf_counter()
    lock_admin_membership_change(db)
    admins_before = count_active_admins(db)
    user = _get_user_or_404(db, user_id)
    username = user.username
    db.query(UserRole).filter(UserRole.user_id == user.id).delete(synchronize_session=False)
    db.query(OnlineSession).filter(OnlineSession.user_id == user.id).delete(synchronize_session=False)
    db.delete(user)
    db.flush()
    if admins_before > 0 and count_active_admins(db) < 1:
        db.rollback()
        raise HTTPException(status_code=400, detail="不能删除最后一个可用管理员")
    db.commit()
    record_operation_log(db, title="用户管理", operation_type="删除", operator=_admin_name(admin), status="成功", detail=f"删除用户：{username}", request=http_request, started_at=started_at)
    return {"message": "用户已删除"}


@router.post("/admin/users/{user_id}/reset-password")
@router.post("/api/admin/users/{user_id}/reset-password")
def reset_password(user_id: int, payload: ResetPasswordRequest, http_request: Request, admin=Depends(require_admin), db: Session = Depends(get_db)):
    started_at = time.perf_counter()
    user = _get_user_or_404(db, user_id)
    user.password_hash = get_password_hash(payload.password)
    user.password_ciphertext = None
    user.failed_login_attempts = 0
    user.locked_until = None
    user.updated_at = datetime.now()
    revoke_active_sessions_for_user(db, user.id)
    db.commit()
    record_operation_log(db, title="用户管理", operation_type="修改", operator=_admin_name(admin), status="成功", detail=f"重置用户密码：{user.username}", request=http_request, started_at=started_at)
    return {"message": "密码已重置"}


@router.post("/admin/users/{user_id}/reveal-password")
@router.post("/api/admin/users/{user_id}/reveal-password")
def reveal_password(user_id: int, http_request: Request, admin=Depends(require_admin), db: Session = Depends(get_db)):
    started_at = time.perf_counter()
    user = _get_user_or_404(db, user_id)
    record_operation_log(
        db,
        title="用户管理",
        operation_type="查看密码",
        operator=_admin_name(admin),
        status="失败",
        detail=f"拒绝查看用户原密码：{user.username}",
        request=http_request,
        started_at=started_at,
        error_message="原密码查看功能已下线",
    )
    raise HTTPException(status_code=410, detail="原密码查看功能已下线，请使用密码重置")
