"""部门、角色、菜单与在线会话管理接口。"""
import time
from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field
from sqlalchemy import or_
from sqlalchemy.orm import Session

from auth_audit import audit_session_consistency
from auth_utils import (
    count_active_admins,
    lock_admin_membership_change,
    require_admin,
    require_any_menu,
    revoke_active_sessions_for_user,
    synchronize_user_compat_role,
)
from database import (
    DatasetResource,
    Department,
    Menu,
    OnlineSession,
    Role,
    RoleDepartment,
    RoleMenu,
    User,
    UserRole,
    get_db,
)
from log_utils import (
    commit_auth_transition,
    get_auth_audit_failure_counts,
    record_auth_audit,
    record_operation_log,
)


require_department_menu = require_any_menu("/system/dept")
require_department_reader = require_any_menu("/system/dept", "/system/user", "/system/role")
require_role_menu = require_any_menu("/system/role")
require_role_reader = require_any_menu("/system/role", "/system/user")
require_menu_menu = require_any_menu("/system/menu")
require_menu_reader = require_any_menu("/system/menu", "/system/role")
require_online_menu = require_any_menu("/system/online")


router = APIRouter(tags=["平台系统管理"])


def _operator(admin) -> str:
    return str((admin or {}).get("username") or "admin")


def _status(value: Optional[str], default: str = "active") -> str:
    normalized = str(value or default).strip().lower()
    if normalized not in {"active", "disabled"}:
        raise HTTPException(status_code=400, detail="状态只能是 active 或 disabled")
    return normalized


def _unique_ids(values) -> List[int]:
    result = []
    for value in values or []:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number not in result:
            result.append(number)
    return result


def _log(db, request, admin, title, operation_type, detail, started_at, status="成功", error=None):
    record_operation_log(
        db,
        title=title,
        operation_type=operation_type,
        operator=_operator(admin),
        status=status,
        detail=detail,
        request=request,
        started_at=started_at,
        error_message=str(error or ""),
    )


class DepartmentPayload(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    parent_id: Optional[int] = Field(None, alias="parentId")
    sort_order: int = Field(0, alias="sortOrder")
    leader: str = ""
    phone: str = ""
    email: str = ""
    status: str = "active"

    model_config = {"populate_by_name": True}


class SortItem(BaseModel):
    id: int
    sort_order: int = Field(..., alias="sortOrder")

    model_config = {"populate_by_name": True}


class SortPayload(BaseModel):
    items: List[SortItem]


def _department_or_404(db: Session, department_id: int) -> Department:
    item = db.query(Department).filter(Department.id == department_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="部门不存在")
    return item


@router.get("/system/departments")
@router.get("/api/system/departments")
def list_departments(
    name: str = "",
    status: str = "",
    admin=Depends(require_department_reader),
    db: Session = Depends(get_db),
):
    query = db.query(Department)
    if name.strip():
        query = query.filter(Department.name.contains(name.strip()))
    if status.strip():
        query = query.filter(Department.status == status.strip())
    rows = query.order_by(Department.parent_id.asc().nullsfirst(), Department.sort_order.asc(), Department.id.asc()).all()
    return {"rows": [item.to_dict() for item in rows], "total": len(rows)}


@router.post("/system/departments")
@router.post("/api/system/departments")
def create_department(
    payload: DepartmentPayload,
    request: Request,
    admin=Depends(require_department_menu),
    db: Session = Depends(get_db),
):
    started_at = time.perf_counter()
    name = payload.name.strip()
    if payload.parent_id:
        _department_or_404(db, payload.parent_id)
    duplicate = db.query(Department).filter(Department.parent_id == payload.parent_id, Department.name == name).first()
    if duplicate:
        raise HTTPException(status_code=400, detail="同级部门名称已存在")
    item = Department(
        parent_id=payload.parent_id,
        name=name,
        sort_order=payload.sort_order,
        leader=payload.leader.strip(),
        phone=payload.phone.strip(),
        email=payload.email.strip(),
        status=_status(payload.status),
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    _log(db, request, admin, "部门管理", "新增", f"新增部门：{name}", started_at)
    return {"message": "部门创建成功", "department": item.to_dict()}


@router.put("/system/departments/{department_id}")
@router.put("/api/system/departments/{department_id}")
def update_department(
    department_id: int,
    payload: DepartmentPayload,
    request: Request,
    admin=Depends(require_department_menu),
    db: Session = Depends(get_db),
):
    started_at = time.perf_counter()
    item = _department_or_404(db, department_id)
    if payload.parent_id == department_id:
        raise HTTPException(status_code=400, detail="上级部门不能选择自身")
    if payload.parent_id:
        _department_or_404(db, payload.parent_id)
        descendant_ids = {department_id}
        while True:
            children = [row[0] for row in db.query(Department.id).filter(Department.parent_id.in_(descendant_ids)).all()]
            new_ids = set(children) - descendant_ids
            if not new_ids:
                break
            descendant_ids.update(new_ids)
        if payload.parent_id in descendant_ids:
            raise HTTPException(status_code=400, detail="不能把部门移动到自身子部门下")
    item.parent_id = payload.parent_id
    item.name = payload.name.strip()
    item.sort_order = payload.sort_order
    item.leader = payload.leader.strip()
    item.phone = payload.phone.strip()
    item.email = payload.email.strip()
    item.status = _status(payload.status)
    item.updated_at = datetime.now()
    db.commit()
    db.refresh(item)
    _log(db, request, admin, "部门管理", "修改", f"修改部门：{item.name}", started_at)
    return {"message": "部门更新成功", "department": item.to_dict()}


@router.delete("/system/departments/{department_id}")
@router.delete("/api/system/departments/{department_id}")
def delete_department(
    department_id: int,
    request: Request,
    admin=Depends(require_department_menu),
    db: Session = Depends(get_db),
):
    started_at = time.perf_counter()
    item = _department_or_404(db, department_id)
    if db.query(Department).filter(Department.parent_id == department_id).first():
        raise HTTPException(status_code=400, detail="存在下级部门，不能删除")
    if db.query(User).filter(User.dept_id == department_id).first():
        raise HTTPException(status_code=400, detail="部门下存在用户，不能删除")
    name = item.name
    db.query(RoleDepartment).filter(RoleDepartment.department_id == department_id).delete(synchronize_session=False)
    db.delete(item)
    db.commit()
    _log(db, request, admin, "部门管理", "删除", f"删除部门：{name}", started_at)
    return {"message": "部门删除成功"}


@router.put("/system/departments/sort/batch")
@router.put("/api/system/departments/sort/batch")
def sort_departments(
    payload: SortPayload,
    request: Request,
    admin=Depends(require_department_menu),
    db: Session = Depends(get_db),
):
    started_at = time.perf_counter()
    changed = 0
    for sort_item in payload.items:
        item = db.query(Department).filter(Department.id == sort_item.id).first()
        if item:
            item.sort_order = sort_item.sort_order
            item.updated_at = datetime.now()
            changed += 1
    db.commit()
    _log(db, request, admin, "部门管理", "排序", f"更新 {changed} 个部门排序", started_at)
    return {"message": "部门排序保存成功", "changed": changed}


class RolePayload(BaseModel):
    role_code: str = Field(..., min_length=1, max_length=64, alias="roleCode")
    name: str = Field(..., min_length=1, max_length=100)
    role_key: str = Field(..., min_length=1, max_length=64, alias="roleKey")
    sort_order: int = Field(0, alias="sortOrder")
    authority_level: int = Field(100, ge=1, le=1000, alias="authorityLevel")
    data_scope: str = Field("self", alias="dataScope")
    status: str = "active"
    remark: str = ""
    menu_ids: List[int] = Field(default_factory=list, alias="menuIds")
    department_ids: List[int] = Field(default_factory=list, alias="departmentIds")

    model_config = {"populate_by_name": True}


class RoleScopePayload(BaseModel):
    data_scope: str = Field(..., alias="dataScope")
    department_ids: List[int] = Field(default_factory=list, alias="departmentIds")

    model_config = {"populate_by_name": True}


class RoleUsersPayload(BaseModel):
    user_ids: List[int] = Field(default_factory=list, alias="userIds")

    model_config = {"populate_by_name": True}


def _role_or_404(db: Session, role_id: int) -> Role:
    item = db.query(Role).filter(Role.id == role_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="角色不存在")
    return item


def _role_dict(db: Session, item: Role):
    data = item.to_dict()
    data["menuIds"] = [row.menu_id for row in db.query(RoleMenu).filter(RoleMenu.role_id == item.id).all()]
    data["departmentIds"] = [row.department_id for row in db.query(RoleDepartment).filter(RoleDepartment.role_id == item.id).all()]
    data["userCount"] = db.query(UserRole).filter(UserRole.role_id == item.id).count()
    return data


def _validate_authority_level(db: Session, role_key: str, authority_level: int, exclude_role_id: Optional[int] = None) -> int:
    level = int(authority_level)
    if role_key == "admin":
        level = 1000
    elif level >= 1000:
        raise HTTPException(status_code=400, detail="非管理员角色的数据集权限级别必须小于 1000")
    query = db.query(Role).filter(Role.authority_level == level)
    if exclude_role_id is not None:
        query = query.filter(Role.id != exclude_role_id)
    if query.first():
        raise HTTPException(status_code=400, detail="数据集权限级别已被其他角色使用")
    return level


@router.get("/system/roles")
@router.get("/api/system/roles")
def list_roles(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=200),
    name: str = "",
    role_key: str = "",
    status: str = "",
    admin=Depends(require_role_menu),
    db: Session = Depends(get_db),
):
    query = db.query(Role)
    if name.strip():
        query = query.filter(Role.name.contains(name.strip()))
    if role_key.strip():
        query = query.filter(Role.role_key.contains(role_key.strip()))
    if status.strip():
        query = query.filter(Role.status == status.strip())
    total = query.count()
    rows = query.order_by(Role.sort_order.asc(), Role.id.asc()).offset((page - 1) * page_size).limit(page_size).all()
    return {"rows": [_role_dict(db, item) for item in rows], "total": total, "page": page, "page_size": page_size}


@router.get("/system/roles/options")
@router.get("/api/system/roles/options")
def role_options(admin=Depends(require_role_reader), db: Session = Depends(get_db)):
    rows = db.query(Role).filter(Role.status == "active").order_by(Role.sort_order.asc(), Role.id.asc()).all()
    return {"rows": [item.to_dict() for item in rows]}


def _replace_role_links(db: Session, role_id: int, menu_ids=None, department_ids=None):
    if menu_ids is not None:
        db.query(RoleMenu).filter(RoleMenu.role_id == role_id).delete(synchronize_session=False)
        for menu_id in _unique_ids(menu_ids):
            if db.query(Menu).filter(Menu.id == menu_id).first():
                db.add(RoleMenu(role_id=role_id, menu_id=menu_id))
    if department_ids is not None:
        db.query(RoleDepartment).filter(RoleDepartment.role_id == role_id).delete(synchronize_session=False)
        for department_id in _unique_ids(department_ids):
            if db.query(Department).filter(Department.id == department_id).first():
                db.add(RoleDepartment(role_id=role_id, department_id=department_id))


@router.post("/system/roles")
@router.post("/api/system/roles")
def create_role(payload: RolePayload, request: Request, admin=Depends(require_admin), db: Session = Depends(get_db)):
    started_at = time.perf_counter()
    role_code = payload.role_code.strip()
    role_key = payload.role_key.strip().lower()
    if not role_code:
        raise HTTPException(status_code=400, detail="角色编号不能为空")
    if db.query(Role).filter(Role.role_code == role_code).first():
        raise HTTPException(status_code=400, detail="角色编号已存在")
    if db.query(Role).filter(Role.role_key == role_key).first():
        raise HTTPException(status_code=400, detail="权限字符已存在")
    authority_level = _validate_authority_level(db, role_key, payload.authority_level)
    item = Role(
        role_code=role_code,
        name=payload.name.strip(),
        role_key=role_key,
        sort_order=payload.sort_order,
        authority_level=authority_level,
        data_scope=payload.data_scope,
        status=_status(payload.status),
        remark=payload.remark.strip(),
    )
    db.add(item)
    db.flush()
    _replace_role_links(db, item.id, payload.menu_ids, payload.department_ids)
    db.commit()
    db.refresh(item)
    _log(db, request, admin, "角色管理", "新增", f"新增角色：{item.name}", started_at)
    return {"message": "角色创建成功", "role": _role_dict(db, item)}


@router.put("/system/roles/{role_id}")
@router.put("/api/system/roles/{role_id}")
def update_role(role_id: int, payload: RolePayload, request: Request, admin=Depends(require_admin), db: Session = Depends(get_db)):
    started_at = time.perf_counter()
    item = _role_or_404(db, role_id)
    role_code = payload.role_code.strip()
    role_key = payload.role_key.strip().lower()
    if not role_code:
        raise HTTPException(status_code=400, detail="角色编号不能为空")
    duplicate_code = db.query(Role).filter(Role.role_code == role_code, Role.id != role_id).first()
    if duplicate_code:
        raise HTTPException(status_code=400, detail="角色编号已存在")
    duplicate = db.query(Role).filter(Role.role_key == role_key, Role.id != role_id).first()
    if duplicate:
        raise HTTPException(status_code=400, detail="权限字符已存在")
    if item.role_key == "admin" and (_status(payload.status) != "active" or role_key != "admin"):
        raise HTTPException(status_code=400, detail="内置管理员角色不能停用或更改权限字符")
    authority_level = _validate_authority_level(db, role_key, payload.authority_level, exclude_role_id=role_id)
    item.name = payload.name.strip()
    item.role_code = role_code
    item.role_key = role_key
    item.sort_order = payload.sort_order
    item.authority_level = authority_level
    item.data_scope = payload.data_scope
    item.status = _status(payload.status)
    item.remark = payload.remark.strip()
    item.updated_at = datetime.now()
    _replace_role_links(db, role_id, payload.menu_ids, payload.department_ids)
    db.commit()
    db.refresh(item)
    _log(db, request, admin, "角色管理", "修改", f"修改角色：{item.name}", started_at)
    return {"message": "角色更新成功", "role": _role_dict(db, item)}


@router.put("/system/roles/{role_id}/scope")
@router.put("/api/system/roles/{role_id}/scope")
def update_role_scope(role_id: int, payload: RoleScopePayload, request: Request, admin=Depends(require_admin), db: Session = Depends(get_db)):
    started_at = time.perf_counter()
    item = _role_or_404(db, role_id)
    item.data_scope = payload.data_scope
    _replace_role_links(db, role_id, department_ids=payload.department_ids)
    db.commit()
    _log(db, request, admin, "角色管理", "数据权限", f"修改角色数据权限：{item.name}", started_at)
    return {"message": "数据权限保存成功"}


@router.delete("/system/roles/{role_id}")
@router.delete("/api/system/roles/{role_id}")
def delete_role(role_id: int, request: Request, admin=Depends(require_admin), db: Session = Depends(get_db)):
    started_at = time.perf_counter()
    lock_admin_membership_change(db)
    item = _role_or_404(db, role_id)
    if item.role_key in {"admin", "user"}:
        raise HTTPException(status_code=400, detail="内置角色不能删除")
    if db.query(DatasetResource).filter(DatasetResource.owner_role_id == role_id).first():
        raise HTTPException(status_code=400, detail="该角色仍有归属数据集，请先转移数据集归属")
    name = item.name
    affected_user_ids = [row.user_id for row in db.query(UserRole).filter(UserRole.role_id == role_id).all()]
    db.query(UserRole).filter(UserRole.role_id == role_id).delete(synchronize_session=False)
    db.query(RoleMenu).filter(RoleMenu.role_id == role_id).delete(synchronize_session=False)
    db.query(RoleDepartment).filter(RoleDepartment.role_id == role_id).delete(synchronize_session=False)
    db.delete(item)
    db.flush()
    for user_id in affected_user_ids:
        user = db.query(User).filter(User.id == user_id).first()
        if user and synchronize_user_compat_role(db, user):
            revoke_active_sessions_for_user(db, user.id)
    db.commit()
    _log(db, request, admin, "角色管理", "删除", f"删除角色：{name}", started_at)
    return {"message": "角色删除成功"}


@router.get("/system/roles/{role_id}/users")
@router.get("/api/system/roles/{role_id}/users")
def role_users(role_id: int, admin=Depends(require_role_menu), db: Session = Depends(get_db)):
    _role_or_404(db, role_id)
    assigned_ids = {row.user_id for row in db.query(UserRole).filter(UserRole.role_id == role_id).all()}
    users = db.query(User).order_by(User.id.asc()).all()
    return {"rows": [{**item.to_dict(), "assigned": item.id in assigned_ids} for item in users]}


@router.put("/system/roles/{role_id}/users")
@router.put("/api/system/roles/{role_id}/users")
def assign_role_users(role_id: int, payload: RoleUsersPayload, request: Request, admin=Depends(require_admin), db: Session = Depends(get_db)):
    started_at = time.perf_counter()
    lock_admin_membership_change(db)
    admins_before = count_active_admins(db)
    role = _role_or_404(db, role_id)
    desired = set(_unique_ids(payload.user_ids))
    previous = {row.user_id for row in db.query(UserRole).filter(UserRole.role_id == role_id).all()}
    db.query(UserRole).filter(UserRole.role_id == role_id).delete(synchronize_session=False)
    existing_desired = set()
    for user_id in desired:
        user = db.query(User).filter(User.id == user_id).first()
        if user:
            existing_desired.add(user_id)
            db.add(UserRole(user_id=user_id, role_id=role_id))
    db.flush()

    for user_id in previous | existing_desired:
        user = db.query(User).filter(User.id == user_id).first()
        if user and synchronize_user_compat_role(db, user):
            revoke_active_sessions_for_user(db, user.id)
    db.flush()

    if admins_before > 0 and count_active_admins(db) < 1:
        db.rollback()
        raise HTTPException(status_code=400, detail="管理员角色至少需要保留一个有效用户")
    db.commit()
    _log(db, request, admin, "角色管理", "分配用户", f"为角色 {role.name} 分配 {len(desired)} 个用户", started_at)
    return {"message": "用户分配成功"}


class MenuPayload(BaseModel):
    name: str = Field(..., min_length=1, max_length=100)
    parent_id: Optional[int] = Field(None, alias="parentId")
    path: str = ""
    route_name: str = Field("", alias="routeName")
    query_params: str = Field("", alias="query")
    component: str = ""
    permission: str = ""
    icon: str = ""
    menu_type: str = Field("menu", alias="menuType")
    visible: str = "visible"
    status: str = "active"
    cache: str = "cache"
    sort_order: int = Field(0, alias="sortOrder")

    model_config = {"populate_by_name": True}


def _menu_or_404(db: Session, menu_id: int) -> Menu:
    item = db.query(Menu).filter(Menu.id == menu_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="菜单不存在")
    return item


@router.get("/system/menus")
@router.get("/api/system/menus")
def list_menus(name: str = "", status: str = "", admin=Depends(require_menu_reader), db: Session = Depends(get_db)):
    query = db.query(Menu)
    if name.strip():
        query = query.filter(Menu.name.contains(name.strip()))
    if status.strip():
        query = query.filter(Menu.status == status.strip())
    rows = query.order_by(Menu.parent_id.asc().nullsfirst(), Menu.sort_order.asc(), Menu.id.asc()).all()
    return {"rows": [item.to_dict() for item in rows], "total": len(rows)}


@router.post("/system/menus")
@router.post("/api/system/menus")
def create_menu(payload: MenuPayload, request: Request, admin=Depends(require_admin), db: Session = Depends(get_db)):
    started_at = time.perf_counter()
    if payload.parent_id:
        _menu_or_404(db, payload.parent_id)
    item = Menu(
        parent_id=payload.parent_id,
        name=payload.name.strip(),
        path=payload.path.strip(),
        route_name=payload.route_name.strip(),
        query_params=payload.query_params.strip(),
        component=payload.component.strip(),
        permission=payload.permission.strip(),
        icon=payload.icon.strip(),
        menu_type=payload.menu_type,
        visible=payload.visible,
        status=_status(payload.status),
        cache=payload.cache,
        sort_order=payload.sort_order,
    )
    db.add(item)
    db.commit()
    db.refresh(item)
    _log(db, request, admin, "菜单管理", "新增", f"新增菜单：{item.name}", started_at)
    return {"message": "菜单创建成功", "menu": item.to_dict()}


@router.put("/system/menus/{menu_id}")
@router.put("/api/system/menus/{menu_id}")
def update_menu(menu_id: int, payload: MenuPayload, request: Request, admin=Depends(require_admin), db: Session = Depends(get_db)):
    started_at = time.perf_counter()
    item = _menu_or_404(db, menu_id)
    if payload.parent_id == menu_id:
        raise HTTPException(status_code=400, detail="上级菜单不能选择自身")
    if payload.parent_id:
        _menu_or_404(db, payload.parent_id)
    item.parent_id = payload.parent_id
    item.name = payload.name.strip()
    item.path = payload.path.strip()
    item.route_name = payload.route_name.strip()
    item.query_params = payload.query_params.strip()
    item.component = payload.component.strip()
    item.permission = payload.permission.strip()
    item.icon = payload.icon.strip()
    item.menu_type = payload.menu_type
    item.visible = payload.visible
    item.status = _status(payload.status)
    item.cache = payload.cache
    item.sort_order = payload.sort_order
    item.updated_at = datetime.now()
    db.commit()
    db.refresh(item)
    _log(db, request, admin, "菜单管理", "修改", f"修改菜单：{item.name}", started_at)
    return {"message": "菜单更新成功", "menu": item.to_dict()}


@router.delete("/system/menus/{menu_id}")
@router.delete("/api/system/menus/{menu_id}")
def delete_menu(menu_id: int, request: Request, admin=Depends(require_admin), db: Session = Depends(get_db)):
    started_at = time.perf_counter()
    item = _menu_or_404(db, menu_id)
    if db.query(Menu).filter(Menu.parent_id == menu_id).first():
        raise HTTPException(status_code=400, detail="存在下级菜单，不能删除")
    name = item.name
    db.query(RoleMenu).filter(RoleMenu.menu_id == menu_id).delete(synchronize_session=False)
    db.delete(item)
    db.commit()
    _log(db, request, admin, "菜单管理", "删除", f"删除菜单：{name}", started_at)
    return {"message": "菜单删除成功"}


@router.put("/system/menus/sort/batch")
@router.put("/api/system/menus/sort/batch")
def sort_menus(payload: SortPayload, request: Request, admin=Depends(require_admin), db: Session = Depends(get_db)):
    started_at = time.perf_counter()
    changed = 0
    for sort_item in payload.items:
        item = db.query(Menu).filter(Menu.id == sort_item.id).first()
        if item:
            item.sort_order = sort_item.sort_order
            item.updated_at = datetime.now()
            changed += 1
    db.commit()
    _log(db, request, admin, "菜单管理", "排序", f"更新 {changed} 个菜单排序", started_at)
    return {"message": "菜单排序保存成功", "changed": changed}


@router.get("/system/online-sessions")
@router.get("/api/system/online-sessions")
def list_online_sessions(
    page: int = Query(1, ge=1),
    page_size: int = Query(10, ge=1, le=200),
    ip: str = "",
    user_name: str = "",
    admin=Depends(require_online_menu),
    db: Session = Depends(get_db),
):
    now = datetime.now()
    db.query(OnlineSession).filter(OnlineSession.status == "active", OnlineSession.expires_at <= now).update(
        {OnlineSession.status: "expired"}, synchronize_session=False
    )
    db.commit()
    query = (
        db.query(OnlineSession, User, Department)
        .outerjoin(User, User.id == OnlineSession.user_id)
        .outerjoin(Department, Department.id == User.dept_id)
        .filter(OnlineSession.status == "active")
    )
    if ip.strip():
        query = query.filter(OnlineSession.ip_address.contains(ip.strip()))
    if user_name.strip():
        query = query.filter(OnlineSession.username.contains(user_name.strip()))
    total = query.count()
    rows = query.order_by(OnlineSession.last_active_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
    result = []
    for session, _user, department in rows:
        item = session.to_dict()
        item["deptName"] = department.name if department else ""
        result.append(item)
    return {"rows": result, "total": total, "page": page, "page_size": page_size}


@router.get("/system/online-sessions/audit")
@router.get("/api/system/online-sessions/audit")
def audit_online_sessions(
    since_hours: int = Query(24, ge=1, le=720),
    limit: int = Query(100, ge=1, le=1000),
    admin=Depends(require_online_menu),
    db: Session = Depends(get_db),
):
    result = audit_session_consistency(db, since_hours=since_hours, limit=limit)
    result["failureCounts"] = get_auth_audit_failure_counts()
    return result


@router.delete("/system/online-sessions/{session_id}")
@router.delete("/api/system/online-sessions/{session_id}")
def force_logout_session(
    session_id: str,
    request: Request,
    admin=Depends(require_online_menu),
    db: Session = Depends(get_db),
):
    started_at = time.perf_counter()
    item = db.query(OnlineSession).filter(OnlineSession.session_id == session_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="在线会话不存在")
    if item.status != "active":
        raise HTTPException(status_code=409, detail="在线会话已失效")
    try:
        item.status = "revoked"
        item.revoked_at = datetime.now()
        record_auth_audit(
            db,
            operator=_operator(admin),
            session_id=item.session_id,
            operation_type="强退",
            reason="forced_logout",
            detail=f"管理员强制用户退出：{item.username}",
            request=request,
            event_key=f"{item.session_id}:forced_logout",
            commit=False,
        )
        commit_auth_transition(db, f"forced_logout:{item.session_id}")
    except Exception:
        db.rollback()
        raise HTTPException(status_code=503, detail="强退审计暂不可用，请稍后重试")
    return {"message": "用户已强制退出"}
