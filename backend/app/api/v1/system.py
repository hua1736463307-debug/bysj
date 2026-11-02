"""Administrative organization, role and audit endpoints."""
from typing import Literal
from fastapi import APIRouter, Depends, Query
from fastapi.encoders import jsonable_encoder
from pydantic import BaseModel, Field, field_validator
from sqlalchemy.orm import Session
from app.api.deps import CurrentUser, require_admin
from app.core.errors import ApiError, ok
from app.db.models import SystemDepartment, SystemRole, SystemUserProfile, SystemAuditLog, SystemMenu
from app.db.session import get_db
from app.domains import system_service

router = APIRouter(prefix="/system", tags=["system"], dependencies=[Depends(require_admin)])


class DepartmentInput(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    parent_id: str | None = None
    status: Literal["active", "disabled"] = "active"
    sort_order: int = Field(default=0, ge=0, le=100000)

    @field_validator("name")
    @classmethod
    def valid_name(cls, value):
        if not value.strip():
            raise ValueError("名称不能为空")
        return value.strip()


class RoleInput(BaseModel):
    name: str = Field(min_length=1, max_length=64)
    base_role: Literal["researcher", "reviewer"] = "researcher"
    status: Literal["active", "disabled"] = "active"
    modules: list[str] = Field(default_factory=list, max_length=8)


class ProfileInput(BaseModel):
    department_id: str | None = None
    role_id: str | None = None


class MenuInput(BaseModel):
    path: str = Field(max_length=128)
    label: str = Field(min_length=1, max_length=64)
    sort_order: int = Field(ge=0, le=100000)
    visible: bool


@router.get("/menus")
def menus(db: Session = Depends(get_db)):
    from app.domains.navigation import list_navigation
    return ok({"items": list_navigation(db)})


@router.put("/menus")
def save_menu(payload: MenuInput, db: Session = Depends(get_db)):
    from app.domains.navigation import ROUTES
    if payload.path not in dict(ROUTES) or not payload.label.strip():
        raise ApiError(422, "invalid_menu", "请选择已注册页面并填写菜单名称")
    row = db.get(SystemMenu, payload.path) or SystemMenu(path=payload.path)
    row.label = payload.label.strip()
    row.sort_order = payload.sort_order
    row.visible = payload.visible
    db.add(row)
    db.commit()
    return ok(serialize(row))


def serialize(row):
    return jsonable_encoder({column.name: getattr(row, column.name) for column in row.__table__.columns})


@router.get("/departments")
def departments(db: Session = Depends(get_db)):
    return ok({"items": [serialize(row) for row in db.query(SystemDepartment).order_by(SystemDepartment.sort_order, SystemDepartment.id).all()]})


@router.post("/departments")
def create_department(payload: DepartmentInput, db: Session = Depends(get_db)):
    return ok(serialize(system_service.save_department(db, payload.model_dump())))


@router.put("/departments/{department_id}")
def update_department(department_id: str, payload: DepartmentInput, db: Session = Depends(get_db)):
    return ok(serialize(system_service.save_department(db, payload.model_dump(), department_id)))


@router.delete("/departments/{department_id}")
def delete_department(department_id: str, db: Session = Depends(get_db)):
    db.query(SystemDepartment).order_by(SystemDepartment.id).populate_existing().with_for_update().all()
    row = db.get(SystemDepartment, department_id)
    if row is None:
        raise ApiError(404, "not_found", "部门不存在")
    if db.query(SystemDepartment).filter_by(parent_id=department_id).first() or db.query(SystemUserProfile).filter_by(department_id=department_id).first():
        raise ApiError(409, "resource_referenced", "部门仍有子部门或关联用户")
    db.delete(row)
    db.commit()
    return ok({"deleted": True})


@router.get("/roles")
def roles(db: Session = Depends(get_db)):
    return ok({"items": [serialize(row) for row in db.query(SystemRole).order_by(SystemRole.name).all()], "modules": system_service.MODULES})


@router.post("/roles")
def create_role(payload: RoleInput, db: Session = Depends(get_db)):
    return ok(serialize(system_service.save_role(db, payload.model_dump())))


@router.put("/roles/{role_id}")
def update_role(role_id: str, payload: RoleInput, db: Session = Depends(get_db)):
    return ok(serialize(system_service.save_role(db, payload.model_dump(), role_id)))


@router.delete("/roles/{role_id}")
def delete_role(role_id: str, db: Session = Depends(get_db)):
    db.query(SystemRole).order_by(SystemRole.id).populate_existing().with_for_update().all()
    row = db.get(SystemRole, role_id)
    if row is None:
        raise ApiError(404, "not_found", "角色不存在")
    if db.query(SystemUserProfile).filter_by(role_id=role_id).first():
        raise ApiError(409, "resource_referenced", "角色仍有关联用户")
    db.delete(row)
    db.commit()
    return ok({"deleted": True})


@router.get("/users/{user_id}/profile")
def profile(user_id: int, db: Session = Depends(get_db)):
    row = db.get(SystemUserProfile, user_id)
    return ok(serialize(row) if row else {"department_id": None, "role_id": None})


@router.put("/users/{user_id}/profile")
def assign(user_id: int, payload: ProfileInput, db: Session = Depends(get_db)):
    return ok(serialize(system_service.assign_profile(db, user_id, payload.model_dump())))


@router.get("/logs")
def logs(
    kind: Literal["login", "operation", "fault"] = "operation",
    search: str = Query(default="", max_length=64),
    page: int = Query(default=1, ge=1), page_size: int = Query(default=20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    query = db.query(SystemAuditLog).filter(SystemAuditLog.kind == kind)
    if search:
        query = query.filter(SystemAuditLog.username.contains(search, autoescape=True))
    total = query.count()
    rows = query.order_by(SystemAuditLog.created_at.desc(), SystemAuditLog.id).offset((page - 1) * page_size).limit(page_size).all()
    return ok({"items": [serialize(row) for row in rows], "total": total})
