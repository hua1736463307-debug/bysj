"""Organization and module authorization, layered over existing research roles."""
from sqlalchemy.orm import Session
from sqlalchemy.exc import IntegrityError
from app.core.errors import ApiError
from app.db.models import SystemDepartment, SystemRole, SystemUserProfile, User
from app.domains.auth_service import _revoke_user_sessions

MODULES = {
    "datasets": "数据管理", "annotations": "数据标注", "training": "模型训练",
    "models": "模型与评估", "inference": "推理验证", "reviews": "检测复核",
    "cxr": "胸片辅助分析", "isic": "皮肤镜研究",
}
JOB_MODULES = {
    "dataset_import": "datasets", "dataset_version": "datasets", "dataset_process": "datasets",
    "dataset_derive": "datasets", "dataset_export": "datasets", "annotation_export": "datasets",
    "annotation_preannotate": "annotations", "feedback_export": "reviews", "training": "training",
    "evaluation": "models", "inference": "inference",
    "cxr_import": "cxr", "cxr_freeze": "cxr", "cxr_train": "cxr", "cxr_evaluate": "cxr",
    "cxr_analyze": "cxr", "cxr_explain": "cxr", "isic_import": "isic", "isic_freeze": "isic",
    "isic_train": "isic", "isic_evaluate": "isic",
}


def permissions(db: Session, user: User) -> list[str]:
    if user.role == "admin":
        return list(MODULES)
    profile = db.get(SystemUserProfile, user.id)
    if profile is None or profile.role_id is None:
        return list(MODULES)
    role = db.get(SystemRole, profile.role_id)
    if role is None or role.status != "active":
        raise ApiError(403, "role_disabled", "所属角色已停用")
    return role.modules


def save_department(db: Session, data: dict, department_id: str | None = None):
    # Serialize tree mutations to prevent concurrent cycles on PostgreSQL.
    departments = db.query(SystemDepartment).order_by(SystemDepartment.id).with_for_update().all()
    existing = {row.id: row for row in departments}
    row = existing.get(department_id) if department_id else SystemDepartment()
    if row is None:
        raise ApiError(404, "not_found", "部门不存在")
    ancestor = data["parent_id"]
    visited = {department_id} if department_id else set()
    while ancestor:
        if ancestor in visited:
            raise ApiError(422, "department_cycle", "部门不能形成循环")
        visited.add(ancestor)
        parent = existing.get(ancestor)
        if parent is None or parent.status != "active":
            raise ApiError(422, "invalid_parent", "上级部门不存在或已停用")
        ancestor = parent.parent_id
    for key, value in data.items():
        setattr(row, key, value)
    db.add(row)
    db.commit()
    return row


def save_role(db: Session, data: dict, role_id: str | None = None):
    data["name"] = data["name"].strip()
    if not data["name"]:
        raise ApiError(422, "invalid_role_name", "角色名称不能为空")
    db.query(SystemRole).order_by(SystemRole.id).populate_existing().with_for_update().all()
    if not set(data["modules"]) <= MODULES.keys():
        raise ApiError(422, "invalid_modules", "包含未知模块")
    modules = set(data["modules"])
    for module in tuple(modules):
        modules.update({
            "annotations": {"datasets", "models"}, "training": {"datasets", "models"},
            "models": {"datasets"}, "reviews": {"datasets", "models", "inference"},
            "inference": {"datasets", "models", "reviews"},
            "cxr": {"datasets", "models"}, "isic": {"datasets", "models"},
        }.get(module, set()))
    data["modules"] = sorted(modules)
    row = db.get(SystemRole, role_id) if role_id else SystemRole()
    if row is None:
        raise ApiError(404, "not_found", "角色不存在")
    duplicate = db.query(SystemRole).filter(SystemRole.name == data["name"], SystemRole.id != role_id).first()
    if duplicate:
        raise ApiError(409, "role_name_taken", "角色名称已存在")
    for key, value in data.items():
        setattr(row, key, value)
    db.add(row)
    try:
        db.flush()
    except IntegrityError as exc:
        db.rollback()
        raise ApiError(409, "role_name_taken", "角色名称已存在") from exc
    profiles = db.query(SystemUserProfile).filter(SystemUserProfile.role_id == row.id).all()
    for profile in profiles:
        user = db.query(User).filter(User.id == profile.user_id).with_for_update().one()
        user.role = row.base_role
        _revoke_user_sessions(db, user.id)
    db.commit()
    return row


def assign_profile(db: Session, user_id: int, data: dict):
    db.query(SystemDepartment).order_by(SystemDepartment.id).populate_existing().with_for_update().all()
    db.query(SystemRole).order_by(SystemRole.id).populate_existing().with_for_update().all()
    user = db.query(User).filter(User.id == user_id).with_for_update().one_or_none()
    if user is None or user.status == "deleted":
        raise ApiError(404, "not_found", "用户不存在")
    if user.role == "admin":
        raise ApiError(409, "protected_admin", "管理员账号受保护")
    for key, model in (("department_id", SystemDepartment), ("role_id", SystemRole)):
        value = data[key]
        if value:
            target = db.get(model, value)
            if target is None or target.status != "active":
                raise ApiError(422, "invalid_assignment", "请选择有效部门和角色")
    profile = db.get(SystemUserProfile, user_id) or SystemUserProfile(user_id=user_id)
    profile.department_id = data["department_id"]
    profile.role_id = data["role_id"]
    if profile.role_id:
        user.role = db.get(SystemRole, profile.role_id).base_role
    db.add(profile)
    _revoke_user_sessions(db, user_id)
    db.commit()
    return profile
