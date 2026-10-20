"""Dataset ownership and role-level authorization helpers."""

from typing import Any, Dict, List, Optional, Set

from fastapi import HTTPException
from sqlalchemy.orm import Session

from database import DatasetResource, DatasetVersion, Role, User, UserRole


def is_admin_user(current_user: Dict[str, Any]) -> bool:
    return current_user.get("role") == "admin" or "admin" in (current_user.get("roleKeys") or [])


def active_user_roles(db: Session, current_user: Dict[str, Any]) -> List[Role]:
    role_ids = []
    for value in current_user.get("roleIds") or []:
        try:
            role_ids.append(int(value))
        except (TypeError, ValueError):
            continue
    if not role_ids and current_user.get("id") is not None:
        role_ids = [
            row.role_id
            for row in db.query(UserRole).filter(UserRole.user_id == int(current_user["id"])).all()
        ]
    if not role_ids:
        return []
    return (
        db.query(Role)
        .filter(Role.id.in_(set(role_ids)), Role.status == "active")
        .order_by(Role.authority_level.desc(), Role.id.asc())
        .all()
    )


def select_upload_role(
    db: Session,
    current_user: Dict[str, Any],
    requested_role_id: Optional[int] = None,
) -> Role:
    roles = active_user_roles(db, current_user)
    if requested_role_id is not None:
        try:
            role_id = int(requested_role_id)
        except (TypeError, ValueError):
            raise HTTPException(status_code=400, detail="归属角色无效")
        role = next((item for item in roles if item.id == role_id), None)
        if role is None:
            raise HTTPException(status_code=403, detail="不能把数据集归属到当前用户未拥有的角色")
        return role
    if len(roles) == 1:
        return roles[0]
    if not roles:
        raise HTTPException(status_code=403, detail="当前用户没有有效角色，不能上传数据集")
    raise HTTPException(status_code=400, detail="当前用户拥有多个角色，上传时必须选择数据集归属角色")


def ensure_dataset_resource(
    db: Session,
    dataset_id: str,
    dataset_role: str,
    current_user: Dict[str, Any],
    owner_role: Role,
) -> DatasetResource:
    resource = (
        db.query(DatasetResource)
        .filter(DatasetResource.dataset_id == dataset_id, DatasetResource.dataset_role == dataset_role)
        .first()
    )
    if resource:
        return resource
    resource = DatasetResource(
        dataset_id=dataset_id,
        dataset_role=dataset_role,
        owner_user_id=current_user.get("id"),
        owner_username=str(current_user.get("username") or "system"),
        owner_role_id=owner_role.id,
        visibility_mode="role_and_higher",
    )
    db.add(resource)
    db.commit()
    db.refresh(resource)
    return resource


def get_dataset_resource(db: Session, dataset_id: str, dataset_role: str) -> Optional[DatasetResource]:
    return (
        db.query(DatasetResource)
        .filter(DatasetResource.dataset_id == dataset_id, DatasetResource.dataset_role == dataset_role)
        .first()
    )


def can_read_dataset(db: Session, current_user: Dict[str, Any], resource: Optional[DatasetResource]) -> bool:
    if is_admin_user(current_user):
        return True
    if resource is None or resource.owner_role_id is None:
        return False
    roles = active_user_roles(db, current_user)
    if any(role.id == resource.owner_role_id for role in roles):
        return True
    owner_role = db.query(Role).filter(Role.id == resource.owner_role_id, Role.status == "active").first()
    if owner_role is None:
        return False
    return any(int(role.authority_level or 0) > int(owner_role.authority_level or 0) for role in roles)


def can_manage_dataset(current_user: Dict[str, Any], resource: Optional[DatasetResource]) -> bool:
    if is_admin_user(current_user):
        return True
    if resource is None or resource.owner_user_id is None:
        return False
    try:
        return int(current_user.get("id")) == int(resource.owner_user_id)
    except (TypeError, ValueError):
        return False


def require_dataset_read(
    db: Session,
    current_user: Dict[str, Any],
    dataset_id: str,
    dataset_role: str,
) -> DatasetResource:
    resource = get_dataset_resource(db, dataset_id, dataset_role)
    if not can_read_dataset(db, current_user, resource):
        raise HTTPException(status_code=403, detail="无权读取该数据集")
    return resource


def require_dataset_manage(
    db: Session,
    current_user: Dict[str, Any],
    dataset_id: str,
    dataset_role: str,
) -> DatasetResource:
    resource = get_dataset_resource(db, dataset_id, dataset_role)
    if not can_manage_dataset(current_user, resource):
        raise HTTPException(status_code=403, detail="仅数据集上传者或管理员可以执行该操作")
    return resource


def readable_dataset_ids(db: Session, current_user: Dict[str, Any], dataset_role: str) -> Set[str]:
    resources = db.query(DatasetResource).filter(DatasetResource.dataset_role == dataset_role).all()
    if is_admin_user(current_user):
        return {item.dataset_id for item in resources}

    user_roles = active_user_roles(db, current_user)
    if not user_roles:
        return set()
    user_role_ids = {role.id for role in user_roles}
    user_authority_levels = [int(role.authority_level or 0) for role in user_roles]
    owner_role_ids = {item.owner_role_id for item in resources if item.owner_role_id is not None}
    owner_authority_levels = {
        role.id: int(role.authority_level or 0)
        for role in db.query(Role).filter(
            Role.id.in_(owner_role_ids), Role.status == "active"
        ).all()
    } if owner_role_ids else {}

    return {
        item.dataset_id
        for item in resources
        if item.owner_role_id in user_role_ids
        or any(
            authority > owner_authority_levels.get(item.owner_role_id, authority)
            for authority in user_authority_levels
        )
    }


def backfill_dataset_resources(db: Session) -> int:
    """Create conservative ownership rows for legacy datasets.

    Ambiguous or unknown ownership is assigned to the admin role for reading.
    A resolved original uploader remains the only non-admin user allowed to manage it.
    """
    admin_role = db.query(Role).filter(Role.role_key == "admin", Role.status == "active").first()
    if admin_role is None:
        return 0
    versions = db.query(DatasetVersion).order_by(DatasetVersion.created_at.asc()).all()
    created = 0
    seen = set()
    for version in versions:
        key = (version.dataset_id, version.dataset_role)
        if key in seen:
            continue
        seen.add(key)
        if get_dataset_resource(db, *key):
            continue
        owner = db.query(User).filter(User.username == version.created_by).first()
        owner_role = admin_role
        if owner:
            role_ids = [row.role_id for row in db.query(UserRole).filter(UserRole.user_id == owner.id).all()]
            roles = db.query(Role).filter(Role.id.in_(role_ids), Role.status == "active").all() if role_ids else []
            if len(roles) == 1:
                owner_role = roles[0]
            elif any(role.role_key == "admin" for role in roles):
                owner_role = next(role for role in roles if role.role_key == "admin")
        db.add(DatasetResource(
            dataset_id=version.dataset_id,
            dataset_role=version.dataset_role,
            owner_user_id=owner.id if owner else None,
            owner_username=owner.username if owner else str(version.created_by or "system"),
            owner_role_id=owner_role.id,
            visibility_mode="role_and_higher",
            created_at=version.created_at,
        ))
        created += 1
    if created:
        db.commit()
    return created
