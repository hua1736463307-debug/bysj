"""
认证与权限工具：密码哈希、JWT、当前用户解析、管理员校验。
"""
import os
import logging
import hashlib
from datetime import datetime, timedelta
from typing import Optional, Dict, Any

from fastapi import Depends, Header, HTTPException, Request, status
from jose import JWTError, jwt
from passlib.context import CryptContext
from sqlalchemy import text
from sqlalchemy.orm import Session

from database import Menu, OnlineSession, Role, RoleMenu, SessionLocal, User, UserRole, get_db
from log_utils import commit_auth_transition, record_auth_audit


DEFAULT_JWT_SECRET_SHA256 = "5e04c17e27031bb268bf64697e8a21c4df8de779d61319fe3195ea049bf404d4"
DEFAULT_BOOTSTRAP_PASSWORD_SHA256 = "ac0e7d037817094e9e0b4441f9bae3209d67b02fa484917065f71b16109a1a78"
MIN_JWT_SECRET_LENGTH = 32
SECRET_KEY = os.getenv("JWT_SECRET_KEY", "").strip()
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("JWT_ACCESS_TOKEN_EXPIRE_MINUTES", "10080"))

logger = logging.getLogger(__name__)

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

VALID_ROLES = {"admin", "user"}
VALID_STATUSES = {"active", "disabled"}
MAX_FAILED_LOGIN_ATTEMPTS = int(os.getenv("MAX_FAILED_LOGIN_ATTEMPTS", "5"))
ACCOUNT_LOCK_MINUTES = int(os.getenv("ACCOUNT_LOCK_MINUTES", "30"))


def validate_security_configuration() -> None:
    secret_digest = hashlib.sha256(SECRET_KEY.encode("utf-8")).hexdigest()
    if not SECRET_KEY or secret_digest == DEFAULT_JWT_SECRET_SHA256 or len(SECRET_KEY) < MIN_JWT_SECRET_LENGTH:
        raise RuntimeError("JWT_SECRET_KEY must be a non-default value of at least 32 characters")


def _require_token_claims(payload: Dict[str, Any]) -> None:
    missing = [name for name in ("sub", "jti", "exp") if payload.get(name) in (None, "")]
    if missing:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="无效的登录凭证")


def verify_password(plain_password: str, password_hash: str) -> bool:
    return pwd_context.verify(plain_password, password_hash)


def get_password_hash(password: str) -> str:
    return pwd_context.hash(password)


def normalize_role(role: str) -> str:
    value = str(role or "user").strip().lower()
    if value not in VALID_ROLES:
        raise HTTPException(status_code=400, detail="角色只能是 admin 或 user")
    return value


def normalize_status(user_status: str) -> str:
    value = str(user_status or "active").strip().lower()
    if value not in VALID_STATUSES:
        raise HTTPException(status_code=400, detail="状态只能是 active 或 disabled")
    return value


ADMIN_MEMBERSHIP_LOCK_KEY = 2026092401


def _user_role_state(db: Session, user: User):
    role_links = db.query(UserRole).filter(UserRole.user_id == user.id).all()
    linked_role_ids = [item.role_id for item in role_links]
    roles = (
        db.query(Role)
        .filter(Role.id.in_(linked_role_ids), Role.status == "active")
        .order_by(Role.sort_order.asc(), Role.id.asc())
        .all()
        if linked_role_ids
        else []
    )
    return role_links, roles


def effective_user_role(user: User, role_links, active_roles) -> str:
    """Resolve the compatibility role from active role relations.

    Existing databases without any relation rows temporarily fall back to the
    legacy users.role value. Once a relation exists, inactive or removed roles
    can no longer grant administrator access.
    """
    role_keys = {str(item.role_key or "").strip().lower() for item in active_roles}
    if role_links:
        return "admin" if "admin" in role_keys else "user"
    return "admin" if str(user.role or "").strip().lower() == "admin" else "user"


def synchronize_user_compat_role(db: Session, user: User) -> bool:
    """Recalculate users.role after an explicit UserRole mutation."""
    _links, roles = _user_role_state(db, user)
    role_keys = {str(item.role_key or "").strip().lower() for item in roles}
    new_role = "admin" if "admin" in role_keys else "user"
    changed = user.role != new_role
    user.role = new_role
    user.updated_at = datetime.now()
    return changed


def count_active_admins(db: Session) -> int:
    return (
        db.query(User.id)
        .join(UserRole, UserRole.user_id == User.id)
        .join(Role, Role.id == UserRole.role_id)
        .filter(User.status == "active", Role.status == "active", Role.role_key == "admin")
        .distinct()
        .count()
    )


def lock_admin_membership_change(db: Session) -> None:
    """Serialize administrator membership changes on PostgreSQL."""
    bind = db.get_bind()
    if bind is not None and bind.dialect.name == "postgresql":
        db.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": ADMIN_MEMBERSHIP_LOCK_KEY})


def revoke_active_sessions_for_user(db: Session, user_id: int) -> int:
    return db.query(OnlineSession).filter(
        OnlineSession.user_id == user_id,
        OnlineSession.status == "active",
    ).update(
        {OnlineSession.status: "revoked", OnlineSession.revoked_at: datetime.now()},
        synchronize_session=False,
    )


def create_access_token(data: Dict[str, Any], expires_delta: Optional[timedelta] = None) -> str:
    validate_security_configuration()
    to_encode = data.copy()
    _require_token_claims({**to_encode, "exp": True})
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


def _filter_menus_with_active_ancestors(menus, db: Session):
    """Exclude active children that belong to an inactive or hidden directory."""
    menu_by_id = {item.id: item for item in db.query(Menu).all()}
    allowed = []
    for menu in menus:
        parent_id = menu.parent_id
        visited = {menu.id}
        while parent_id:
            if parent_id in visited:
                break
            visited.add(parent_id)
            parent = menu_by_id.get(parent_id)
            if not parent or parent.status != "active" or parent.visible != "visible":
                break
            parent_id = parent.parent_id
        else:
            allowed.append(menu)
            continue
        if not parent_id:
            allowed.append(menu)
    return allowed


def _include_active_visible_ancestors(menus, db: Session):
    """Add usable directory ancestors so the frontend can render a complete tree."""
    menu_by_id = {item.id: item for item in db.query(Menu).all()}
    included = {item.id: item for item in menus}
    for menu in menus:
        parent_id = menu.parent_id
        visited = {menu.id}
        while parent_id:
            if parent_id in visited:
                break
            visited.add(parent_id)
            parent = menu_by_id.get(parent_id)
            if not parent or parent.status != "active" or parent.visible != "visible":
                break
            included[parent.id] = parent
            parent_id = parent.parent_id
    return list(included.values())


def _build_menu_tree(menus):
    """Serialize active menu and directory records as a stable, sorted tree."""
    nodes = {}
    for item in menus:
        if item.menu_type not in {"directory", "menu"}:
            continue
        nodes[item.id] = {**item.to_dict(), "children": []}

    roots = []
    ordered = sorted(nodes.values(), key=lambda item: (
        item.get("parentId") is not None,
        item.get("sortOrder") or 0,
        item.get("id") or 0,
    ))
    for node in ordered:
        parent_id = node.get("parentId")
        parent = nodes.get(parent_id)
        if parent:
            parent["children"].append(node)
        else:
            roots.append(node)

    def sort_children(items):
        items.sort(key=lambda item: (item.get("sortOrder") or 0, item.get("id") or 0))
        for item in items:
            sort_children(item["children"])
        return items

    return sort_children(roots)


def build_user_payload(user: User, db: Session) -> Dict[str, Any]:
    data = user.to_dict()
    role_links, roles = _user_role_state(db, user)
    role_ids = [item.id for item in roles]
    role_keys = [item.role_key for item in roles]
    effective_role = effective_user_role(user, role_links, roles)
    is_admin = effective_role == "admin"

    if is_admin:
        menus = db.query(Menu).filter(Menu.status == "active", Menu.visible == "visible").all()
    elif role_ids:
        menu_ids = [
            item.menu_id
            for item in db.query(RoleMenu).filter(RoleMenu.role_id.in_(role_ids)).all()
        ]
        menus = (
            db.query(Menu)
            .filter(Menu.id.in_(menu_ids), Menu.status == "active", Menu.visible == "visible")
            .all()
            if menu_ids
            else []
        )
    else:
        menus = []

    if effective_role != "admin":
        menus = [item for item in menus if item.path != "/system/algorithms"]
    menus = _filter_menus_with_active_ancestors(menus, db)
    menu_tree = _build_menu_tree(_include_active_visible_ancestors(menus, db))

    data.update({
        "role": effective_role,
        "roleIds": role_ids,
        "roleKeys": role_keys,
        "roles": [
            {
                "id": item.id,
                "name": item.name,
                "roleKey": item.role_key,
                "authorityLevel": item.authority_level,
            }
            for item in roles
        ],
        "menuIds": sorted({item.id for item in menus}),
        "menuPaths": sorted({item.path for item in menus if item.menu_type == "menu" and item.path}),
        "menuTree": menu_tree,
    })
    return data


def authenticate_user(db: Session, username: str, password: str) -> Optional[User]:
    user = db.query(User).filter(User.username == username).first()
    if not user:
        return None
    if user.status != "active":
        raise HTTPException(status_code=403, detail="该账号已被禁用")
    if user.locked_until and user.locked_until > datetime.now():
        raise HTTPException(status_code=423, detail=f"账号已锁定，请在 {user.locked_until.strftime('%Y-%m-%d %H:%M:%S')} 后重试")
    if not verify_password(password, user.password_hash):
        user.failed_login_attempts = int(user.failed_login_attempts or 0) + 1
        if user.failed_login_attempts >= MAX_FAILED_LOGIN_ATTEMPTS:
            user.locked_until = datetime.now() + timedelta(minutes=ACCOUNT_LOCK_MINUTES)
        db.commit()
        return None
    user.failed_login_attempts = 0
    user.locked_until = None
    db.commit()
    return user


def _extract_token_from_authorization(authorization: Optional[str]) -> str:
    if not authorization:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="未登录或登录已过期")

    parts = authorization.strip().split()
    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="认证信息格式错误")

    return parts[1]


def _decode_token(token: str) -> Dict[str, Any]:
    validate_security_configuration()
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        _require_token_claims(payload)
        return payload
    except HTTPException:
        raise
    except JWTError:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录已过期，请重新登录")


def _get_user_from_payload(db: Session, payload: Dict[str, Any], request: Optional[Request] = None) -> User:
    _require_token_claims(payload)
    user_id = payload.get("sub")

    try:
        user_id_value = int(user_id)
    except (TypeError, ValueError):
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="无效的登录凭证")

    user = db.query(User).filter(User.id == user_id_value).first()
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="用户不存在")
    if user.status != "active":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="该账号已被禁用")

    username_claim = str(payload.get("username") or "").strip()
    if username_claim and username_claim != user.username:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="无效的登录凭证")

    session_id = str(payload["jti"]).strip()
    if session_id:
        online_session = db.query(OnlineSession).filter(
            OnlineSession.session_id == session_id,
            OnlineSession.user_id == user.id,
        ).first()
        if not online_session:
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="会话已失效，请重新登录")
        if online_session.status != "active":
            if online_session.status == "revoked":
                try:
                    record_auth_audit(
                        db,
                        operator=user.username,
                        session_id=session_id,
                        operation_type="会话撤销",
                        reason="session_revoked",
                        detail="使用已撤销会话访问系统",
                        request=request,
                        status="失败",
                        event_key=f"{session_id}:session_revoked",
                        commit=True,
                    )
                except Exception:
                    db.rollback()
                    logger.exception("记录已撤销会话访问审计失败: session_id=%s", session_id)
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="会话已失效，请重新登录")
        if online_session.expires_at <= datetime.now():
            online_session.status = "expired"
            try:
                record_auth_audit(
                    db,
                    operator=user.username,
                    session_id=session_id,
                    operation_type="过期",
                    reason="token_expired",
                    detail="登录令牌已过期，会话自动失效",
                    request=request,
                    event_key=f"{session_id}:token_expired",
                    commit=False,
                )
                commit_auth_transition(db, f"token_expired:{session_id}")
            except Exception:
                db.rollback()
                raise HTTPException(status_code=503, detail="会话审计暂不可用，请稍后重试")
            raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="登录已过期，请重新登录")
        if not online_session.last_active_at or (datetime.now() - online_session.last_active_at).total_seconds() >= 60:
            online_session.last_active_at = datetime.now()
            db.commit()
    return user


def get_token_payload(authorization: Optional[str]) -> Dict[str, Any]:
    token = _extract_token_from_authorization(authorization)
    return _decode_token(token)


def revoke_session_from_authorization(
    db: Session,
    authorization: Optional[str],
    *,
    commit: bool = True,
) -> Optional[OnlineSession]:
    payload = get_token_payload(authorization)
    session_id = str(payload.get("jti") or "").strip()
    if not session_id:
        return None
    online_session = db.query(OnlineSession).filter(OnlineSession.session_id == session_id).first()
    if online_session and online_session.status == "active":
        online_session.status = "revoked"
        online_session.revoked_at = datetime.now()
        if commit:
            db.commit()
        return online_session
    return None


def get_current_user_from_authorization(
    authorization: Optional[str],
    request: Optional[Request] = None,
) -> Dict[str, Any]:
    """供 FastAPI middleware 使用。"""
    token = _extract_token_from_authorization(authorization)
    payload = _decode_token(token)

    db = SessionLocal()
    try:
        user = _get_user_from_payload(db, payload, request)
        return build_user_payload(user, db)
    finally:
        db.close()


def get_current_user(
    request: Request,
    authorization: Optional[str] = Header(default=None),
    db: Session = Depends(get_db),
) -> Dict[str, Any]:
    """供路由 Depends 使用。"""
    token = _extract_token_from_authorization(authorization)
    payload = _decode_token(token)
    user = _get_user_from_payload(db, payload, request)
    return build_user_payload(user, db)


def require_any_menu(*menu_paths: str):
    allowed_paths = {str(path).strip() for path in menu_paths if str(path).strip()}

    def dependency(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
        if current_user.get("role") == "admin":
            return current_user
        if allowed_paths.intersection(current_user.get("menuPaths") or []):
            return current_user
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="菜单权限不足")

    return dependency


def require_admin(current_user: Dict[str, Any] = Depends(get_current_user)) -> Dict[str, Any]:
    if current_user.get("role") != "admin":
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="管理员权限不足")
    return current_user


def ensure_default_admin() -> None:
    """仅在空库首次启动时创建显式配置的管理员，不修改已有账号。"""
    validate_security_configuration()
    db = SessionLocal()
    try:
        if db.query(User.id).limit(1).first() is not None:
            return

        username = os.getenv("DEFAULT_ADMIN_USERNAME", "").strip()
        password = os.getenv("DEFAULT_ADMIN_PASSWORD", "").strip()
        password_digest = hashlib.sha256(password.encode("utf-8")).hexdigest()
        if not username or not password or password_digest == DEFAULT_BOOTSTRAP_PASSWORD_SHA256:
            raise RuntimeError("DEFAULT_ADMIN_USERNAME and a non-default DEFAULT_ADMIN_PASSWORD are required for empty databases")

        admin = User(
            username=username,
            password_hash=get_password_hash(password),
            password_ciphertext=None,
            role="admin",
            status="active",
            created_at=datetime.now(),
            updated_at=datetime.now(),
        )
        db.add(admin)
        db.commit()
    finally:
        db.close()
