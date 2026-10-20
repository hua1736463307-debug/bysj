"""PostgreSQL 数据库连接、系统管理模型与兼容迁移。"""
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Generator

from sqlalchemy import BigInteger, Boolean, Column, DateTime, Float, ForeignKey, Index, Integer, JSON, String, Text, UniqueConstraint, create_engine, text
from sqlalchemy.exc import OperationalError
from sqlalchemy.orm import Session, declarative_base, sessionmaker


DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "sqlite:///./bysj.db",
)

engine_options = {
    "pool_pre_ping": True,
    "pool_recycle": 3600,
    "future": True,
}
if not DATABASE_URL.startswith("sqlite"):
    engine_options.update({"pool_size": 10, "max_overflow": 20})
engine = create_engine(DATABASE_URL, **engine_options)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)
Base = declarative_base()


def _format_time(value):
    return value.strftime("%Y-%m-%d %H:%M:%S") if value else ""


class Department(Base):
    __tablename__ = "departments"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    parent_id = Column(BigInteger, nullable=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    sort_order = Column(Integer, nullable=False, default=0)
    leader = Column(String(64), nullable=True)
    phone = Column(String(32), nullable=True)
    email = Column(String(128), nullable=True)
    status = Column(String(16), nullable=False, default="active", index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=True, default=datetime.now, onupdate=datetime.now)

    def to_dict(self):
        return {
            "id": self.id,
            "parentId": self.parent_id,
            "name": self.name,
            "sortOrder": self.sort_order,
            "leader": self.leader or "",
            "phone": self.phone or "",
            "email": self.email or "",
            "status": self.status,
            "createdAt": _format_time(self.created_at),
        }


class User(Base):
    """保留原 users 表字段，同时扩展 RuoYi 风格管理信息。"""
    __tablename__ = "users"

    id = Column(BigInteger, primary_key=True, autoincrement=True, index=True)
    user_no = Column(String, unique=True, nullable=True, index=True)
    username = Column(String(64), unique=True, nullable=False, index=True)
    password_hash = Column(String(255), nullable=False)
    password_ciphertext = Column(Text, nullable=True)
    role = Column(String(32), nullable=False, default="user")
    status = Column(String(32), nullable=False, default="active")
    dept_id = Column(BigInteger, nullable=True, index=True)
    nickname = Column(String(64), nullable=True)
    phone = Column(String(32), nullable=True, index=True)
    email = Column(String(128), nullable=True)
    remark = Column(String(255), nullable=True)
    last_login_ip = Column(String(64), nullable=True)
    last_login_at = Column(DateTime, nullable=True)
    failed_login_attempts = Column(Integer, nullable=False, default=0)
    locked_until = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=True, default=datetime.now, onupdate=datetime.now)

    def to_dict(self):
        return {
            "id": self.id,
            "userNo": self.user_no or str(self.id),
            "username": self.username,
            "nickName": self.nickname or self.username,
            "deptId": self.dept_id,
            "phone": self.phone or "",
            "email": self.email or "",
            "remark": self.remark or "",
            "role": self.role,
            "status": self.status,
            "lastLoginIp": self.last_login_ip or "",
            "lastLoginAt": _format_time(self.last_login_at),
            "locked": bool(self.locked_until and self.locked_until > datetime.now()),
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "createTime": _format_time(self.created_at),
        }


class Role(Base):
    __tablename__ = "roles"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    role_code = Column(String(64), nullable=False, index=True)
    name = Column(String(100), nullable=False)
    role_key = Column(String(64), nullable=False, unique=True, index=True)
    sort_order = Column(Integer, nullable=False, default=0)
    authority_level = Column(Integer, nullable=False, default=100, index=True)
    data_scope = Column(String(16), nullable=False, default="self")
    status = Column(String(16), nullable=False, default="active", index=True)
    remark = Column(String(255), nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=True, default=datetime.now, onupdate=datetime.now)

    def to_dict(self):
        return {
            "id": self.id,
            "roleCode": self.role_code,
            "name": self.name,
            "roleKey": self.role_key,
            "sortOrder": self.sort_order,
            "authorityLevel": self.authority_level,
            "dataScope": self.data_scope,
            "status": self.status,
            "remark": self.remark or "",
            "createdAt": _format_time(self.created_at),
        }


class Menu(Base):
    __tablename__ = "menus"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    parent_id = Column(BigInteger, nullable=True, index=True)
    name = Column(String(100), nullable=False, index=True)
    path = Column(String(255), nullable=True)
    route_name = Column(String(100), nullable=True)
    query_params = Column(String(255), nullable=True)
    component = Column(String(255), nullable=True)
    permission = Column(String(128), nullable=True)
    icon = Column(String(64), nullable=True)
    menu_type = Column(String(16), nullable=False, default="menu")
    visible = Column(String(16), nullable=False, default="visible")
    status = Column(String(16), nullable=False, default="active", index=True)
    cache = Column(String(16), nullable=False, default="cache")
    sort_order = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=True, default=datetime.now, onupdate=datetime.now)

    def to_dict(self):
        return {
            "id": self.id,
            "parentId": self.parent_id,
            "name": self.name,
            "path": self.path or "",
            "routeName": self.route_name or "",
            "query": self.query_params or "",
            "component": self.component or "",
            "permission": self.permission or "",
            "icon": self.icon or "",
            "menuType": self.menu_type,
            "visible": self.visible,
            "status": self.status,
            "cache": self.cache,
            "sortOrder": self.sort_order,
            "createdAt": _format_time(self.created_at),
        }


class SystemSeedState(Base):
    __tablename__ = "system_seed_state"

    key = Column(String(64), primary_key=True)
    initialized_at = Column(DateTime, nullable=False, default=datetime.now)


class UserRole(Base):
    __tablename__ = "user_roles"
    __table_args__ = (UniqueConstraint("user_id", "role_id", name="uq_user_role"),)

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    user_id = Column(BigInteger, nullable=False, index=True)
    role_id = Column(BigInteger, nullable=False, index=True)


class RoleMenu(Base):
    __tablename__ = "role_menus"
    __table_args__ = (UniqueConstraint("role_id", "menu_id", name="uq_role_menu"),)

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    role_id = Column(BigInteger, nullable=False, index=True)
    menu_id = Column(BigInteger, nullable=False, index=True)


class RoleDepartment(Base):
    __tablename__ = "role_departments"
    __table_args__ = (UniqueConstraint("role_id", "department_id", name="uq_role_department"),)

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    role_id = Column(BigInteger, nullable=False, index=True)
    department_id = Column(BigInteger, nullable=False, index=True)


class OnlineSession(Base):
    __tablename__ = "online_sessions"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    session_id = Column(String(64), nullable=False, unique=True, index=True)
    user_id = Column(BigInteger, nullable=False, index=True)
    username = Column(String(64), nullable=False, index=True)
    ip_address = Column(String(64), nullable=True, index=True)
    location = Column(String(128), nullable=False, default="内网IP")
    browser = Column(String(128), nullable=True)
    os = Column(String(128), nullable=True)
    status = Column(String(16), nullable=False, default="active", index=True)
    login_time = Column(DateTime, nullable=False, default=datetime.now)
    last_active_at = Column(DateTime, nullable=False, default=datetime.now)
    expires_at = Column(DateTime, nullable=False)
    revoked_at = Column(DateTime, nullable=True)

    def to_dict(self):
        return {
            "id": self.id,
            "sessionId": self.session_id,
            "userId": self.user_id,
            "userName": self.username,
            "ip": self.ip_address or "",
            "location": self.location or "内网IP",
            "browser": self.browser or "",
            "os": self.os or "",
            "status": self.status,
            "loginTime": _format_time(self.login_time),
            "lastActiveAt": _format_time(self.last_active_at),
        }


class LoginLog(Base):
    __tablename__ = "login_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True, index=True)
    username = Column(String(64), nullable=False, index=True)
    ip_address = Column(String(64), nullable=True, index=True)
    location = Column(String(128), nullable=False, default="内网IP")
    browser = Column(String(128), nullable=True)
    os = Column(String(128), nullable=True)
    status = Column(String(16), nullable=False, index=True)
    message = Column(String(255), nullable=True)
    login_time = Column(DateTime, nullable=False, default=datetime.now, index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "userName": self.username,
            "ip": self.ip_address or "",
            "location": self.location or "",
            "browser": self.browser or "",
            "os": self.os or "",
            "status": self.status,
            "message": self.message or "",
            "loginTime": _format_time(self.login_time),
        }


class OperationLog(Base):
    __tablename__ = "operation_logs"

    id = Column(BigInteger, primary_key=True, autoincrement=True, index=True)
    title = Column(String(128), nullable=False, index=True)
    operation_type = Column(String(32), nullable=False, index=True)
    operator = Column(String(64), nullable=False, index=True)
    ip_address = Column(String(64), nullable=True, index=True)
    location = Column(String(128), nullable=False, default="内网IP")
    status = Column(String(16), nullable=False, index=True)
    detail = Column(String(255), nullable=True)
    request_method = Column(String(16), nullable=True)
    request_url = Column(String(512), nullable=True)
    request_params = Column(Text, nullable=True)
    response_data = Column(Text, nullable=True)
    error_message = Column(Text, nullable=True)
    session_id = Column(String(64), nullable=True, index=True)
    request_id = Column(String(128), nullable=True, index=True)
    reason = Column(String(64), nullable=True, index=True)
    user_agent = Column(Text, nullable=True)
    event_key = Column(String(192), nullable=True)
    cost_ms = Column(BigInteger, nullable=False, default=0)
    operation_time = Column(DateTime, nullable=False, default=datetime.now, index=True)

    def to_dict(self):
        return {
            "id": self.id,
            "title": self.title,
            "type": self.operation_type,
            "operator": self.operator,
            "ip": self.ip_address or "",
            "location": self.location or "",
            "status": self.status,
            "detail": self.detail or "",
            "sessionId": self.session_id or "",
            "requestId": self.request_id or "",
            "reason": self.reason or "",
            "userAgent": self.user_agent or "",
            "eventKey": self.event_key or "",
            "method": self.request_method or "",
            "url": self.request_url or "",
            "params": self.request_params or "",
            "result": self.response_data or "",
            "error": self.error_message or "",
            "operTime": _format_time(self.operation_time),
            "cost": f"{self.cost_ms}毫秒",
        }


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id = Column(String(36), primary_key=True)
    dataset_id = Column(String(255), nullable=False, index=True)
    parent_version_id = Column(String(36), nullable=True, index=True)
    dataset_role = Column(String(32), nullable=False, index=True)
    status = Column(String(32), nullable=False, default="ready", index=True)
    root_path = Column(Text, nullable=False)
    manifest_path = Column(Text, nullable=False)
    class_map_json = Column(JSON, nullable=False, default=dict)
    split_config_json = Column(JSON, nullable=False, default=dict)
    metadata_json = Column(JSON, nullable=False, default=dict)
    content_hash = Column(String(64), nullable=False, index=True)
    created_by = Column(String(64), nullable=False, default="system")
    created_at = Column(DateTime, nullable=False, default=datetime.now, index=True)

    __table_args__ = (
        UniqueConstraint("dataset_id", "dataset_role", "content_hash", name="uq_dataset_version_content"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "parent_version_id": self.parent_version_id,
            "dataset_role": self.dataset_role,
            "status": self.status,
            "root_path": self.root_path,
            "manifest_path": self.manifest_path,
            "class_map": self.class_map_json or {},
            "split_config": self.split_config_json or {},
            "metadata": self.metadata_json or {},
            "content_hash": self.content_hash,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class DatasetResource(Base):
    __tablename__ = "dataset_resources"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    dataset_id = Column(String(255), nullable=False, index=True)
    dataset_role = Column(String(32), nullable=False, index=True)
    owner_user_id = Column(BigInteger, nullable=True, index=True)
    owner_username = Column(String(64), nullable=False, default="system", index=True)
    owner_role_id = Column(BigInteger, nullable=True, index=True)
    visibility_mode = Column(String(32), nullable=False, default="role_and_higher")
    created_at = Column(DateTime, nullable=False, default=datetime.now)

    __table_args__ = (
        UniqueConstraint("dataset_id", "dataset_role", name="uq_dataset_resource_identity"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "dataset_id": self.dataset_id,
            "dataset_role": self.dataset_role,
            "owner_user_id": self.owner_user_id,
            "owner_username": self.owner_username,
            "owner_role_id": self.owner_role_id,
            "visibility_mode": self.visibility_mode,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class DatasetAsset(Base):
    __tablename__ = "dataset_assets"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    dataset_version_id = Column(String(36), nullable=False, index=True)
    relative_path = Column(Text, nullable=False)
    basename = Column(String(255), nullable=False, index=True)
    size = Column(BigInteger, nullable=False, default=0)
    width = Column(Integer, nullable=False, default=0)
    height = Column(Integer, nullable=False, default=0)
    label_path = Column(Text, nullable=True)
    source_group = Column(String(255), nullable=True, index=True)
    content_hash = Column(String(64), nullable=False, index=True)
    annotations_json = Column(JSON, nullable=False, default=list)
    created_at = Column(DateTime, nullable=False, default=datetime.now)

    __table_args__ = (
        UniqueConstraint("dataset_version_id", "relative_path", name="uq_dataset_asset_path"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "dataset_version_id": self.dataset_version_id,
            "relative_path": self.relative_path,
            "basename": self.basename,
            "size": self.size,
            "width": self.width,
            "height": self.height,
            "label_path": self.label_path,
            "source_group": self.source_group,
            "content_hash": self.content_hash,
            "annotations": self.annotations_json or [],
        }


class AnnotationDraft(Base):
    """Per-user, per-dataset, per-annotation-kind resumable annotation state."""
    __tablename__ = "annotation_drafts"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_user_id = Column(BigInteger, nullable=False, index=True)
    owner_username = Column(String(64), nullable=False, index=True)
    dataset_id = Column(String(255), nullable=False, index=True)
    source_version_id = Column(String(36), nullable=False, index=True)
    temporary_dataset_id = Column(String(255), nullable=True, index=True)
    annotation_kind = Column(String(32), nullable=False, index=True)
    model_path = Column(Text, nullable=False)
    model_name = Column(String(255), nullable=False)
    annotation_geometry = Column(String(16), nullable=False, default="rect")
    class_names_json = Column(JSON, nullable=False, default=list)
    class_bindings_json = Column(JSON, nullable=False, default=list)
    class_context_json = Column(JSON, nullable=True)
    last_page = Column(Integer, nullable=False, default=1)
    last_image_path = Column(Text, nullable=False, default="")
    revision = Column(Integer, nullable=False, default=0)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now, index=True)

    __table_args__ = (
        UniqueConstraint(
            "owner_user_id",
            "dataset_id",
            "annotation_kind",
            name="uq_annotation_draft_owner_dataset_kind",
        ),
        Index("ix_annotation_draft_dataset_kind", "dataset_id", "annotation_kind"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "owner_user_id": self.owner_user_id,
            "owner_username": self.owner_username,
            "dataset_id": self.dataset_id,
            "source_version_id": self.source_version_id,
            "temporary_dataset_id": self.temporary_dataset_id,
            "annotation_kind": self.annotation_kind,
            "model_path": self.model_path,
            "model_name": self.model_name,
            "annotation_geometry": self.annotation_geometry,
            "class_names": self.class_names_json or [],
            "class_bindings": self.class_bindings_json or [],
            "class_scope": (self.class_context_json or {}).get("scope", "dataset"),
            "class_context": self.class_context_json or {},
            "last_page": self.last_page,
            "last_image_path": self.last_image_path or "",
            "revision": self.revision,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class AnnotationDraftAsset(Base):
    """Incremental full-image snapshots belonging to an annotation draft."""
    __tablename__ = "annotation_draft_assets"

    id = Column(BigInteger().with_variant(Integer, "sqlite"), primary_key=True, autoincrement=True)
    draft_id = Column(String(36), ForeignKey("annotation_drafts.id", ondelete="CASCADE"), nullable=False, index=True)
    relative_path = Column(Text, nullable=False)
    width = Column(Integer, nullable=False, default=0)
    height = Column(Integer, nullable=False, default=0)
    annotations_json = Column(JSON, nullable=False, default=list)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now, index=True)

    __table_args__ = (
        UniqueConstraint("draft_id", "relative_path", name="uq_annotation_draft_asset_path"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "draft_id": self.draft_id,
            "relative_path": self.relative_path,
            "width": self.width,
            "height": self.height,
            "annotations": self.annotations_json or [],
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class AlgorithmCatalog(Base):
    __tablename__ = 'monitoring_algorithm_catalog'
    id = Column(String(32), primary_key=True, default='global')
    config_json = Column(JSON, nullable=False, default=dict)
    revision = Column(Integer, nullable=False, default=1)
    updated_by = Column(String(64), nullable=False)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)


class RecorderConfig(Base):
    """Vendor-neutral NVR configuration; recordings remain on the recorder."""
    __tablename__ = "recorder_configs"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id = Column(String(64), nullable=False, index=True)
    config_json = Column(JSON, nullable=False, default=dict)
    password_ciphertext = Column(Text, nullable=False)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)


class ContinuousMonitor(Base):
    """One reusable, server-owned monitoring plan per user; never stores frames."""
    __tablename__ = "continuous_monitors"
    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id = Column(String(64), nullable=False, unique=True)
    owner_username = Column(String(64), nullable=False)
    name = Column(String(100), nullable=False, default="持续监控")
    desired_state = Column(String(16), nullable=False, default="stopped", index=True)
    status = Column(String(24), nullable=False, default="stopped")
    config_json = Column(JSON, nullable=False, default=dict)
    result_json = Column(JSON, nullable=False, default=dict)
    heartbeat_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)


class JobTask(Base):
    __tablename__ = "job_tasks"

    id = Column(String(36), primary_key=True)
    type = Column(String(32), nullable=False, index=True)
    status = Column(String(32), nullable=False, default="pending", index=True)
    stage = Column(String(64), nullable=False, default="queued")
    progress = Column(Float, nullable=False, default=0.0)
    message = Column(Text, nullable=False, default="")
    owner_id = Column(String(64), nullable=True, index=True)
    owner_username = Column(String(64), nullable=False, index=True)
    worker_pid = Column(Integer, nullable=True)
    cancel_requested = Column(Boolean, nullable=False, default=False, index=True)
    error_code = Column(String(64), nullable=True)
    error_detail = Column(Text, nullable=True)
    input_json = Column(JSON, nullable=False, default=dict)
    result_json = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, default=datetime.now, index=True)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    heartbeat_at = Column(DateTime, nullable=True, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)

    def to_dict(self):
        return {
            "id": self.id,
            "task_id": self.id,
            "type": self.type,
            "status": self.status,
            "stage": self.stage,
            "progress": float(self.progress or 0.0),
            "message": self.message or "",
            "owner_id": self.owner_id,
            "owner_username": self.owner_username,
            "worker_pid": self.worker_pid,
            "cancel_requested": bool(self.cancel_requested),
            "error_code": self.error_code,
            "error_detail": self.error_detail,
            "input": self.input_json or {},
            "result": self.result_json or {},
            "results": self.result_json or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "heartbeat_at": self.heartbeat_at.isoformat() if self.heartbeat_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class JobEvent(Base):
    __tablename__ = "job_events"

    id = Column(BigInteger, primary_key=True, autoincrement=True)
    task_id = Column(String(36), nullable=False, index=True)
    sequence = Column(Integer, nullable=False)
    level = Column(String(16), nullable=False, default="info")
    event_type = Column(String(64), nullable=False, default="progress", index=True)
    message = Column(Text, nullable=False, default="")
    payload_json = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, default=datetime.now, index=True)

    __table_args__ = (
        UniqueConstraint("task_id", "sequence", name="uq_job_event_sequence"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "task_id": self.task_id,
            "sequence": self.sequence,
            "level": self.level,
            "event_type": self.event_type,
            "message": self.message or "",
            "payload": self.payload_json or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class BehaviorEvent(Base):
    __tablename__ = "behavior_events"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    event_id = Column(String(36), nullable=False, unique=True, index=True)
    owner_id = Column(BigInteger, nullable=True, index=True)
    owner_username = Column(String(64), nullable=False, default="system", index=True)
    mode = Column(String(16), nullable=False, index=True)
    source_id = Column(String(255), nullable=False, index=True)
    session_id = Column(String(128), nullable=True, index=True)
    grid_id = Column(String(128), nullable=True, index=True)
    grid_position = Column(String(32), nullable=True)
    event_type = Column(String(64), nullable=False, index=True)
    state = Column(String(32), nullable=False, default="confirmed", index=True)
    severity = Column(String(16), nullable=False, default="low", index=True)
    rule_score = Column(Float, nullable=False, default=0.0)
    should_alert = Column(Boolean, nullable=False, default=False)
    alerted_at = Column(DateTime, nullable=True, index=True)
    handling_status = Column(String(32), nullable=False, default="pending", index=True)
    handling_note = Column(Text, nullable=False, default="")
    handled_by = Column(String(64), nullable=True, index=True)
    handled_at = Column(DateTime, nullable=True)
    feedback_verdict = Column(String(32), nullable=True, index=True)
    feedback_label = Column(String(64), nullable=True, index=True)
    feedback_note = Column(Text, nullable=False, default="")
    feedback_by = Column(String(64), nullable=True, index=True)
    feedback_at = Column(DateTime, nullable=True)
    decision_model_id = Column(String(36), nullable=True, index=True)
    decision_model_version = Column(String(255), nullable=True)
    decision_json = Column(JSON, nullable=False, default=dict)
    persistence_status = Column(String(32), nullable=False, default="persisted", index=True)
    track_ids_json = Column(JSON, nullable=False, default=list)
    object_json = Column(JSON, nullable=False, default=dict)
    reason_json = Column(JSON, nullable=False, default=dict)
    started_at = Column(DateTime, nullable=False, index=True)
    confirmed_at = Column(DateTime, nullable=False, index=True)
    ended_at = Column(DateTime, nullable=True, index=True)
    snapshot_path = Column(Text, nullable=True)
    clip_path = Column(Text, nullable=True)
    evidence_status = Column(String(32), nullable=False, default="collecting")
    evidence_error = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.now, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)

    __table_args__ = (
        Index("ix_behavior_events_owner_confirmed", "owner_id", "confirmed_at"),
        Index("ix_behavior_events_source_confirmed", "source_id", "confirmed_at"),
        Index("ix_behavior_events_type_confirmed", "event_type", "confirmed_at"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "event_id": self.event_id,
            "owner_id": self.owner_id,
            "owner_username": self.owner_username,
            "mode": self.mode,
            "source_id": self.source_id,
            "session_id": self.session_id,
            "grid_id": self.grid_id,
            "grid_position": self.grid_position,
            "event_type": self.event_type,
            "state": self.state,
            "severity": self.severity,
            "rule_score": float(self.rule_score or 0.0),
            "confidence": float(self.rule_score or 0.0),
            "should_alert": bool(self.should_alert),
            "alerted_at": self.alerted_at.isoformat() if self.alerted_at else None,
            "handling_status": self.handling_status or "pending",
            "handling_note": self.handling_note or "",
            "handled_by": self.handled_by,
            "handled_at": self.handled_at.isoformat() if self.handled_at else None,
            "feedback_verdict": self.feedback_verdict,
            "feedback_label": self.feedback_label,
            "feedback_note": self.feedback_note or "",
            "feedback_by": self.feedback_by,
            "feedback_at": self.feedback_at.isoformat() if self.feedback_at else None,
            "decision_model_id": self.decision_model_id,
            "decision_model_version": self.decision_model_version,
            "decision": self.decision_json or {},
            "persistence_status": self.persistence_status or "persisted",
            "track_ids": self.track_ids_json or [],
            "object": self.object_json or {},
            "reason": self.reason_json or {},
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "confirmed_at": self.confirmed_at.isoformat() if self.confirmed_at else None,
            "ended_at": self.ended_at.isoformat() if self.ended_at else None,
            "evidence_status": self.evidence_status,
            "evidence_error": self.evidence_error,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class AlarmPushDelivery(Base):
    __tablename__ = "alarm_push_deliveries"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    event_id = Column(String(36), nullable=False, unique=True, index=True)
    destination_url = Column(String(512), nullable=False)
    status = Column(String(32), nullable=False, default="pending", index=True)
    attempt_count = Column(Integer, nullable=False, default=0)
    next_attempt_at = Column(DateTime, nullable=False, default=datetime.now, index=True)
    last_attempt_at = Column(DateTime, nullable=True)
    delivered_at = Column(DateTime, nullable=True, index=True)
    last_response_code = Column(Integer, nullable=True)
    last_error = Column(Text, nullable=True)
    created_at = Column(DateTime, nullable=False, default=datetime.now, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)

    __table_args__ = (
        Index("ix_alarm_push_deliveries_due", "status", "next_attempt_at"),
    )


class BehaviorEventAction(Base):
    __tablename__ = "behavior_event_actions"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    event_id = Column(String(36), nullable=False, index=True)
    owner_id = Column(BigInteger, nullable=True, index=True)
    action_type = Column(String(32), nullable=False, index=True)
    from_value_json = Column(JSON, nullable=False, default=dict)
    to_value_json = Column(JSON, nullable=False, default=dict)
    note = Column(Text, nullable=False, default="")
    actor_username = Column(String(64), nullable=False, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.now, index=True)

    __table_args__ = (
        Index("ix_behavior_event_actions_event_created", "event_id", "created_at"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "event_id": self.event_id,
            "owner_id": self.owner_id,
            "action_type": self.action_type,
            "from_value": self.from_value_json or {},
            "to_value": self.to_value_json or {},
            "note": self.note or "",
            "actor_username": self.actor_username,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }


class BehaviorReviewSample(Base):
    __tablename__ = "behavior_review_samples"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id = Column(BigInteger, nullable=True, index=True)
    owner_username = Column(String(64), nullable=False, index=True)
    source_task_id = Column(String(128), nullable=False, index=True)
    source_event_id = Column(String(128), nullable=False)
    source_event_type = Column(String(64), nullable=True, index=True)
    source_kind = Column(String(32), nullable=False, default="batch_review", index=True)
    behavior_event_id = Column(String(36), nullable=True, index=True)
    source_id = Column(String(255), nullable=True, index=True)
    model_family = Column(String(64), nullable=True, index=True)
    content_hash = Column(String(64), nullable=True, index=True)
    review_label = Column(String(64), nullable=False, index=True)
    review_status = Column(String(32), nullable=False, default="reviewed", index=True)
    sample_type = Column(String(16), nullable=False, default="clip")
    clip_path = Column(Text, nullable=True)
    image_path = Column(Text, nullable=True)
    evidence_json = Column(JSON, nullable=False, default=dict)
    note = Column(Text, nullable=False, default="")
    reviewed_by = Column(String(64), nullable=False, index=True)
    reviewed_at = Column(DateTime, nullable=False, default=datetime.now, index=True)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)

    __table_args__ = (
        UniqueConstraint(
            "owner_username",
            "source_task_id",
            "source_event_id",
            name="uq_behavior_review_sample_source",
        ),
        Index("ix_behavior_review_owner_label", "owner_username", "review_label"),
    )

    def to_training_sample(self):
        evidence = self.evidence_json or {}
        return {
            "id": self.id,
            "sample_type": self.sample_type,
            "clip_path": self.clip_path,
            "image_path": self.image_path,
            "label": self.review_label,
            "source_task_id": self.source_task_id,
            "source_event_id": self.source_event_id,
            "source_event_type": self.source_event_type,
            "source_kind": self.source_kind or "batch_review",
            "behavior_event_id": self.behavior_event_id,
            "source_id": self.source_id,
            "model_family": self.model_family,
            "content_hash": self.content_hash,
            "owner_username": self.owner_username,
            "reviewed_by": self.reviewed_by,
            "reviewed_at": self.reviewed_at.isoformat() if self.reviewed_at else None,
            "note": self.note or "",
            "evidence": evidence,
        }

    def to_dict(self):
        return {
            **self.to_training_sample(),
            "owner_id": self.owner_id,
            "review_status": self.review_status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class BehaviorRuleProfile(Base):
    __tablename__ = "behavior_rule_profiles"

    id = Column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    owner_id = Column(BigInteger, nullable=False, index=True)
    owner_username = Column(String(64), nullable=False, index=True)
    scope = Column(String(16), nullable=False, default="default", index=True)
    source_id = Column(String(255), nullable=False, default="", index=True)
    config_json = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, default=datetime.now)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)

    __table_args__ = (
        UniqueConstraint("owner_id", "scope", "source_id", name="uq_behavior_rule_profile_scope"),
    )

    def to_dict(self):
        return {
            "id": self.id,
            "owner_id": self.owner_id,
            "owner_username": self.owner_username,
            "scope": self.scope,
            "source_id": self.source_id or None,
            "config": self.config_json or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
        }


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id = Column(String(36), primary_key=True)
    name = Column(String(255), nullable=False, index=True)
    path = Column(Text, nullable=False, unique=True)
    sha256 = Column(String(64), nullable=False, index=True)
    size = Column(BigInteger, nullable=False, default=0)
    source = Column(String(32), nullable=False, index=True)
    status = Column(String(32), nullable=False, default="active", index=True)
    format = Column(String(32), nullable=False, default="pt")
    task_id = Column(String(36), nullable=True, index=True)
    dataset_version_id = Column(String(36), nullable=True, index=True)
    created_by = Column(String(64), nullable=False, default="system")
    metadata_json = Column(JSON, nullable=False, default=dict)
    created_at = Column(DateTime, nullable=False, default=datetime.now, index=True)
    updated_at = Column(DateTime, nullable=False, default=datetime.now, onupdate=datetime.now)

    def to_dict(self):
        metadata = self.metadata_json or {}
        timestamp = self.updated_at.timestamp() if self.updated_at else 0
        return {
            "id": self.id,
            "name": self.name,
            "type": self.source,
            "source": self.source,
            "path": self.path,
            "sha256": self.sha256,
            "size": self.size,
            "status": self.status,
            "format": self.format,
            "task_id": self.task_id,
            "dataset_version_id": self.dataset_version_id,
            "created_by": self.created_by,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "updated_at": self.updated_at.isoformat() if self.updated_at else None,
            "created_time": self.created_at.timestamp() if self.created_at else timestamp,
            "modified_time": timestamp,
            "display_name": metadata.get("display_name") or Path(self.name).stem.replace("_", " "),
            **metadata,
        }


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


USER_COMPAT_COLUMNS = (
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS user_no TEXT",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS dept_id BIGINT",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS nickname VARCHAR(64)",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS phone VARCHAR(32)",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS email VARCHAR(128)",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS remark VARCHAR(255)",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login_ip VARCHAR(64)",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS last_login_at TIMESTAMP",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS failed_login_attempts INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS locked_until TIMESTAMP",
    "ALTER TABLE users ADD COLUMN IF NOT EXISTS password_ciphertext TEXT",
)

OPERATION_LOG_COMPAT_COLUMNS = (
    "ALTER TABLE operation_logs ADD COLUMN IF NOT EXISTS request_method VARCHAR(16)",
    "ALTER TABLE operation_logs ADD COLUMN IF NOT EXISTS request_url VARCHAR(512)",
    "ALTER TABLE operation_logs ADD COLUMN IF NOT EXISTS request_params TEXT",
    "ALTER TABLE operation_logs ADD COLUMN IF NOT EXISTS response_data TEXT",
    "ALTER TABLE operation_logs ADD COLUMN IF NOT EXISTS error_message TEXT",
)

ROLE_COMPAT_COLUMNS = (
    "ALTER TABLE roles ADD COLUMN IF NOT EXISTS role_code VARCHAR(64)",
    "ALTER TABLE roles ADD COLUMN IF NOT EXISTS authority_level INTEGER NOT NULL DEFAULT 100",
)

MENU_COMPAT_COLUMNS = (
    "ALTER TABLE menus ADD COLUMN IF NOT EXISTS route_name VARCHAR(100)",
    "ALTER TABLE menus ADD COLUMN IF NOT EXISTS query_params VARCHAR(255)",
)

DATASET_VERSION_COMPAT_COLUMNS = (
    "ALTER TABLE dataset_versions ADD COLUMN IF NOT EXISTS metadata_json JSONB NOT NULL DEFAULT '{}'::jsonb",
    "ALTER TABLE dataset_versions DROP CONSTRAINT IF EXISTS uq_dataset_version_content",
    "ALTER TABLE dataset_versions ADD CONSTRAINT uq_dataset_version_content UNIQUE (dataset_id, dataset_role, content_hash)",
)

BEHAVIOR_EVENT_COMPAT_COLUMNS = (
    "ALTER TABLE behavior_events ADD COLUMN IF NOT EXISTS alerted_at TIMESTAMP",
    "ALTER TABLE behavior_events ADD COLUMN IF NOT EXISTS handling_status VARCHAR(32) NOT NULL DEFAULT 'pending'",
    "ALTER TABLE behavior_events ADD COLUMN IF NOT EXISTS handling_note TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE behavior_events ADD COLUMN IF NOT EXISTS handled_by VARCHAR(64)",
    "ALTER TABLE behavior_events ADD COLUMN IF NOT EXISTS handled_at TIMESTAMP",
    "ALTER TABLE behavior_events ADD COLUMN IF NOT EXISTS feedback_verdict VARCHAR(32)",
    "ALTER TABLE behavior_events ADD COLUMN IF NOT EXISTS feedback_label VARCHAR(64)",
    "ALTER TABLE behavior_events ADD COLUMN IF NOT EXISTS feedback_note TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE behavior_events ADD COLUMN IF NOT EXISTS feedback_by VARCHAR(64)",
    "ALTER TABLE behavior_events ADD COLUMN IF NOT EXISTS feedback_at TIMESTAMP",
    "ALTER TABLE behavior_events ADD COLUMN IF NOT EXISTS decision_model_id VARCHAR(36)",
    "ALTER TABLE behavior_events ADD COLUMN IF NOT EXISTS decision_model_version VARCHAR(255)",
    "ALTER TABLE behavior_events ADD COLUMN IF NOT EXISTS decision_json JSONB NOT NULL DEFAULT '{}'::jsonb",
    "ALTER TABLE behavior_events ADD COLUMN IF NOT EXISTS persistence_status VARCHAR(32) NOT NULL DEFAULT 'legacy_unknown'",
    "CREATE INDEX IF NOT EXISTS ix_behavior_events_handling_status ON behavior_events (handling_status)",
    "CREATE INDEX IF NOT EXISTS ix_behavior_events_feedback_verdict ON behavior_events (feedback_verdict)",
    "CREATE INDEX IF NOT EXISTS ix_behavior_events_decision_model_id ON behavior_events (decision_model_id)",
    "CREATE INDEX IF NOT EXISTS ix_behavior_events_alerted_at ON behavior_events (alerted_at)",
    "CREATE INDEX IF NOT EXISTS ix_behavior_events_persistence_status ON behavior_events (persistence_status)",
    "CREATE INDEX IF NOT EXISTS ix_behavior_events_alarm_center ON behavior_events (should_alert, confirmed_at DESC)",
    "CREATE INDEX IF NOT EXISTS ix_behavior_events_source_confirmed ON behavior_events (source_id, confirmed_at)",
)

ALARM_PUSH_DELIVERY_COMPAT_COLUMNS = (
    "CREATE TABLE IF NOT EXISTS alarm_push_deliveries ("
    "id VARCHAR(36) PRIMARY KEY, event_id VARCHAR(36) NOT NULL UNIQUE, "
    "destination_url VARCHAR(512) NOT NULL, status VARCHAR(32) NOT NULL DEFAULT 'pending', "
    "attempt_count INTEGER NOT NULL DEFAULT 0, next_attempt_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, "
    "last_attempt_at TIMESTAMP, delivered_at TIMESTAMP, last_response_code INTEGER, last_error TEXT, "
    "created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP)",
    "CREATE INDEX IF NOT EXISTS ix_alarm_push_deliveries_due ON alarm_push_deliveries (status, next_attempt_at)",
)

BEHAVIOR_REVIEW_COMPAT_COLUMNS = (
    "ALTER TABLE behavior_review_samples ADD COLUMN IF NOT EXISTS source_kind VARCHAR(32) NOT NULL DEFAULT 'batch_review'",
    "ALTER TABLE behavior_review_samples ADD COLUMN IF NOT EXISTS behavior_event_id VARCHAR(36)",
    "ALTER TABLE behavior_review_samples ADD COLUMN IF NOT EXISTS source_id VARCHAR(255)",
    "ALTER TABLE behavior_review_samples ADD COLUMN IF NOT EXISTS model_family VARCHAR(64)",
    "ALTER TABLE behavior_review_samples ADD COLUMN IF NOT EXISTS content_hash VARCHAR(64)",
    "CREATE INDEX IF NOT EXISTS ix_behavior_review_event_id ON behavior_review_samples (behavior_event_id)",
    "CREATE INDEX IF NOT EXISTS ix_behavior_review_source_kind ON behavior_review_samples (source_kind)",
    "CREATE INDEX IF NOT EXISTS ix_behavior_review_content_hash ON behavior_review_samples (content_hash)",
)

ANNOTATION_DRAFT_COMPAT_COLUMNS = (
    "CREATE TABLE IF NOT EXISTS annotation_drafts ("
    "id VARCHAR(36) PRIMARY KEY, owner_user_id BIGINT NOT NULL, owner_username VARCHAR(64) NOT NULL, "
    "dataset_id VARCHAR(255) NOT NULL, source_version_id VARCHAR(36) NOT NULL, temporary_dataset_id VARCHAR(255), annotation_kind VARCHAR(32) NOT NULL, "
    "model_path TEXT NOT NULL, model_name VARCHAR(255) NOT NULL, annotation_geometry VARCHAR(16) NOT NULL DEFAULT 'rect', "
    "class_names_json JSONB NOT NULL DEFAULT '[]'::jsonb, class_bindings_json JSONB NOT NULL DEFAULT '[]'::jsonb, "
    "last_page INTEGER NOT NULL DEFAULT 1, last_image_path TEXT NOT NULL DEFAULT '', revision INTEGER NOT NULL DEFAULT 0, "
    "created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP) ",
    "CREATE TABLE IF NOT EXISTS annotation_draft_assets ("
    "id BIGSERIAL PRIMARY KEY, draft_id VARCHAR(36) NOT NULL REFERENCES annotation_drafts(id) ON DELETE CASCADE, "
    "relative_path TEXT NOT NULL, width INTEGER NOT NULL DEFAULT 0, height INTEGER NOT NULL DEFAULT 0, "
    "annotations_json JSONB NOT NULL DEFAULT '[]'::jsonb, updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP, "
    "CONSTRAINT uq_annotation_draft_asset_path UNIQUE (draft_id, relative_path))",
    "CREATE INDEX IF NOT EXISTS ix_annotation_draft_assets_draft_id ON annotation_draft_assets (draft_id)",
    "ALTER TABLE annotation_drafts ADD COLUMN IF NOT EXISTS owner_user_id BIGINT",
    "ALTER TABLE annotation_drafts ADD COLUMN IF NOT EXISTS source_version_id VARCHAR(36)",
    "ALTER TABLE annotation_drafts ADD COLUMN IF NOT EXISTS temporary_dataset_id VARCHAR(255)",
    "ALTER TABLE annotation_drafts ADD COLUMN IF NOT EXISTS annotation_kind VARCHAR(32)",
    "ALTER TABLE annotation_drafts ADD COLUMN IF NOT EXISTS model_path TEXT",
    "ALTER TABLE annotation_drafts ADD COLUMN IF NOT EXISTS model_name VARCHAR(255)",
    "ALTER TABLE annotation_drafts ADD COLUMN IF NOT EXISTS annotation_geometry VARCHAR(16) DEFAULT 'rect'",
    "ALTER TABLE annotation_drafts ADD COLUMN IF NOT EXISTS class_names_json JSONB NOT NULL DEFAULT '[]'::jsonb",
    "ALTER TABLE annotation_drafts ADD COLUMN IF NOT EXISTS class_bindings_json JSONB NOT NULL DEFAULT '[]'::jsonb",
    "ALTER TABLE annotation_drafts ADD COLUMN IF NOT EXISTS class_context_json JSONB",
    "ALTER TABLE annotation_drafts ADD COLUMN IF NOT EXISTS last_page INTEGER NOT NULL DEFAULT 1",
    "ALTER TABLE annotation_drafts ADD COLUMN IF NOT EXISTS last_image_path TEXT NOT NULL DEFAULT ''",
    "ALTER TABLE annotation_drafts ADD COLUMN IF NOT EXISTS revision INTEGER NOT NULL DEFAULT 0",
    "ALTER TABLE annotation_drafts ADD COLUMN IF NOT EXISTS created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP",
    "ALTER TABLE annotation_drafts ADD COLUMN IF NOT EXISTS updated_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP",
    "ALTER TABLE annotation_drafts ALTER COLUMN annotation_kind SET DEFAULT 'yolo'",
    "ALTER TABLE annotation_drafts ALTER COLUMN model_path SET DEFAULT ''",
    "ALTER TABLE annotation_drafts ALTER COLUMN model_name SET DEFAULT ''",
    # The rollback baseline may have the earlier owner_username/model_key/payload_json
    # draft table. Keep those legacy columns insertable while the new ORM contract
    # uses owner_user_id + annotation_kind and its own unique index.
    "DO $$ BEGIN IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'annotation_drafts' AND column_name = 'model_key') THEN ALTER TABLE annotation_drafts ALTER COLUMN model_key SET DEFAULT ''; END IF; END $$",
    "DO $$ BEGIN IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'annotation_drafts' AND column_name = 'payload_json') THEN ALTER TABLE annotation_drafts ALTER COLUMN payload_json SET DEFAULT '{}'::jsonb; END IF; END $$",
    "ALTER TABLE annotation_drafts DROP CONSTRAINT IF EXISTS uq_annotation_draft_owner_dataset_model",
    "UPDATE annotation_drafts d SET owner_user_id = u.id FROM users u WHERE d.owner_user_id IS NULL AND d.owner_username = u.username",
    "UPDATE annotation_drafts SET annotation_kind = 'yolo' WHERE annotation_kind IS NULL",
    "DO $$ BEGIN IF EXISTS (SELECT 1 FROM information_schema.columns WHERE table_name = 'annotation_drafts' AND column_name = 'model_key') THEN UPDATE annotation_drafts SET model_path = COALESCE(NULLIF(model_path, ''), model_key, '') WHERE model_path IS NULL; UPDATE annotation_drafts SET model_name = COALESCE(NULLIF(model_name, ''), model_key, '') WHERE model_name IS NULL; END IF; END $$",
    "CREATE UNIQUE INDEX IF NOT EXISTS uq_annotation_draft_owner_dataset_kind ON annotation_drafts (owner_user_id, dataset_id, annotation_kind)",
    "CREATE INDEX IF NOT EXISTS ix_annotation_draft_dataset_kind ON annotation_drafts (dataset_id, annotation_kind)",
    "CREATE INDEX IF NOT EXISTS ix_annotation_drafts_temporary_dataset_id ON annotation_drafts (temporary_dataset_id)",
)


def _run_compat_migrations() -> None:
    """为已部署数据库补列；全部语句均为 PostgreSQL 幂等迁移。"""
    with engine.begin() as conn:
        for statement in USER_COMPAT_COLUMNS:
            conn.execute(text(statement))
        for statement in OPERATION_LOG_COMPAT_COLUMNS:
            conn.execute(text(statement))
        for statement in ROLE_COMPAT_COLUMNS:
            conn.execute(text(statement))
        for statement in MENU_COMPAT_COLUMNS:
            conn.execute(text(statement))
        for statement in DATASET_VERSION_COMPAT_COLUMNS:
            conn.execute(text(statement))
        for statement in BEHAVIOR_EVENT_COMPAT_COLUMNS:
            conn.execute(text(statement))
        for statement in ALARM_PUSH_DELIVERY_COMPAT_COLUMNS:
            conn.execute(text(statement))
        for statement in BEHAVIOR_REVIEW_COMPAT_COLUMNS:
            conn.execute(text(statement))
        for statement in ANNOTATION_DRAFT_COMPAT_COLUMNS:
            conn.execute(text(statement))
        conn.execute(text("ALTER TABLE users ALTER COLUMN user_no TYPE TEXT"))
        conn.execute(text("UPDATE users SET user_no = CAST(id AS TEXT) WHERE user_no IS NULL OR BTRIM(user_no) = ''"))
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_users_user_no ON users (user_no)"))
        conn.execute(text("UPDATE roles SET role_code = CAST(id AS TEXT) WHERE role_code IS NULL OR BTRIM(role_code) = ''"))
        conn.execute(text("ALTER TABLE roles ALTER COLUMN role_code SET NOT NULL"))
        conn.execute(text("CREATE UNIQUE INDEX IF NOT EXISTS uq_roles_role_code ON roles (role_code)"))


def init_db(max_retries: int = 30, interval_seconds: int = 2) -> None:
    last_error = None
    for _attempt in range(1, max_retries + 1):
        try:
            with engine.connect() as conn:
                conn.execute(text("SELECT 1"))
            Base.metadata.create_all(bind=engine)
            _run_compat_migrations()
            return
        except OperationalError as exc:
            last_error = exc
            time.sleep(interval_seconds)

    raise RuntimeError(f"数据库初始化失败，PostgreSQL 连接不可用: {last_error}")


DEFAULT_MENUS = (
    (None, "首页", "/", "House", "menu", 1, "active"),
    (None, "原始数据集", "/raw-dataset", "Histogram", "menu", 2, "active"),
    (None, "数据标注", "/dataset-annotation", "EditPen", "menu", 3, "active"),
    (None, "训练数据集", "/train-dataset", "DocumentCopy", "menu", 4, "active"),
    (None, "模型训练", "/model-training", "Setting", "menu", 5, "active"),
    (None, "模型中心", "/model-hub", "Box", "menu", 6, "active"),
    (None, "监控管理", "/monitoring", "View", "directory", 7, "active"),
    ("监控管理", "监控配置", "/monitoring/config", "Setting", "menu", 1, "active"),
    ("监控管理", "实时监控", "/monitoring/live", "Monitor", "menu", 2, "active"),
    ("监控管理", "推理服务（旧版）", "/inference-service", "View", "menu", 3, "disabled"),
    ("监控管理", "告警中心", "/alarm-center", "Bell", "menu", 4, "active"),
    ("监控管理", "告警算法配置", "/system/algorithms", "DataAnalysis", "menu", 6, "active"),
    (None, "行为分析", "/behavior", "VideoCamera", "directory", 8, "active"),
    ("行为分析", "行为分析", "/behavior", "VideoCamera", "menu", 1, "disabled"),
    ("行为分析", "算法优化", "/algorithm-training", "DataAnalysis", "menu", 3, "active"),
    (None, "平台管理", "/system", "Setting", "directory", 9, "active"),
    ("平台管理", "部门管理", "/system/dept", "Connection", "menu", 1, "active"),
    ("平台管理", "用户管理", "/system/user", "UserFilled", "menu", 2, "active"),
    ("平台管理", "角色管理", "/system/role", "Avatar", "menu", 3, "active"),
    ("平台管理", "菜单管理", "/system/menu", "Menu", "menu", 4, "active"),
    ("平台管理", "在线用户", "/system/online", "Monitor", "menu", 5, "active"),
    ("平台管理", "日志管理", "/system/log", "Tickets", "directory", 6, "active"),
    ("日志管理", "操作日志", "/system/log/operlog", "Document", "menu", 1, "active"),
    ("日志管理", "登录日志", "/system/log/logininfor", "DocumentChecked", "menu", 2, "active"),
    ("日志管理", "系统故障日志", "/system/log/faults", "Warning", "menu", 3, "active"),
)

ROOT_MENU_SORT_ORDERS = {
    "首页": 1,
    "原始数据集": 2,
    "数据标注": 3,
    "训练数据集": 4,
    "模型训练": 5,
    "模型中心": 6,
    "监控管理": 7,
    "行为分析": 8,
    "平台管理": 9,
}

MENU_SEED_STATE_KEY = "default_menus_v1"


def _reconcile_legacy_behavior_menus(db: Session) -> None:
    """Merge the obsolete behavior root into the current behavior group once."""
    group = (
        db.query(Menu)
        .filter(Menu.parent_id.is_(None), Menu.name == "行为分析模块", Menu.menu_type == "directory")
        .order_by(Menu.id.asc())
        .first()
    )
    if not group:
        return

    legacy_roots = (
        db.query(Menu)
        .filter(Menu.parent_id.is_(None), Menu.name == "行为分析", Menu.menu_type == "directory")
        .order_by(Menu.id.asc())
        .all()
    )
    if not legacy_roots:
        return

    canonical_behavior = (
        db.query(Menu)
        .filter(Menu.parent_id == group.id, Menu.path == "/behavior", Menu.menu_type == "menu")
        .order_by(Menu.id.asc())
        .first()
    )
    if canonical_behavior:
        canonical_behavior.name = "行为分析"
        canonical_behavior.icon = canonical_behavior.icon or "VideoCamera"
        canonical_behavior.status = "active"
        canonical_behavior.visible = "visible"
        canonical_behavior.sort_order = 1
    else:
        canonical_behavior = Menu(
            parent_id=group.id,
            name="行为分析",
            path="/behavior",
            icon="VideoCamera",
            menu_type="menu",
            sort_order=1,
            status="active",
            visible="visible",
        )
        db.add(canonical_behavior)
        db.flush()

    for legacy_root in legacy_roots:
        for child in db.query(Menu).filter(Menu.parent_id == legacy_root.id).all():
            duplicate = (
                db.query(Menu)
                .filter(
                    Menu.id != child.id,
                    Menu.parent_id == group.id,
                    Menu.path == child.path,
                    Menu.menu_type == child.menu_type,
                )
                .first()
            )
            target = duplicate or child
            if duplicate:
                for link in db.query(RoleMenu).filter(RoleMenu.menu_id == child.id).all():
                    exists = db.query(RoleMenu).filter(
                        RoleMenu.role_id == link.role_id,
                        RoleMenu.menu_id == duplicate.id,
                    ).first()
                    if not exists:
                        db.add(RoleMenu(role_id=link.role_id, menu_id=duplicate.id))
                db.query(RoleMenu).filter(RoleMenu.menu_id == child.id).delete(synchronize_session=False)
                db.delete(child)
            else:
                child.parent_id = group.id
            if target.path == "/alarm-center":
                target.sort_order = 2
            elif target.path == "/algorithm-training":
                target.sort_order = 3

        for link in db.query(RoleMenu).filter(RoleMenu.menu_id == legacy_root.id).all():
            exists = db.query(RoleMenu).filter(
                RoleMenu.role_id == link.role_id,
                RoleMenu.menu_id == group.id,
            ).first()
            if not exists:
                db.add(RoleMenu(role_id=link.role_id, menu_id=group.id))
        db.query(RoleMenu).filter(RoleMenu.menu_id == legacy_root.id).delete(synchronize_session=False)
        db.delete(legacy_root)

    db.flush()


def seed_default_menus_if_uninitialized(db: Session) -> bool:
    """Create the built-in menu tree only for a brand-new menu table.

    Once any menu exists, absence is treated as an intentional administrator
    deletion (or an explicitly managed custom configuration), not as a reason
    to recreate the default menu on every backend restart.
    """
    if db.query(SystemSeedState).filter(SystemSeedState.key == MENU_SEED_STATE_KEY).first():
        return False

    menu_table_has_rows = db.query(Menu.id).first() is not None
    menu_by_name = {}
    if not menu_table_has_rows:
        for parent_name, name, path, icon, menu_type, sort_order, status in DEFAULT_MENUS:
            parent = menu_by_name.get(parent_name) if parent_name else None
            menu = Menu(
                parent_id=parent.id if parent else None,
                name=name,
                path=path,
                icon=icon,
                menu_type=menu_type,
                sort_order=sort_order,
                status=status,
            )
            db.add(menu)
            db.flush()
            if menu_type == "directory":
                menu_by_name[name] = menu

    db.add(SystemSeedState(key=MENU_SEED_STATE_KEY))
    db.flush()
    return not menu_table_has_rows


def ensure_system_seed_data() -> None:
    """创建基础部门、角色、菜单，并把旧 role 字段同步到关联表。"""
    db = SessionLocal()
    try:
        root_department = db.query(Department).filter(Department.parent_id.is_(None)).order_by(Department.id).first()
        if not root_department:
            root_department = Department(name="总公司", sort_order=0, status="active")
            db.add(root_department)
            db.flush()

        role_specs = (("超级管理员", "1", "admin", "all", 1, 1000), ("普通用户", "2", "user", "self", 2, 100))
        role_by_key = {}
        for name, code, key, scope, order, authority_level in role_specs:
            role = db.query(Role).filter(Role.role_key == key).first()
            if not role:
                role = Role(
                    name=name,
                    role_code=code,
                    role_key=key,
                    data_scope=scope,
                    sort_order=order,
                    authority_level=authority_level,
                    status="active",
                )
                db.add(role)
                db.flush()
            elif key == "admin" or not int(role.authority_level or 0):
                role.authority_level = authority_level
            role_by_key[key] = role

        used_levels = {int(role.authority_level or 0) for role in role_by_key.values()}
        next_level = 900
        custom_roles = (
            db.query(Role)
            .filter(~Role.role_key.in_(["admin", "user"]))
            .order_by(Role.sort_order.asc(), Role.id.asc())
            .all()
        )
        for role in custom_roles:
            current_level = int(role.authority_level or 0)
            if current_level <= 0 or current_level in used_levels:
                while next_level in used_levels and next_level > 100:
                    next_level -= 1
                role.authority_level = max(1, next_level)
            used_levels.add(role.authority_level)
            next_level -= 1

        seed_default_menus_if_uninitialized(db)
        log_parent = db.query(Menu).filter(Menu.path == "/system/log", Menu.menu_type == "directory").first()
        if log_parent and not db.query(Menu.id).filter(Menu.path == "/system/log/faults").first():
            db.add(Menu(parent_id=log_parent.id, name="系统故障日志", path="/system/log/faults",
                        icon="Warning", menu_type="menu", sort_order=3, status="active"))
            db.flush()
        from monitoring_menu_migration import migrate_embedded_algorithm_config, migrate_monitoring_management
        migrate_monitoring_management(db, Menu, RoleMenu, SystemSeedState)
        migrate_embedded_algorithm_config(db, Menu, RoleMenu, SystemSeedState)
        from algorithm_catalog import migrate_algorithm_menu
        migrate_algorithm_menu(db, Menu, SystemSeedState)
        from menu_structure_migration import migrate_current_menu_structure
        migrate_current_menu_structure(db, Menu, RoleMenu, SystemSeedState)

        db.flush()
        all_menu_ids = [item.id for item in db.query(Menu).all()]
        admin_role = role_by_key["admin"]
        for menu_id in all_menu_ids:
            exists = db.query(RoleMenu).filter(RoleMenu.role_id == admin_role.id, RoleMenu.menu_id == menu_id).first()
            if not exists:
                db.add(RoleMenu(role_id=admin_role.id, menu_id=menu_id))

        for user in db.query(User).all():
            if not user.user_no:
                user.user_no = str(user.id)
            if not user.dept_id:
                user.dept_id = root_department.id

            existing_links = db.query(UserRole).filter(UserRole.user_id == user.id).all()
            if not existing_links:
                # One-time compatibility bridge for databases that predate user_roles.
                role_key = user.role if user.role in role_by_key else "user"
                role = role_by_key[role_key]
                db.add(UserRole(user_id=user.id, role_id=role.id))
                user.role = "admin" if role.role_key == "admin" else "user"
                continue

            linked_role_ids = [link.role_id for link in existing_links]
            active_roles = (
                db.query(Role)
                .filter(Role.id.in_(linked_role_ids), Role.status == "active")
                .all()
                if linked_role_ids
                else []
            )
            user.role = "admin" if any(role.role_key == "admin" for role in active_roles) else "user"

        db.commit()
    finally:
        db.close()
