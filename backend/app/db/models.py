"""数据库模型。

表名沿用既有物理表（users/dataset_resources/dataset_versions/dataset_assets/
annotation_drafts/annotation_draft_assets/job_tasks/job_events/model_registry），
避免出现第二套数据源；新增 UUID 列及医学域表见 CONTRACTS C02。
"""
from __future__ import annotations

from datetime import datetime

from sqlalchemy import (
    JSON,
    BigInteger,
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, new_uuid, utcnow

BIGINT_PK = BigInteger().with_variant(Integer, "sqlite")
BIGINT_FK = BigInteger().with_variant(Integer, "sqlite")
UUID = String(36)


class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(BIGINT_PK, primary_key=True, autoincrement=True)
    username: Mapped[str] = mapped_column(String(64), unique=True, nullable=False, index=True)
    password_hash: Mapped[str] = mapped_column(String(255), nullable=False)
    role: Mapped[str] = mapped_column(String(32), nullable=False, default="researcher")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active")
    nickname: Mapped[str | None] = mapped_column(String(64), nullable=True)
    email: Mapped[str | None] = mapped_column(String(128), nullable=True)
    failed_login_attempts: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    locked_until: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    last_login_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)
    __table_args__ = (Index("uq_users_username_lower", func.lower(username), unique=True),)

    def to_public(self) -> dict:
        from app.core.auth import product_role

        return {
            "id": self.id,
            "username": self.username,
            "display_name": self.nickname or self.username,
            "role": product_role(self.role),
            "status": self.status,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "last_login_at": self.last_login_at.isoformat() if self.last_login_at else None,
        }


class SystemDepartment(Base):
    __tablename__ = "system_departments"
    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(64), nullable=False)
    parent_id: Mapped[str | None] = mapped_column(UUID, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)


class SystemRole(Base):
    __tablename__ = "system_roles"
    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(64), nullable=False, unique=True)
    base_role: Mapped[str] = mapped_column(String(32), nullable=False, default="researcher")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="active")
    modules: Mapped[list] = mapped_column(JSON, nullable=False, default=list)


class SystemUserProfile(Base):
    __tablename__ = "system_user_profiles"
    user_id: Mapped[int] = mapped_column(BIGINT_FK, ForeignKey("users.id", ondelete="RESTRICT"), primary_key=True)
    department_id: Mapped[str | None] = mapped_column(UUID, ForeignKey("system_departments.id", ondelete="RESTRICT"), nullable=True)
    role_id: Mapped[str | None] = mapped_column(UUID, ForeignKey("system_roles.id", ondelete="RESTRICT"), nullable=True)


class SystemMenu(Base):
    __tablename__ = "system_menus"
    path: Mapped[str] = mapped_column(String(128), primary_key=True)
    label: Mapped[str] = mapped_column(String(64), nullable=False)
    sort_order: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    visible: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)


class SystemAuditLog(Base):
    __tablename__ = "system_audit_logs"
    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_uuid)
    kind: Mapped[str] = mapped_column(String(16), nullable=False, index=True)
    username: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    action: Mapped[str] = mapped_column(String(160), nullable=False)
    status: Mapped[int] = mapped_column(Integer, nullable=False)
    client_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)
    request_id: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow, index=True)


class AuthSession(Base):
    __tablename__ = "auth_sessions"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_uuid)
    user_id: Mapped[int] = mapped_column(BIGINT_FK, nullable=False, index=True)
    jti: Mapped[str] = mapped_column(String(64), nullable=False, unique=True, index=True)
    issued_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)
    revoked_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    user_agent: Mapped[str | None] = mapped_column(String(255), nullable=True)
    client_ip: Mapped[str | None] = mapped_column(String(64), nullable=True)


class DatasetResource(Base):
    """数据集主实体；v1 的 dataset_id 就是本表 id。"""

    __tablename__ = "dataset_resources"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_uuid)
    dataset_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    dataset_role: Mapped[str] = mapped_column(String(32), nullable=False, index=True, default="medical")
    owner_user_id: Mapped[int | None] = mapped_column(BIGINT_FK, nullable=True, index=True)
    owner_username: Mapped[str] = mapped_column(String(64), nullable=False, default="system", index=True)
    owner_role_id: Mapped[int | None] = mapped_column(Integer, nullable=True)
    visibility_mode: Mapped[str] = mapped_column(String(32), nullable=False, default="private")

    name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_url: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_version: Mapped[str | None] = mapped_column(String(128), nullable=True)
    license: Mapped[str | None] = mapped_column(Text, nullable=True)
    modality: Mapped[str | None] = mapped_column(String(64), nullable=True)
    task_type: Mapped[str] = mapped_column(String(32), nullable=False, default="detect")
    group_level: Mapped[str] = mapped_column(String(32), nullable=False, default="original_image")

    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)

    __table_args__ = (
        UniqueConstraint("dataset_id", "dataset_role", name="uq_dataset_resource_identity"),
        Index("ix_dataset_resource_owner", "owner_user_id", "deleted_at"),
    )


class DatasetVersion(Base):
    __tablename__ = "dataset_versions"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_uuid)
    dataset_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    parent_version_id: Mapped[str | None] = mapped_column(UUID, nullable=True, index=True)
    dataset_role: Mapped[str] = mapped_column(String(32), nullable=False, index=True, default="medical")
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="ready", index=True)
    root_path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    manifest_path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    class_map_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    split_config_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True, default="")
    created_by: Mapped[str] = mapped_column(String(64), nullable=False, default="system")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow, index=True)

    state: Mapped[str] = mapped_column(String(16), nullable=False, default="draft", index=True)
    frozen_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    manifest_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    split_manifest_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    class_map_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    asset_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    split_counts_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_by_user_id: Mapped[int | None] = mapped_column(BIGINT_FK, nullable=True, index=True)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)

    __table_args__ = (
        UniqueConstraint("dataset_id", "dataset_role", "content_hash", name="uq_dataset_version_content"),
        Index("ix_dataset_version_state", "dataset_id", "state"),
    )


class DatasetAsset(Base):
    __tablename__ = "dataset_assets"

    id: Mapped[int] = mapped_column(BIGINT_PK, primary_key=True, autoincrement=True)
    dataset_version_id: Mapped[str] = mapped_column(UUID, nullable=False, index=True)
    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    basename: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    size: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    width: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    height: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    label_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    source_group: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    content_hash: Mapped[str] = mapped_column(String(64), nullable=False, index=True, default="")
    annotations_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)

    root_asset_id: Mapped[int | None] = mapped_column(BIGINT_FK, nullable=True, index=True)
    original_asset_id: Mapped[int | None] = mapped_column(BIGINT_FK, nullable=True, index=True)
    group_level: Mapped[str] = mapped_column(String(32), nullable=False, default="original_image")
    split: Mapped[str | None] = mapped_column(String(16), nullable=True, index=True)
    transform_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)

    __table_args__ = (
        UniqueConstraint("dataset_version_id", "relative_path", name="uq_dataset_asset_path"),
    )


class DatasetImport(Base):
    __tablename__ = "dataset_imports"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_uuid)
    owner_id: Mapped[int | None] = mapped_column(BIGINT_FK, nullable=True, index=True)
    job_id: Mapped[str | None] = mapped_column(UUID, nullable=True, index=True)
    dataset_id: Mapped[str | None] = mapped_column(UUID, nullable=True, index=True)
    source_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    staging_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", index=True)
    report_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class AnnotationDraft(Base):
    __tablename__ = "annotation_drafts"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_uuid)
    owner_user_id: Mapped[int] = mapped_column(BIGINT_FK, nullable=False, index=True)
    owner_username: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    dataset_id: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    source_version_id: Mapped[str] = mapped_column(UUID, nullable=False, index=True)
    temporary_dataset_id: Mapped[str | None] = mapped_column(String(255), nullable=True, index=True)
    annotation_kind: Mapped[str] = mapped_column(String(32), nullable=False, index=True, default="bbox")
    model_path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    model_name: Mapped[str] = mapped_column(String(255), nullable=False, default="")
    annotation_geometry: Mapped[str] = mapped_column(String(16), nullable=False, default="rect")
    class_names_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    class_bindings_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    class_context_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    last_page: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    last_image_path: Mapped[str] = mapped_column(Text, nullable=False, default="")
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    __table_args__ = (
        UniqueConstraint(
            "owner_user_id",
            "dataset_id",
            "annotation_kind",
            "source_version_id",
            "annotation_geometry",
            name="uq_annotation_draft_session",
        ),
    )


class AnnotationDraftAsset(Base):
    __tablename__ = "annotation_draft_assets"

    id: Mapped[int] = mapped_column(BIGINT_PK, primary_key=True, autoincrement=True)
    draft_id: Mapped[str] = mapped_column(UUID, nullable=False, index=True)
    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    width: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    height: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    annotations_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    deleted_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    __table_args__ = (
        UniqueConstraint("draft_id", "relative_path", name="uq_annotation_draft_asset_path"),
    )


class JobTask(Base):
    __tablename__ = "job_tasks"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_uuid)
    type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    archived_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="pending", index=True)
    stage: Mapped[str] = mapped_column(String(64), nullable=False, default="queued")
    progress: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    message: Mapped[str] = mapped_column(Text, nullable=False, default="")
    owner_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    owner_username: Mapped[str] = mapped_column(String(64), nullable=False, default="", index=True)
    worker_pid: Mapped[int | None] = mapped_column(Integer, nullable=True)
    cancel_requested: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False, index=True)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    input_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    result_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow, index=True)
    started_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    heartbeat_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True, index=True)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow, onupdate=utcnow)

    owner_user_id: Mapped[int | None] = mapped_column(BIGINT_FK, nullable=True, index=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    request_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    lease_owner: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    lease_expires_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("owner_id", "type", "idempotency_key", name="uq_job_idempotency"),
        Index("ix_job_queue", "type", "status", "created_at"),
    )

    def to_public(self) -> dict:
        return {
            "job_id": self.id,
            "type": self.type,
            "owner": self.owner_username,
            "archived_at": self.archived_at.isoformat() if self.archived_at else None,
            "status": self.status,
            "stage": self.stage,
            "progress": float(self.progress or 0.0),
            "message": self.message or "",
            "attempt": self.attempt,
            "input": self.input_json or {},
            "result": self.result_json or {},
            "error": {"code": self.error_code, "detail": self.error_detail}
            if self.error_code
            else None,
            "created_at": self.created_at.isoformat() if self.created_at else None,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "finished_at": self.finished_at.isoformat() if self.finished_at else None,
            "status_url": f"/api/v1/jobs/{self.id}",
        }


class JobEvent(Base):
    __tablename__ = "job_events"

    id: Mapped[int] = mapped_column(BIGINT_PK, primary_key=True, autoincrement=True)
    task_id: Mapped[str] = mapped_column(UUID, nullable=False, index=True)
    sequence: Mapped[int] = mapped_column(Integer, nullable=False)
    level: Mapped[str] = mapped_column(String(16), nullable=False, default="info")
    event_type: Mapped[str] = mapped_column(String(64), nullable=False, default="progress", index=True)
    message: Mapped[str] = mapped_column(Text, nullable=False, default="")
    payload_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow, index=True)

    __table_args__ = (UniqueConstraint("task_id", "sequence", name="uq_job_event_sequence"),)


class ModelRegistry(Base):
    __tablename__ = "model_registry"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_uuid)
    name: Mapped[str] = mapped_column(String(255), nullable=False, index=True)
    path: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False, index=True)
    size: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    source: Mapped[str] = mapped_column(String(32), nullable=False, default="training", index=True)
    status: Mapped[str] = mapped_column(String(32), nullable=False, default="active", index=True)
    format: Mapped[str] = mapped_column(String(32), nullable=False, default="pt")
    task_id: Mapped[str | None] = mapped_column(String(36), nullable=True, index=True)
    dataset_version_id: Mapped[str | None] = mapped_column(UUID, nullable=True, index=True)
    created_by: Mapped[str] = mapped_column(String(64), nullable=False, default="system")
    metadata_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow, index=True)
    updated_at: Mapped[datetime | None] = mapped_column(DateTime, default=utcnow, onupdate=utcnow)

    owner_id: Mapped[int | None] = mapped_column(BIGINT_FK, nullable=True, index=True)
    model_task: Mapped[str] = mapped_column(String(32), nullable=False, default="detect")
    trusted_source: Mapped[str] = mapped_column(String(32), nullable=False, default="platform")
    training_job_id: Mapped[str | None] = mapped_column(UUID, nullable=True, unique=True)
    available: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    class_map_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)

    def to_public(self, include_lineage: bool = True) -> dict:
        data = {
            "model_id": self.id,
            "name": self.name,
            "task": self.model_task,
            "source": self.source,
            "trusted_source": self.trusted_source,
            "sha256": self.sha256,
            "size": self.size,
            "format": self.format,
            "available": bool(self.available) and self.status == "active",
            "class_map": self.class_map_json or {},
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
        if include_lineage:
            data["lineage"] = {
                "training_job_id": self.training_job_id or self.task_id,
                "dataset_version_id": self.dataset_version_id,
                "created_by": self.created_by,
            }
        return data


class EvaluationRun(Base):
    __tablename__ = "evaluation_runs"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_uuid)
    owner_id: Mapped[int | None] = mapped_column(BIGINT_FK, nullable=True, index=True)
    job_id: Mapped[str | None] = mapped_column(UUID, nullable=True, index=True)
    model_id: Mapped[str] = mapped_column(UUID, nullable=False, index=True)
    dataset_version_id: Mapped[str] = mapped_column(UUID, nullable=False, index=True)
    split: Mapped[str] = mapped_column(String(16), nullable=False, default="val")
    params_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    metrics_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    artifacts_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    context_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending", index=True)
    purpose: Mapped[str | None] = mapped_column(Text, nullable=True)
    approved_by: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class InferenceRun(Base):
    __tablename__ = "inference_runs"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_uuid)
    owner_id: Mapped[int | None] = mapped_column(BIGINT_FK, nullable=True, index=True)
    job_id: Mapped[str | None] = mapped_column(UUID, nullable=True, index=True)
    model_id: Mapped[str] = mapped_column(UUID, nullable=False, index=True)
    model_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    params_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending", index=True)
    total_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed_count: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)


class InferenceItem(Base):
    __tablename__ = "inference_items"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_uuid)
    run_id: Mapped[str] = mapped_column(UUID, nullable=False, index=True)
    asset_id: Mapped[int | None] = mapped_column(BIGINT_FK, nullable=True, index=True)
    image_relative_path: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    image_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending", index=True)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    timing_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    artifact_id: Mapped[str | None] = mapped_column(UUID, nullable=True, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (UniqueConstraint("run_id", "asset_id", "attempt", name="uq_inference_item"),)


class InferencePrediction(Base):
    __tablename__ = "inference_predictions"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_uuid)
    item_id: Mapped[str] = mapped_column(UUID, nullable=False, index=True)
    class_id: Mapped[int] = mapped_column(Integer, nullable=False)
    class_name: Mapped[str | None] = mapped_column(String(128), nullable=True)
    confidence: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    bbox_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    raw_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)


class ReviewRecord(Base):
    __tablename__ = "review_records"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_uuid)
    owner_id: Mapped[int | None] = mapped_column(BIGINT_FK, nullable=True, index=True)
    reviewer_id: Mapped[int | None] = mapped_column(BIGINT_FK, nullable=True, index=True)
    inference_item_id: Mapped[str | None] = mapped_column(UUID, nullable=True, index=True)
    image_asset_id: Mapped[int | None] = mapped_column(BIGINT_FK, nullable=True, index=True)
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="draft", index=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    final_annotations_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    decided_by: Mapped[int | None] = mapped_column(BIGINT_FK, nullable=True)


class ReviewItem(Base):
    __tablename__ = "review_items"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_uuid)
    review_id: Mapped[str] = mapped_column(UUID, nullable=False, index=True)
    prediction_id: Mapped[str | None] = mapped_column(UUID, nullable=True, index=True)
    verdict: Mapped[str] = mapped_column(String(32), nullable=False)
    original_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    corrected_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)


class FeedbackExport(Base):
    __tablename__ = "feedback_exports"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_uuid)
    job_id: Mapped[str | None] = mapped_column(UUID, nullable=True, index=True)
    parent_version_id: Mapped[str] = mapped_column(UUID, nullable=False, index=True)
    child_version_id: Mapped[str | None] = mapped_column(UUID, nullable=True, index=True)
    review_ids_hash: Mapped[str | None] = mapped_column(String(64), nullable=True)
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True, unique=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending", index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)


class ExperimentRun(Base):
    __tablename__ = "experiment_runs"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_uuid)
    owner_id: Mapped[int | None] = mapped_column(BIGINT_FK, nullable=True, index=True)
    training_job_id: Mapped[str | None] = mapped_column(UUID, nullable=True, index=True)
    model_id: Mapped[str | None] = mapped_column(UUID, nullable=True, index=True)
    dataset_version_id: Mapped[str | None] = mapped_column(UUID, nullable=True, index=True)
    code_sha: Mapped[str | None] = mapped_column(String(64), nullable=True)
    env_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    config_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    metrics_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    artifact_refs: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow, index=True)


class ArtifactRecord(Base):
    __tablename__ = "artifact_records"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_uuid)
    owner_id: Mapped[int | None] = mapped_column(BIGINT_FK, nullable=True, index=True)
    resource_type: Mapped[str] = mapped_column(String(32), nullable=False, index=True)
    resource_id: Mapped[str | None] = mapped_column(String(64), nullable=True, index=True)
    relative_path: Mapped[str] = mapped_column(Text, nullable=False)
    sha256: Mapped[str] = mapped_column(String(64), nullable=False)
    mime: Mapped[str | None] = mapped_column(String(128), nullable=True)
    size: Mapped[int] = mapped_column(BigInteger, nullable=False, default=0)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)


class ResourceLease(Base):
    __tablename__ = "resource_leases"

    resource: Mapped[str] = mapped_column(String(160), primary_key=True)
    holder: Mapped[str] = mapped_column(String(160), nullable=False)
    generation: Mapped[int] = mapped_column(BigInteger, nullable=False, default=1, server_default="1")
    job_id: Mapped[str | None] = mapped_column(UUID, nullable=True)
    acquired_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    heartbeat_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    expires_at: Mapped[datetime] = mapped_column(DateTime, nullable=False)


# ---------------------------------------------------------------------------
# CXR 五标签（cxr_multilabel / cxr-five-v1）——与 detect/BBox 表并存，不改旧义。
# 患者/检查/分组只保存稳定伪标识，不保存姓名等真实身份。
# ---------------------------------------------------------------------------


class CxrSample(Base):
    """胸片样本与原始四态标签；annotations_json 对 CXR 恒为空。"""

    __tablename__ = "cxr_samples"

    id: Mapped[int] = mapped_column(BIGINT_PK, primary_key=True, autoincrement=True)
    asset_id: Mapped[int] = mapped_column(
        BIGINT_FK, ForeignKey("dataset_assets.id", ondelete="RESTRICT"), nullable=False
    )
    dataset_version_id: Mapped[str] = mapped_column(UUID, nullable=False, index=True)
    patient_key: Mapped[str] = mapped_column(String(64), nullable=False)
    study_key: Mapped[str] = mapped_column(String(64), nullable=False)
    component_key: Mapped[str] = mapped_column(String(64), nullable=False)
    view: Mapped[str] = mapped_column(String(8), nullable=False, default="PA")
    raw_labels_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False, default="cxr-five-v1")
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    pixel_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)

    __table_args__ = (
        UniqueConstraint("asset_id", name="uq_cxr_sample_asset"),
        Index("ix_cxr_sample_version_patient", "dataset_version_id", "patient_key"),
        Index("ix_cxr_sample_version_component", "dataset_version_id", "component_key"),
    )


class CxrAnalysisBatch(Base):
    """一次胸片分析请求；按 owner 作用域幂等。"""

    __tablename__ = "cxr_analysis_batches"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_uuid)
    owner_id: Mapped[int] = mapped_column(BIGINT_FK, nullable=False, index=True)
    owner_username: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    job_id: Mapped[str | None] = mapped_column(UUID, nullable=True, unique=True, index=True)
    model_id: Mapped[str] = mapped_column(UUID, nullable=False, index=True)
    model_sha256: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    request_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    idempotency_key: Mapped[str | None] = mapped_column(String(128), nullable=True)
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending", index=True)
    total: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    failed: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    error_code: Mapped[str | None] = mapped_column(String(64), nullable=True)
    error_detail: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow, index=True)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("owner_id", "idempotency_key", name="uq_cxr_batch_idempotency"),
    )


class CxrAnalysis(Base):
    """单图五标签结果；原始评分不可被复核覆盖。"""

    __tablename__ = "cxr_analyses"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_uuid)
    batch_id: Mapped[str] = mapped_column(
        UUID, ForeignKey("cxr_analysis_batches.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    owner_id: Mapped[int] = mapped_column(BIGINT_FK, nullable=False, index=True)
    asset_id: Mapped[int | None] = mapped_column(
        BIGINT_FK, ForeignKey("dataset_assets.id", ondelete="RESTRICT"), nullable=True, index=True
    )
    item_key: Mapped[str] = mapped_column(String(160), nullable=False)
    upload_ref: Mapped[str | None] = mapped_column(Text, nullable=True)
    image_sha256: Mapped[str | None] = mapped_column(String(64), nullable=True)
    image_width: Mapped[int | None] = mapped_column(Integer, nullable=True)
    image_height: Mapped[int | None] = mapped_column(Integer, nullable=True)
    model_id: Mapped[str] = mapped_column(UUID, nullable=False, index=True)
    model_sha256: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    task: Mapped[str] = mapped_column(String(32), nullable=False, default="cxr_multilabel")
    label_schema: Mapped[str] = mapped_column(String(32), nullable=False, default="cxr-five-v1")
    raw_scores_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    calibrated_scores_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    thresholds_json: Mapped[dict | None] = mapped_column(JSON, nullable=True)
    calibration_status: Mapped[str] = mapped_column(String(32), nullable=False, default="not_calibrated")
    threshold_source: Mapped[str] = mapped_column(String(64), nullable=False, default="default_no_calibration")
    calibration_artifact_id: Mapped[str | None] = mapped_column(UUID, nullable=True)
    quality_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    explanation_artifact_id: Mapped[str | None] = mapped_column(UUID, nullable=True)
    explanation_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    backend: Mapped[str] = mapped_column(String(32), nullable=False, default="unknown")
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="pending", index=True)
    attempt: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    error: Mapped[str | None] = mapped_column(Text, nullable=True)
    timing_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    finished_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("batch_id", "item_key", "attempt", name="uq_cxr_analysis_item_attempt"),
    )


class CxrReview(Base):
    """逐标签人工复核；原模型分数以快照形式独立保存，不被覆盖。"""

    __tablename__ = "cxr_reviews"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_uuid)
    analysis_id: Mapped[str] = mapped_column(
        UUID, ForeignKey("cxr_analyses.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    owner_id: Mapped[int] = mapped_column(BIGINT_FK, nullable=False, index=True)
    reviewer_id: Mapped[int | None] = mapped_column(BIGINT_FK, nullable=True, index=True)
    reviewer_username: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="draft", index=True)
    revision: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    items_json: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    model_snapshot_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    reason: Mapped[str | None] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow, onupdate=utcnow)
    submitted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    decided_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    decided_by_id: Mapped[int | None] = mapped_column(BIGINT_FK, nullable=True)
    decided_by_username: Mapped[str | None] = mapped_column(String(64), nullable=True)

    __table_args__ = (
        UniqueConstraint("analysis_id", "owner_id", name="uq_cxr_review_analysis_owner"),
    )


class CxrReport(Base):
    """不可变报告快照；最终报告引用指定复核 revision，修订出新版本。"""

    __tablename__ = "cxr_reports"

    id: Mapped[str] = mapped_column(UUID, primary_key=True, default=new_uuid)
    analysis_id: Mapped[str] = mapped_column(
        UUID, ForeignKey("cxr_analyses.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    review_id: Mapped[str] = mapped_column(
        UUID, ForeignKey("cxr_reviews.id", ondelete="RESTRICT"), nullable=False, index=True
    )
    review_revision: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    state: Mapped[str] = mapped_column(String(16), nullable=False, default="draft", index=True)
    version: Mapped[int] = mapped_column(Integer, nullable=False, default=1)
    supersedes_id: Mapped[str | None] = mapped_column(UUID, nullable=True, index=True)
    snapshot_json: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    snapshot_sha256: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    artifact_id: Mapped[str | None] = mapped_column(UUID, nullable=True)
    owner_id: Mapped[int] = mapped_column(BIGINT_FK, nullable=False, index=True)
    issued_by_id: Mapped[int | None] = mapped_column(BIGINT_FK, nullable=True)
    issued_by_username: Mapped[str | None] = mapped_column(String(64), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow, index=True)
    issued_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    __table_args__ = (
        UniqueConstraint("analysis_id", "version", name="uq_cxr_report_version"),
    )


# ---------------------------------------------------------------------------
# 皮肤镜方向（ISIC 2018 Task 3 / 七类单标签）——与 CXR 多标签表并列，
# 不复用四态，也不把七类塞进胸片标签列。
# ---------------------------------------------------------------------------


class IsicSample(Base):
    """皮肤镜样本：单标签互斥，按 lesion（或单图）分组。"""

    __tablename__ = "isic_samples"

    id: Mapped[int] = mapped_column(BIGINT_PK, primary_key=True, autoincrement=True)
    asset_id: Mapped[int] = mapped_column(
        BIGINT_FK, ForeignKey("dataset_assets.id", ondelete="RESTRICT"), nullable=False
    )
    dataset_version_id: Mapped[str] = mapped_column(UUID, nullable=False, index=True)
    #: 官方 lesion_id；validation/test 无分组表时为 NULL。
    lesion_key: Mapped[str | None] = mapped_column(String(64), nullable=True)
    #: 切分用的分组键：有 lesion_id 用 lesion，否则退化为单图键。
    group_key: Mapped[str] = mapped_column(String(96), nullable=False)
    class_key: Mapped[str] = mapped_column(String(32), nullable=False)
    class_index: Mapped[int] = mapped_column(Integer, nullable=False)
    schema_version: Mapped[str] = mapped_column(String(32), nullable=False, default="isic2018-task3-v1")
    pixel_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    source_hash: Mapped[str] = mapped_column(String(64), nullable=False, default="")
    created_at: Mapped[datetime] = mapped_column(DateTime, nullable=False, default=utcnow)

    __table_args__ = (
        UniqueConstraint("asset_id", name="uq_isic_sample_asset"),
        Index("ix_isic_sample_version_group", "dataset_version_id", "group_key"),
        Index("ix_isic_sample_version_class", "dataset_version_id", "class_index"),
    )
