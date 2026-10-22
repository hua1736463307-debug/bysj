"""单卡 GPU / worker 资源租约。

用数据库行实现互斥，避免依赖进程内锁；PostgreSQL 下由唯一主键 + 条件更新
保证并发安全，SQLite 下用同一逻辑做功能测试（不作为 PG 竞争证据）。
"""
from __future__ import annotations

from datetime import timedelta

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.db.base import utcnow
from app.db.models import ResourceLease

GPU_RESOURCE = "gpu:0"
WORKER_RESOURCE_PREFIX = "worker:"


def acquire_lease(
    db: Session,
    resource: str,
    *,
    holder: str,
    job_id: str | None = None,
    ttl_seconds: int = 600,
) -> bool:
    now = utcnow()
    lease = db.get(ResourceLease, resource)
    if lease is None:
        db.add(
            ResourceLease(
                resource=resource,
                holder=holder,
                job_id=job_id,
                acquired_at=now,
                heartbeat_at=now,
                expires_at=now + timedelta(seconds=ttl_seconds),
            )
        )
        try:
            db.commit()
            return True
        except IntegrityError:
            db.rollback()
            return False
    if lease.expires_at <= now or lease.holder == holder:
        lease.holder = holder
        lease.job_id = job_id
        lease.acquired_at = now
        lease.heartbeat_at = now
        lease.expires_at = now + timedelta(seconds=ttl_seconds)
        db.commit()
        return True
    return False


def renew_lease(db: Session, resource: str, *, holder: str, ttl_seconds: int = 600) -> bool:
    now = utcnow()
    lease = db.get(ResourceLease, resource)
    if lease is None or lease.holder != holder:
        return False
    lease.heartbeat_at = now
    lease.expires_at = now + timedelta(seconds=ttl_seconds)
    db.commit()
    return True


def release_lease(db: Session, resource: str, *, holder: str) -> bool:
    lease = db.get(ResourceLease, resource)
    if lease is None or lease.holder != holder:
        return False
    db.delete(lease)
    db.commit()
    return True


def lease_holder(db: Session, resource: str) -> str | None:
    lease = db.get(ResourceLease, resource)
    if lease is None or lease.expires_at <= utcnow():
        return None
    return lease.holder


def active_worker_leases(db: Session) -> int:
    now = utcnow()
    return (
        db.query(ResourceLease)
        .filter(ResourceLease.resource.startswith(WORKER_RESOURCE_PREFIX))
        .filter(ResourceLease.expires_at > now)
        .count()
    )
