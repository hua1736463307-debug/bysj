"""Atomic, fenced leases. Expired GPU owners remain quarantined until released.

A DB timeout alone cannot prove a CUDA process has stopped. GPU leases are only
reused after the old owner explicitly releases (after process cleanup), or an
operator verifies termination and releases that exact holder/generation.
"""
from datetime import timedelta
from sqlalchemy import or_, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.dialects.sqlite import insert as sqlite_insert
from sqlalchemy.orm import Session
from app.db.base import utcnow
from app.db.models import ResourceLease

GPU_RESOURCE = "gpu:0"
WORKER_RESOURCE_PREFIX = "worker:"


def acquire_lease(db: Session, resource: str, *, holder: str,
                  job_id: str | None = None, ttl_seconds: int = 600) -> int | None:
    if not holder or ttl_seconds <= 0:
        raise ValueError("holder and positive TTL required")
    now = utcnow()
    insert = pg_insert if db.bind.dialect.name == "postgresql" else sqlite_insert
    values = dict(resource=resource, holder=holder, job_id=job_id, generation=1,
                  acquired_at=now, heartbeat_at=now, expires_at=now + timedelta(seconds=ttl_seconds))
    # GPU expiry is quarantine, not proof of process termination.
    reusable = ResourceLease.holder == ""
    if resource != GPU_RESOURCE:
        reusable = or_(reusable, ResourceLease.expires_at <= now)
    statement = insert(ResourceLease).values(**values).on_conflict_do_update(
        index_elements=[ResourceLease.resource], where=reusable,
        set_={**{k: v for k, v in values.items() if k not in {"resource", "generation"}},
              "generation": ResourceLease.generation + 1},
    ).returning(ResourceLease.generation)
    token = db.execute(statement).scalar_one_or_none()
    db.commit()
    return token


def renew_lease(db: Session, resource: str, *, holder: str, generation: int,
                job_id: str | None = None, ttl_seconds: int = 600) -> bool:
    now = utcnow()
    result = db.execute(update(ResourceLease).where(
        ResourceLease.resource == resource, ResourceLease.holder == holder,
        ResourceLease.generation == generation, ResourceLease.job_id == job_id,
        ResourceLease.expires_at > now,
    ).values(heartbeat_at=now, expires_at=now + timedelta(seconds=ttl_seconds)))
    db.commit()
    return result.rowcount == 1


def release_lease(db: Session, resource: str, *, holder: str, generation: int,
                  job_id: str | None = None) -> bool:
    result = db.execute(update(ResourceLease).where(
        ResourceLease.resource == resource, ResourceLease.holder == holder,
        ResourceLease.generation == generation, ResourceLease.job_id == job_id,
    ).values(holder="", job_id=None, expires_at=utcnow()))
    db.commit()
    return result.rowcount == 1


def lease_holder(db: Session, resource: str) -> str | None:
    lease = db.get(ResourceLease, resource, populate_existing=True)
    if lease is None or not lease.holder or lease.expires_at <= utcnow():
        return None
    return lease.holder


def active_worker_leases(db: Session) -> int:
    return db.query(ResourceLease).filter(ResourceLease.resource.startswith(WORKER_RESOURCE_PREFIX),
        ResourceLease.holder != "", ResourceLease.expires_at > utcnow()).count()


def worker_roles(db: Session) -> dict:
    rows = db.query(ResourceLease).filter(ResourceLease.resource.startswith(WORKER_RESOURCE_PREFIX),
        ResourceLease.holder != "", ResourceLease.expires_at > utcnow()).all()
    return {role: sum(row.resource.startswith(f"worker:{role}:") for row in rows)
            for role in ("dataset", "trainer", "inference")}
