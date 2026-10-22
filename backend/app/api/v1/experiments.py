"""v1 实验索引。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, get_current_user
from app.core.errors import ApiError, ok
from app.db.models import ExperimentRun
from app.db.session import get_db
from app.schemas.common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE

router = APIRouter(prefix="/experiments", tags=["experiments"])


def _serialize(run: ExperimentRun) -> dict:
    return {
        "experiment_id": run.id,
        "training_job_id": run.training_job_id,
        "model_id": run.model_id,
        "dataset_version_id": run.dataset_version_id,
        "code_sha": run.code_sha,
        "config": run.config_json or {},
        "environment": run.env_json or {},
        "metrics": run.metrics_json or {},
        "artifacts": run.artifact_refs or {},
        "created_at": run.created_at.isoformat() if run.created_at else None,
    }


@router.get("", response_model=None)
def list_experiments(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    query = db.query(ExperimentRun)
    if not user.is_admin:
        query = query.filter(ExperimentRun.owner_id == user.id)
    total = query.count()
    items = (
        query.order_by(ExperimentRun.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return ok(
        {
            "items": [_serialize(item) for item in items],
            "total": total,
            "page": page,
            "page_size": page_size,
        }
    )


@router.get("/{experiment_id}", response_model=None)
def get_experiment(
    experiment_id: str,
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    run = db.get(ExperimentRun, experiment_id)
    if run is None or (not user.is_admin and run.owner_id != user.id):
        raise ApiError(404, "not_found", "实验记录不存在")
    return ok(_serialize(run))
