"""评估域：独立 val/test 运行记录。"""
from __future__ import annotations
import uuid
from sqlalchemy.exc import IntegrityError
from pydantic import ValidationError
from app.schemas.inference import EvaluationParams

from sqlalchemy.orm import Session

from app.api.deps import CurrentUser
from app.core.config import Settings
from app.core.errors import ApiError
from app.db.models import EvaluationRun, JobTask
from app.domains import job_service
from app.domains.datasets import service as dataset_service
from app.domains.models import service as model_service


def serialize(run: EvaluationRun) -> dict:
    return {
        "evaluation_id": run.id,
        "model_id": run.model_id,
        "dataset_version_id": run.dataset_version_id,
        "split": run.split,
        "status": run.status,
        "metrics": run.metrics_json or {},
        "params": run.params_json or {},
        "artifacts": run.artifacts_json or {},
        "purpose": run.purpose,
        "approved_by": run.approved_by,
        "context_hash": run.context_hash,
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
    }


def create_evaluation(
    db: Session,
    settings: Settings,
    user: CurrentUser,
    payload: dict,
) -> tuple[EvaluationRun, JobTask, bool]:
    try:
        params = EvaluationParams.model_validate(payload.get("params") or {}).model_dump()
    except ValidationError as exc:
        raise ApiError(422, "invalid_params", str(exc)) from exc
    split = str(payload.get("split") or "val")
    if split not in {"val", "test"}:
        raise ApiError(422, "invalid_split", "split 只能是 val 或 test")
    model = model_service.get_model(db, payload["model_id"], user)
    model_service.verify_model_file(model, settings)
    version = dataset_service.get_version(db, payload["dataset_version_id"], user)
    if version.state != "frozen":
        raise ApiError(422, "version_not_frozen", "评估只接受冻结版本")

    model_classes = {str(key): str(value) for key, value in (model.class_map_json or {}).items()}
    version_classes = {str(key): str(value) for key, value in (version.class_map_json or {}).items()}
    if model_classes and version_classes and model_classes != version_classes:
        raise ApiError(422, "class_map_mismatch", "模型类别与数据版本类别不一致")

    if split == "test":
        purpose = str(payload.get("purpose") or "").strip()
        if not purpose:
            raise ApiError(422, "purpose_required", "test 评估必须记录目的")
        if not user.is_admin:
            raise ApiError(403, "approval_required", "test 评估需要管理员批准")
        approved_by = user.username
    else:
        purpose = payload.get("purpose")
        approved_by = None

    key = payload.get("idempotency_key")
    existing_job = db.query(JobTask).filter(JobTask.owner_id == str(user.id), JobTask.type == "evaluation",
        JobTask.idempotency_key == key).one_or_none() if key else None
    run_id = (existing_job.input_json or {}).get("evaluation_id") if existing_job else str(uuid.uuid5(uuid.NAMESPACE_URL, f"bysj:evaluation:{user.id}:{key}")) if key else str(uuid.uuid4())
    expected = {"model_id": model.id, "dataset_version_id": version.id, "split": split, "params": params, "purpose": purpose}
    run = db.get(EvaluationRun, run_id)
    if run is None:
        run = EvaluationRun(
            id=run_id, owner_id=user.id, model_id=model.id,
            dataset_version_id=version.id, split=split, params_json=params,
            status="pending", purpose=purpose, approved_by=approved_by,
        )
        db.add(run)
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            run = db.get(EvaluationRun, run_id)
            if run is None:
                raise
    actual = {"model_id": run.model_id, "dataset_version_id": run.dataset_version_id, "split": run.split,
              "params": EvaluationParams.model_validate(run.params_json or {}).model_dump(), "purpose": run.purpose}
    if actual != expected:
        raise ApiError(409, "idempotency_conflict", "相同幂等键对应了不同评估请求")
    if existing_job:
        return run, existing_job, False

    job, created = job_service.create_job(
        db,
        job_type="evaluation",
        owner_id=user.id,
        owner_username=user.username,
        payload={
            "evaluation_id": run.id,
            "dataset_version_id": version.id,
            "model_id": model.id,
            "evaluation_request": expected,
        },
        idempotency_key=payload.get("idempotency_key"),
    )
    run.job_id = job.id
    db.commit()
    return run, job, created


def get_evaluation(db: Session, evaluation_id: str, user: CurrentUser) -> EvaluationRun:
    run = db.get(EvaluationRun, evaluation_id)
    if run is None or (not user.is_admin and run.owner_id != user.id):
        raise ApiError(404, "not_found", "评估记录不存在")
    return run
