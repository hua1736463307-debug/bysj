"""v1 推理接口与结果导出。"""
from __future__ import annotations

import json

from fastapi import APIRouter, Depends, File, Form, Query, UploadFile
from fastapi.responses import Response
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser, get_current_user, settings_dep
from app.core.config import Settings
from app.core.errors import ApiError, accepted, ok
from app.db.models import InferenceItem
from app.db.session import get_db
from app.domains import job_service
from app.domains.inference import service as inference_service
from app.schemas.common import DEFAULT_PAGE_SIZE, MAX_PAGE_SIZE

router = APIRouter(prefix="/inference", tags=["inference"])


@router.post("/images", response_model=None)
def create_inference(
    model_id: str = Form(...),
    files: list[UploadFile] = File(default=[]),
    asset_ids: str = Form(default=""),
    params: str = Form(default="{}"),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
    settings: Settings = Depends(settings_dep),
):
    payloads: list[tuple[str, bytes]] = []
    for upload in files:
        if not upload.filename:
            continue
        data = upload.file.read(settings.max_image_bytes + 1)
        if len(data) > settings.max_image_bytes:
            raise ApiError(413, "file_too_large", f"图片超过上限: {upload.filename}")
        payloads.append((upload.filename, data))
    ids = []
    for chunk in asset_ids.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        try:
            ids.append(int(chunk))
        except ValueError as exc:
            raise ApiError(422, "invalid_asset_id", "asset_ids 必须是逗号分隔的整数") from exc
    try:
        request_params = json.loads(params)
    except json.JSONDecodeError as exc:
        raise ApiError(422, "invalid_params", "params 必须是 JSON 对象") from exc

    run = inference_service.create_run(
        db,
        settings,
        user,
        model_id=model_id,
        files=payloads,
        asset_ids=ids,
        params=request_params,
    )
    job, created = job_service.create_job(
        db,
        job_type="inference",
        owner_id=user.id,
        owner_username=user.username,
        payload={"run_id": run.id, "model_id": model_id},
    )
    run.job_id = job.id
    db.commit()
    if settings.inline_jobs and created:
        from app.workers.dispatcher import execute_job_inline

        execute_job_inline(job.id)
        db.refresh(job)
        db.refresh(run)
    return accepted(
        {"run_id": run.id, "job": job.to_public(), "status_url": f"/api/v1/inference/runs/{run.id}"},
        message="推理任务已接收",
    )


@router.get("/runs", response_model=None)
def list_runs(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    items, total = inference_service.list_runs(db, user, page=page, page_size=page_size)
    return ok(
        {
            "items": [inference_service.serialize_run(item) for item in items],
            "total": total,
            "page": page,
            "page_size": page_size,
        }
    )


@router.get("/runs/{run_id}", response_model=None)
def get_run(
    run_id: str,
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=DEFAULT_PAGE_SIZE, ge=1, le=MAX_PAGE_SIZE),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    run = inference_service.get_run(db, run_id, user)
    items = db.query(InferenceItem).filter(InferenceItem.run_id == run.id).order_by(InferenceItem.created_at, InferenceItem.id).offset((page - 1) * page_size).limit(page_size).all()
    data = inference_service.serialize_run(run)
    data["items"] = [inference_service.serialize_item(db, item) for item in items]
    data["page"] = page
    data["page_size"] = page_size
    return ok(data)


@router.get("/runs/{run_id}/export", response_model=None)
def export_run(
    run_id: str,
    format: str = Query(default="json", pattern="^(json|csv)$"),
    user: CurrentUser = Depends(get_current_user),
    db: Session = Depends(get_db),
):
    run = inference_service.get_run(db, run_id, user)
    body, media_type, filename = inference_service.export_run(db, run, format)
    return Response(
        content=body,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
