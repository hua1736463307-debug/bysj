"""推理域：输入持久化、分项状态与结果导出。"""
from __future__ import annotations

import csv
import io
import json
import shutil
import uuid
from pathlib import Path

from sqlalchemy.orm import Session

from app.api.deps import CurrentUser
from app.core.config import Settings
from app.core.errors import ApiError
from app.core.storage import ensure_dir, normalize_relative
from app.db.models import ArtifactRecord, DatasetAsset, InferenceItem, InferencePrediction, InferenceRun
from app.domains.datasets import service as dataset_service
from app.domains.models import service as model_service


def create_run(
    db: Session,
    settings: Settings,
    user: CurrentUser,
    *,
    model_id: str,
    files: list[tuple[str, bytes]] | None = None,
    asset_ids: list[int] | None = None,
    params: dict | None = None,
) -> InferenceRun:
    model = model_service.get_model(db, model_id, user)
    model_service.verify_model_file(model, settings)

    items: list[dict] = []
    for filename, payload in files or []:
        items.append({"kind": "upload", "filename": filename, "payload": payload})
    for asset_id in asset_ids or []:
        asset = dataset_service.get_asset(db, asset_id, user)
        items.append({"kind": "asset", "asset_id": asset.id, "relative_path": asset.relative_path})
    if not items:
        raise ApiError(422, "no_inputs", "至少提供一张图片或资产 ID")
    if len(items) > settings.max_batch_inference:
        raise ApiError(422, "batch_too_large", f"批量推理最多 {settings.max_batch_inference} 张")

    run = InferenceRun(
        owner_id=user.id,
        model_id=model.id,
        model_hash=model.sha256,
        params_json=params or {},
        status="pending",
        total_count=len(items),
    )
    db.add(run)
    db.commit()
    db.refresh(run)

    run_dir = ensure_dir(Path(settings.data_root) / "inference" / run.id / "inputs")
    for index, item in enumerate(items):
        if item["kind"] == "upload":
            name = normalize_relative(item["filename"]).name
            target = run_dir / f"{index:04d}-{name}"
            target.write_bytes(item["payload"])
            db.add(
                InferenceItem(
                    run_id=run.id,
                    image_relative_path=str(target),
                    image_sha256=__import__("hashlib").sha256(item["payload"]).hexdigest(),
                    status="pending",
                )
            )
        else:
            db.add(
                InferenceItem(
                    run_id=run.id,
                    asset_id=item["asset_id"],
                    image_relative_path=item["relative_path"],
                    status="pending",
                )
            )
    db.commit()
    return run


def list_runs(db: Session, user: CurrentUser, *, page: int, page_size: int):
    query = db.query(InferenceRun)
    if not user.is_admin:
        query = query.filter(InferenceRun.owner_id == user.id)
    total = query.count()
    items = (
        query.order_by(InferenceRun.created_at.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return items, total


def get_run(db: Session, run_id: str, user: CurrentUser) -> InferenceRun:
    run = db.get(InferenceRun, run_id)
    if run is None or (not user.is_admin and run.owner_id != user.id):
        raise ApiError(404, "not_found", "推理记录不存在")
    return run


def serialize_run(run: InferenceRun) -> dict:
    return {
        "run_id": run.id,
        "model_id": run.model_id,
        "model_sha256": run.model_hash,
        "status": run.status,
        "total": run.total_count,
        "completed": run.completed_count,
        "failed": run.failed_count,
        "params": run.params_json or {},
        "created_at": run.created_at.isoformat() if run.created_at else None,
        "finished_at": run.finished_at.isoformat() if run.finished_at else None,
    }


def serialize_item(db: Session, item: InferenceItem) -> dict:
    predictions = (
        db.query(InferencePrediction).filter(InferencePrediction.item_id == item.id).all()
    )
    return {
        "item_id": item.id,
        "asset_id": item.asset_id,
        "status": item.status,
        "image_sha256": item.image_sha256,
        "width": item.image_width,
        "height": item.image_height,
        "timing": item.timing_json or {},
        "error": item.error,
        "artifact_id": item.artifact_id,
        "predictions": [
            {
                "id": prediction.id,
                "class_id": prediction.class_id,
                "class_name": prediction.class_name,
                "confidence": prediction.confidence,
                "bbox_xyxy": prediction.bbox_json,
            }
            for prediction in predictions
        ],
    }


def resolve_item_path(db: Session, settings: Settings, item: InferenceItem) -> Path:
    if item.asset_id:
        asset = db.get(DatasetAsset, item.asset_id)
        if asset is None:
            raise ApiError(404, "not_found", "源图像不存在")
        from app.db.models import DatasetVersion

        version = db.get(DatasetVersion, asset.dataset_version_id)
        return Path(version.root_path) / "images" / asset.relative_path
    path = Path(item.image_relative_path or "")
    try:
        path.resolve().relative_to(Path(settings.data_root).resolve())
    except ValueError as exc:
        raise ApiError(409, "invalid_path", "输入路径越界") from exc
    return path


def save_annotated_image(source: Path, predictions: list[dict], target: Path) -> None:
    from PIL import Image, ImageDraw

    ensure_dir(target.parent)
    with Image.open(source) as image:
        canvas = image.convert("RGB")
    draw = ImageDraw.Draw(canvas)
    for prediction in predictions:
        box = prediction["bbox_xyxy"]
        draw.rectangle(box, outline=(255, 64, 64), width=3)
    canvas.save(target, format="JPEG", quality=90)


def export_run(db: Session, run: InferenceRun, fmt: str) -> tuple[bytes, str, str]:
    items = db.query(InferenceItem).filter(InferenceItem.run_id == run.id).order_by(InferenceItem.id).all()
    payload = {
        "schema_version": 1,
        "run": serialize_run(run),
        "items": [serialize_item(db, item) for item in items],
    }
    if fmt == "json":
        return json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"), "application/json", f"{run.id}.json"
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        ["item_id", "status", "class_id", "class_name", "confidence", "x1", "y1", "x2", "y2", "image_sha256"]
    )
    for item in payload["items"]:
        if not item["predictions"]:
            writer.writerow([item["item_id"], item["status"], "", "", "", "", "", "", "", item["image_sha256"]])
        for prediction in item["predictions"]:
            writer.writerow(
                [
                    item["item_id"],
                    item["status"],
                    prediction["class_id"],
                    _csv_safe(prediction.get("class_name") or ""),
                    prediction["confidence"],
                    *prediction["bbox_xyxy"],
                    item["image_sha256"],
                ]
            )
    return buffer.getvalue().encode("utf-8"), "text/csv", f"{run.id}.csv"


def _csv_safe(value: str) -> str:
    text = str(value)
    if text[:1] in {"=", "+", "-", "@"}:
        return "'" + text
    return text


def record_artifact(
    db: Session,
    settings: Settings,
    *,
    owner_id: int | None,
    resource_type: str,
    resource_id: str,
    path: Path,
    mime: str,
) -> ArtifactRecord:
    root = Path(settings.artifact_root).resolve()
    try:
        relative = path.resolve().relative_to(root)
    except ValueError as exc:
        raise ApiError(500, "artifact_path", "产物不在受控目录") from exc
    record = ArtifactRecord(
        owner_id=owner_id,
        resource_type=resource_type,
        resource_id=resource_id,
        relative_path=relative.as_posix(),
        sha256=__import__("hashlib").sha256(path.read_bytes()).hexdigest(),
        mime=mime,
        size=path.stat().st_size,
    )
    db.add(record)
    db.commit()
    db.refresh(record)
    return record
