"""标注域：草稿会话与修订 compare-and-swap。"""
from __future__ import annotations
import math

from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from app.api.deps import CurrentUser
from app.core.errors import ApiError
from app.db.base import utcnow
from app.db.models import AnnotationDraft, AnnotationDraftAsset, DatasetAsset, DatasetVersion
from app.domains.datasets import service as dataset_service

GEOMETRY_KIND = "bbox"


def _validate_box(box: dict, class_map: dict) -> dict:
    polygon = None
    if box.get("polygon") is not None:
        from app.ml.geometry import normalize_polygon
        try:
            polygon, bounds = normalize_polygon(box["polygon"])
            box = {**box, **bounds}
        except (ValueError, TypeError) as exc:
            raise ApiError(422, "invalid_polygon", str(exc)) from exc
    try:
        class_id = int(box.get("class_id"))
    except (TypeError, ValueError) as exc:
        raise ApiError(422, "invalid_box", "class_id 必须是整数") from exc
    if str(class_id) not in {str(key) for key in class_map}:
        raise ApiError(422, "unknown_class", f"类别 {class_id} 不在 class_map 中")
    values = []
    for name in ("cx", "cy", "w", "h"):
        try:
            values.append(float(box.get(name)))
        except (TypeError, ValueError) as exc:
            raise ApiError(422, "invalid_box", f"{name} 必须是数值") from exc
    cx, cy, w, h = values
    if not all(math.isfinite(value) for value in values):
        raise ApiError(422, "invalid_box", "bbox 坐标必须是有限数值")
    if w <= 0 or h <= 0:
        raise ApiError(422, "invalid_box", "bbox 宽高必须大于 0")
    if not (-1e-6 <= cx <= 1 + 1e-6 and -1e-6 <= cy <= 1 + 1e-6):
        raise ApiError(422, "invalid_box", "bbox 中心点越界")
    if cx - w / 2 < -1e-6 or cy - h / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6 or cy + h / 2 > 1 + 1e-6:
        raise ApiError(422, "invalid_box", "bbox 超出图像范围")
    return {
        "client_id": str(box.get("client_id") or ""),
        "class_id": class_id,
        "cx": round(cx, 6),
        "cy": round(cy, 6),
        "w": round(w, 6),
        "h": round(h, 6),
        "source": str(box.get("source") or "manual"),
        "prediction_id": box.get("prediction_id"),
        **({"polygon": polygon} if polygon is not None else {}),
    }


def _get_draft(
    db: Session,
    *,
    user: CurrentUser,
    dataset_id: str,
    version_id: str,
    class_names: list[str],
    geometry: str,
    for_update: bool = False,
) -> AnnotationDraft | None:
    query = db.query(AnnotationDraft).filter(
        AnnotationDraft.owner_user_id == user.id,
        AnnotationDraft.dataset_id == dataset_id,
        AnnotationDraft.annotation_kind == GEOMETRY_KIND,
        AnnotationDraft.source_version_id == version_id,
        AnnotationDraft.annotation_geometry == geometry,
    )
    if for_update:
        query = query.with_for_update()
    return query.one_or_none()


def get_annotation_state(db: Session, asset_id: int, user: CurrentUser) -> dict:
    asset = dataset_service.get_asset(db, asset_id, user)
    version = dataset_service.get_version(db, asset.dataset_version_id, user)
    version = db.query(DatasetVersion).filter_by(id=version.id).populate_existing().with_for_update().one()
    class_map = version.class_map_json or {}
    draft = _get_draft(
        db,
        user=user,
        dataset_id=version.dataset_id,
        version_id=version.id,
        class_names=list(class_map.values()),
        geometry="rect",
    )
    draft_asset = None
    if draft is not None and version.state != "frozen":
        draft_asset = (
            db.query(AnnotationDraftAsset)
            .filter(
                AnnotationDraftAsset.draft_id == draft.id,
                AnnotationDraftAsset.relative_path == asset.relative_path,
            )
            .one_or_none()
        )
    annotations = draft_asset.annotations_json if draft_asset is not None else (asset.annotations_json or [])
    return {
        "image_id": asset.id,
        "asset": {
            "asset_id": asset.id,
            "relative_path": asset.relative_path,
            "width": asset.width,
            "height": asset.height,
            "sha256": asset.content_hash,
            "split": asset.split,
        },
        "version": {"version_id": version.id, "dataset_id": version.dataset_id, "state": version.state},
        "class_map": class_map,
        "revision": draft.revision if draft is not None else 0,
        "draft_id": draft.id if draft is not None else None,
        "annotations": annotations,
        "deleted_ids": (draft_asset.deleted_json or []) if draft_asset is not None else [],
        "editable": version.state in {"draft", "failed"},
    }


def save_annotations(
    db: Session,
    asset_id: int,
    user: CurrentUser,
    *,
    expected_revision: int,
    boxes: list[dict],
    deleted_ids: list[str] | None = None,
    page: int = 1,
    geometry: str = "rect",
) -> dict:
    asset = dataset_service.get_asset(db, asset_id, user)
    version = dataset_service.get_version(db, asset.dataset_version_id, user)
    dataset_service.assert_detect_task(db, version)
    version = db.query(DatasetVersion).filter(DatasetVersion.id == version.id).populate_existing().with_for_update().one()
    if version.state not in {"draft", "failed"}:
        raise ApiError(409, "version_frozen", "冻结版本内容只读；请创建子版本后再修改标注")
    class_map = version.class_map_json or {}
    validated = [_validate_box(box, class_map) for box in boxes]

    for attempt in range(2):
        draft = _get_draft(
            db,
            user=user,
            dataset_id=version.dataset_id,
            version_id=version.id,
            class_names=list(class_map.values()),
            geometry=geometry,
            for_update=True,
        )
        if draft is None:
            draft = AnnotationDraft(
                owner_user_id=user.id,
                owner_username=user.username,
                dataset_id=version.dataset_id,
                source_version_id=version.id,
                annotation_kind=GEOMETRY_KIND,
                annotation_geometry=geometry,
                model_path="",
                model_name="",
                class_names_json=list(class_map.values()),
                class_bindings_json=[
                    {"class_id": int(key), "name": name} for key, name in class_map.items()
                ],
                class_context_json={"scope": "dataset"},
                revision=0,
            )
            db.add(draft)
            try:
                db.flush()
            except IntegrityError:
                db.rollback()
                if attempt == 1:
                    raise ApiError(409, "draft_conflict", "草稿会话创建冲突，请重试")
                continue
            db.refresh(draft)

        if int(draft.revision) != int(expected_revision):
            raise ApiError(
                409,
                "revision_conflict",
                "标注已被其他会话修改",
                {"server_revision": draft.revision, "expected_revision": expected_revision},
            )

        draft_asset = (
            db.query(AnnotationDraftAsset)
            .filter(
                AnnotationDraftAsset.draft_id == draft.id,
                AnnotationDraftAsset.relative_path == asset.relative_path,
            )
            .one_or_none()
        )
        if draft_asset is None:
            draft_asset = AnnotationDraftAsset(
                draft_id=draft.id,
                relative_path=asset.relative_path,
                width=asset.width,
                height=asset.height,
                annotations_json=validated,
                deleted_json=list(deleted_ids or []),
            )
            db.add(draft_asset)
        else:
            draft_asset.annotations_json = validated
            draft_asset.deleted_json = list(deleted_ids or [])
            draft_asset.width = asset.width
            draft_asset.height = asset.height
            draft_asset.updated_at = utcnow()

        draft.revision = int(draft.revision) + 1
        draft.last_page = max(1, int(page))
        draft.last_image_path = asset.relative_path
        draft.updated_at = utcnow()
        # The version is locked until commit. Freeze/export/training must see
        # the saved labels, not the original import labels hidden by a draft.
        asset.annotations_json = validated
        try:
            db.commit()
        except IntegrityError:
            db.rollback()
            if attempt == 1:
                raise ApiError(409, "draft_conflict", "标注保存冲突，请重试")
            continue
        return {
            "image_id": asset.id,
            "revision": draft.revision,
            "annotations": validated,
            "deleted_ids": list(deleted_ids or []),
            "saved_at": utcnow().isoformat(),
        }
    raise ApiError(409, "draft_conflict", "标注保存冲突，请重试")
