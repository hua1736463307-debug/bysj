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


def shared_revision(db: Session, version: DatasetVersion) -> int:
    """Read/initialize the CAS token while the caller holds the version row lock."""
    metadata = version.metadata_json or {}
    token = metadata.get("annotation_revision") or {}
    if isinstance(token, dict) and token.get("version_id") == version.id:
        return int(token["value"])
    # Every old client token must expire, including drafts belonging to admins.
    # Binding to this version also prevents copied metadata reusing a parent's token.
    revisions = [row.revision for row in db.query(AnnotationDraft).filter_by(source_version_id=version.id).all()]
    revision = max(revisions) + 1 if revisions else 0
    version.metadata_json = {**metadata, "annotation_revision": {"version_id": version.id, "value": revision}}
    return revision


def advance_revision(db: Session, version: DatasetVersion) -> int:
    revision = shared_revision(db, version) + 1
    version.metadata_json = {**(version.metadata_json or {}),
                             "annotation_revision": {"version_id": version.id, "value": revision}}
    return revision


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
    db.refresh(asset)
    if asset.deleted_at is not None:
        raise ApiError(404, "not_found", "图像资产不存在")
    revision = shared_revision(db, version)
    annotations = asset.annotations_json or []
    deleted_ids = (draft_asset.deleted_json or []) if draft_asset is not None and draft.revision == revision else []
    db.commit()
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
        "revision": revision,
        "draft_id": draft.id if draft is not None else None,
        "annotations": annotations,
        "deleted_ids": deleted_ids,
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
    db.refresh(asset)
    if asset.deleted_at is not None:
        raise ApiError(404, "not_found", "图像资产不存在")
    class_map = version.class_map_json or {}
    revision = shared_revision(db, version)
    if revision != int(expected_revision):
        raise ApiError(409, "revision_conflict", "标注已被其他会话修改",
                       {"server_revision": revision, "expected_revision": expected_revision})
    validated = [_validate_box(box, class_map) for box in boxes]
    draft = _get_draft(db, user=user, dataset_id=version.dataset_id, version_id=version.id,
                       class_names=list(class_map.values()), geometry=geometry, for_update=True)
    if draft is None:
        draft = AnnotationDraft(
            owner_user_id=user.id, owner_username=user.username, dataset_id=version.dataset_id,
            source_version_id=version.id, annotation_kind=GEOMETRY_KIND, annotation_geometry=geometry,
            model_path="", model_name="", class_names_json=list(class_map.values()),
            class_bindings_json=[{"class_id": int(key), "name": name} for key, name in class_map.items()],
            class_context_json={"scope": "dataset"}, revision=revision,
        )
        db.add(draft)
        db.flush()
    draft_asset = db.query(AnnotationDraftAsset).filter(
        AnnotationDraftAsset.draft_id == draft.id,
        AnnotationDraftAsset.relative_path == asset.relative_path,
    ).one_or_none()
    if draft_asset is None:
        draft_asset = AnnotationDraftAsset(draft_id=draft.id, relative_path=asset.relative_path)
        db.add(draft_asset)
    draft_asset.annotations_json = validated
    draft_asset.deleted_json = list(deleted_ids or [])
    draft_asset.width, draft_asset.height = asset.width, asset.height
    draft_asset.updated_at = utcnow()
    revision = advance_revision(db, version)
    draft.revision = revision
    draft.last_page = max(1, int(page))
    draft.last_image_path = asset.relative_path
    draft.updated_at = utcnow()
    asset.annotations_json = validated
    try:
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise ApiError(409, "draft_conflict", "标注保存冲突，请刷新后重试") from exc
    return {"image_id": asset.id, "revision": revision, "annotations": validated,
            "deleted_ids": list(deleted_ids or []), "saved_at": utcnow().isoformat()}
