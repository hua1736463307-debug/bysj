"""Validation and persistence helpers for resumable annotation drafts."""

import json
import re
from datetime import datetime
from typing import Any, Dict, Iterable, Optional

from sqlalchemy import func
from sqlalchemy.orm import Session

from database import AnnotationDraft, AnnotationDraftAsset


ANNOTATION_KINDS = {"multimodal", "yolo"}
MAX_DRAFT_ASSETS_PER_REQUEST = 500
MAX_DRAFT_ANNOTATIONS_PER_IMAGE = 2_000
MAX_DRAFT_CLASSES = 500
MAX_DRAFT_PAYLOAD_BYTES = 20 * 1024 * 1024


def safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value or default)
    except (TypeError, ValueError):
        return default


def normalize_annotation_kind(value: Any) -> str:
    kind = str(value or "").strip().lower()
    if kind not in ANNOTATION_KINDS:
        raise ValueError("annotation_kind 仅支持 multimodal 或 yolo")
    return kind


def normalize_relative_path(value: Any) -> str:
    path = re.sub(r"/{2,}", "/", str(value or "").replace("\\", "/").strip())
    if not path or path.startswith("/") or (len(path) > 1 and path[1] == ":"):
        raise ValueError("标注草稿图片路径必须是相对路径")
    if any(part in {"", ".", ".."} for part in path.split("/")):
        raise ValueError("标注草稿图片路径无效")
    return path


def json_copy(value: Any) -> Any:
    try:
        return json.loads(json.dumps(value, ensure_ascii=False, allow_nan=False))
    except (TypeError, ValueError) as exc:
        raise ValueError("标注草稿包含不可保存的数据") from exc


def sanitize_draft_asset(value: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("标注草稿图片必须是对象")
    relative_path = normalize_relative_path(
        value.get("relative_path") or value.get("saved_path") or value.get("path") or value.get("filename")
    )
    raw_annotations = value.get("annotations")
    if raw_annotations is None:
        raw_annotations = []
    if not isinstance(raw_annotations, list):
        raise ValueError("标注草稿 annotations 必须是数组")
    if len(raw_annotations) > MAX_DRAFT_ANNOTATIONS_PER_IMAGE:
        raise ValueError("单张图片标注数量超过临时保存上限")
    annotations = [json_copy(item) for item in raw_annotations if isinstance(item, dict)]
    return {
        "relative_path": relative_path,
        "width": max(0, safe_int(value.get("width"))),
        "height": max(0, safe_int(value.get("height"))),
        "annotations": annotations,
    }


def sanitize_draft_payload(payload: Any) -> Dict[str, Any]:
    if not isinstance(payload, dict):
        raise ValueError("标注草稿 payload 必须是对象")
    raw_assets = payload.get("assets") or payload.get("images") or []
    if not isinstance(raw_assets, list) or len(raw_assets) > MAX_DRAFT_ASSETS_PER_REQUEST:
        raise ValueError(f"单次临时保存最多支持 {MAX_DRAFT_ASSETS_PER_REQUEST} 张图片")
    assets = {}
    for value in raw_assets:
        asset = sanitize_draft_asset(value)
        assets[asset["relative_path"]] = asset
    raw_classes = payload.get("class_names") or payload.get("classes") or []
    if not isinstance(raw_classes, list):
        raw_classes = []
    class_names = [str(item or "").strip() for item in raw_classes[:MAX_DRAFT_CLASSES] if str(item or "").strip()]
    raw_bindings = payload.get("class_bindings") or payload.get("annotation_class_bindings") or []
    if not isinstance(raw_bindings, list):
        raw_bindings = []
    normalized = {
        "assets": list(assets.values()),
        "class_names": class_names,
        "class_bindings": json_copy(raw_bindings),
        "last_page": max(1, safe_int(payload.get("last_page") or payload.get("image_page"), 1)),
        "last_image_path": str(payload.get("last_image_path") or payload.get("selected_image_path") or "").strip(),
    }
    encoded = json.dumps(normalized, ensure_ascii=False, allow_nan=False, separators=(",", ":"))
    if len(encoded.encode("utf-8")) > MAX_DRAFT_PAYLOAD_BYTES:
        raise ValueError("标注草稿内容超过临时保存大小上限")
    return normalized


def user_id(current_user: Dict[str, Any]) -> int:
    try:
        value = int(current_user.get("id"))
    except (TypeError, ValueError):
        value = 0
    if value <= 0:
        raise ValueError("当前登录用户缺少有效用户 ID")
    return value


def get_draft(db: Session, owner_id: int, dataset_id: str, kind: str) -> Optional[AnnotationDraft]:
    return (
        db.query(AnnotationDraft)
        .filter(
            AnnotationDraft.owner_user_id == owner_id,
            AnnotationDraft.dataset_id == str(dataset_id),
            AnnotationDraft.annotation_kind == normalize_annotation_kind(kind),
        )
        .first()
    )


def get_draft_for_update(db: Session, owner_id: int, dataset_id: str, kind: str) -> Optional[AnnotationDraft]:
    return (
        db.query(AnnotationDraft)
        .filter(
            AnnotationDraft.owner_user_id == owner_id,
            AnnotationDraft.dataset_id == str(dataset_id),
            AnnotationDraft.annotation_kind == normalize_annotation_kind(kind),
        )
        .with_for_update()
        .first()
    )


def get_draft_by_id(db: Session, owner_id: int, draft_id: str) -> Optional[AnnotationDraft]:
    return db.query(AnnotationDraft).filter(
        AnnotationDraft.id == str(draft_id),
        AnnotationDraft.owner_user_id == owner_id,
    ).first()


def count_assets(db: Session, draft_id: str) -> int:
    return int(db.query(func.count(AnnotationDraftAsset.id)).filter(AnnotationDraftAsset.draft_id == draft_id).scalar() or 0)


def serialize_draft(db: Session, draft: AnnotationDraft) -> Dict[str, Any]:
    payload = draft.to_dict()
    payload["saved_image_count"] = count_assets(db, draft.id)
    payload["exists"] = True
    return payload


def delete_draft(db: Session, draft: Optional[AnnotationDraft]) -> bool:
    if draft is None:
        return False
    db.query(AnnotationDraftAsset).filter(AnnotationDraftAsset.draft_id == draft.id).delete(synchronize_session=False)
    db.delete(draft)
    return True


def delete_drafts_for_dataset(db: Session, dataset_id: str) -> int:
    draft_ids = [row[0] for row in db.query(AnnotationDraft.id).filter(AnnotationDraft.dataset_id == str(dataset_id)).all()]
    if not draft_ids:
        return 0
    db.query(AnnotationDraftAsset).filter(AnnotationDraftAsset.draft_id.in_(draft_ids)).delete(synchronize_session=False)
    return int(db.query(AnnotationDraft).filter(AnnotationDraft.id.in_(draft_ids)).delete(synchronize_session=False) or 0)


def upsert_draft(
    db: Session,
    *,
    draft: Optional[AnnotationDraft],
    owner_id: int,
    owner_username: str,
    dataset_id: str,
    source_version_id: str,
    annotation_kind: str,
    model_path: str,
    model_name: str,
    annotation_geometry: str,
    class_names: Iterable[str],
    class_bindings: Any,
    last_page: int,
    last_image_path: str,
    assets: Iterable[Dict[str, Any]],
) -> AnnotationDraft:
    now = datetime.now()
    if draft is None:
        draft = AnnotationDraft(
            owner_user_id=owner_id,
            owner_username=owner_username,
            dataset_id=str(dataset_id),
            source_version_id=str(source_version_id),
            annotation_kind=annotation_kind,
            model_path=model_path,
            model_name=model_name or model_path,
            annotation_geometry=annotation_geometry,
            class_names_json=list(class_names),
            class_bindings_json=class_bindings or [],
            last_page=max(1, int(last_page or 1)),
            last_image_path=last_image_path or "",
            revision=0,
            created_at=now,
            updated_at=now,
        )
        db.add(draft)
        db.flush()
    else:
        draft.owner_username = owner_username
        draft.source_version_id = str(source_version_id)
        draft.model_path = model_path
        draft.model_name = model_name or model_path
        draft.annotation_geometry = annotation_geometry
        draft.class_names_json = list(class_names)
        draft.class_bindings_json = class_bindings or []
        draft.last_page = max(1, int(last_page or 1))
        draft.last_image_path = last_image_path or ""
        draft.updated_at = now
    for value in assets:
        item = sanitize_draft_asset(value)
        asset = db.query(AnnotationDraftAsset).filter(
            AnnotationDraftAsset.draft_id == draft.id,
            AnnotationDraftAsset.relative_path == item["relative_path"],
        ).first()
        if asset is None:
            asset = AnnotationDraftAsset(draft_id=draft.id, relative_path=item["relative_path"])
            db.add(asset)
        asset.width = item["width"]
        asset.height = item["height"]
        asset.annotations_json = item["annotations"]
        asset.updated_at = now
    draft.revision = int(draft.revision or 0) + 1
    draft.updated_at = now
    db.flush()
    return draft
