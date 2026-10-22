"""医学人工复核：图像级审阅、审批与最终标签。"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.api.deps import CurrentUser
from app.core.errors import ApiError
from app.db.base import utcnow
from app.db.models import (
    DatasetAsset,
    InferenceItem,
    InferencePrediction,
    InferenceRun,
    ReviewItem,
    ReviewRecord,
)
from app.domains.datasets import service as dataset_service
from app.ml.geometry import xyxy_to_cxcywh_norm

VERDICTS = {"correct", "false_positive", "wrong_class", "missed_object", "unreviewable"}
STATES = {"draft", "submitted", "approved", "rejected"}


def _box_from_prediction(prediction: InferencePrediction, asset: DatasetAsset) -> dict:
    normalized = xyxy_to_cxcywh_norm(prediction.bbox_json, asset.width, asset.height)
    return {
        "class_id": prediction.class_id,
        "cx": round(normalized[0], 6),
        "cy": round(normalized[1], 6),
        "w": round(normalized[2], 6),
        "h": round(normalized[3], 6),
    }


def _validate_box(box: dict, class_map: dict) -> dict:
    try:
        class_id = int(box.get("class_id"))
    except (TypeError, ValueError) as exc:
        raise ApiError(422, "invalid_box", "class_id 必须是整数") from exc
    if str(class_id) not in {str(key) for key in class_map}:
        raise ApiError(422, "unknown_class", f"类别 {class_id} 不在 class_map 中")
    try:
        cx, cy, w, h = (float(box.get(name)) for name in ("cx", "cy", "w", "h"))
    except (TypeError, ValueError) as exc:
        raise ApiError(422, "invalid_box", "坐标必须是数值") from exc
    if w <= 0 or h <= 0:
        raise ApiError(422, "invalid_box", "bbox 宽高必须大于 0")
    if cx - w / 2 < -1e-6 or cy - h / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6 or cy + h / 2 > 1 + 1e-6:
        raise ApiError(422, "invalid_box", "bbox 超出图像范围")
    return {"class_id": class_id, "cx": cx, "cy": cy, "w": w, "h": h}


def _resolve_target(db: Session, user: CurrentUser, payload: dict) -> tuple[DatasetAsset, InferenceItem | None]:
    item = None
    asset = None
    if payload.get("inference_item_id"):
        item = db.get(InferenceItem, payload["inference_item_id"])
        if item is None:
            raise ApiError(404, "not_found", "推理结果不存在")
        run = db.get(InferenceRun, item.run_id)
        if run is None or (not user.is_admin and run.owner_id != user.id):
            raise ApiError(404, "not_found", "推理结果不存在")
        if item.asset_id:
            asset = db.get(DatasetAsset, item.asset_id)
    if asset is None and payload.get("image_asset_id"):
        asset = dataset_service.get_asset(db, int(payload["image_asset_id"]), user)
    if asset is None:
        raise ApiError(422, "asset_required", "必须提供可访问的图像资产")
    # 归属校验：资产所属数据集必须属于当前用户（管理员除外）
    dataset_service.get_version(db, asset.dataset_version_id, user)
    return asset, item


def _derive_annotations(db: Session, asset: DatasetAsset, item: InferenceItem | None, items: list[dict], class_map: dict) -> tuple[list[dict], list[ReviewItem]]:
    base: list[dict] = []
    predictions: dict[str, InferencePrediction] = {}
    if item is not None:
        rows = db.query(InferencePrediction).filter(InferencePrediction.item_id == item.id).all()
        predictions = {row.id: row for row in rows}
        base = [_box_from_prediction(row, asset) for row in rows]
    else:
        base = [dict(annotation) for annotation in (asset.annotations_json or [])]

    final: list[dict] = list(base)
    rows: list[ReviewItem] = []
    for entry in items:
        verdict = str(entry.get("verdict") or "").strip()
        if verdict not in VERDICTS:
            raise ApiError(422, "invalid_verdict", f"不支持的复核结论: {verdict}")
        prediction = predictions.get(str(entry.get("prediction_id") or ""))
        original = _box_from_prediction(prediction, asset) if prediction else None
        corrected = None
        if verdict == "unreviewable":
            final = []
        elif verdict == "correct":
            if original is None:
                raise ApiError(422, "prediction_required", "correct 结论必须关联预测")
        elif verdict == "false_positive":
            if original is None:
                raise ApiError(422, "prediction_required", "false_positive 必须关联预测")
            final = [box for box in final if not _same_box(box, original)]
        elif verdict == "wrong_class":
            if original is None:
                raise ApiError(422, "prediction_required", "wrong_class 必须关联预测")
            corrected = _validate_box(entry.get("corrected") or {**original, "class_id": entry.get("class_id")}, class_map)
            final = [box for box in final if not _same_box(box, original)] + [corrected]
        elif verdict == "missed_object":
            corrected = _validate_box(entry.get("corrected") or {}, class_map)
            final.append(corrected)
        rows.append(
            ReviewItem(
                prediction_id=prediction.id if prediction else None,
                verdict=verdict,
                original_json=original or {},
                corrected_json=corrected,
            )
        )
    return final, rows


def _same_box(a: dict, b: dict, tolerance: float = 1e-4) -> bool:
    return (
        int(a["class_id"]) == int(b["class_id"])
        and abs(float(a["cx"]) - float(b["cx"])) < tolerance
        and abs(float(a["cy"]) - float(b["cy"])) < tolerance
        and abs(float(a["w"]) - float(b["w"])) < tolerance
        and abs(float(a["h"]) - float(b["h"])) < tolerance
    )


def create_review(db: Session, user: CurrentUser, payload: dict) -> ReviewRecord:
    asset, item = _resolve_target(db, user, payload)
    version = dataset_service.get_version(db, asset.dataset_version_id, user)
    class_map = version.class_map_json or {}
    final, rows = _derive_annotations(db, asset, item, payload.get("items") or [], class_map)
    if payload.get("final_annotations") is not None:
        final = [_validate_box(box, class_map) for box in payload["final_annotations"]]
    if not rows:
        raise ApiError(422, "items_required", "至少提交一条复核结论")

    review = ReviewRecord(
        owner_id=user.id,
        reviewer_id=user.id,
        inference_item_id=item.id if item else None,
        image_asset_id=asset.id,
        state="draft",
        revision=0,
        reason=payload.get("reason"),
    )
    db.add(review)
    db.flush()
    for row in rows:
        row.review_id = review.id
        db.add(row)
    review.final_annotations_json = final
    db.commit()
    db.refresh(review)
    return review


def update_review(db: Session, review: ReviewRecord, user: CurrentUser, payload: dict) -> ReviewRecord:
    if review.state != "draft":
        raise ApiError(409, "review_locked", "只有 draft 状态可以修改")
    if review.owner_id != user.id and not user.is_admin:
        raise ApiError(404, "not_found", "复核记录不存在")
    if int(review.revision) != int(payload.get("expected_revision", -1)):
        raise ApiError(409, "revision_conflict", "复核记录已被修改", {"server_revision": review.revision})

    asset = db.get(DatasetAsset, review.image_asset_id)
    version = dataset_service.get_version(db, asset.dataset_version_id, user)
    item = db.get(InferenceItem, review.inference_item_id) if review.inference_item_id else None
    final, rows = _derive_annotations(db, asset, item, payload.get("items") or [], version.class_map_json or {})
    if payload.get("final_annotations") is not None:
        final = [_validate_box(box, version.class_map_json or {}) for box in payload["final_annotations"]]

    db.query(ReviewItem).filter(ReviewItem.review_id == review.id).delete()
    for row in rows:
        row.review_id = review.id
        db.add(row)
    review.reason = payload.get("reason", review.reason)
    review.final_annotations_json = final
    review.revision = int(review.revision) + 1
    db.commit()
    db.refresh(review)
    return review


def submit_review(db: Session, review: ReviewRecord, user: CurrentUser) -> ReviewRecord:
    if review.owner_id != user.id and not user.is_admin:
        raise ApiError(404, "not_found", "复核记录不存在")
    if review.state != "draft":
        raise ApiError(409, "invalid_state", f"当前状态不可提交: {review.state}")
    review.state = "submitted"
    review.submitted_at = utcnow()
    review.revision = int(review.revision) + 1
    db.commit()
    return review


def decide_review(db: Session, review: ReviewRecord, user: CurrentUser, *, approve: bool, reason: str | None) -> ReviewRecord:
    if not user.can_review:
        raise ApiError(403, "forbidden", "需要复核审批权限")
    if review.state != "submitted":
        raise ApiError(409, "invalid_state", "只有 submitted 状态可以审批")
    review.state = "approved" if approve else "rejected"
    review.decided_at = utcnow()
    review.decided_by = user.id
    if reason:
        review.reason = reason
    review.revision = int(review.revision) + 1
    db.commit()
    return review


def serialize(db: Session, review: ReviewRecord) -> dict:
    items = db.query(ReviewItem).filter(ReviewItem.review_id == review.id).all()
    return {
        "review_id": review.id,
        "state": review.state,
        "revision": review.revision,
        "reason": review.reason,
        "inference_item_id": review.inference_item_id,
        "image_asset_id": review.image_asset_id,
        "final_annotations": review.final_annotations_json or [],
        "items": [
            {
                "review_item_id": item.id,
                "prediction_id": item.prediction_id,
                "verdict": item.verdict,
                "original": item.original_json or {},
                "corrected": item.corrected_json,
            }
            for item in items
        ],
        "created_at": review.created_at.isoformat() if review.created_at else None,
        "submitted_at": review.submitted_at.isoformat() if review.submitted_at else None,
        "decided_at": review.decided_at.isoformat() if review.decided_at else None,
    }
