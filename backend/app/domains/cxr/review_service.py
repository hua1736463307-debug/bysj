"""CXR 逐标签人工复核：draft → submitted → approved/rejected，CAS 保存。

- 原模型分数以 `model_snapshot_json` 独立保存，任何编辑都不覆盖它；
- 提交/审批记录**实际 actor**，不冒用医生署名；
- 未完成全部标签意见或含 indeterminate 时不允许批准，不自动通过。
"""
from __future__ import annotations

from sqlalchemy.orm import Session

from app.api.deps import CurrentUser
from app.core.errors import ApiError
from app.db.base import utcnow
from app.db.models import CxrAnalysis, CxrReview
from app.domains.cxr import service as cxr_service
from app.schemas.cxr import LABEL_KEYS, LABEL_INDEX, build_label_results

EDITABLE_STATES = {"draft"}
DECIDABLE_STATES = {"submitted"}


def _snapshot(analysis: CxrAnalysis) -> dict:
    return {
        "analysis_id": analysis.id,
        "model_id": analysis.model_id,
        "model_sha256": analysis.model_sha256,
        "label_schema": analysis.label_schema,
        "calibration_status": analysis.calibration_status,
        "threshold_source": analysis.threshold_source,
        "raw_scores": analysis.raw_scores_json or {},
        "calibrated_scores": analysis.calibrated_scores_json,
        "thresholds": analysis.thresholds_json,
        "labels": build_label_results(
            analysis.raw_scores_json or {},
            analysis.calibrated_scores_json,
            analysis.thresholds_json,
            calibration_status=analysis.calibration_status,
            threshold_source=analysis.threshold_source,
        ),
    }


def _normalize_items(items: list[dict]) -> list[dict]:
    seen: dict[str, dict] = {}
    for item in items:
        key = item["label_key"]
        if key in seen:
            raise ApiError(422, "cxr_duplicate_label", f"标签 {key} 重复填写意见")
        seen[key] = {
            "label_key": key,
            "display_name": item.get("display_name") or None,
            "opinion": item["opinion"],
            "reason": item.get("reason"),
        }
    order = {key: index for index, key in enumerate(LABEL_KEYS)}
    return sorted(seen.values(), key=lambda item: order[item["label_key"]])


def create_review(db: Session, user: CurrentUser, payload: dict) -> CxrReview:
    analysis = cxr_service.get_analysis(db, payload["analysis_id"], user)
    if analysis.status != "completed":
        raise ApiError(409, "cxr_analysis_not_ready", "只有已完成的分析可以复核")
    existing = (
        db.query(CxrReview)
        .filter(CxrReview.analysis_id == analysis.id, CxrReview.owner_id == user.id)
        .one_or_none()
    )
    if existing is not None:
        raise ApiError(409, "cxr_review_exists", "该分析已有复核记录，请使用 PATCH 更新草稿")
    review = CxrReview(
        analysis_id=analysis.id,
        owner_id=user.id,
        reviewer_id=user.id,
        reviewer_username=user.username,
        state="draft",
        revision=0,
        items_json=_normalize_items(payload.get("items") or []),
        model_snapshot_json=_snapshot(analysis),
        reason=payload.get("reason"),
    )
    db.add(review)
    db.commit()
    db.refresh(review)
    return review


def update_review(db: Session, review: CxrReview, user: CurrentUser, payload: dict) -> CxrReview:
    if review.owner_id != user.id and not user.is_admin:
        raise ApiError(404, "not_found", "复核记录不存在")
    if review.state not in EDITABLE_STATES:
        raise ApiError(409, "cxr_review_locked", f"当前状态 {review.state} 不可编辑，草稿保留在提交前的版本")
    changed = (
        db.query(CxrReview)
        .filter(CxrReview.id == review.id, CxrReview.revision == payload["expected_revision"],
                CxrReview.state == "draft")
        .update(
            {
                "items_json": _normalize_items(payload.get("items") or []),
                "reason": payload.get("reason"),
                "revision": review.revision + 1,
                "updated_at": utcnow(),
            },
            synchronize_session=False,
        )
    )
    if not changed:
        db.rollback()
        raise ApiError(409, "cxr_revision_conflict", "复核记录已被修改，草稿已保留，请刷新后重试")
    db.commit()
    db.refresh(review)
    return review


def submit_review(db: Session, review: CxrReview, user: CurrentUser) -> CxrReview:
    if review.owner_id != user.id and not user.is_admin:
        raise ApiError(404, "not_found", "复核记录不存在")
    if review.state != "draft":
        raise ApiError(409, "cxr_review_state", f"只有草稿可以提交，当前 {review.state}")
    now = utcnow()
    changed = (
        db.query(CxrReview)
        .filter(CxrReview.id == review.id, CxrReview.state == "draft")
        .update(
            {
                "state": "submitted",
                "submitted_at": now,
                "updated_at": now,
                "reviewer_id": user.id,
                "reviewer_username": user.username,
            },
            synchronize_session=False,
        )
    )
    if not changed:
        db.rollback()
        raise ApiError(409, "cxr_review_state", "复核状态已变化，请刷新")
    db.commit()
    db.refresh(review)
    return review


def decide_review(db: Session, review: CxrReview, user: CurrentUser, *, approve: bool, reason: str | None) -> CxrReview:
    if not user.can_review:
        raise ApiError(403, "forbidden", "需要复核权限")
    if review.state not in DECIDABLE_STATES:
        raise ApiError(409, "cxr_review_state", f"只有已提交的复核可以审批，当前 {review.state}")
    if approve:
        items = review.items_json or []
        covered = {item["label_key"] for item in items}
        missing = [key for key in LABEL_KEYS if key not in covered]
        indeterminate = [item["label_key"] for item in items if item.get("opinion") == "indeterminate"]
        if missing:
            raise ApiError(409, "cxr_review_incomplete", "五项标签均需给出意见后才能批准", {"missing": missing})
        if indeterminate:
            raise ApiError(
                409,
                "cxr_review_indeterminate",
                "存在无法判断的标签，不能自动批准",
                {"indeterminate": indeterminate},
            )
    now = utcnow()
    changed = (
        db.query(CxrReview)
        .filter(CxrReview.id == review.id, CxrReview.state == "submitted")
        .update(
            {
                "state": "approved" if approve else "rejected",
                "decided_at": now,
                "decided_by_id": user.id,
                "decided_by_username": user.username,
                "updated_at": now,
            },
            synchronize_session=False,
        )
    )
    if not changed:
        db.rollback()
        raise ApiError(409, "cxr_review_state", "复核状态已变化，请刷新")
    db.commit()
    db.refresh(review)
    return review


def label_matrix(review: CxrReview) -> list[dict]:
    """逐标签并列模型值与人工意见（只读展示）。"""
    snapshot = review.model_snapshot_json or {}
    labels = {item["label_key"]: item for item in snapshot.get("labels") or []}
    opinions = {item["label_key"]: item for item in review.items_json or []}
    output: list[dict] = []
    for key in LABEL_KEYS:
        model = labels.get(key) or {}
        opinion = opinions.get(key) or {}
        output.append(
            {
                "label_key": key,
                "index": LABEL_INDEX[key],
                "display_name": model.get("display_name"),
                "raw_score": model.get("raw_score"),
                "calibrated_score": model.get("calibrated_score"),
                "threshold": model.get("threshold"),
                "flag": model.get("flag"),
                "calibration_status": model.get("calibration_status"),
                "opinion": opinion.get("opinion"),
                "opinion_reason": opinion.get("reason"),
            }
        )
    return output
