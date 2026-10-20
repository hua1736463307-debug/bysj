"""Persistent offline behavior-analysis job execution shared by API and worker."""

from __future__ import annotations

import logging
import os
import time
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from database import BehaviorReviewSample, JobTask, ModelRegistry, SessionLocal
from job_store import is_cancel_requested, update_job
from model_registry import sha256_file
from utils import constants

from .analyzer import BehaviorAnalysisCancelled, BehaviorAnalyzer
from .event_store import persist_behavior_transitions
from .realtime_engine import normalize_behavior_config
from .temporal_fusion import TEMPORAL_EVENT_TYPES, train_temporal_fusion_from_samples


logger = logging.getLogger(__name__)


def _safe_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        normalized = float(value)
    except (TypeError, ValueError):
        normalized = default
    return max(minimum, min(maximum, normalized))


def _safe_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        normalized = int(value)
    except (TypeError, ValueError):
        normalized = default
    return max(minimum, min(maximum, normalized))


def run_behavior_analysis_payload(
    payload: Dict[str, Any],
    *,
    output_task_id: Optional[str] = None,
    progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
    cancel_callback: Optional[Callable[[], bool]] = None,
) -> Dict[str, Any]:
    """Execute one normalized behavior request without importing the API app."""
    behavior_config = normalize_behavior_config(payload.get("behavior_config")) if payload.get("behavior_config") else None
    verifier_mode = str(payload.get("verifier_mode") or "off").strip().lower()
    if verifier_mode == "augment":
        verifier_mode = "filter"
    if verifier_mode not in {"off", "shadow", "filter"}:
        verifier_mode = "off"
    analyzer = BehaviorAnalyzer(
        model_path=str(payload.get("model_path") or "yolov8n-pose.pt"),
        output_root=str(Path(constants.STATIC_DIR) / "behavior_results"),
        conf=_safe_float(payload.get("conf"), 0.35, 0.0, 1.0),
        iou=_safe_float(payload.get("iou"), 0.50, 0.0, 1.0),
        tracker=str(payload.get("tracker") or "bytetrack.yaml"),
        imgsz=_safe_int(payload.get("imgsz"), 640, 160, 1920),
        object_model_path=str(payload.get("object_model_path") or "/app/models/yolov8n.pt"),
        object_conf=_safe_float(payload.get("object_conf"), 0.25, 0.0, 1.0),
        object_imgsz=payload.get("object_imgsz"),
        transfer_classes=payload.get("phone_classes") or payload.get("transfer_classes"),
        object_owner_distance_ratio=_safe_float(payload.get("object_owner_distance_ratio"), 0.35, 0.0, 2.0),
        transfer_max_seconds=_safe_float(payload.get("transfer_max_seconds"), 3.0, 0.1, 60.0),
        transfer_min_owner_frames=_safe_int(payload.get("transfer_min_owner_frames"), 3, 1, 300),
        transfer_person_distance_ratio=_safe_float(payload.get("transfer_person_distance_ratio"), 0.85, 0.0, 4.0),
        draw_objects=bool(payload.get("draw_objects", False)),
        extract_event_clips=bool(payload.get("extract_event_clips", True)),
        clip_pre_seconds=_safe_float(payload.get("clip_pre_seconds"), 1.0, 0.0, 30.0),
        clip_post_seconds=_safe_float(payload.get("clip_post_seconds"), 1.0, 0.0, 30.0),
        clip_min_seconds=_safe_float(payload.get("clip_min_seconds"), 2.0, 0.1, 120.0),
        clip_max_seconds=_safe_float(payload.get("clip_max_seconds"), 8.0, 0.0, 300.0),
        enable_model_review=verifier_mode != "off",
        require_model_review=bool(payload.get("require_model_review", False)),
        classifier_model_path=str(payload.get("verifier_model_path") or payload.get("classifier_model_path") or ""),
        classifier_threshold=_safe_float(payload.get("verifier_threshold", payload.get("classifier_threshold")), 0.65, 0.0, 1.0),
        classifier_max_frames=_safe_int(payload.get("classifier_max_frames"), 96, 8, 512),
        verifier_mode=verifier_mode,
        temporal_mode=str(payload.get("temporal_mode") or "off").strip().lower(),
        temporal_model_id=str(payload.get("temporal_model_id") or ""),
        temporal_model_path=str(payload.get("temporal_model_path") or ""),
        temporal_thresholds=payload.get("temporal_thresholds") if isinstance(payload.get("temporal_thresholds"), dict) else None,
        temporal_clip_length=_safe_int(payload.get("temporal_clip_length"), 12, 4, 48),
        temporal_sample_fps=_safe_float(payload.get("temporal_sample_fps"), 6.0, 1.0, 15.0),
        keep_candidate_events=bool(payload.get("keep_candidate_events", True)),
        keep_rejected_events=bool(payload.get("keep_rejected_events", False)),
        fallback_to_rule_events=True if verifier_mode == "shadow" else bool(payload.get("fallback_to_rule_events", False)),
    )
    return analyzer.analyze_video(
        video_path=str(payload.get("video_path") or ""),
        enabled_events=(behavior_config.get("enabled_events") if behavior_config else payload.get("enabled_events")),
        sample_stride=_safe_int(payload.get("sample_stride"), 1, 1, 30),
        output_task_id=output_task_id,
        max_frames=payload.get("max_frames"),
        progress_callback=progress_callback,
        cancel_callback=cancel_callback,
        generate_annotated_video=bool(payload.get("generate_annotated_video", False)),
        annotated_video_fps=_safe_float(payload.get("annotated_video_fps"), 6.0, 3.0, 6.0),
        object_model_path=(behavior_config or {}).get("object_model_path", payload.get("object_model_path")),
        object_conf=(behavior_config or {}).get("object_confidence", payload.get("object_conf")),
        transfer_classes=(behavior_config or {}).get(
            "phone_classes",
            payload.get("phone_classes") or payload.get("transfer_classes"),
        ),
        extract_event_clips=bool(payload.get("extract_event_clips", True)),
        clip_pre_seconds=payload.get("clip_pre_seconds"),
        clip_post_seconds=payload.get("clip_post_seconds"),
        clip_min_seconds=payload.get("clip_min_seconds"),
        clip_max_seconds=payload.get("clip_max_seconds"),
        keep_candidate_events=bool(payload.get("keep_candidate_events", True)),
        keep_rejected_events=bool(payload.get("keep_rejected_events", False)),
        behavior_config=behavior_config,
        temporal_mode=str(payload.get("temporal_mode") or "off").strip().lower(),
        temporal_model_id=str(payload.get("temporal_model_id") or ""),
        temporal_model_path=str(payload.get("temporal_model_path") or ""),
        temporal_thresholds=payload.get("temporal_thresholds") if isinstance(payload.get("temporal_thresholds"), dict) else None,
        temporal_clip_length=_safe_int(payload.get("temporal_clip_length"), 12, 4, 48),
        temporal_sample_fps=_safe_float(payload.get("temporal_sample_fps"), 6.0, 1.0, 15.0),
    )


def _media_path(value: Any) -> Optional[str]:
    text_value = str(value or "").strip()
    if not text_value:
        return None
    if text_value.startswith("/api/static/"):
        return "/app/static/" + text_value[len("/api/static/"):]
    if text_value.startswith("/static/"):
        return "/app/static/" + text_value[len("/static/"):]
    return text_value


def persist_batch_behavior_events(task: JobTask, result: Dict[str, Any]) -> None:
    """Preserve the existing batch-to-behavior-event projection."""
    try:
        owner_id = int(task.owner_id) if task.owner_id is not None else None
    except (TypeError, ValueError):
        owner_id = None
    created_at = task.created_at or datetime.now()
    base_timestamp = created_at.timestamp()
    confirmed = []
    resolved = []
    for event in result.get("events") or []:
        event_id = str(event.get("event_id") or uuid.uuid4())
        start_time = base_timestamp + float(event.get("start_time") or 0.0)
        confirmed_time = base_timestamp + float(event.get("confirmed_time", event.get("start_time")) or 0.0)
        end_time = base_timestamp + float(event.get("end_time") or event.get("start_time") or 0.0)
        item = {
            "event_id": event_id,
            "event_type": event.get("event_type"),
            "state": "confirmed",
            "severity": event.get("severity") or "low",
            "rule_score": float(event.get("rule_score", event.get("confidence", 0.0)) or 0.0),
            "track_ids": list(event.get("persons") or []),
            "object_id": event.get("object_id"),
            "object_class": event.get("object_class"),
            "reason": dict(event.get("reason") or {}),
            "decision": {
                "rule": event.get("rule_decision") or {},
                "temporal": event.get("temporal_decision") or (
                    event.get("reason") if (event.get("reason") or {}).get("model_family") == "temporal_fusion" else {}
                ),
                "final": event.get("final_decision") or {},
                "input_quality": event.get("input_quality") or {},
                "temporal_span": event.get("temporal_span") or {},
                "fallback": bool(event.get("fallback", True)),
                "training_feature_vector": event.get("training_feature_vector") or [],
            },
            "decision_model_id": event.get("model_id") or (event.get("reason") or {}).get("model_id"),
            "decision_model_version": event.get("model_version") or (event.get("reason") or {}).get("model_version"),
            "started_at": start_time,
            "confirmed_at": confirmed_time,
            "snapshot_path": _media_path(event.get("snapshot")),
            "clip_path": _media_path(event.get("clip")),
            "evidence_status": "ready" if event.get("snapshot") or event.get("clip") else "failed",
            "evidence_error": event.get("clip_error"),
        }
        confirmed.append(item)
        resolved.append({**item, "state": "resolved", "ended_at": end_time})
    context = {
        "owner_id": owner_id,
        "owner_username": str(task.owner_username or "unknown"),
        "mode": "batch",
        "source_id": str((task.input_json or {}).get("source_id") or (task.input_json or {}).get("video_path") or task.id),
        "session_id": task.id,
    }
    persist_behavior_transitions(confirmed, **context)
    persist_behavior_transitions(resolved, **context)


def execute_behavior_job(task_id: str) -> Dict[str, Any]:
    """Run one claimed behavior job and persist all state transitions."""
    db = SessionLocal()
    try:
        task = db.query(JobTask).filter(
            JobTask.id == task_id,
            JobTask.type.in_({"behavior_analysis", "behavior_temporal_training"}),
        ).first()
        if task is None:
            raise KeyError(f"behavior job not found: {task_id}")
        payload = dict(task.input_json or {})
    finally:
        db.close()

    if task.type == "behavior_temporal_training":
        return execute_temporal_training_job(task_id)

    def update_progress(progress: Dict[str, Any]) -> None:
        progress_db = SessionLocal()
        try:
            current = progress_db.query(JobTask).filter(JobTask.id == task_id).first()
            if current is None or current.status in {"completed", "failed", "cancelled"}:
                return
            update_job(
                progress_db,
                task_id,
                stage="analyzing",
                progress=float(progress.get("progress") or 0.0),
                message=str(progress.get("message") or "正在分析视频"),
                worker_pid=os.getpid(),
                event_type="progress",
                event_payload={
                    "frame_idx": int(progress.get("frame_idx") or 0),
                    "processed_frames": int(progress.get("processed_frames") or 0),
                    "total_frames": int(progress.get("total_frames") or 0),
                    "elapsed_seconds": float(progress.get("elapsed_seconds") or 0.0),
                },
            )
        finally:
            progress_db.close()

    try:
        result = run_behavior_analysis_payload(
            payload,
            output_task_id=task_id,
            progress_callback=update_progress,
            cancel_callback=lambda: is_cancel_requested(task_id),
        )
        if is_cancel_requested(task_id):
            raise BehaviorAnalysisCancelled("行为分析任务已取消")
        result_db = SessionLocal()
        try:
            current = result_db.query(JobTask).filter(JobTask.id == task_id).first()
            if current is None:
                raise KeyError(f"behavior job not found: {task_id}")
            persist_batch_behavior_events(current, result)
            update_job(
                result_db,
                task_id,
                status="completed",
                stage="completed",
                progress=100.0,
                message="行为分析完成",
                result=result,
                worker_pid=os.getpid(),
                event_type="completed",
                event_payload={"result_task_id": result.get("task_id")},
            )
        finally:
            result_db.close()
        return result
    except BehaviorAnalysisCancelled:
        cancel_db = SessionLocal()
        try:
            update_job(
                cancel_db,
                task_id,
                status="cancelled",
                stage="cancelled",
                message="行为分析任务已取消",
                worker_pid=os.getpid(),
                event_type="cancelled",
            )
        finally:
            cancel_db.close()
        raise


def _temporal_training_samples(config: Dict[str, Any]) -> list[Dict[str, Any]]:
    db = SessionLocal()
    try:
        query = db.query(BehaviorReviewSample).filter(
            BehaviorReviewSample.review_status == "reviewed",
            ~BehaviorReviewSample.review_label.in_(("uncertain", "delete")),
        )
        if not bool(config.get("include_all_reviewers", True)):
            query = query.filter(
                BehaviorReviewSample.owner_username == str(config.get("owner_username") or "")
            )
        return [row.to_training_sample() for row in query.order_by(BehaviorReviewSample.reviewed_at.asc()).all()]
    finally:
        db.close()


def execute_temporal_training_job(task_id: str) -> Dict[str, Any]:
    """Train/register one temporal candidate inside the behavior Worker."""
    db = SessionLocal()
    try:
        task = db.query(JobTask).filter(
            JobTask.id == task_id,
            JobTask.type == "behavior_temporal_training",
        ).first()
        if task is None:
            raise KeyError(f"temporal training job not found: {task_id}")
        config = dict(task.input_json or {})
    finally:
        db.close()

    def progress(value: float, message: str) -> None:
        progress_db = SessionLocal()
        try:
            current = progress_db.query(JobTask).filter(JobTask.id == task_id).first()
            if current is None or current.status in {"completed", "failed", "cancelled"}:
                return
            update_job(
                progress_db,
                task_id,
                stage="training_temporal_model",
                progress=float(value),
                message=message,
                worker_pid=os.getpid(),
                event_type="progress",
            )
        finally:
            progress_db.close()

    try:
        progress(5.0, "正在读取行为复核样本...")
        samples = _temporal_training_samples(config)
        if not samples:
            raise ValueError("没有可用于端到端训练的已复核样本")
        if set(config.get("event_types") or TEMPORAL_EVENT_TYPES) != set(TEMPORAL_EVENT_TYPES):
            raise ValueError(f"端到端模型只支持三类行为: {list(TEMPORAL_EVENT_TYPES)}")

        def cancelled() -> bool:
            if is_cancel_requested(task_id):
                raise BehaviorAnalysisCancelled("端到端行为训练已取消")
            return False

        result = train_temporal_fusion_from_samples(
            samples,
            str(config.get("model_path") or ""),
            clip_length=int(config.get("clip_length") or 12),
            image_size=int(config.get("image_size") or 64),
            epochs=int(config.get("epochs") or 3),
            batch_size=int(config.get("batch_size") or 2),
            learning_rate=float(config.get("learning_rate") or 0.0005),
            device=str(config.get("device") or "auto"),
            thresholds=dict(config.get("thresholds") or {}),
            progress_callback=progress,
            cancel_callback=cancelled,
        )
        model_path = Path(str(result["model_path"])).resolve()
        if not model_path.is_file():
            raise FileNotFoundError("端到端模型训练完成但模型文件不存在")
        metadata = dict(result.get("metadata") or {})
        model_id = str(uuid.uuid4())
        model_record = ModelRegistry(
            id=model_id,
            name=model_path.name,
            path=str(model_path),
            sha256=sha256_file(model_path),
            size=model_path.stat().st_size,
            source="behavior_temporal",
            status="candidate",
            format="pt",
            task_id=task_id,
            created_by=str(config.get("owner_username") or "unknown"),
            metadata_json={
                **metadata,
                "display_name": f"行为时序融合模型 {task_id[:8]}",
                "model_family": "temporal_fusion",
                "verifier_mode": "shadow",
                "supported_event_types": list(TEMPORAL_EVENT_TYPES),
                "activation_scope": "per_event_head",
                "parent_model_id": config.get("parent_model_id"),
            },
        )
        db = SessionLocal()
        try:
            db.add(model_record)
            db.flush()
            result.update({
                "model_id": model_id,
                "model_status": "candidate",
                "model_sha256": model_record.sha256,
            })
            update_job(
                db,
                task_id,
                status="completed",
                stage="candidate_ready",
                progress=100.0,
                message="端到端行为模型训练完成，已登记为候选版本",
                result=result,
                worker_pid=os.getpid(),
                event_type="completed",
                event_payload={"model_id": model_id, "model_status": "candidate"},
            )
        finally:
            db.close()
        return result
    except BehaviorAnalysisCancelled:
        cancel_db = SessionLocal()
        try:
            current = cancel_db.query(JobTask).filter(JobTask.id == task_id).first()
            if current is not None and current.status not in {"completed", "cancelled"}:
                update_job(
                    cancel_db,
                    task_id,
                    status="cancelled",
                    stage="cancelled",
                    progress=100.0,
                    message="端到端行为训练已取消",
                    worker_pid=os.getpid(),
                    event_type="cancelled",
                )
        finally:
            cancel_db.close()
        raise
    except Exception as exc:
        logger.exception("端到端行为训练失败: task_id=%s", task_id)
        from gpu_probe_service import is_gpu_fault
        if is_gpu_fault(exc):
            raise
        failed_db = SessionLocal()
        try:
            current = failed_db.query(JobTask).filter(JobTask.id == task_id).first()
            if current is not None and current.status not in {"completed", "failed", "cancelled"}:
                update_job(
                    failed_db,
                    task_id,
                    status="failed",
                    stage="failed",
                    progress=100.0,
                    message=f"端到端行为训练失败: {exc}",
                    error_code="behavior_temporal_training_failed",
                    error_detail=str(exc),
                    worker_pid=os.getpid(),
                    event_type="failed",
                    level="error",
                )
        finally:
            failed_db.close()
        raise
    except Exception as exc:
        logger.exception("行为分析任务失败: task_id=%s", task_id)
        failed_db = SessionLocal()
        try:
            update_job(
                failed_db,
                task_id,
                status="failed",
                stage="failed",
                message=f"行为分析失败: {exc}",
                error_code="behavior_analysis_failed",
                error_detail=str(exc),
                worker_pid=os.getpid(),
                event_type="failed",
                level="error",
            )
        finally:
            failed_db.close()
        raise
