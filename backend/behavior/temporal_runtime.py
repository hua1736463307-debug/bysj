"""Registry-aware temporal-fusion runtime for batch and realtime behavior paths."""

from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np

from .temporal_fusion import (
    DEFAULT_CLIP_LENGTH,
    DEFAULT_SAMPLE_FPS,
    DEFAULT_THRESHOLDS,
    MODEL_FAMILY,
    TEMPORAL_EVENT_TYPES,
    TemporalFusionStream,
    load_temporal_model,
)

logger = logging.getLogger(__name__)


def _find_temporal_record(model_id: Optional[str] = None) -> Optional[Any]:
    try:
        from database import ModelRegistry, SessionLocal

        db = SessionLocal()
        try:
            query = db.query(ModelRegistry).filter(
                ModelRegistry.source == "behavior_temporal",
                ModelRegistry.status.in_(("active", "candidate")),
            )
            if model_id:
                return query.filter(ModelRegistry.id == str(model_id)).first()
            return query.filter(ModelRegistry.status == "active").order_by(ModelRegistry.updated_at.desc()).first()
        finally:
            db.close()
    except Exception as exc:
        logger.warning("读取端到端行为模型注册信息失败: %s", exc)
        return None


class TemporalModelRuntime:
    """One model/stream instance per batch task or realtime source session."""

    def __init__(
        self,
        *,
        mode: str = "off",
        model_id: Optional[str] = None,
        model_path: Optional[str] = None,
        thresholds: Optional[Dict[str, float]] = None,
        clip_length: int = DEFAULT_CLIP_LENGTH,
        sample_fps: float = DEFAULT_SAMPLE_FPS,
    ):
        self.mode = str(mode or "off").strip().lower()
        if self.mode == "augment":
            self.mode = "filter"
        if self.mode not in {"off", "shadow", "filter"}:
            self.mode = "off"
        self.requested_model_id = str(model_id or "")
        self.model_id = ""
        self.model_sha256 = ""
        self.model_path = str(model_path or "")
        self.metadata: Dict[str, Any] = {}
        self.stream: Optional[TemporalFusionStream] = None
        self.error: Optional[str] = None
        self._lock = threading.RLock()
        if self.mode != "off":
            self._load(
                thresholds=thresholds,
                clip_length=clip_length,
                sample_fps=sample_fps,
            )

    @property
    def available(self) -> bool:
        return self.stream is not None and not self.error

    def _load(self, *, thresholds: Optional[Dict[str, float]], clip_length: int, sample_fps: float) -> None:
        record = _find_temporal_record(self.requested_model_id or None)
        if record is not None:
            if self.mode == "filter" and str(record.status or "") != "active":
                self.error = "过滤模式只能使用已激活的端到端行为模型"
                return
            self.model_id = str(record.id)
            self.model_sha256 = str(record.sha256 or "")
            self.model_path = str(record.path)
            self.metadata = dict(record.metadata_json or {})
        if not self.model_path:
            self.error = "未找到激活的端到端行为模型"
            return
        try:
            model, metadata = load_temporal_model(self.model_path)
            self.metadata = {**metadata, **self.metadata}
            self.stream = TemporalFusionStream(
                model,
                model_id=self.model_id or Path(self.model_path).stem,
                metadata=self.metadata,
                clip_length=int(self.metadata.get("clip_length") or clip_length),
                sample_fps=float(self.metadata.get("sample_fps") or sample_fps),
                thresholds=thresholds or DEFAULT_THRESHOLDS,
            )
            self.error = None
        except Exception as exc:
            self.error = str(exc)
            logger.warning("端到端行为模型加载失败，保持规则兜底: %s", exc)

    def update(
        self,
        frame: np.ndarray,
        persons: Dict[int, Dict[str, Any]],
        objects: Optional[Dict[Any, Dict[str, Any]]],
        timestamp: float,
        *,
        enabled_events: Optional[Iterable[str]] = None,
        input_is_rgb: bool = False,
    ) -> Dict[str, Any]:
        with self._lock:
            if not self.available:
                return {
                    "events": [],
                    "score_snapshots": [],
                    "mode": self.mode,
                    "available": False,
                    "model_id": self.model_id,
                    "model_sha256": self.model_sha256,
                    "error": self.error,
                }
            result = self.stream.update(
                frame,
                persons,
                objects,
                timestamp,
                enabled_events=enabled_events,
                input_is_rgb=input_is_rgb,
            )
            result.update({
                "mode": self.mode,
                "available": True,
                "model_id": self.model_id,
                "model_sha256": self.model_sha256,
                "model_version": self.metadata.get("display_name") or self.model_id,
            })
            return result

    def status(self) -> Dict[str, Any]:
        return {
            "enabled": self.mode != "off",
            "mode": self.mode,
            "available": self.available,
            "model_id": self.model_id,
            "model_sha256": self.model_sha256,
            "model_path": self.model_path,
            "model_family": MODEL_FAMILY,
            "supported_event_types": list(TEMPORAL_EVENT_TYPES),
            "metadata": self.metadata,
            "error": self.error,
        }
