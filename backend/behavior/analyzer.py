"""Object + Pose temporal behavior analyzer.

Backward compatible with the stage-1 API:
- Existing endpoints can keep instantiating BehaviorAnalyzer(model_path, conf, iou, ...)
- analyze_video(video_path, enabled_events, sample_stride) still works

The primary detection path combines a configured YOLO object model with a
configured YOLO Pose model. A human-feedback verifier can optionally score
the resulting rule candidates in shadow mode or filter them after detection.
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from datetime import datetime
from collections import defaultdict, deque
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

from .object_tracker import SimpleObjectTracker
from .clip_extractor import extract_event_clips as run_extract_event_clips
from .video_output import BrowserCompatibleVideoWriter
from .realtime_engine import (
    BehaviorRuleEngine,
    PHONE_CLASS_ALIASES,
    assign_badge_detections,
    matching_class_names,
    normalized_class_names,
    normalize_behavior_config,
)
from .feedback_mapping import SUPPORTED_BEHAVIOR_EVENTS, normalize_behavior_events
from .temporal_runtime import TemporalModelRuntime

logger = logging.getLogger(__name__)


class BehaviorAnalysisCancelled(RuntimeError):
    """Raised when a batch behavior analysis observes a cooperative cancel request."""

try:
    from .action_classifier import (
        DEFAULT_MODEL_PATH as ACTION_CLASSIFIER_DEFAULT_MODEL_PATH,
        LABEL_NAME_MAP as ACTION_LABEL_NAME_MAP,
        load_action_verifier,
    )
except Exception as exc:
    ACTION_CLASSIFIER_DEFAULT_MODEL_PATH = "/app/models/action_event_classifier.pkl"
    ACTION_LABEL_NAME_MAP: Dict[str, str] = {}
    load_action_verifier = None
    logger.warning("人工反馈验证器模块导入失败: %s", exc)

# COCO-style pose keypoint indices used by YOLO pose models.
NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16

DEFAULT_EVENTS = list(SUPPORTED_BEHAVIOR_EVENTS)

DEFAULT_TRANSFER_CLASSES = {
    "cell phone",
}

EVENT_NAME_MAP = {
    "phone_use": "\u4f7f\u7528\u624b\u673a",
    "fall_down": "\u4eba\u5458\u6454\u5012",
    "missing_badge": "\u672a\u4f69\u6234\u5de5\u724c",
    "normal": "\u6b63\u5e38\u884c\u4e3a",
    "uncertain": "\u4e0d\u786e\u5b9a",
}

RULE_TO_FINAL_LABEL = {
    "phone_use": "phone_use",
    "fall_down": "fall_down",
    "missing_badge": "missing_badge",
}

REJECT_LABELS = {"normal", "uncertain", "delete", ""}


def _normalize_phone_classes(values: Iterable[Any]) -> set:
    configured = normalized_class_names(values)
    return configured & PHONE_CLASS_ALIASES or {"cell phone"}


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def dist(p1: Any, p2: Any) -> float:
    if p1 is None or p2 is None:
        return 1e9
    return math.sqrt((float(p1[0]) - float(p2[0])) ** 2 + (float(p1[1]) - float(p2[1])) ** 2)


def bbox_center(box: Iterable[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_size(box: Iterable[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    return max(1.0, x2 - x1), max(1.0, y2 - y1)


def bbox_iou(box_a: Iterable[float], box_b: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(v) for v in box_a]
    bx1, by1, bx2, by2 = [float(v) for v in box_b]

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih

    area_a = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    area_b = max(1.0, (bx2 - bx1) * (by2 - by1))
    return float(inter / (area_a + area_b - inter + 1e-6))


def point_in_upper_body(point: Any, box: Iterable[float]) -> bool:
    if point is None:
        return False
    x, y = float(point[0]), float(point[1])
    x1, y1, x2, y2 = [float(v) for v in box]
    h = max(1.0, y2 - y1)
    return x1 <= x <= x2 and y1 <= y <= y1 + 0.65 * h


def valid_point(point: Any) -> bool:
    if point is None or len(point) < 2:
        return False
    if len(point) >= 3 and float(point[2]) < 0.25:
        return False
    return float(point[0]) > 1 and float(point[1]) > 1


def get_kpt(person: Dict[str, Any], idx: int) -> Optional[Tuple[float, float]]:
    kpts = person.get("keypoints") or []
    if len(kpts) <= idx:
        return None
    p = kpts[idx]
    min_confidence = float(person.get("keypoint_confidence", 0.25) or 0.25)
    if not valid_point(p) or (len(p) >= 3 and float(p[2]) < min_confidence):
        return None
    return float(p[0]), float(p[1])


def ensure_dir(path: Any) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def create_time_named_output_dir(output_root: Any, preferred_task_id: Optional[str] = None) -> Tuple[str, Path]:
    """
    Create a behavior analysis output directory named by processing start time.

    Directory name format:
    - 2026-05-18_15-41-23
    - 2026-05-18_15-41-23_01 when the same second already exists

    The function creates the directory with exist_ok=False to avoid collisions
    when multiple analysis tasks start in the same second.
    """
    root = ensure_dir(output_root)
    safe_preferred = "".join(
        character for character in str(preferred_task_id or "")
        if character.isalnum() or character in {"-", "_"}
    ).strip("-_")
    time_name = safe_preferred or datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for index in range(1000):
        task_id = time_name if index == 0 else f"{time_name}_{index:02d}"
        candidate = root / task_id
        try:
            candidate.mkdir(parents=True, exist_ok=False)
            return task_id, candidate
        except FileExistsError:
            continue

    # Extremely unlikely fallback. Keep it filesystem-safe and still time based.
    fallback_id = f"{time_name}_{int(time.time() * 1000)}"
    fallback_dir = root / fallback_id
    fallback_dir.mkdir(parents=True, exist_ok=False)
    return fallback_id, fallback_dir


def model_path_exists(model_path: str) -> bool:
    if not model_path:
        return False
    if os.path.exists(model_path):
        return True
    if os.path.exists(os.path.join("models", model_path)):
        return True
    if os.path.exists(os.path.join("/app/models", os.path.basename(model_path))):
        return True
    if os.path.exists(os.path.join("/app/models/pretrained", os.path.basename(model_path))):
        return True
    return False


def resolve_model_path(model_path: str) -> str:
    text = str(model_path or "").strip()
    if not text:
        return text
    if os.path.exists(text):
        return text
    candidates = [
        os.path.join("models", text),
        os.path.join("/app/models", os.path.basename(text)),
        os.path.join("/app/models/pretrained", os.path.basename(text)),
        os.path.basename(text),
    ]
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    return text


class BehaviorAnalyzer:
    def __init__(
        self,
        model_path: str = "/app/models/yolov8n-pose.pt",
        output_root: str = "/app/static/behavior_results",
        conf: float = 0.35,
        iou: float = 0.5,
        tracker: str = "bytetrack.yaml",
        imgsz: int = 640,
        object_model_path: str = "/app/models/yolov8n.pt",
        object_conf: float = 0.25,
        object_imgsz: Optional[int] = None,
        transfer_classes: Optional[List[str]] = None,
        object_owner_distance_ratio: float = 0.35,
        transfer_max_seconds: float = 3.0,
        transfer_min_owner_frames: int = 3,
        transfer_person_distance_ratio: float = 0.85,
        draw_objects: bool = False,
        extract_event_clips: bool = True,
        clip_pre_seconds: float = 1.0,
        clip_post_seconds: float = 1.0,
        clip_min_seconds: float = 2.0,
        clip_max_seconds: float = 8.0,
        enable_model_review: bool = False,
        require_model_review: bool = False,
        classifier_model_path: str = ACTION_CLASSIFIER_DEFAULT_MODEL_PATH,
        classifier_threshold: float = 0.65,
        classifier_max_frames: int = 96,
        verifier_mode: str = "off",
        keep_candidate_events: bool = True,
        keep_rejected_events: bool = True,
        fallback_to_rule_events: bool = False,
        temporal_mode: str = "off",
        temporal_model_id: Optional[str] = None,
        temporal_model_path: Optional[str] = None,
        temporal_thresholds: Optional[Dict[str, float]] = None,
        temporal_clip_length: int = 12,
        temporal_sample_fps: float = 6.0,
        **_: Any,
    ):
        self.model_path = resolve_model_path(model_path)
        self.object_model_path = resolve_model_path(object_model_path)
        self.output_root = str(output_root)
        self.conf = max(0.01, min(1.0, float(conf)))
        self.iou = max(0.01, min(1.0, float(iou)))
        self.tracker = tracker or "bytetrack.yaml"
        self.imgsz = int(imgsz or 640)
        self.object_imgsz = int(object_imgsz or self.imgsz)
        self.object_conf = max(0.01, min(1.0, float(object_conf)))
        self.phone_classes = _normalize_phone_classes(transfer_classes or DEFAULT_TRANSFER_CLASSES)
        # Preserve the attribute name for older callers.
        self.transfer_classes = set(self.phone_classes)
        self.object_owner_distance_ratio = float(object_owner_distance_ratio)
        self.transfer_max_seconds = float(transfer_max_seconds)
        self.transfer_min_owner_frames = int(transfer_min_owner_frames)
        self.transfer_person_distance_ratio = float(transfer_person_distance_ratio)
        self.draw_objects = bool(draw_objects)
        self.extract_event_clips_enabled = bool(extract_event_clips)
        self.clip_pre_seconds = max(0.0, float(clip_pre_seconds))
        self.clip_post_seconds = max(0.0, float(clip_post_seconds))
        self.clip_min_seconds = max(0.1, float(clip_min_seconds))
        self.clip_max_seconds = max(0.0, float(clip_max_seconds))
        normalized_verifier_mode = str(verifier_mode or "off").strip().lower()
        if normalized_verifier_mode == "augment":
            normalized_verifier_mode = "filter"
        if normalized_verifier_mode not in {"off", "shadow", "filter"}:
            normalized_verifier_mode = "off"
        if enable_model_review and normalized_verifier_mode == "off":
            normalized_verifier_mode = "shadow"
        self.verifier_mode = normalized_verifier_mode
        self.enable_model_review = self.verifier_mode != "off"
        self.require_model_review = bool(require_model_review)
        self.classifier_model_path = resolve_model_path(classifier_model_path or ACTION_CLASSIFIER_DEFAULT_MODEL_PATH)
        self.classifier_threshold = max(0.0, min(1.0, float(classifier_threshold)))
        self.classifier_max_frames = max(8, int(classifier_max_frames or 96))
        self.keep_candidate_events = bool(keep_candidate_events)
        self.keep_rejected_events = bool(keep_rejected_events)
        self.fallback_to_rule_events = bool(fallback_to_rule_events)
        self.temporal_mode = str(temporal_mode or "off").strip().lower()
        if self.temporal_mode not in {"off", "shadow", "filter"}:
            self.temporal_mode = "off"
        self.temporal_model_id = str(temporal_model_id or "")
        self.temporal_model_path = str(temporal_model_path or "")
        self.temporal_thresholds = dict(temporal_thresholds or {})
        self.temporal_clip_length = max(4, min(int(temporal_clip_length or 12), 48))
        self.temporal_sample_fps = max(1.0, min(float(temporal_sample_fps or 6.0), 15.0))
        self.temporal_runtime: Optional[TemporalModelRuntime] = None

        self.pose_model = YOLO(self.model_path)
        self.object_model = None
        self.object_model_load_error: Optional[str] = None
        self.badge_detector_status: Dict[str, Any] = {
            "enabled": False,
            "available": None,
            "error": None,
            "mode": "object_model",
        }
        self.object_tracker = SimpleObjectTracker(max_missed_frames=18)

        ensure_dir(self.output_root)
        self.reset_state()

    def reset_state(self) -> None:
        self.history = defaultdict(lambda: deque(maxlen=90))
        self.active_events: Dict[str, Dict[str, Any]] = {}
        self.finished_events: List[Dict[str, Any]] = []
        self.object_tracker.reset()
        self.current_output_dir: Optional[Path] = None
        self.current_snapshot_dir: Optional[Path] = None
        self.current_clips_dir: Optional[Path] = None
        self.current_task_id: str = ""

    def _should_enable_object_pipeline(self, enabled_events: List[str]) -> bool:
        return self.draw_objects or bool({"phone_use", "missing_badge"} & set(enabled_events))

    @staticmethod
    def _should_enable_badge_pipeline(enabled_events: List[str]) -> bool:
        return "missing_badge" in enabled_events

    def _ensure_object_model(self) -> bool:
        if self.object_model is not None:
            return True
        if self.object_model_load_error:
            return False

        # Avoid silent internet downloads when users run offline containers.
        if not model_path_exists(self.object_model_path):
            self.object_model_load_error = f"物品检测模型不存在: {self.object_model_path}"
            logger.warning(self.object_model_load_error)
            return False
        try:
            self.object_model = YOLO(resolve_model_path(self.object_model_path))
            return True
        except Exception as exc:
            self.object_model_load_error = f"物品检测模型加载失败: {exc}"
            logger.warning(self.object_model_load_error)
            return False

    def analyze_video(
        self,
        video_path: str,
        enabled_events: Optional[List[str]] = None,
        sample_stride: int = 1,
        object_model_path: Optional[str] = None,
        object_conf: Optional[float] = None,
        transfer_classes: Optional[List[str]] = None,
        extract_event_clips: Optional[bool] = None,
        clip_pre_seconds: Optional[float] = None,
        clip_post_seconds: Optional[float] = None,
        clip_min_seconds: Optional[float] = None,
        clip_max_seconds: Optional[float] = None,
        enable_model_review: Optional[bool] = None,
        require_model_review: Optional[bool] = None,
        classifier_model_path: Optional[str] = None,
        classifier_threshold: Optional[float] = None,
        classifier_max_frames: Optional[int] = None,
        verifier_mode: Optional[str] = None,
        keep_candidate_events: Optional[bool] = None,
        keep_rejected_events: Optional[bool] = None,
        fallback_to_rule_events: Optional[bool] = None,
        temporal_mode: Optional[str] = None,
        temporal_model_id: Optional[str] = None,
        temporal_model_path: Optional[str] = None,
        temporal_thresholds: Optional[Dict[str, float]] = None,
        temporal_clip_length: Optional[int] = None,
        temporal_sample_fps: Optional[float] = None,
        behavior_config: Optional[Dict[str, Any]] = None,
        max_frames: Optional[int] = None,
        progress_callback: Optional[Callable[[Dict[str, Any]], None]] = None,
        cancel_callback: Optional[Callable[[], bool]] = None,
        output_task_id: Optional[str] = None,
        generate_annotated_video: bool = False,
        annotated_video_fps: float = 6.0,
        **_: Any,
    ) -> Dict[str, Any]:
        """Analyze a video and return the same response shape as stage 1."""
        self.reset_state()

        normalized_behavior = normalize_behavior_config(
            {
                "enabled": True,
                "pose_model_path": self.model_path,
                "object_model_path": self.object_model_path,
                "pose_confidence": self.conf,
                "pose_iou": self.iou,
                "object_confidence": self.object_conf,
                "phone_classes": sorted(self.phone_classes),
                "transfer_classes": sorted(self.phone_classes),
            },
            behavior_config,
        )
        self.phone_classes = _normalize_phone_classes(normalized_behavior.get("phone_classes"))
        self.transfer_classes = set(self.phone_classes)
        if enabled_events is None:
            enabled_events = normalized_behavior["enabled_events"] or list(DEFAULT_EVENTS)
        enabled_events = normalize_behavior_events(list(enabled_events))
        badge_pipeline_enabled = self._should_enable_badge_pipeline(enabled_events)
        self.badge_detector_status = {
            "enabled": badge_pipeline_enabled,
            "available": None,
            "error": None,
            "mode": "object_model",
            "model_path": normalized_behavior["object_model_path"],
            "badge_classes": list(normalized_behavior["badge_classes"]),
        }
        normalized_pose_path = resolve_model_path(normalized_behavior["pose_model_path"])
        if normalized_pose_path != self.model_path:
            self.model_path = normalized_pose_path
            self.pose_model = YOLO(self.model_path)
        self.object_model_path = resolve_model_path(normalized_behavior["object_model_path"])
        self.conf = float(normalized_behavior["pose_confidence"])
        self.iou = float(normalized_behavior["pose_iou"])
        self.object_conf = float(normalized_behavior["object_confidence"])
        self.keypoint_confidence = float(normalized_behavior["keypoint_confidence"])
        self.badge_classes = normalized_class_names(normalized_behavior["badge_classes"])

        if object_model_path:
            self.object_model_path = resolve_model_path(object_model_path)
            self.object_model = None
            self.object_model_load_error = None
        if object_conf is not None:
            self.object_conf = max(0.01, min(1.0, float(object_conf)))
        if transfer_classes:
            self.phone_classes = _normalize_phone_classes(transfer_classes)
            self.transfer_classes = set(self.phone_classes)
        if extract_event_clips is not None:
            self.extract_event_clips_enabled = bool(extract_event_clips)
        if clip_pre_seconds is not None:
            self.clip_pre_seconds = max(0.0, float(clip_pre_seconds))
        if clip_post_seconds is not None:
            self.clip_post_seconds = max(0.0, float(clip_post_seconds))
        if clip_min_seconds is not None:
            self.clip_min_seconds = max(0.1, float(clip_min_seconds))
        if clip_max_seconds is not None:
            self.clip_max_seconds = max(0.0, float(clip_max_seconds))
        if enable_model_review is not None:
            self.enable_model_review = bool(enable_model_review)
            if self.enable_model_review and self.verifier_mode == "off":
                self.verifier_mode = "shadow"
            elif not self.enable_model_review:
                self.verifier_mode = "off"
        if verifier_mode is not None:
            normalized_verifier_mode = str(verifier_mode or "off").strip().lower()
            if normalized_verifier_mode == "augment":
                normalized_verifier_mode = "filter"
            if normalized_verifier_mode not in {"off", "shadow", "filter"}:
                raise ValueError(f"不支持的人工反馈验证模式: {verifier_mode}")
            self.verifier_mode = normalized_verifier_mode
            self.enable_model_review = self.verifier_mode != "off"
        if require_model_review is not None:
            self.require_model_review = bool(require_model_review)
        if classifier_model_path:
            self.classifier_model_path = resolve_model_path(classifier_model_path)
        if classifier_threshold is not None:
            self.classifier_threshold = max(0.0, min(1.0, float(classifier_threshold)))
        if classifier_max_frames is not None:
            self.classifier_max_frames = max(8, int(classifier_max_frames))
        if keep_candidate_events is not None:
            self.keep_candidate_events = bool(keep_candidate_events)
        if keep_rejected_events is not None:
            self.keep_rejected_events = bool(keep_rejected_events)
        if fallback_to_rule_events is not None:
            self.fallback_to_rule_events = bool(fallback_to_rule_events)
        if temporal_mode is not None:
            normalized_temporal_mode = str(temporal_mode or "off").strip().lower()
            self.temporal_mode = normalized_temporal_mode if normalized_temporal_mode in {"off", "shadow", "filter"} else "off"
        if temporal_model_id is not None:
            self.temporal_model_id = str(temporal_model_id or "")
        if temporal_model_path is not None:
            self.temporal_model_path = str(temporal_model_path or "")
        if temporal_thresholds is not None:
            self.temporal_thresholds = dict(temporal_thresholds or {})
        if temporal_clip_length is not None:
            self.temporal_clip_length = max(4, min(int(temporal_clip_length or 12), 48))
        if temporal_sample_fps is not None:
            self.temporal_sample_fps = max(1.0, min(float(temporal_sample_fps or 6.0), 15.0))

        video_path = str(video_path or "").strip()
        if not video_path:
            raise FileNotFoundError("视频路径为空")
        if video_path.startswith("/static/"):
            video_path = "/app" + video_path
        elif video_path.startswith("static/"):
            video_path = "/app/" + video_path

        if not (video_path.startswith("rtsp://") or video_path.startswith("http://") or video_path.startswith("https://")):
            if not os.path.exists(video_path):
                raise FileNotFoundError(f"视频不存在: {video_path}")

        self.current_task_id, self.current_output_dir = create_time_named_output_dir(
            self.output_root,
            preferred_task_id=output_task_id,
        )
        self.current_snapshot_dir = ensure_dir(self.current_output_dir / "snapshots")
        self.current_clips_dir = ensure_dir(self.current_output_dir / "clips")

        annotated_video_path = self.current_output_dir / "annotated.mp4"
        events_json_path = self.current_output_dir / "events.json"

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"无法打开视频: {video_path}")

        fps = _safe_float(cap.get(cv2.CAP_PROP_FPS), 25.0)
        if fps <= 1:
            fps = 25.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        normalized_max_frames = int(max_frames) if max_frames is not None else 0
        frame_limit = normalized_max_frames if normalized_max_frames > 0 else None
        progress_total = min(total_frames, frame_limit) if total_frames > 0 and frame_limit else (frame_limit or total_frames)

        if width <= 0 or height <= 0:
            ret_probe, probe = cap.read()
            if not ret_probe or probe is None:
                cap.release()
                raise RuntimeError("无法读取视频首帧")
            height, width = probe.shape[:2]
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

        write_full_video = bool(generate_annotated_video)
        target_output_fps = max(3.0, min(6.0, float(annotated_video_fps or 6.0)))
        output_stride = max(1, int(math.ceil(fps / target_output_fps)))
        output_fps = max(0.1, fps / output_stride)
        writer = (
            BrowserCompatibleVideoWriter(annotated_video_path, output_fps, width, height)
            if write_full_video
            else None
        )

        sample_stride = max(1, int(sample_stride or 1))
        if behavior_config is not None:
            target_behavior_fps = float(normalized_behavior["behavior_fps"])
            sample_stride = max(sample_stride, max(1, int(round(fps / target_behavior_fps))))
        object_pipeline_enabled = self._should_enable_object_pipeline(enabled_events)
        if object_pipeline_enabled:
            self._ensure_object_model()
        rule_engine = None
        events: List[Dict[str, Any]] = []
        rule_event_records: Dict[str, Dict[str, Any]] = {}
        engine_config = normalize_behavior_config(
            normalized_behavior,
            {
                "enabled": True,
                "enabled_events": enabled_events,
                "behavior_fps": max(1.0, fps / sample_stride),
                "record_only": True,
                "rules": {
                    "phone_use": {
                        "stable_owner_frames": self.transfer_min_owner_frames,
                        "owner_distance_ratio": self.object_owner_distance_ratio,
                    }
                },
            },
        )
        rule_engine = BehaviorRuleEngine(
            str(output_task_id or self.current_task_id),
            engine_config,
        )
        self.temporal_runtime = TemporalModelRuntime(
            mode=self.temporal_mode,
            model_id=self.temporal_model_id,
            model_path=self.temporal_model_path,
            thresholds=self.temporal_thresholds,
            clip_length=self.temporal_clip_length,
            sample_fps=self.temporal_sample_fps,
        )

        frame_idx = 0
        processed_frames = 0
        start_ts = time.time()
        last_progress_at = 0.0
        last_persons: Dict[int, Dict[str, Any]] = {}
        last_objects: Dict[int, Dict[str, Any]] = {}
        temporal_event_records: Dict[str, Dict[str, Any]] = {}
        temporal_shadow_count = 0

        def emit_progress(*, force: bool = False, message: str = "正在分析视频") -> None:
            nonlocal last_progress_at
            if progress_callback is None:
                return
            now = time.monotonic()
            if not force and now - last_progress_at < 1.0:
                return
            last_progress_at = now
            progress = 0.0
            if progress_total > 0:
                progress = min(99.0, (frame_idx / progress_total) * 99.0)
            payload = {
                "progress": round(progress, 2),
                "message": message,
                "frame_idx": frame_idx,
                "processed_frames": processed_frames,
                "total_frames": progress_total or total_frames,
                "elapsed_seconds": round(time.time() - start_ts, 2),
            }
            try:
                progress_callback(payload)
            except Exception as exc:
                logger.warning("行为分析进度回调失败: %s", exc)

        emit_progress(force=True, message="视频已打开，开始分析")

        try:
            while True:
                if cancel_callback is not None and cancel_callback():
                    raise BehaviorAnalysisCancelled("行为分析任务已取消")
                if frame_limit is not None and frame_idx >= frame_limit:
                    break

                ret, frame = cap.read()
                if not ret or frame is None:
                    break

                frame_idx += 1
                timestamp = frame_idx / fps

                should_process = sample_stride <= 1 or frame_idx % sample_stride == 0
                if should_process:
                    persons = self._infer_persons(frame)
                    self._update_person_history(persons, frame_idx, timestamp)
                    scene_detections: List[Dict[str, Any]] = []
                    model_classes: set = set()
                    objects: Dict[int, Dict[str, Any]] = {}
                    rule_objects: Dict[int, Dict[str, Any]] = {}
                    if object_pipeline_enabled:
                        objects, scene_detections, model_classes = self._infer_and_track_objects(
                            frame,
                            frame_idx,
                            timestamp,
                        )
                        rule_objects = {
                            object_id: item for object_id, item in objects.items()
                            if str(item.get("class_name") or "").casefold() in self.phone_classes
                        }
                        self._update_object_owners(rule_objects, persons, frame_idx, timestamp)
                    if badge_pipeline_enabled:
                        self._annotate_badge_presence(
                            persons,
                            normalized_behavior,
                            scene_detections,
                            model_classes,
                        )

                    if rule_engine is not None:
                        rule_result = rule_engine.update(
                            persons,
                            rule_objects,
                            frame_index=frame_idx,
                            timestamp=timestamp,
                        )
                        self._consume_rule_transitions(
                            rule_result["transitions"],
                            rule_event_records,
                            frame,
                            persons,
                            objects,
                            frame_idx,
                            fps,
                        )
                    if self.temporal_runtime is not None and self.temporal_mode != "off":
                        temporal_result = self.temporal_runtime.update(
                            frame,
                            persons,
                            objects,
                            timestamp,
                            enabled_events=enabled_events,
                        )
                        temporal_transitions = list(temporal_result.get("events") or [])
                        temporal_shadow_count += len(temporal_transitions)
                        for transition in temporal_transitions:
                            event_id = str(transition.get("event_id") or "")
                            if not event_id:
                                continue
                            transition["snapshot"] = self._save_snapshot(
                                transition,
                                frame,
                                persons,
                                objects,
                            )
                            transition["state"] = "resolved"
                            transition["end_time"] = float(transition.get("confirmed_at") or timestamp)
                            temporal_event_records[event_id] = transition
                    last_persons = persons
                    last_objects = objects
                    processed_frames += 1
                else:
                    persons = last_persons
                    objects = last_objects

                if writer is not None and (frame_idx - 1) % output_stride == 0:
                    annotated = self._draw_frame(frame, persons, objects, timestamp)
                    writer.write(annotated)
                emit_progress()
        except BaseException:
            if writer is not None:
                writer.abort()
            raise
        finally:
            cap.release()

        emit_progress(force=True, message="视频分析完成，正在整理结果")
        if writer is not None:
            video_ok, video_format = writer.release()
        else:
            video_ok, video_format = False, "disabled"
        if writer is not None and not video_ok:
            logger.warning("标注视频生成失败: %s", video_format)
        if rule_engine is not None:
            self._consume_rule_transitions(
                rule_engine.close(frame_idx / fps if fps else 0.0),
                rule_event_records,
                None,
                last_persons,
                last_objects,
                frame_idx,
                fps,
            )
            events = self._format_events(list(rule_event_records.values()))
        rule_events = list(events)
        temporal_events = self._format_events(list(temporal_event_records.values()))
        if self.temporal_mode == "filter":
            events = self._merge_temporal_batch_events(rule_events, temporal_events)
        else:
            events = rule_events
        clip_summary = {
            "enabled": bool(self.extract_event_clips_enabled),
            "clips_generated": 0,
            "clips_failed": 0,
            "output_dir": self._to_static_url(self.current_clips_dir) if self.current_clips_dir else None,
            "error": None,
        }
        if self.extract_event_clips_enabled and self.current_clips_dir is not None:
            events, clip_summary = run_extract_event_clips(
                video_path=video_path,
                events=events,
                output_dir=self.current_clips_dir,
                pre_seconds=self.clip_pre_seconds,
                post_seconds=self.clip_post_seconds,
                min_seconds=self.clip_min_seconds,
                max_seconds=self.clip_max_seconds,
            )

        candidate_events = [dict(event) for event in events]
        output_events, rejected_events, review_summary = self._review_events_with_classifier(candidate_events)

        result_payload = {
            "task_id": self.current_task_id,
            "video_path": video_path,
            "annotated_video": self._to_static_url(annotated_video_path) if video_ok else None,
            "annotated_video_status": video_ok,
            "annotated_video_format": video_format,
            "events_json": self._to_static_url(events_json_path),
            # Keep rule and temporal decisions separately visible; filter mode merges them.
            "events": output_events,
            "candidate_events": candidate_events if self.keep_candidate_events else [],
            "rejected_events": rejected_events if self.keep_rejected_events else [],
            "summary": {
                "total_events": len(output_events),
                "total_final_events": len(output_events),
                "total_candidate_events": len(candidate_events),
                "total_rejected_events": len(rejected_events),
                "processed_frames": processed_frames,
                "total_frames": total_frames,
                "fps": round(fps, 3),
                "width": width,
                "height": height,
                "elapsed_seconds": round(time.time() - start_ts, 2),
                "annotated_video_enabled": write_full_video,
                "annotated_video_fps": round(output_fps, 3) if write_full_video else None,
                "enabled_events": enabled_events,
                "stage": "object_pose_rules_with_feedback_verifier" if self.verifier_mode != "off" else "object_pose_rules",
                "object_pipeline_enabled": object_pipeline_enabled,
                "object_model_path": self.object_model_path,
                "object_model_load_error": self.object_model_load_error,
                "badge_detector": dict(self.badge_detector_status),
                "transfer_classes": sorted(list(self.transfer_classes)),
                "clip_extraction": clip_summary,
                "model_review": review_summary,
                "temporal_model": self.temporal_runtime.status() if self.temporal_runtime else {
                    "enabled": False,
                    "mode": "off",
                },
                "temporal_shadow_events": temporal_shadow_count,
                "temporal_comparison": {
                    "rule_events": len(rule_events),
                    "temporal_events": len(temporal_events),
                    "added_events": len([event for event in events if event.get("reason", {}).get("model_family") == "temporal_fusion"]),
                },
            },
            "temporal_shadow_events": temporal_events if self.temporal_mode == "shadow" else [],
            "rule_events": rule_events,
            "temporal_events": temporal_events,
        }

        with open(events_json_path, "w", encoding="utf-8") as f:
            json.dump(result_payload, f, ensure_ascii=False, indent=2)

        return result_payload

    def _infer_persons(self, frame: np.ndarray) -> Dict[int, Dict[str, Any]]:
        persons: Dict[int, Dict[str, Any]] = {}
        results = self.pose_model.track(
            frame,
            persist=True,
            tracker=self.tracker,
            conf=self.conf,
            iou=self.iou,
            imgsz=self.imgsz,
            verbose=False,
        )
        if not results:
            return persons

        result = results[0]
        if result.boxes is None or result.boxes.xyxy is None:
            return persons

        boxes = result.boxes.xyxy.cpu().numpy()
        if result.boxes.id is not None:
            ids = result.boxes.id.cpu().numpy().astype(int)
        else:
            # Fallback when tracker does not provide IDs.
            ids = np.arange(1, len(boxes) + 1).astype(int)

        keypoints = None
        keypoint_confidences = None
        if result.keypoints is not None and result.keypoints.xy is not None:
            keypoints = result.keypoints.xy.cpu().numpy()
            if result.keypoints.conf is not None:
                keypoint_confidences = result.keypoints.conf.cpu().numpy()

        for i, track_id in enumerate(ids):
            box = [float(v) for v in boxes[i].tolist()]
            w, h = bbox_size(box)
            kpts = keypoints[i].tolist() if keypoints is not None and i < len(keypoints) else []
            confidences = (
                keypoint_confidences[i].tolist()
                if keypoint_confidences is not None and i < len(keypoint_confidences)
                else []
            )
            if confidences:
                kpts = [
                    [float(point[0]), float(point[1]), float(confidences[index])]
                    for index, point in enumerate(kpts)
                ]
            persons[int(track_id)] = {
                "track_id": int(track_id),
                "bbox": box,
                "center": bbox_center(box),
                "width": float(w),
                "height": float(h),
                "keypoints": kpts,
                "keypoint_confidence": self.keypoint_confidence,
            }
        return persons

    def _annotate_badge_presence(
        self,
        persons: Dict[int, Dict[str, Any]],
        behavior_config: Dict[str, Any],
        detections: List[Dict[str, Any]],
        model_classes: set,
    ) -> None:
        """Mark pose-valid people using badge classes from the object model."""
        if not persons:
            return
        matched_classes = matching_class_names(model_classes, self.badge_classes)
        if not matched_classes:
            self.badge_detector_status.update({
                "available": False,
                "error": "configured object model exposes no badge class",
            })
            return
        self.badge_detector_status.update({
            "available": True,
            "error": None,
            "matched_classes": matched_classes,
        })

        padding = float(behavior_config["rules"]["missing_badge"]["chest_padding_ratio"])
        badge_detections = [
            item for item in detections
            if str(item.get("class_name") or "").strip().casefold() in self.badge_classes
        ]
        for person in persons.values():
            chest_roi = self._chest_roi(person, padding)
            if chest_roi is None:
                continue
            person["badge_evaluable"] = True
            person["chest_roi"] = chest_roi
            person["badge_detector_mode"] = "object_model"
            person["badge_classes"] = matched_classes
            person["badge_visibility_score"] = 0.72
        assign_badge_detections(persons, badge_detections)

    @staticmethod
    def _box_center_in(box: Iterable[float], region: Iterable[float]) -> bool:
        center = bbox_center(box)
        x1, y1, x2, y2 = [float(value) for value in region]
        return x1 <= center[0] <= x2 and y1 <= center[1] <= y2

    @staticmethod
    def _chest_roi(person: Dict[str, Any], padding: float) -> Optional[List[float]]:
        left_shoulder = get_kpt(person, LEFT_SHOULDER)
        right_shoulder = get_kpt(person, RIGHT_SHOULDER)
        left_hip = get_kpt(person, LEFT_HIP)
        right_hip = get_kpt(person, RIGHT_HIP)
        if left_shoulder is None or right_shoulder is None:
            return None
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2.0
        shoulder_width = max(1.0, dist(left_shoulder, right_shoulder))
        hips = [point for point in (left_hip, right_hip) if point is not None]
        lower_y = (
            min(point[1] for point in hips)
            if hips
            else shoulder_y + max(
                shoulder_width * 1.8,
                float(person.get("height") or 0.0) * 0.32,
            )
        )
        if lower_y <= shoulder_y + 4.0:
            return None
        extra = shoulder_width * max(0.0, padding)
        return [
            round(min(left_shoulder[0], right_shoulder[0]) - extra, 2),
            round(shoulder_y - extra * 0.35, 2),
            round(max(left_shoulder[0], right_shoulder[0]) + extra, 2),
            round(lower_y, 2),
        ]

    def _infer_and_track_objects(
        self,
        frame: np.ndarray,
        frame_idx: int,
        timestamp: float,
    ) -> Tuple[Dict[int, Dict[str, Any]], List[Dict[str, Any]], set]:
        if self.object_model is None:
            if self.badge_detector_status.get("enabled"):
                self.badge_detector_status.update({
                    "available": False,
                    "error": self.object_model_load_error or "object model unavailable",
                })
            return {}, [], set()

        try:
            results = self.object_model(
                frame,
                conf=self.object_conf,
                iou=self.iou,
                imgsz=self.object_imgsz,
                verbose=False,
            )
        except Exception as exc:
            logger.warning(f"物品检测失败: {exc}")
            if self.badge_detector_status.get("enabled"):
                self.badge_detector_status.update({
                    "available": False,
                    "error": f"object detection failed: {exc}"[:300],
                })
            return {}, [], set()

        detections: List[Dict[str, Any]] = []
        if not results:
            return self.object_tracker.update(detections, frame_idx, timestamp), detections, set()

        result = results[0]
        names = getattr(result, "names", None) or getattr(self.object_model, "names", {}) or {}
        model_classes = {
            str(value).strip().casefold()
            for value in (names.values() if isinstance(names, dict) else names)
        }
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return self.object_tracker.update(detections, frame_idx, timestamp), detections, model_classes

        for box in boxes:
            cls_id = int(box.cls[0].cpu().numpy()) if hasattr(box, "cls") else -1
            conf = float(box.conf[0].cpu().numpy()) if hasattr(box, "conf") else 0.0
            class_name = names.get(cls_id, f"class_{cls_id}") if isinstance(names, dict) else f"class_{cls_id}"
            class_name = str(class_name)

            if class_name == "person":
                continue

            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().tolist()
            detections.append({
                "class_id": cls_id,
                "class_name": class_name,
                "bbox": [float(x1), float(y1), float(x2), float(y2)],
                "confidence": conf,
            })

        return self.object_tracker.update(detections, frame_idx, timestamp), detections, model_classes

    def _update_person_history(self, persons: Dict[int, Dict[str, Any]], frame_idx: int, timestamp: float) -> None:
        for tid, person in persons.items():
            self.history[tid].append({
                "frame_idx": int(frame_idx),
                "timestamp": float(timestamp),
                "bbox": person["bbox"],
                "center": person["center"],
                "height": person["height"],
                "width": person["width"],
                "keypoints": person.get("keypoints", []),
            })

    def _update_object_owners(
        self,
        objects: Dict[int, Dict[str, Any]],
        persons: Dict[int, Dict[str, Any]],
        frame_idx: int,
        timestamp: float,
    ) -> None:
        for object_id, obj in objects.items():
            owner_id, distance_ratio = self._find_object_owner(obj, persons)
            obj["owner_id"] = owner_id
            obj["owner_distance_ratio"] = distance_ratio
            self.object_tracker.add_owner_observation(
                object_id=object_id,
                owner_id=owner_id,
                frame_idx=frame_idx,
                timestamp=timestamp,
                distance_ratio=distance_ratio,
            )
            track = self.object_tracker.tracks.get(object_id)
            obj["owner_history"] = list(track.owner_history) if track else []

    def _find_object_owner(
        self,
        obj: Dict[str, Any],
        persons: Dict[int, Dict[str, Any]],
    ) -> Tuple[Optional[int], Optional[float]]:
        obj_center = obj.get("center") or bbox_center(obj.get("bbox", [0, 0, 0, 0]))
        candidates: List[Tuple[float, int]] = []

        for tid, person in persons.items():
            h = max(1.0, float(person.get("height", 1)))
            wrist_points = [get_kpt(person, LEFT_WRIST), get_kpt(person, RIGHT_WRIST)]
            wrist_points = [p for p in wrist_points if p is not None]
            if not wrist_points:
                continue
            nearest = min(dist(obj_center, p) for p in wrist_points)
            ratio = nearest / h
            if ratio <= self.object_owner_distance_ratio:
                candidates.append((ratio, int(tid)))

        if not candidates:
            return None, None

        candidates.sort(key=lambda item: item[0])
        best_ratio, best_tid = candidates[0]

        # Ambiguity guard: if two hands are almost equally close, do not assign owner.
        if len(candidates) >= 2:
            second_ratio, _ = candidates[1]
            if best_ratio > second_ratio * 0.85:
                return None, best_ratio

        return best_tid, best_ratio

    def _resolve_clip_path_for_classifier(self, clip_value: Any) -> Optional[str]:
        """Convert /static/... or http://localhost:8000/static/... to a local container path."""
        if not clip_value:
            return None
        text = str(clip_value).strip().replace("\\", "/")
        if not text:
            return None
        if text.startswith("/app/static/"):
            return text
        if text.startswith("/static/"):
            return "/app" + text
        if text.startswith("static/"):
            return "/app/" + text
        if "/static/" in text:
            return "/app" + text[text.index("/static/"):]
        return text

    def _build_model_review_summary(self) -> Dict[str, Any]:
        return {
            "enabled": bool(self.enable_model_review),
            "available": False,
            "mode": self.verifier_mode,
            "verifier_path": self.classifier_model_path,
            "classifier_path": self.classifier_model_path,
            "threshold": float(self.classifier_threshold),
            "require_model_review": bool(self.require_model_review),
            "fallback_to_rule_events": bool(self.fallback_to_rule_events),
            "accepted": 0,
            "rejected": 0,
            "skipped": 0,
            "error": None,
        }

    def _enrich_rule_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        enriched = dict(event)
        rule_type = str(enriched.get("event_type") or "")
        rule_conf = float(enriched.get("confidence", 0.0) or 0.0)
        enriched.setdefault("rule_event_type", rule_type)
        enriched.setdefault("rule_event_name", EVENT_NAME_MAP.get(rule_type, rule_type))
        enriched.setdefault("rule_confidence", rule_conf)
        return enriched

    @staticmethod
    def _restore_rule_event(event: Dict[str, Any]) -> Dict[str, Any]:
        rule_type = str(event.get("rule_event_type") or event.get("event_type") or "")
        rule_confidence = float(event.get("rule_confidence", event.get("confidence", 0.0)) or 0.0)
        event.update({
            "event_type": rule_type,
            "event_name": EVENT_NAME_MAP.get(rule_type, rule_type),
            "confidence": rule_confidence,
            "final_event_type": rule_type,
            "final_event_name": EVENT_NAME_MAP.get(rule_type, rule_type),
            "final_confidence": rule_confidence,
        })
        return event

    def _model_label_name(self, label: str) -> str:
        return EVENT_NAME_MAP.get(label) or ACTION_LABEL_NAME_MAP.get(label) or label

    def _should_accept_model_review(self, rule_event_type: str, model_event_type: str, model_confidence: float) -> bool:
        expected_label = RULE_TO_FINAL_LABEL.get(str(rule_event_type or ""))
        model_label = str(model_event_type or "")
        if not expected_label:
            return False
        if model_label in REJECT_LABELS:
            return False
        if float(model_confidence or 0.0) < self.classifier_threshold:
            return False
        return model_label == expected_label

    def _review_events_with_classifier(
        self,
        candidate_events: List[Dict[str, Any]],
        *,
        enabled: Optional[bool] = None,
    ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
        """Score object + Pose rule candidates with an optional feedback verifier.

        Shadow mode never changes the rule result. Filter mode can suppress a
        candidate only after the primary rule pipeline has produced it.
        """
        summary = self._build_model_review_summary()
        review_enabled = self.enable_model_review if enabled is None else bool(enabled)
        summary["enabled"] = review_enabled
        enriched_candidates = [self._enrich_rule_event(event) for event in candidate_events]

        if not review_enabled:
            summary["skipped"] = len(enriched_candidates)
            final_events: List[Dict[str, Any]] = []
            for event in enriched_candidates:
                rule_type = str(event.get("rule_event_type") or event.get("event_type") or "")
                rule_conf = float(event.get("rule_confidence", event.get("confidence", 0.0)) or 0.0)
                event.update({
                    "verified_by_model": False,
                    "review_decision": "skipped",
                    "model_event_type": None,
                    "model_event_name": None,
                    "model_confidence": None,
                    "final_event_type": rule_type,
                    "final_event_name": EVENT_NAME_MAP.get(rule_type, rule_type),
                    "final_confidence": rule_conf,
                })
                final_events.append(event)
            return final_events, [], summary

        if load_action_verifier is None:
            summary["error"] = "人工反馈验证器模块不可用"
            if self.require_model_review:
                raise RuntimeError(summary["error"])
            rejected = []
            for event in enriched_candidates:
                event.update({
                    "verified_by_model": False,
                    "review_decision": "classifier_unavailable",
                    "model_event_type": None,
                    "model_event_name": None,
                    "model_confidence": None,
                    "final_event_type": None,
                    "final_event_name": None,
                    "final_confidence": 0.0,
                })
                rejected.append(event)
            summary["rejected"] = len(rejected)
            if self.verifier_mode == "shadow" or self.fallback_to_rule_events:
                return [self._restore_rule_event(event) for event in enriched_candidates], rejected, summary
            return [], rejected, summary

        try:
            classifier = load_action_verifier(self.classifier_model_path)
            summary["available"] = True
            summary["labels"] = list(getattr(classifier, "labels", []) or [])
            summary["counts"] = dict(getattr(classifier, "counts", {}) or {})
            summary["model_type"] = (getattr(classifier, "metadata", {}) or {}).get("model_type")
        except Exception as exc:
            summary["error"] = f"分类器模型不可用: {exc}"
            if self.require_model_review:
                raise RuntimeError(summary["error"])
            rejected = []
            for event in enriched_candidates:
                event.update({
                    "verified_by_model": False,
                    "review_decision": "classifier_unavailable",
                    "model_event_type": None,
                    "model_event_name": None,
                    "model_confidence": None,
                    "final_event_type": None,
                    "final_event_name": None,
                    "final_confidence": 0.0,
                })
                rejected.append(event)
            summary["rejected"] = len(rejected)
            if self.verifier_mode == "shadow" or self.fallback_to_rule_events:
                return [self._restore_rule_event(event) for event in enriched_candidates], rejected, summary
            return [], rejected, summary

        final_events: List[Dict[str, Any]] = []
        rejected_events: List[Dict[str, Any]] = []
        verifier_passed = 0
        for event in enriched_candidates:
            rule_type = str(event.get("rule_event_type") or event.get("event_type") or "")
            clip_path = self._resolve_clip_path_for_classifier(event.get("clip"))
            reviewed = dict(event)

            if not clip_path or not os.path.exists(clip_path):
                reviewed.update({
                    "verified_by_model": False,
                    "review_decision": "rejected_no_clip",
                    "model_event_type": None,
                    "model_event_name": None,
                    "model_confidence": None,
                    "final_event_type": None,
                    "final_event_name": None,
                    "final_confidence": 0.0,
                    "model_review_error": f"clip 不存在或不可访问: {clip_path}",
                })
                rejected_events.append(reviewed)
                if self.verifier_mode == "shadow" or self.fallback_to_rule_events:
                    final_events.append(self._restore_rule_event(reviewed))
                continue

            try:
                prediction = classifier.predict_clip(clip_path, max_frames=self.classifier_max_frames)
                model_label = str(prediction.get("label") or "")
                model_confidence = float(prediction.get("confidence", 0.0) or 0.0)
                accepted = self._should_accept_model_review(rule_type, model_label, model_confidence)
                reviewed.update({
                    "model_event_type": model_label,
                    "model_event_name": self._model_label_name(model_label),
                    "model_confidence": round(model_confidence, 4),
                    "model_probabilities": prediction.get("scores", {}),
                    "model_distances": prediction.get("distances", {}),
                    "verified_by_model": bool(accepted),
                    "review_decision": (
                        "shadow_accepted" if accepted else "shadow_rejected"
                    ) if self.verifier_mode == "shadow" else ("accepted" if accepted else "rejected"),
                })
                if accepted:
                    verifier_passed += 1
                if self.verifier_mode == "shadow":
                    final_events.append(self._restore_rule_event(reviewed))
                    if not accepted:
                        rejected_events.append(dict(reviewed))
                    continue
                if accepted:
                    reviewed.update({
                        "event_type": model_label,
                        "event_name": self._model_label_name(model_label),
                        "confidence": round(model_confidence, 4),
                        "final_event_type": model_label,
                        "final_event_name": self._model_label_name(model_label),
                        "final_confidence": round(model_confidence, 4),
                    })
                    final_events.append(reviewed)
                else:
                    reviewed.update({
                        "final_event_type": None,
                        "final_event_name": None,
                        "final_confidence": 0.0,
                    })
                    rejected_events.append(reviewed)
            except Exception as exc:
                reviewed.update({
                    "verified_by_model": False,
                    "review_decision": "rejected_prediction_error",
                    "model_event_type": None,
                    "model_event_name": None,
                    "model_confidence": None,
                    "final_event_type": None,
                    "final_event_name": None,
                    "final_confidence": 0.0,
                    "model_review_error": str(exc),
                })
                rejected_events.append(reviewed)
                if self.verifier_mode == "shadow" or self.fallback_to_rule_events:
                    final_events.append(self._restore_rule_event(reviewed))

        summary["accepted"] = verifier_passed
        summary["rejected"] = len(rejected_events)
        summary["final_event_count"] = len(final_events)
        return final_events, rejected_events, summary

    @staticmethod
    def _merge_temporal_batch_events(
        rule_events: List[Dict[str, Any]],
        temporal_events: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """Add temporal detections while avoiding duplicate rule events."""
        merged = [dict(event) for event in rule_events]
        for candidate in temporal_events:
            event_type = str(candidate.get("event_type") or "")
            candidate_tracks = set(candidate.get("persons") or [])
            candidate_time = float(candidate.get("confirmed_time", candidate.get("start_time", 0.0)) or 0.0)
            duplicate = False
            for existing in merged:
                if str(existing.get("event_type") or "") != event_type:
                    continue
                existing_tracks = set(existing.get("persons") or [])
                if candidate_tracks and existing_tracks and not candidate_tracks.intersection(existing_tracks):
                    continue
                existing_time = float(existing.get("confirmed_time", existing.get("start_time", 0.0)) or 0.0)
                if abs(existing_time - candidate_time) <= 3.0:
                    duplicate = True
                    break
            if not duplicate:
                merged.append(dict(candidate))
        return merged

    def _consume_rule_transitions(
        self,
        transitions: List[Dict[str, Any]],
        records: Dict[str, Dict[str, Any]],
        frame: Optional[np.ndarray],
        persons: Dict[int, Dict[str, Any]],
        objects: Dict[int, Dict[str, Any]],
        frame_idx: int,
        fps: float,
    ) -> None:
        for transition in transitions or []:
            event_id = str(transition.get("event_id") or "")
            if not event_id:
                continue
            if transition.get("state") == "confirmed":
                start_time = float(transition.get("started_at") or 0.0)
                confirmed_at = float(transition.get("confirmed_at") or start_time)
                item = {
                    **transition,
                    "start_time": start_time,
                    "end_time": confirmed_at,
                    "start_frame": max(1, int(round(start_time * max(fps, 1.0)))),
                    "end_frame": int(frame_idx),
                    "frame_idx": int(frame_idx),
                    "timestamp": confirmed_at,
                    "confidence": float(transition.get("rule_score") or 0.0),
                }
                item["snapshot"] = (
                    self._save_snapshot(item, frame, persons, objects)
                    if frame is not None
                    else None
                )
                records[event_id] = item
            elif transition.get("state") == "resolved" and event_id in records:
                item = records[event_id]
                item["state"] = "resolved"
                item["end_time"] = float(transition.get("ended_at") or item.get("end_time") or 0.0)
                item["end_frame"] = int(frame_idx)
                item["duration_seconds"] = float(transition.get("duration_seconds") or 0.0)
                item["rule_score"] = max(
                    float(item.get("rule_score") or 0.0),
                    float(transition.get("rule_score") or 0.0),
                )
                item["confidence"] = item["rule_score"]
                item["reason"] = transition.get("reason") or item.get("reason") or {}

    def _format_events(self, events: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        formatted: List[Dict[str, Any]] = []
        events_sorted = sorted(events, key=lambda e: (float(e.get("start_time", 0)), str(e.get("event_type", ""))))
        for idx, event in enumerate(events_sorted, start=1):
            reason = dict(event.get("reason") or {})
            is_temporal = reason.get("model_family") == "temporal_fusion"
            model_id = reason.get("model_id")
            model_version = reason.get("model_version") or model_id
            item = {
                "id": idx,
                "event_id": event.get("event_id"),
                "event_type": event.get("event_type"),
                "event_name": EVENT_NAME_MAP.get(str(event.get("event_type")), str(event.get("event_type"))),
                "persons": event.get("persons", []),
                "start_frame": int(event.get("start_frame", event.get("frame_idx", 0))),
                "end_frame": int(event.get("end_frame", event.get("frame_idx", 0))),
                "start_time": round(float(event.get("start_time", event.get("timestamp", 0))), 2),
                "confirmed_time": round(float(event.get("confirmed_at", event.get("start_time", 0))), 2),
                "end_time": round(float(event.get("end_time", event.get("timestamp", 0))), 2),
                "duration": round(float(event.get("end_time", 0)) - float(event.get("start_time", 0)), 2),
                "confidence": round(float(event.get("confidence", 0.0)), 3),
                "rule_score": round(float(event.get("rule_score", event.get("confidence", 0.0))), 3),
                "state": event.get("state", "resolved"),
                "severity": event.get("severity"),
                "should_alert": False,
                "snapshot": event.get("snapshot"),
                "reason": reason,
                "rule_decision": {
                    "accepted": not is_temporal,
                    "score": round(float(event.get("rule_score") or 0.0), 4),
                },
                "temporal_decision": {
                    "accepted": is_temporal,
                    "confidence": round(float(event.get("confidence") or 0.0), 4) if is_temporal else None,
                    "available": is_temporal,
                    "model_id": model_id,
                    "model_version": model_version,
                },
                "final_decision": {
                    "accepted": True,
                    "source": "temporal_fusion" if is_temporal else "rule",
                },
                "model_id": model_id,
                "model_version": model_version,
                "input_quality": reason.get("input_quality") or {},
                "temporal_span": reason.get("temporal_span") or {},
                "fallback": not is_temporal,
            }
            feature_vector = event.get("training_feature_vector") or event.get("pose_feature_vector")
            if isinstance(feature_vector, (list, tuple)) and feature_vector:
                item["training_feature_vector"] = [float(value) for value in feature_vector[:96]]
            if event.get("object_id") is not None:
                item["object_id"] = event.get("object_id")
                item["object_class"] = event.get("object_class")
            formatted.append(item)
        return formatted

    def _save_snapshot(
        self,
        event: Dict[str, Any],
        frame: np.ndarray,
        persons: Dict[int, Dict[str, Any]],
        objects: Dict[int, Dict[str, Any]],
    ) -> Optional[str]:
        if self.current_snapshot_dir is None:
            return None
        try:
            persons_part = "_".join(map(str, event.get("persons", []))) or "none"
            obj_part = f"_obj{event.get('object_id')}" if event.get("object_id") is not None else ""
            filename = f"{event.get('event_type')}_{event.get('frame_idx')}_p{persons_part}{obj_part}.png"
            path = self.current_snapshot_dir / filename
            img = self._draw_frame(frame, persons, objects, float(event.get("timestamp", 0)), highlight_event=event)
            cv2.imwrite(str(path), img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
            return self._to_static_url(path)
        except Exception as exc:
            logger.warning(f"保存行为事件截图失败: {exc}")
            return None

    def _draw_frame(
        self,
        frame: np.ndarray,
        persons: Dict[int, Dict[str, Any]],
        objects: Dict[int, Dict[str, Any]],
        timestamp: float,
        highlight_event: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        img = frame.copy()

        # Draw person boxes/keypoints.
        for tid, p in persons.items():
            x1, y1, x2, y2 = [int(round(v)) for v in p.get("bbox", [0, 0, 0, 0])]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 220, 0), 2)
            cv2.putText(img, f"ID {tid}", (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 220, 0), 2)
            for kp in p.get("keypoints", []) or []:
                if valid_point(kp):
                    cv2.circle(img, (int(kp[0]), int(kp[1])), 3, (255, 0, 0), -1)

        # Draw object boxes/tracks.
        for object_id, obj in objects.items():
            x1, y1, x2, y2 = [int(round(v)) for v in obj.get("bbox", [0, 0, 0, 0])]
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 165, 255), 2)
            label = f"OBJ {object_id} {obj.get('class_name', 'object')} {float(obj.get('confidence', 0.0)):.2f}"
            owner = obj.get("owner_id")
            if owner is not None:
                label += f" -> ID {owner}"
            cv2.putText(img, label[:60], (x1, max(18, y1 - 8)), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 165, 255), 2)

        cv2.putText(img, f"time: {timestamp:.2f}s", (20, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 255), 2)

        if highlight_event:
            text = EVENT_NAME_MAP.get(str(highlight_event.get("event_type")), str(highlight_event.get("event_type")))
            persons_text = ",".join(map(str, highlight_event.get("persons", [])))
            extra = ""
            if highlight_event.get("object_id") is not None:
                extra = f" OBJ {highlight_event.get('object_id')} {highlight_event.get('object_class', '')}"
            cv2.putText(img, f"EVENT: {text} P[{persons_text}]{extra}", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        else:
            # Show currently active event count to make video review easier.
            active_count = len(self.active_events)
            if active_count:
                cv2.putText(img, f"active events: {active_count}", (20, 62), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        return img

    def _to_static_url(self, path: Any) -> str:
        text = str(path).replace("\\", "/")
        if text.startswith("/app/static/"):
            return text.replace("/app", "", 1)
        marker = "/static/"
        if marker in text:
            return text[text.index(marker):]
        return text
