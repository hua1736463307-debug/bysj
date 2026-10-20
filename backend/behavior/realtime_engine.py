"""Object + Pose temporal behavior rules shared by batch and realtime inference."""

from __future__ import annotations

import copy
import math
import time
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Any, Deque, Dict, Iterable, List, Optional, Tuple

from .pose_feature_encoder import aggregate_pose_features, encode_single_pose
from .feedback_mapping import SUPPORTED_BEHAVIOR_EVENTS, normalize_behavior_events

NOSE = 0
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12


def dist(left: Any, right: Any) -> float:
    if left is None or right is None:
        return 1e9
    return math.hypot(float(left[0]) - float(right[0]), float(left[1]) - float(right[1]))


def bbox_center(box: Iterable[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = [float(value) for value in box]
    return ((x1 + x2) / 2.0, (y1 + y2) / 2.0)


def bbox_iou(left: Iterable[float], right: Iterable[float]) -> float:
    ax1, ay1, ax2, ay2 = [float(value) for value in left]
    bx1, by1, bx2, by2 = [float(value) for value in right]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    intersection = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    left_area = max(1.0, (ax2 - ax1) * (ay2 - ay1))
    right_area = max(1.0, (bx2 - bx1) * (by2 - by1))
    return intersection / (left_area + right_area - intersection + 1e-6)


def get_kpt(person: Dict[str, Any], index: int) -> Optional[Tuple[float, float]]:
    points = person.get("keypoints") or []
    if index >= len(points) or len(points[index]) < 2:
        return None
    confidence = points[index][2] if len(points[index]) >= 3 else None
    min_confidence = float(person.get("keypoint_confidence", 0.25) or 0.25)
    if confidence is not None and float(confidence) < min_confidence:
        return None
    x, y = float(points[index][0]), float(points[index][1])
    return (x, y) if x > 1.0 and y > 1.0 else None


def point_in_upper_body(point: Any, box: Iterable[float]) -> bool:
    if point is None:
        return False
    x1, y1, x2, y2 = [float(value) for value in box]
    return x1 <= float(point[0]) <= x2 and y1 <= float(point[1]) <= y1 + 0.65 * max(1.0, y2 - y1)


def _badge_code_from_detection(detection: Dict[str, Any]) -> Tuple[Optional[str], float]:
    for key in ("badge_code", "badge_id", "badge_number", "decoded_text"):
        value = detection.get(key)
        if value is None:
            continue
        code = str(value).strip()
        if not code:
            continue
        confidence = detection.get("badge_code_confidence", detection.get("identity_confidence", 1.0))
        try:
            return code[:128], max(0.0, min(1.0, float(confidence)))
        except (TypeError, ValueError):
            return code[:128], 0.0
    return None, 0.0


def assign_badge_detections(
    persons: Dict[int, Dict[str, Any]],
    badge_detections: Iterable[Dict[str, Any]],
) -> None:
    """Assign each badge box to at most one evaluable chest region."""
    assignments: Dict[int, List[Dict[str, Any]]] = defaultdict(list)
    ambiguous_track_ids = set()
    for person in persons.values():
        if person.get("badge_evaluable") and person.get("chest_roi"):
            person["badge_present"] = False
            person["badge_association_state"] = "unseen"
            person.pop("badge_code", None)
            person.pop("badge_code_confidence", None)

    for detection in badge_detections or []:
        bbox = detection.get("bbox") or []
        if len(bbox) < 4:
            continue
        center = bbox_center(bbox)
        candidates = [
            track_id
            for track_id, person in persons.items()
            if person.get("badge_evaluable")
            and person.get("chest_roi")
            and _point_in_box(center, person["chest_roi"])
        ]
        if len(candidates) == 1:
            assignments[candidates[0]].append(detection)
        elif len(candidates) > 1:
            ambiguous_track_ids.update(candidates)

    for track_id in ambiguous_track_ids:
        person = persons[track_id]
        person["badge_evaluable"] = False
        person["badge_association_state"] = "ambiguous"

    for track_id, detections in assignments.items():
        person = persons[track_id]
        if track_id in ambiguous_track_ids:
            continue
        codes = {
            code: confidence
            for code, confidence in (_badge_code_from_detection(item) for item in detections)
            if code
        }
        if len(codes) > 1:
            person["badge_evaluable"] = False
            person["badge_association_state"] = "ambiguous"
            continue
        person["badge_present"] = True
        person["badge_association_state"] = "assigned"
        if codes:
            code, confidence = next(iter(codes.items()))
            person["badge_code"] = code
            person["badge_code_confidence"] = confidence


def _point_in_box(point: Tuple[float, float], box: Iterable[float]) -> bool:
    x1, y1, x2, y2 = [float(value) for value in box]
    return x1 <= point[0] <= x2 and y1 <= point[1] <= y2


def normalized_class_names(values: Iterable[Any]) -> set:
    return {
        str(value).strip().casefold()
        for value in (values or [])
        if str(value).strip()
    }


def matching_class_names(model_classes: Iterable[Any], configured_classes: Iterable[Any]) -> List[str]:
    return sorted(normalized_class_names(model_classes) & normalized_class_names(configured_classes))


PHONE_CLASS_ALIASES = {
    "mobilephone",
    "手机",
    "cell phone",
    "cellphone",
    "mobile phone",
    "phone",
    "smartphone",
}

SUPPORTED_EVENTS = SUPPORTED_BEHAVIOR_EVENTS
DEFAULT_ENABLED_EVENTS = SUPPORTED_EVENTS

EVENT_SEVERITY = {
    "phone_use": "medium",
    "fall_down": "high",
    "missing_badge": "medium",
}

DEFAULT_BEHAVIOR_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "enabled_events": list(DEFAULT_ENABLED_EVENTS),
    "behavior_fps": 6.0,
    "record_only": True,
    "pose_model_path": "/app/models/pretrained/yolov8n-pose.pt",
    "object_model_path": "/app/models/yolov8n.pt",
    "badge_classes": ["badge", "work badge", "id badge", "employee badge", "name badge", "工牌", "胸牌"],
    "pose_confidence": 0.35,
    "keypoint_confidence": 0.25,
    "pose_iou": 0.5,
    "object_confidence": 0.25,
    "tracker": "bytetrack.yaml",
    "temporal_mode": "off",
    "temporal_model_id": "",
    "temporal_clip_length": 12,
    "temporal_sample_fps": 6.0,
    "temporal_thresholds": {
        "phone_use": 0.35,
        "missing_badge": 0.35,
        "fall_down": 0.35,
    },
    "phone_classes": ["cell phone"],
    # Kept as an input/output compatibility field for older clients.
    "transfer_classes": ["cell phone"],
    "evidence": {
        "snapshot": True,
        "clip_pre_seconds": 2.0,
        "clip_post_seconds": 3.0,
        "retention_days": 90,
        "max_width": 960,
        "jpeg_quality": 70,
    },
    "rules": {
        "phone_use": {
            "window_size": 1,
            "required_hits": 1,
            "min_duration": 0.0,
            "resolve_gap": 1.0,
            "cooldown": 10.0,
            # Kept for compatibility with existing clients; no longer gates alerts.
            "stable_owner_frames": 3,
            "owner_distance_ratio": 0.35,
        },
        "fall_down": {
            "window_size": 8,
            "required_hits": 5,
            "min_duration": 1.2,
            "resolve_gap": 1.0,
            "cooldown": 30.0,
            "upright_lookback": 1.5,
            "transition_window": 1.0,
            "upright_angle": 35.0,
            "horizontal_angle": 55.0,
            "horizontal_aspect_ratio": 1.20,
            "fallback_aspect_ratio": 1.35,
            "center_drop_ratio": 0.20,
            "fallback_drop_ratio": 0.25,
            "hip_speed_ratio": 0.30,
        },
        "missing_badge": {
            "window_size": 10,
            "required_hits": 7,
            "min_duration": 1.2,
            "resolve_gap": 1.0,
            "cooldown": 15.0,
            "chest_padding_ratio": 0.18,
            "min_person_height": 120.0,
            "recent_badge_grace_seconds": 8.0,
            "session_grace_seconds": 3.0,
            "session_match_iou": 0.25,
            "session_match_distance_ratio": 0.35,
            "identity_min_confidence": 0.90,
        },
    },
}


def _deep_merge(base: Dict[str, Any], override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    merged = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = copy.deepcopy(value)
    return merged


def _bounded_float(value: Any, default: float, minimum: float, maximum: float) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    return max(minimum, min(maximum, result))


def _bounded_int(value: Any, default: int, minimum: int, maximum: int) -> int:
    try:
        result = int(float(value))
    except (TypeError, ValueError):
        result = default
    return max(minimum, min(maximum, result))


def normalize_behavior_config(*overrides: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Deep-merge and validate behavior settings in precedence order."""
    config = copy.deepcopy(DEFAULT_BEHAVIOR_CONFIG)
    for override in overrides:
        config = _deep_merge(config, override)

    config["enabled"] = bool(config.get("enabled", False))
    config["independent_snapshot_alerts"] = bool(config.get("independent_snapshot_alerts", False))
    config["require_face_for_missing_badge"] = bool(config.get("require_face_for_missing_badge", False))
    face_required=config.get('face_required_events') or []
    config['face_required_events']=sorted({code for code in face_required if isinstance(code,str)}) if isinstance(face_required,(list,tuple)) else []
    config["record_only"] = True
    config["behavior_fps"] = _bounded_float(config.get("behavior_fps"), 6.0, 1.0, 15.0)
    config["pose_confidence"] = _bounded_float(config.get("pose_confidence"), 0.35, 0.01, 1.0)
    config["keypoint_confidence"] = _bounded_float(config.get("keypoint_confidence"), 0.25, 0.01, 1.0)
    config["pose_iou"] = _bounded_float(config.get("pose_iou"), 0.5, 0.01, 1.0)
    config["object_confidence"] = _bounded_float(config.get("object_confidence"), 0.25, 0.01, 1.0)
    config["temporal_mode"] = str(config.get("temporal_mode") or "off").strip().lower()
    if config["temporal_mode"] == "augment":
        # Canonical name for new clients; retain filter internally for API
        # compatibility with stored profiles and runtime branches.
        config["temporal_mode"] = "filter"
    if config["temporal_mode"] not in {"off", "shadow", "filter"}:
        config["temporal_mode"] = "off"
    config["temporal_model_id"] = str(config.get("temporal_model_id") or "").strip()
    config["temporal_clip_length"] = _bounded_int(config.get("temporal_clip_length"), 12, 4, 48)
    config["temporal_sample_fps"] = _bounded_float(config.get("temporal_sample_fps"), 6.0, 1.0, 15.0)
    temporal_thresholds = config.get("temporal_thresholds") if isinstance(config.get("temporal_thresholds"), dict) else {}
    config["temporal_thresholds"] = {
        event_type: _bounded_float(temporal_thresholds.get(event_type), 0.35, 0.05, 0.95)
        for event_type in DEFAULT_BEHAVIOR_CONFIG["temporal_thresholds"]
    }
    config["pose_model_path"] = str(config.get("pose_model_path") or DEFAULT_BEHAVIOR_CONFIG["pose_model_path"])
    config["object_model_path"] = str(config.get("object_model_path") or DEFAULT_BEHAVIOR_CONFIG["object_model_path"])
    # Legacy badge_detector/YOLOE prompt settings are intentionally ignored.
    config.pop("badge_detector", None)
    config["badge_classes"] = sorted({
        str(item).strip()
        for item in (config.get("badge_classes") or DEFAULT_BEHAVIOR_CONFIG["badge_classes"])
        if str(item).strip()
    }, key=str.casefold)
    config["tracker"] = str(config.get("tracker") or "bytetrack.yaml")
    config["enabled_events"] = normalize_behavior_events(config.get("enabled_events"))
    config['face_required_events']=[code for code in config['face_required_events'] if code in config['enabled_events']]
    configured_phone_classes = config.get("phone_classes") or config.get("transfer_classes")
    phone_classes = {
        str(item).strip()
        for item in (configured_phone_classes or DEFAULT_BEHAVIOR_CONFIG["phone_classes"])
        if str(item).strip().casefold() in PHONE_CLASS_ALIASES
    }
    config["phone_classes"] = sorted(phone_classes, key=str.casefold) or list(
        DEFAULT_BEHAVIOR_CONFIG["phone_classes"]
    )
    # Preserve the old field for clients that still submit/read transfer_classes.
    config["transfer_classes"] = list(config["phone_classes"])

    evidence = config["evidence"]
    evidence["snapshot"] = bool(evidence.get("snapshot", True))
    evidence["clip_pre_seconds"] = _bounded_float(evidence.get("clip_pre_seconds"), 2.0, 0.0, 10.0)
    evidence["clip_post_seconds"] = _bounded_float(evidence.get("clip_post_seconds"), 3.0, 0.0, 10.0)
    evidence["retention_days"] = max(90, _bounded_int(evidence.get("retention_days"), 90, 1, 3650))
    evidence["max_width"] = _bounded_int(evidence.get("max_width"), 960, 320, 1920)
    evidence["jpeg_quality"] = _bounded_int(evidence.get("jpeg_quality"), 70, 40, 95)

    for event_type in SUPPORTED_EVENTS:
        defaults = DEFAULT_BEHAVIOR_CONFIG["rules"][event_type]
        rule = config["rules"].setdefault(event_type, {})
        rule["window_size"] = _bounded_int(rule.get("window_size"), defaults["window_size"], 1, 120)
        rule["required_hits"] = _bounded_int(
            rule.get("required_hits"),
            defaults["required_hits"],
            1,
            rule["window_size"],
        )
        rule["min_duration"] = _bounded_float(rule.get("min_duration"), defaults["min_duration"], 0.0, 60.0)
        rule["resolve_gap"] = _bounded_float(rule.get("resolve_gap"), defaults["resolve_gap"], 0.05, 30.0)
        rule["cooldown"] = _bounded_float(rule.get("cooldown"), defaults["cooldown"], 0.0, 3600.0)
    # A recognized phone is itself sufficient evidence; ignore legacy temporal gates.
    phone_rule = config["rules"]["phone_use"]
    phone_rule["window_size"] = 1
    phone_rule["required_hits"] = 1
    phone_rule["min_duration"] = 0.0
    phone = config["rules"]["phone_use"]
    phone_defaults = DEFAULT_BEHAVIOR_CONFIG["rules"]["phone_use"]
    phone["stable_owner_frames"] = _bounded_int(
        phone.get("stable_owner_frames"),
        phone_defaults["stable_owner_frames"],
        1,
        30,
    )
    phone["owner_distance_ratio"] = _bounded_float(
        phone.get("owner_distance_ratio"),
        phone_defaults["owner_distance_ratio"],
        0.05,
        1.0,
    )
    missing = config["rules"]["missing_badge"]
    missing_defaults = DEFAULT_BEHAVIOR_CONFIG["rules"]["missing_badge"]
    missing["chest_padding_ratio"] = _bounded_float(
        missing.get("chest_padding_ratio"),
        missing_defaults["chest_padding_ratio"],
        0.0,
        0.5,
    )
    missing["min_person_height"] = _bounded_float(
        missing.get("min_person_height"),
        missing_defaults["min_person_height"],
        40.0,
        2000.0,
    )
    missing["recent_badge_grace_seconds"] = _bounded_float(
        missing.get("recent_badge_grace_seconds"),
        missing_defaults["recent_badge_grace_seconds"],
        0.0,
        60.0,
    )
    missing["session_grace_seconds"] = _bounded_float(
        missing.get("session_grace_seconds"),
        missing_defaults["session_grace_seconds"],
        0.1,
        30.0,
    )
    missing["session_match_iou"] = _bounded_float(
        missing.get("session_match_iou"),
        missing_defaults["session_match_iou"],
        0.01,
        1.0,
    )
    missing["session_match_distance_ratio"] = _bounded_float(
        missing.get("session_match_distance_ratio"),
        missing_defaults["session_match_distance_ratio"],
        0.01,
        2.0,
    )
    missing["identity_min_confidence"] = _bounded_float(
        missing.get("identity_min_confidence"),
        missing_defaults["identity_min_confidence"],
        0.0,
        1.0,
    )
    return config


def normalize_behavior_override(override: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    """Validate an override while preserving only explicitly supplied keys."""
    original = override if isinstance(override, dict) else {}
    normalized = normalize_behavior_config(original)

    def project(source: Dict[str, Any], validated: Dict[str, Any]) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        for key, value in source.items():
            if key not in validated:
                continue
            if isinstance(value, dict) and isinstance(validated[key], dict):
                result[key] = project(value, validated[key])
            else:
                result[key] = copy.deepcopy(validated[key])
        return result

    result = project(original, normalized)
    if "record_only" in original:
        result["record_only"] = True
    return result


def _safe_track_id(value: Any) -> Optional[int]:
    try:
        result = int(value)
        return result if result >= 0 else None
    except (TypeError, ValueError):
        return None


class BadgeIdentitySessionManager:
    """Short-lived, conservative badge-code ownership across tracker ID changes."""

    def __init__(self, rule: Dict[str, Any]):
        self.rule = rule
        self.sessions: Dict[str, Dict[str, Any]] = {}
        self.track_sessions: Dict[int, str] = {}
        self.code_sessions: Dict[str, str] = {}
        self.next_session = 1

    def reset(self) -> None:
        self.sessions.clear()
        self.track_sessions.clear()
        self.code_sessions.clear()
        self.next_session = 1

    def update(self, persons: Dict[int, Dict[str, Any]], timestamp: float) -> None:
        self._expire(timestamp)
        used_sessions = set()
        for track_id, person in sorted(persons.items()):
            session_id = self.track_sessions.get(track_id)
            if session_id not in self.sessions or session_id in used_sessions:
                session_id = self._match_session(person, used_sessions, timestamp)
            if session_id is None:
                session_id = f"badge-session-{self.next_session}"
                self.next_session += 1
                self.sessions[session_id] = {
                    "bbox": list(person["bbox"]),
                    "last_seen": timestamp,
                    "badge_seen_at": None,
                    "badge_code": None,
                    "code_seen_at": None,
                    "conflict_until": 0.0,
                }
            session = self.sessions[session_id]
            session["bbox"] = list(person["bbox"])
            session["last_seen"] = timestamp
            self.track_sessions[track_id] = session_id
            person["person_session_id"] = session_id
            used_sessions.add(session_id)

        observed_codes: Dict[str, set] = defaultdict(set)
        for person in persons.values():
            session = self.sessions[person["person_session_id"]]
            if person.get("badge_association_state") == "ambiguous":
                continue
            if person.get("badge_present"):
                session["badge_seen_at"] = timestamp
            code = str(person.get("badge_code") or "").strip()
            if not code:
                continue
            confidence = float(person.get("badge_code_confidence") or 0.0)
            if confidence < float(self.rule["identity_min_confidence"]):
                person["badge_identity_state"] = "code_unverified"
                continue
            observed_codes[code].add(person["person_session_id"])

        for code, observed_sessions in observed_codes.items():
            previous_session = self.code_sessions.get(code)
            claimed_sessions = set(observed_sessions)
            if previous_session in self.sessions:
                previous = self.sessions[previous_session]
                if timestamp - float(previous.get("last_seen") or 0.0) <= float(self.rule["recent_badge_grace_seconds"]):
                    claimed_sessions.add(previous_session)
            if len(claimed_sessions) != 1:
                for session_id in claimed_sessions:
                    if session_id in self.sessions:
                        self.sessions[session_id]["conflict_until"] = timestamp + float(self.rule["recent_badge_grace_seconds"])
                continue
            session_id = next(iter(claimed_sessions))
            session = self.sessions[session_id]
            previous_code = session.get("badge_code")
            if previous_code and previous_code != code and timestamp - float(session.get("code_seen_at") or 0.0) <= float(self.rule["recent_badge_grace_seconds"]):
                session["conflict_until"] = timestamp + float(self.rule["recent_badge_grace_seconds"])
                continue
            if previous_code and previous_code != code:
                self.code_sessions.pop(previous_code, None)
            self.code_sessions[code] = session_id
            session["badge_code"] = code
            session["code_seen_at"] = timestamp

        for person in persons.values():
            session = self.sessions[person["person_session_id"]]
            if person.get("badge_association_state") == "ambiguous":
                person["badge_identity_state"] = "ambiguous"
                person["badge_evaluable"] = False
            elif timestamp < float(session.get("conflict_until") or 0.0):
                person["badge_identity_state"] = "conflict"
                person["badge_evaluable"] = False
            elif person.get("badge_present"):
                person["badge_identity_state"] = (
                    "bound" if session.get("badge_code") else "unidentified"
                )
            elif (
                session.get("badge_seen_at") is not None
                and timestamp - float(session["badge_seen_at"]) <= float(self.rule["recent_badge_grace_seconds"])
            ):
                person["badge_identity_state"] = "recently_confirmed"
                # Previous frames may explain identity, but the current image
                # determines whether a visible badge is present for alerts.
            else:
                person["badge_identity_state"] = "unseen"

    def stats(self) -> Dict[str, int]:
        return {
            "active_sessions": len(self.sessions),
            "active_bindings": len(self.code_sessions),
            "conflicted_sessions": sum(
                1 for session in self.sessions.values() if float(session.get("conflict_until") or 0.0) > 0.0
            ),
        }

    def _expire(self, timestamp: float) -> None:
        retention = max(
            float(self.rule["session_grace_seconds"]),
            float(self.rule["recent_badge_grace_seconds"]),
        )
        expired = [
            session_id
            for session_id, session in self.sessions.items()
            if timestamp - float(session.get("last_seen") or 0.0) > retention
        ]
        for session_id in expired:
            session = self.sessions.pop(session_id)
            code = session.get("badge_code")
            if code and self.code_sessions.get(code) == session_id:
                self.code_sessions.pop(code, None)
        self.track_sessions = {
            track_id: session_id
            for track_id, session_id in self.track_sessions.items()
            if session_id in self.sessions
        }

    def _match_session(
        self,
        person: Dict[str, Any],
        used_sessions: set,
        timestamp: float,
    ) -> Optional[str]:
        candidates = []
        person_box = person["bbox"]
        person_center = bbox_center(person_box)
        person_scale = max(1.0, math.sqrt(max(1.0, (person_box[2] - person_box[0]) * (person_box[3] - person_box[1]))))
        for session_id, session in self.sessions.items():
            if session_id in used_sessions:
                continue
            if timestamp - float(session.get("last_seen") or 0.0) > float(self.rule["session_grace_seconds"]):
                continue
            overlap = bbox_iou(person_box, session["bbox"])
            distance_ratio = dist(person_center, bbox_center(session["bbox"])) / person_scale
            if overlap < float(self.rule["session_match_iou"]) and distance_ratio > float(self.rule["session_match_distance_ratio"]):
                continue
            candidates.append((overlap + max(0.0, 1.0 - distance_ratio) * 0.1, session_id))
        candidates.sort(reverse=True)
        if not candidates:
            return None
        if len(candidates) > 1 and candidates[0][0] - candidates[1][0] < 0.12:
            return None
        return candidates[0][1]


def _midpoint(a: Optional[Tuple[float, float]], b: Optional[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2.0, (a[1] + b[1]) / 2.0)


def _torso_angle(person: Dict[str, Any]) -> Optional[float]:
    shoulders = _midpoint(get_kpt(person, LEFT_SHOULDER), get_kpt(person, RIGHT_SHOULDER))
    hips = _midpoint(get_kpt(person, LEFT_HIP), get_kpt(person, RIGHT_HIP))
    if shoulders is None or hips is None:
        return None
    dx = abs(hips[0] - shoulders[0])
    dy = abs(hips[1] - shoulders[1])
    if dx + dy < 1e-6:
        return None
    return math.degrees(math.atan2(dx, dy))


def _hip_y(person: Dict[str, Any]) -> Optional[float]:
    hips = _midpoint(get_kpt(person, LEFT_HIP), get_kpt(person, RIGHT_HIP))
    return hips[1] if hips else None


def _shoulder_width(person: Dict[str, Any]) -> Optional[float]:
    left = get_kpt(person, LEFT_SHOULDER)
    right = get_kpt(person, RIGHT_SHOULDER)
    if left is None or right is None:
        return None
    width = dist(left, right)
    return width if width > 1.0 else None


def _event_key(event_type: str, persons: Iterable[int], object_id: Optional[int] = None) -> str:
    if event_type == "phone_use" and object_id is not None:
        # Phone alarms are object-driven; owner changes must not create duplicates.
        return f"{event_type}::object:{int(object_id)}"
    person_part = ",".join(str(item) for item in sorted(set(int(item) for item in persons)))
    object_part = f":{int(object_id)}" if object_id is not None else ""
    return f"{event_type}:{person_part}{object_part}"


@dataclass
class CandidateState:
    event_type: str
    key: str
    persons: List[int]
    object_id: Optional[int]
    first_hit: float
    last_hit: float
    hits: Deque[bool]
    best_score: float = 0.0
    reason: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ActiveState:
    event: Dict[str, Any]
    last_hit: float


class BehaviorRuleEngine:
    """Stateful rule evaluator. Model inference and file I/O stay outside this class."""

    def __init__(self, source_id: str, config: Optional[Dict[str, Any]] = None):
        self.source_id = str(source_id)
        self.config = normalize_behavior_config(config)
        self.person_history: Dict[int, Deque[Dict[str, Any]]] = defaultdict(lambda: deque(maxlen=120))
        self.badge_identity = BadgeIdentitySessionManager(self.config["rules"]["missing_badge"])
        self.candidates: Dict[str, CandidateState] = {}
        self.active: Dict[str, ActiveState] = {}
        self.cooldowns: Dict[str, float] = {}
        self.transferred_pairs: set = set()
        self.frames_processed = 0
        self.events_confirmed = 0
        self.last_timestamp: Optional[float] = None

    def reset(self) -> None:
        self.person_history.clear()
        self.badge_identity.reset()
        self.candidates.clear()
        self.active.clear()
        self.cooldowns.clear()
        self.transferred_pairs.clear()
        self.frames_processed = 0
        self.events_confirmed = 0
        self.last_timestamp = None

    def update(
        self,
        persons: Dict[int, Dict[str, Any]],
        objects: Optional[Dict[int, Dict[str, Any]]] = None,
        *,
        timestamp: Optional[float] = None,
        frame_index: Optional[int] = None,
    ) -> Dict[str, Any]:
        timestamp = float(time.time() if timestamp is None else timestamp)
        if self.last_timestamp is not None and timestamp < self.last_timestamp:
            self.reset()
        self.last_timestamp = timestamp
        self.frames_processed += 1
        frame_index = int(frame_index if frame_index is not None else self.frames_processed)

        normalized_persons: Dict[int, Dict[str, Any]] = {}
        for raw_id, person in (persons or {}).items():
            track_id = _safe_track_id(person.get("track_id", raw_id))
            if track_id is None:
                continue
            item = dict(person)
            item["track_id"] = track_id
            item.setdefault("bbox", [0.0, 0.0, 0.0, 0.0])
            item.setdefault("center", bbox_center(item["bbox"]))
            item.setdefault("width", max(1.0, float(item["bbox"][2]) - float(item["bbox"][0])))
            item.setdefault("height", max(1.0, float(item["bbox"][3]) - float(item["bbox"][1])))
            item.setdefault("keypoints", [])
            normalized_persons[track_id] = item
            self.person_history[track_id].append({
                **item,
                "timestamp": timestamp,
                "frame_index": frame_index,
                "torso_angle": _torso_angle(item),
                "hip_y": _hip_y(item),
            })

        self.badge_identity.update(normalized_persons, timestamp)
        # Recorder snapshots are independent decisions. Resolve the preceding
        # snapshot before evaluating this one, even when the violation persists.
        transitions = self.close(timestamp) if self.config["independent_snapshot_alerts"] else []
        raw_events = self._detect(normalized_persons, objects or {}, timestamp)
        transitions.extend(self._advance(raw_events, timestamp))
        return {
            "transitions": transitions,
            "active_events": [copy.deepcopy(item.event) for item in self.active.values()],
            "stats": self.stats(),
        }

    def close(self, timestamp: Optional[float] = None) -> List[Dict[str, Any]]:
        timestamp = float(time.time() if timestamp is None else timestamp)
        transitions: List[Dict[str, Any]] = []
        for key, state in list(self.active.items()):
            event = copy.deepcopy(state.event)
            event["state"] = "resolved"
            event["ended_at"] = timestamp
            event["duration_seconds"] = round(max(0.0, timestamp - float(event["started_at"])), 3)
            transitions.append(event)
            self.active.pop(key, None)
        self.candidates.clear()
        return transitions

    def stats(self) -> Dict[str, Any]:
        return {
            "frames_processed": self.frames_processed,
            "events_confirmed": self.events_confirmed,
            "active_events": len(self.active),
            "candidate_events": len(self.candidates),
            "behavior_fps": self.config["behavior_fps"],
            "badge_identity": self.badge_identity.stats(),
        }

    def _detect(
        self,
        persons: Dict[int, Dict[str, Any]],
        objects: Dict[int, Dict[str, Any]],
        timestamp: float,
    ) -> List[Dict[str, Any]]:
        enabled = set(self.config["enabled_events"])
        events: List[Dict[str, Any]] = []
        if "fall_down" in enabled:
            events.extend(self._detect_falls(persons, timestamp))
        if "phone_use" in enabled:
            events.extend(self._detect_phone_use(persons, objects))
        if "missing_badge" in enabled:
            events.extend(self._detect_missing_badge(persons))
        return events

    def _detect_missing_badge(self, persons: Dict[int, Dict[str, Any]]) -> List[Dict[str, Any]]:
        rule = self.config["rules"]["missing_badge"]
        events = []
        for track_id, person in persons.items():
            if not person.get("badge_evaluable") or person.get("badge_present"):
                continue
            if (self.config["require_face_for_missing_badge"] or 'missing_badge' in self.config['face_required_events']) and not person.get("matched_face"):
                continue
            if float(person.get("height") or 0.0) < rule["min_person_height"]:
                continue
            face = person.get("matched_face") or {}
            events.append(self._raw_event(
                "missing_badge",
                [track_id],
                min(0.95, max(0.55, float(person.get("badge_visibility_score") or 0.70))),
                {
                    "badge_present": False,
                    "chest_roi": person.get("chest_roi"),
                    "detector_mode": "object_model",
                    "badge_classes": person.get("badge_classes") or [],
                    "person_session_id": person.get("person_session_id"),
                    "badge_identity_state": person.get("badge_identity_state"),
                    "person_bbox": list(person.get("bbox") or []),
                    "face_bbox": list(face.get("bbox") or []),
                    "face_crop_bbox": list(face.get("crop_bbox") or []),
                    "face_confidence": face.get("confidence"),
                    "face_model_sha256": face.get("model_sha256"),
                },
            ))
        return events

    def _detect_falls(self, persons: Dict[int, Dict[str, Any]], timestamp: float) -> List[Dict[str, Any]]:
        rule = self.config["rules"]["fall_down"]
        events = []
        for track_id, person in persons.items():
            if self.config["independent_snapshot_alerts"]:
                aspect = float(person.get("width") or 0.0) / max(float(person.get("height") or 0.0), 1.0)
                torso_angle = _torso_angle(person)
                horizontal = ((torso_angle is not None and torso_angle >= rule["horizontal_angle"])
                              or aspect >= rule["fallback_aspect_ratio"])
                if not horizontal:
                    continue
                events.append(self._raw_event(
                    "fall_down", [track_id],
                    min(0.95, 0.55 + min(0.25, max(0.0, aspect - 1.0) * 0.25)),
                    {"single_frame_posture": True,
                     "torso_angle": None if torso_angle is None else round(torso_angle, 3),
                     "aspect_ratio": round(aspect, 3),
                     "person_bbox": list(person.get("bbox") or [])},
                ))
                continue
            history = list(self.person_history[track_id])
            if len(history) < 2:
                continue
            current = history[-1]
            recent = [item for item in history if timestamp - float(item["timestamp"]) <= rule["upright_lookback"]]
            upright_seen = any(
                (item.get("torso_angle") is not None and item["torso_angle"] < rule["upright_angle"])
                or float(item.get("width", 1.0)) / max(float(item.get("height", 1.0)), 1.0) < 0.9
                for item in recent[:-1]
            )
            transition_history = [
                item for item in history
                if timestamp - float(item["timestamp"]) <= rule["transition_window"]
            ]
            old = transition_history[0]
            old_height = max(float(old.get("height", 1.0)), 1.0)
            aspect = float(current.get("width", 1.0)) / max(float(current.get("height", 1.0)), 1.0)
            center_drop = (float(current["center"][1]) - float(old["center"][1])) / old_height
            torso_angle = current.get("torso_angle")
            hip_speed = 0.0
            if old.get("hip_y") is not None and current.get("hip_y") is not None:
                dt = max(timestamp - float(old["timestamp"]), 1e-6)
                hip_speed = (float(current["hip_y"]) - float(old["hip_y"])) / dt / old_height
            pose_ready = torso_angle is not None
            posture = (
                (pose_ready and torso_angle >= rule["horizontal_angle"])
                or aspect >= rule["horizontal_aspect_ratio"]
            )
            transition = center_drop >= rule["center_drop_ratio"] or hip_speed >= rule["hip_speed_ratio"]
            fallback = (
                not pose_ready
                and aspect >= rule["fallback_aspect_ratio"]
                and center_drop >= rule["fallback_drop_ratio"]
            )
            key = _event_key("fall_down", [track_id])
            tracking_fall = key in self.candidates or key in self.active
            if not ((upright_seen and ((posture and transition) or fallback)) or (tracking_fall and posture)):
                continue
            score = min(
                0.98,
                0.45
                + min(0.20, max(0.0, aspect - 1.0) * 0.30)
                + min(0.18, max(0.0, center_drop) * 0.50)
                + min(0.15, max(0.0, hip_speed) * 0.20),
            )
            events.append(self._raw_event(
                "fall_down",
                [track_id],
                score,
                {
                    "upright_seen": upright_seen,
                    "torso_angle": None if torso_angle is None else round(torso_angle, 3),
                    "aspect_ratio": round(aspect, 3),
                    "center_drop_ratio": round(center_drop, 3),
                    "hip_speed_ratio": round(hip_speed, 3),
                    "fallback_used": bool(fallback),
                    "transition_latched": bool(tracking_fall),
                    "person_bbox": list(current.get("bbox") or person.get("bbox") or []),
                },
            ))
        return events

    def _detect_phone_use(
        self,
        persons: Dict[int, Dict[str, Any]],
        objects: Dict[int, Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        events = []
        for object_id, obj in objects.items():
            if str(obj.get("class_name") or "").strip().casefold() not in {
                str(item).casefold() for item in self.config["phone_classes"]
            }:
                continue
            owner_id = obj.get("owner_id")
            try:
                owner_id = int(owner_id) if owner_id is not None and int(owner_id) in persons else None
            except (TypeError, ValueError):
                owner_id = None
            if 'phone_use' in self.config['face_required_events'] and (owner_id is None or not persons[owner_id].get('matched_face')):
                continue
            try:
                detection_confidence = float(obj.get("confidence") or 0.0)
            except (TypeError, ValueError):
                detection_confidence = 0.0
            distance_ratio = obj.get("owner_distance_ratio")
            try:
                distance_ratio = round(float(distance_ratio), 3) if distance_ratio is not None else None
            except (TypeError, ValueError):
                distance_ratio = None
            score = min(
                0.99,
                max(0.50, detection_confidence),
            )
            events.append(self._raw_event(
                "phone_use",
                [owner_id] if owner_id is not None else [],
                score,
                {
                    "owner_id": owner_id,
                    "owner_distance_ratio": distance_ratio,
                    "object_class": obj.get("class_name"),
                    "detection_confidence": round(detection_confidence, 4),
                    "person_bbox": list(persons.get(owner_id, {}).get("bbox") or []) if owner_id is not None else [],
                    "object_bbox": list(obj.get("bbox") or []),
                },
                object_id=int(object_id),
                metadata={"object_class": obj.get("class_name")},
            ))
        return events

    def _raw_event(
        self,
        event_type: str,
        persons: List[int],
        score: float,
        reason: Dict[str, Any],
        *,
        object_id: Optional[int] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        return {
            "key": _event_key(event_type, persons, object_id),
            "event_type": event_type,
            "persons": sorted(set(int(item) for item in persons)),
            "object_id": object_id,
            "rule_score": round(max(0.0, min(1.0, float(score))), 4),
            "reason": reason,
            "metadata": metadata or {},
        }

    def _pose_feature_vector(self, person_ids: Iterable[int], max_frames: int = 96) -> Optional[List[float]]:
        frames: List[Tuple[float, int, Any]] = []
        for track_id in person_ids or []:
            for item in list(self.person_history.get(int(track_id), []))[-max(1, int(max_frames)):]:
                keypoints = item.get("keypoints") or []
                if keypoints:
                    frames.append((float(item.get("timestamp") or 0.0), int(track_id), keypoints))
        if not frames:
            return None
        frames.sort(key=lambda item: (item[0], item[1]))
        try:
            encoded = [encode_single_pose(item[2]) for item in frames[-max(1, int(max_frames)):]]
            return aggregate_pose_features(encoded).astype(float).tolist()
        except (TypeError, ValueError):
            return None

    def _advance(self, raw_events: List[Dict[str, Any]], timestamp: float) -> List[Dict[str, Any]]:
        transitions: List[Dict[str, Any]] = []
        current = {event["key"]: event for event in raw_events}
        tracked_keys = set(self.active) | set(current)

        for key in tracked_keys:
            raw = current.get(key)
            event_type = (
                raw.get("event_type")
                if raw
                else self.active[key].event["event_type"]
            )
            rule = self.config["rules"][event_type]
            if key in self.active:
                state = self.active[key]
                if raw:
                    state.last_hit = timestamp
                    state.event["ended_at"] = None
                    if raw["rule_score"] >= state.event["rule_score"]:
                        state.event["rule_score"] = raw["rule_score"]
                        state.event["confidence"] = raw["rule_score"]
                        state.event["reason"] = raw["reason"]
                    continue
                if timestamp - state.last_hit <= rule["resolve_gap"]:
                    continue
                event = copy.deepcopy(state.event)
                event["state"] = "resolved"
                event["ended_at"] = timestamp
                event["duration_seconds"] = round(timestamp - float(event["started_at"]), 3)
                transitions.append(event)
                self.active.pop(key, None)
                continue

            if not raw:
                continue
            event_id = str(uuid.uuid4())
            event = {
                    "event_id": event_id,
                    "event_type": event_type,
                    "state": "confirmed",
                    "severity": EVENT_SEVERITY[event_type],
                    "rule_score": round(float(raw["rule_score"]), 4),
                    "confidence": round(float(raw["rule_score"]), 4),
                    "should_alert": True,
                    "source_id": self.source_id,
                    "persons": list(raw["persons"]),
                    "track_ids": list(raw["persons"]),
                    "object_id": raw.get("object_id"),
                    "started_at": timestamp,
                    "confirmed_at": timestamp,
                    "ended_at": None,
                    "duration_seconds": 0.0,
                    "reason": copy.deepcopy(raw["reason"]),
                    "pose_feature_vector": self._pose_feature_vector(raw["persons"]),
                    **copy.deepcopy(raw.get("metadata") or {}),
            }
            self.active[key] = ActiveState(event={**event, "state": "active"}, last_hit=timestamp)
            self.events_confirmed += 1
            transitions.append(event)
        return transitions

    def _max_wrist_speed(self, track_id: int) -> float:
        history = list(self.person_history[track_id])
        if len(history) < 2:
            return 0.0
        current = history[-1]
        previous = next(
            (item for item in reversed(history[:-1]) if float(current["timestamp"]) - float(item["timestamp"]) >= 0.25),
            history[0],
        )
        dt = max(float(current["timestamp"]) - float(previous["timestamp"]), 1e-6)
        height = max(float(current.get("height", 1.0)), 1.0)
        speeds = []
        for index in (LEFT_WRIST, RIGHT_WRIST):
            left = get_kpt(previous, index)
            right = get_kpt(current, index)
            if left is not None and right is not None:
                speeds.append(dist(left, right) / dt / height)
        return max(speeds, default=0.0)

    @staticmethod
    def _wrist_in_body(person: Dict[str, Any], other_bbox: Iterable[float]) -> bool:
        return any(
            point is not None and point_in_upper_body(point, other_bbox)
            for point in (get_kpt(person, LEFT_WRIST), get_kpt(person, RIGHT_WRIST))
        )

    @staticmethod
    def _stable_owner(window: List[Dict[str, Any]], required: int) -> Optional[int]:
        ids = [int(item["owner_id"]) for item in window if item.get("owner_id") is not None]
        if len(ids) < required:
            return None
        counts: Dict[int, int] = {}
        for owner_id in ids:
            counts[owner_id] = counts.get(owner_id, 0) + 1
        owner_id, count = max(counts.items(), key=lambda item: item[1])
        return owner_id if count >= required else None
