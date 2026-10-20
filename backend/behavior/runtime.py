"""Shared object + Pose runtime for temporal behavior rules."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import cv2
import numpy as np

from monitoring_policy import behavior_detections
from .evidence import BehaviorEvidenceRecorder
from .object_tracker import SimpleObjectTracker
from .realtime_engine import (
    LEFT_HIP,
    LEFT_SHOULDER,
    LEFT_WRIST,
    RIGHT_HIP,
    RIGHT_SHOULDER,
    RIGHT_WRIST,
    BehaviorRuleEngine,
    assign_badge_detections,
    bbox_center,
    bbox_iou,
    dist,
    get_kpt,
    matching_class_names,
    normalized_class_names,
    normalize_behavior_config,
)
from .temporal_runtime import TemporalModelRuntime


def _bbox_from_detection(detection: Dict[str, Any]) -> Optional[List[float]]:
    box = detection.get("bbox") or detection.get("box") or detection.get("xyxy")
    if isinstance(box, dict):
        x1 = float(box.get("x1", box.get("left", box.get("x", 0.0))))
        y1 = float(box.get("y1", box.get("top", box.get("y", 0.0))))
        x2 = float(box.get("x2", box.get("right", x1 + float(box.get("width", 0.0)))))
        y2 = float(box.get("y2", box.get("bottom", y1 + float(box.get("height", 0.0)))))
        return [x1, y1, x2, y2]
    if isinstance(box, (list, tuple)) and len(box) >= 4:
        return [float(value) for value in box[:4]]
    return None


def associate_faces_to_persons(persons: Dict[int, Dict[str, Any]], faces: List[Dict[str, Any]]) -> None:
    """Associate a face only when its center lies in one person's upper body."""
    for person in persons.values():
        person.pop("matched_face", None)
    for face in faces:
        box = face.get("detected_bbox") or face.get("bbox")
        if not isinstance(box, (list, tuple)) or len(box) != 4:
            continue
        left, top, right, bottom = [float(value) for value in box]
        if right <= left or bottom <= top:
            continue
        center_x, center_y = (left + right) / 2.0, (top + bottom) / 2.0
        owners = []
        for person in persons.values():
            person_box = person.get("bbox") or []
            if len(person_box) != 4:
                continue
            x1, y1, x2, y2 = [float(value) for value in person_box]
            if x1 <= center_x <= x2 and y1 <= center_y <= y1 + (y2 - y1) * 0.5:
                owners.append(person)
        if len(owners) != 1:
            continue
        matched = {"bbox": [left, top, right, bottom],
                   "crop_bbox": list(face.get("bbox") or box),
                   "confidence": float(face.get("confidence") or 0.0),
                   "model_sha256": face.get("model_sha256")}
        previous = owners[0].get("matched_face") or {}
        if matched["confidence"] > float(previous.get("confidence") or 0.0):
            owners[0]["matched_face"] = matched


class SharedBehaviorModelPool:
    def __init__(self):
        self.models: Dict[str, Any] = {}
        self.model_locks: Dict[str, threading.RLock] = {}
        self.lock = threading.RLock()
        from inference_service.client import inference_client
        self.inference_client = inference_client

    def predict(self, model_path: str, image_rgb: np.ndarray, **kwargs: Any):
        model_id = kwargs.pop("model_id", None)
        revision = kwargs.pop("revision", None)
        enabled_events = kwargs.pop("enabled_events", None)
        model_ids = kwargs.pop("model_ids", None)
        confidence = kwargs.get("conf", .25)
        iou = kwargs.get("iou", .7)
        if model_ids:
            if revision is not None:
                results = self.inference_client.infer_pinned_sync(image_rgb, model_ids, revision=revision,
                                                                 confidence=confidence, iou=iou,
                                                                 enabled_events=enabled_events)
            else:
                results = self.inference_client.infer_current_sync(image_rgb, model_ids,
                                                                  confidence=confidence, iou=iou,
                                                                  enabled_events=enabled_events)
            detections = [item for result in results for item in result.get("detections", [])]
            class_names = [name for result in results for name in result.get("class_names", [])]
            return [{"detections": detections, "class_names": class_names, "poses": []}]
        if model_id and revision is not None:
            results = self.inference_client.infer_pinned_sync(image_rgb, [model_id], revision=revision,
                                                              confidence=confidence, iou=iou,
                                                              enabled_events=enabled_events)
            if not results:
                return []
            item = results[0]
            return [{"detections": item.get("detections", []), "names": {},
                     "poses": item.get("poses", [])}]
        if model_id:
            results = self.inference_client.infer_current_sync(image_rgb, [model_id],
                                                              confidence=confidence, iou=iou,
                                                              enabled_events=enabled_events)
            if not results:
                return []
            item = results[0]
            return [{"detections": item.get("detections", []), "names": {},
                     "poses": item.get("poses", [])}]
        resolved_path = self._resolve_path(model_path)
        model = self._get_model(resolved_path)
        with self.lock:
            model_lock = self.model_locks[resolved_path]
        with model_lock:
            bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            return model.predict(source=bgr, verbose=False, **kwargs)

    def _get_model(self, model_path: str):
        path = str(model_path or "")
        if not Path(path).exists():
            raise FileNotFoundError(f"behavior model does not exist: {path}")
        with self.lock:
            if path not in self.models:
                from ultralytics import YOLO

                self.models[path] = YOLO(path)
                self.model_locks[path] = threading.RLock()
            return self.models[path]

    @staticmethod
    def _resolve_path(model_path: str) -> str:
        requested = Path(str(model_path or ""))
        candidates = (
            requested,
            Path.cwd() / "models" / requested.name,
            Path(__file__).resolve().parents[1] / "models" / requested.name,
            Path("/app/models") / requested.name,
            Path("/app/models/pretrained") / requested.name,
        )
        for candidate in candidates:
            if candidate.is_file():
                return str(candidate)
        checked = ", ".join(str(item) for item in candidates)
        raise FileNotFoundError(
            f"behavior model does not exist: {model_path}; checked: {checked}"
        )


class SimplePersonTracker:
    """Per-source fallback used only when primary ByteTrack IDs are unavailable."""

    def __init__(self, max_missed_frames: int = 12):
        self.max_missed_frames = max_missed_frames
        self.next_id = 1
        self.tracks: Dict[int, Dict[str, Any]] = {}

    def update(self, detections: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
        unmatched_tracks = set(self.tracks)
        unmatched_detections = set(range(len(detections)))
        candidates: List[Tuple[float, int, int]] = []
        for track_id, track in self.tracks.items():
            for index, detection in enumerate(detections):
                overlap = bbox_iou(track["bbox"], detection["bbox"])
                scale = max(float(track["height"]), float(detection["height"]), 1.0)
                center_score = max(0.0, 1.0 - dist(track["center"], detection["center"]) / scale)
                score = max(overlap, center_score * 0.75)
                if overlap >= 0.15 or center_score >= 0.40:
                    candidates.append((score, track_id, index))
        candidates.sort(reverse=True)
        for _, track_id, index in candidates:
            if track_id not in unmatched_tracks or index not in unmatched_detections:
                continue
            item = dict(detections[index])
            item["track_id"] = track_id
            item["missed"] = 0
            self.tracks[track_id] = item
            unmatched_tracks.remove(track_id)
            unmatched_detections.remove(index)
        for index in unmatched_detections:
            track_id = self.next_id
            self.next_id += 1
            item = dict(detections[index])
            item["track_id"] = track_id
            item["missed"] = 0
            self.tracks[track_id] = item
        for track_id in list(unmatched_tracks):
            self.tracks[track_id]["missed"] = int(self.tracks[track_id].get("missed", 0)) + 1
            if self.tracks[track_id]["missed"] > self.max_missed_frames:
                self.tracks.pop(track_id, None)
        return {
            track_id: {key: value for key, value in item.items() if key != "missed"}
            for track_id, item in self.tracks.items()
            if int(item.get("missed", 0)) == 0
        }


class BehaviorSessionRuntime:
    def __init__(
        self,
        source_id: str,
        config: Optional[Dict[str, Any]],
        model_pool: SharedBehaviorModelPool,
        output_root: Optional[str] = None,
    ):
        self.source_id = str(source_id)
        self.config = normalize_behavior_config(config)
        self.model_pool = model_pool
        self.engine = BehaviorRuleEngine(source_id, self.config)
        self.temporal_model = TemporalModelRuntime(
            mode=self.config.get("temporal_mode", "off"),
            model_id=self.config.get("temporal_model_id"),
            thresholds=self.config.get("temporal_thresholds"),
            clip_length=self.config.get("temporal_clip_length", 12),
            sample_fps=self.config.get("temporal_sample_fps", self.config.get("behavior_fps", 6.0)),
        )
        self.person_tracker = SimplePersonTracker()
        self.object_tracker = SimpleObjectTracker(max_missed_frames=18)
        self.evidence = BehaviorEvidenceRecorder(source_id, self.config, output_root)
        self.last_processed_at: Optional[float] = None
        self.frame_index = 0
        self.frames_dropped = 0
        self.processing_times: List[float] = []
        self.badge_detector_status: Dict[str, Any] = {
            "enabled": "missing_badge" in self.config["enabled_events"],
            "available": None,
            "error": None,
            "mode": "object_model",
            "model_path": self.config["object_model_path"],
            "badge_classes": list(self.config["badge_classes"]),
        }
        self.face_detector_status: Dict[str, Any] = {"enabled": bool(self.config["require_face_for_missing_badge"]),
                                                     "available": None, "error": None, "count": 0}
        self.lock = threading.RLock()

    def process(
        self,
        image_rgb: np.ndarray,
        primary_detections: Optional[List[Dict[str, Any]]] = None,
        *,
        timestamp: Optional[float] = None,
    ) -> Dict[str, Any]:
        timestamp = float(time.time() if timestamp is None else timestamp)
        with self.lock:
            interval = 1.0 / max(float(self.config["behavior_fps"]), 1.0)
            if (not self.config.get("process_every_sampled_frame")
                    and self.last_processed_at is not None
                    and timestamp - self.last_processed_at < interval):
                self.frames_dropped += 1
                return self._response([], [], skipped=True)
            self.last_processed_at = timestamp
            self.frame_index += 1
            started = time.perf_counter()
            evidence_updates = self.evidence.add_frame(image_rgb, timestamp)
            try:
                persons = self._infer_persons(image_rgb, primary_detections or [])
                scene_detections: Optional[List[Dict[str, Any]]] = None
                model_classes: Optional[set] = None
                if ("missing_badge" in self.config["enabled_events"] or "phone_use" in self.config["enabled_events"]):
                    try:
                        scene_detections, model_classes = self._detect_scene_objects(image_rgb)
                    except Exception as exc:
                        self.badge_detector_status.update({
                            "available": False,
                            "error": str(exc)[:300],
                        })
                        if self.config.get("scene_model_ids"):
                            raise
                self._annotate_badge_presence(persons, scene_detections, model_classes)
                if self.config.get('face_required_events'):
                    from face_capture.alarm_gate import FacePresenceGate
                    gate=FacePresenceGate(image_rgb,f'{self.source_id}:{self.frame_index}:{timestamp:.9f}')
                    gate.observe_people(p.get('bbox') for p in persons.values())
                    faces=[]
                    for person in persons.values():
                        person['alarm_face_gate']=gate.check(person.get('bbox'))
                        checked=person['alarm_face_gate']
                        if checked['passed']:
                            faces.append({'bbox':checked['face_crop_bbox'],'detected_bbox':checked['face_bbox'],
                                          'confidence':checked['face_confidence'],'model_sha256':checked['face_model_sha256']})
                    associate_faces_to_persons(persons,faces)
                    self.face_detector_status.update({'enabled':True,'available':not any(p['alarm_face_gate'].get('reason')=='face_detection_failed' for p in persons.values()),'error':None,'count':len(faces)})
                elif self.config["require_face_for_missing_badge"] and "missing_badge" in self.config["enabled_events"]:
                    try:
                        from face_capture.detector import detect as detect_faces
                        faces = detect_faces(cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR), threshold=0.6, min_size=20)
                        associate_faces_to_persons(persons, faces)
                        self.face_detector_status.update({"available": True, "error": None, "count": len(faces)})
                    except Exception as exc:
                        associate_faces_to_persons(persons, [])
                        self.face_detector_status.update({"available": False, "error": str(exc)[:300], "count": 0})
                objects = self._infer_objects(
                    image_rgb,
                    primary_detections or [],
                    timestamp,
                    persons,
                    scene_detections=scene_detections,
                )
                result = self.engine.update(
                    persons,
                    objects,
                    timestamp=timestamp,
                    frame_index=self.frame_index,
                )
                temporal_result = self.temporal_model.update(
                    image_rgb,
                    persons,
                    objects,
                    timestamp,
                    enabled_events=self.config.get("enabled_events"),
                    input_is_rgb=True,
                )
                rule_transitions = [
                    transition
                    for transition in result["transitions"]
                    if transition.get("state") in {"confirmed", "resolved"}
                ]
                public_transitions = list(rule_transitions)
                temporal_events = list(temporal_result.get("events") or [])
                if self.config.get("temporal_mode") == "filter":
                    public_transitions.extend(self._merge_temporal_events(public_transitions, temporal_events))
                public_transitions=self._face_admitted_events(public_transitions,persons)
                self._decorate_decisions(
                    public_transitions,
                    rule_transitions,
                    temporal_events,
                    temporal_result,
                )
                for transition in public_transitions:
                    if transition["state"] == "confirmed":
                        evidence = self.evidence.confirm(transition, image_rgb)
                        transition.update(evidence)
                elapsed = (time.perf_counter() - started) * 1000.0
                self.processing_times.append(elapsed)
                self.processing_times = self.processing_times[-120:]
                response = self._response(
                    public_transitions,
                    evidence_updates,
                    active_events=self._face_admitted_events(result["active_events"],persons),
                    temporal_shadow_events=(
                        temporal_events if self.config.get("temporal_mode") == "shadow" else []
                    ),
                    temporal_score_snapshots=temporal_result.get("score_snapshots") or [],
                    temporal_model=temporal_result,
                )
                response['_monitor_detections'] = behavior_detections(persons, primary_detections, list(scene_detections or []) + list((objects or {}).values()), self.config['enabled_events'], self.config)
                return response
            except Exception as exc:
                elapsed = (time.perf_counter() - started) * 1000.0
                self.processing_times.append(elapsed)
                self.processing_times = self.processing_times[-120:]
                response = self._response([], evidence_updates, temporal_model=self.temporal_model.status())
                response["error"] = str(exc)
                return response

    def _face_admitted_events(self, events, persons):
        """Guard rule/model final results with this frame's person face evidence."""
        required=set(self.config.get('face_required_events') or [])
        if not required:
            return events
        admitted=[]
        for event in events:
            if event.get('event_type') not in required or event.get('state')=='resolved':
                admitted.append(event)
                continue
            ids=event.get('persons') or event.get('track_ids') or []
            try:
                targets=[persons[int(track)] for track in ids]
            except (TypeError,ValueError,KeyError):
                continue
            if not targets or not all(p.get('matched_face') and (p.get('alarm_face_gate') or {}).get('passed') for p in targets):
                continue
            gate=targets[0]['alarm_face_gate']
            event['reason']={**(event.get('reason') or {}),'face_gate':dict(gate),
                             **{k:gate[k] for k in ('face_bbox','face_crop_bbox','face_confidence','face_model_sha256')}}
            event.setdefault('decision',{})['face_gate']=dict(gate)
            admitted.append(event)
        return admitted

    def close(self, timestamp: Optional[float] = None) -> Dict[str, Any]:
        with self.lock:
            transitions = self.engine.close(timestamp)
            temporal_status = self.temporal_model.status()
            self._decorate_decisions(transitions, transitions, [], temporal_status)
            evidence_updates = self.evidence.close()
            return self._response(transitions, evidence_updates, temporal_model=temporal_status)

    def _response(
        self,
        transitions: List[Dict[str, Any]],
        evidence_updates: List[Dict[str, Any]],
        *,
        active_events: Optional[List[Dict[str, Any]]] = None,
        skipped: bool = False,
        temporal_shadow_events: Optional[List[Dict[str, Any]]] = None,
        temporal_score_snapshots: Optional[List[Dict[str, Any]]] = None,
        temporal_model: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        stats = self.engine.stats()
        stats.update({
            "frames_dropped": self.frames_dropped,
            "avg_processing_ms": round(
                sum(self.processing_times) / len(self.processing_times),
                3,
            ) if self.processing_times else 0.0,
            "skipped": skipped,
        })
        stats["badge_detector"] = dict(self.badge_detector_status)
        stats["face_detector"] = dict(self.face_detector_status)
        stats["temporal_model"] = temporal_model or self.temporal_model.status()
        return {
            "action_events": transitions,
            "active_events": active_events if active_events is not None else [
                item.event.copy() for item in self.engine.active.values()
            ],
            "temporal_shadow_events": temporal_shadow_events or [],
            "temporal_score_snapshots": temporal_score_snapshots or [],
            "temporal_model": temporal_model or self.temporal_model.status(),
            "behavior_stats": stats,
            "evidence_updates": evidence_updates,
        }

    @staticmethod
    def _decorate_decisions(
        events: List[Dict[str, Any]],
        rule_events: List[Dict[str, Any]],
        temporal_events: List[Dict[str, Any]],
        temporal_result: Dict[str, Any],
    ) -> None:
        """Expose rule/model/final decisions without hiding the rule fallback."""
        model_id = str(temporal_result.get("model_id") or "")
        model_sha256 = str(temporal_result.get("model_sha256") or "")
        model_version = str(temporal_result.get("model_version") or model_id)
        available = bool(temporal_result.get("available"))
        score_snapshots = list(temporal_result.get("score_snapshots") or [])
        for event in events:
            event_type = str(event.get("event_type") or "")
            tracks = set(event.get("persons") or event.get("track_ids") or [])
            timestamp = float(event.get("confirmed_at") or event.get("start_time") or 0.0)
            matching_temporal = next(
                (
                    candidate for candidate in temporal_events
                    if str(candidate.get("event_type") or "") == event_type
                    and (
                        (tracks and set(candidate.get("persons") or []) and tracks.intersection(set(candidate.get("persons") or [])))
                        or (not tracks and not set(candidate.get("persons") or []))
                    )
                    and abs(float(candidate.get("confirmed_at") or candidate.get("start_time") or 0.0) - timestamp) <= 3.0
                ),
                None,
            )
            matching_snapshot = next(
                (
                    candidate for candidate in score_snapshots
                    if str(candidate.get("event_type") or "") == event_type
                    and (
                        (tracks and set(candidate.get("persons") or []) and tracks.intersection(set(candidate.get("persons") or [])))
                        or (not tracks and not set(candidate.get("persons") or []))
                    )
                    and abs(float(candidate.get("timestamp") or candidate.get("start_time") or 0.0) - timestamp) <= 3.0
                ),
                None,
            )
            is_rule = any(item.get("event_id") == event.get("event_id") for item in rule_events)
            temporal_reason = dict((matching_temporal or {}).get("reason") or {})
            event["rule_decision"] = {
                "accepted": bool(is_rule),
                "score": round(float(event.get("rule_score") or 0.0), 4) if is_rule else 0.0,
            }
            event["temporal_decision"] = {
                "accepted": bool(matching_temporal),
                "confidence": (matching_temporal or {}).get("confidence"),
                "score": (matching_snapshot or {}).get("score"),
                "threshold": (matching_snapshot or {}).get("threshold"),
                "above_threshold": (matching_snapshot or {}).get("above_threshold"),
                "would_emit": (matching_snapshot or {}).get("would_emit"),
                "available": available,
                "model_id": model_id,
                "model_sha256": model_sha256,
                "model_version": model_version,
            }
            event["final_decision"] = {
                "accepted": True,
                "source": "temporal_fusion" if matching_temporal and not is_rule else "rule",
            }
            event["model_id"] = model_id or None
            event["model_version"] = model_version or None
            event["input_quality"] = temporal_reason.get("input_quality") or {}
            event["temporal_span"] = temporal_reason.get("temporal_span") or {}
            event["fallback"] = bool(not available or (not matching_temporal and is_rule))
            existing_decision = event.get("decision") if isinstance(event.get("decision"), dict) else {}
            existing_decision["rule"] = event["rule_decision"]
            existing_decision["temporal"] = event["temporal_decision"]
            existing_decision["final"] = event["final_decision"]
            event["decision"] = existing_decision

    @staticmethod
    def _merge_temporal_events(
        rule_events: List[Dict[str, Any]],
        temporal_events: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        additions: List[Dict[str, Any]] = []
        for candidate in temporal_events:
            event_type = str(candidate.get("event_type") or "")
            tracks = set(candidate.get("persons") or [])
            confirmed_at = float(candidate.get("confirmed_at") or 0.0)
            duplicate = False
            for existing in rule_events + additions:
                if str(existing.get("event_type") or "") != event_type:
                    continue
                if tracks and set(existing.get("persons") or []) and not tracks.intersection(set(existing.get("persons") or [])):
                    continue
                if abs(float(existing.get("confirmed_at") or 0.0) - confirmed_at) <= 3.0:
                    duplicate = True
                    break
            if not duplicate:
                additions.append(candidate)
        return additions

    def _infer_persons(
        self,
        image_rgb: np.ndarray,
        primary_detections: List[Dict[str, Any]],
    ) -> Dict[int, Dict[str, Any]]:
        results = self.model_pool.predict(
            self.config["pose_model_path"],
            image_rgb,
            conf=self.config["pose_confidence"],
            iou=self.config["pose_iou"],
            model_id=self.config.get("pose_model_id"), revision=self.config.get("inference_revision"),
            enabled_events=self.config.get("enabled_events", []),
        )
        pose_people: List[Dict[str, Any]] = []
        if results:
            result = results[0]
            if isinstance(result, dict):
                for pose in result.get("poses", []):
                    bbox = [float(value) for value in pose.get("bbox", [])[:4]]
                    if len(bbox) != 4:
                        continue
                    width, height = max(1.0, bbox[2] - bbox[0]), max(1.0, bbox[3] - bbox[1])
                    pose_people.append({"bbox": bbox, "center": bbox_center(bbox),
                                        "width": width, "height": height,
                                        "keypoints": pose.get("keypoints", []),
                                        "keypoint_confidence": self.config["keypoint_confidence"]})
                # Rule matching and fallback tracking below use the same normalized pose data.
                result = None
            if result is not None:
                boxes = getattr(result, "boxes", None)
                keypoints = getattr(result, "keypoints", None)
                box_values = boxes.xyxy.cpu().numpy() if boxes is not None and boxes.xyxy is not None else []
                point_values = keypoints.xy.cpu().numpy() if keypoints is not None and keypoints.xy is not None else []
                point_confidences = keypoints.conf.cpu().numpy() if keypoints is not None and keypoints.conf is not None else []
                for index, values in enumerate(box_values):
                    bbox = [float(value) for value in values.tolist()]
                    width = max(1.0, bbox[2] - bbox[0])
                    height = max(1.0, bbox[3] - bbox[1])
                    points = point_values[index].tolist() if index < len(point_values) else []
                    confidences = point_confidences[index].tolist() if index < len(point_confidences) else []
                    if confidences:
                        points = [[float(point[0]), float(point[1]), float(confidences[point_index])]
                                  for point_index, point in enumerate(points)]
                    pose_people.append({"bbox": bbox, "center": bbox_center(bbox),
                                        "width": width, "height": height, "keypoints": points,
                                        "keypoint_confidence": self.config["keypoint_confidence"]})
        primary_people = []
        for detection in primary_detections:
            class_name = str(detection.get("class") or detection.get("class_name") or "").lower()
            track_id = detection.get("track_id")
            bbox = _bbox_from_detection(detection)
            if class_name in {"person", "people", "human"} and bbox and track_id is not None and int(track_id) >= 0:
                primary_people.append((int(track_id), bbox))
        assigned: Dict[int, Dict[str, Any]] = {}
        used = set()
        matches: List[Tuple[float, int, int]] = []
        for pose_index, person in enumerate(pose_people):
            for track_id, bbox in primary_people:
                overlap = bbox_iou(person["bbox"], bbox)
                if overlap >= 0.20:
                    matches.append((overlap, track_id, pose_index))
        for _, track_id, pose_index in sorted(matches, reverse=True):
            if track_id in assigned or pose_index in used:
                continue
            item = dict(pose_people[pose_index])
            item["track_id"] = track_id
            assigned[track_id] = item
            used.add(pose_index)
        unmatched = [item for index, item in enumerate(pose_people) if index not in used]
        fallback = self.person_tracker.update(unmatched)
        for fallback_id, item in fallback.items():
            synthetic_id = 1_000_000 + fallback_id
            while synthetic_id in assigned:
                synthetic_id += 1_000_000
            item["track_id"] = synthetic_id
            assigned[synthetic_id] = item
        return assigned

    def _annotate_badge_presence(
        self,
        persons: Dict[int, Dict[str, Any]],
        detections: Optional[List[Dict[str, Any]]],
        model_classes: Optional[set],
    ) -> None:
        if "missing_badge" not in self.config["enabled_events"] or not persons:
            return
        if detections is None or model_classes is None:
            return
        badge_classes = normalized_class_names(self.config["badge_classes"])
        matched_classes = matching_class_names(model_classes, badge_classes)
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
        badge_detections = [
            item for item in detections
            if str(item.get("class_name") or "").strip().casefold() in badge_classes
        ]

        padding = float(self.config["rules"]["missing_badge"]["chest_padding_ratio"])
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

    def _infer_objects(
        self,
        image_rgb: np.ndarray,
        primary_detections: List[Dict[str, Any]],
        timestamp: float,
        persons: Dict[int, Dict[str, Any]],
        *,
        scene_detections: Optional[List[Dict[str, Any]]] = None,
    ) -> Dict[int, Dict[str, Any]]:
        if "phone_use" not in self.config["enabled_events"]:
            return {}
        phone_classes = {str(item).casefold() for item in self.config["phone_classes"]}
        detections = self._objects_from_primary(primary_detections, phone_classes)
        if not detections:
            if scene_detections is None:
                scene_detections, _ = self._detect_scene_objects(image_rgb)
            detections = [
                item for item in scene_detections
                if str(item.get("class_name") or "").casefold() in phone_classes
            ]
        objects = self.object_tracker.update(detections, self.frame_index, timestamp)
        for object_id, item in objects.items():
            owner_id, distance_ratio = self._find_owner(item, persons)
            item["owner_id"] = owner_id
            item["owner_distance_ratio"] = distance_ratio
            self.object_tracker.add_owner_observation(
                object_id,
                owner_id,
                self.frame_index,
                timestamp,
                distance_ratio,
            )
            track = self.object_tracker.tracks.get(object_id)
            item["owner_history"] = list(track.owner_history) if track else []
        return objects

    @staticmethod
    def _objects_from_primary(
        detections: Iterable[Dict[str, Any]],
        phone_classes: set,
    ) -> List[Dict[str, Any]]:
        objects = []
        for detection in detections:
            class_name = str(detection.get("class") or detection.get("class_name") or "")
            bbox = _bbox_from_detection(detection)
            if class_name.casefold() not in phone_classes or not bbox:
                continue
            objects.append({
                "class_id": int(detection.get("class_id", -1)),
                "class_name": class_name,
                "bbox": bbox,
                "confidence": float(detection.get("confidence", 0.0)),
            })
        return objects

    def _detect_scene_objects(self, image_rgb: np.ndarray) -> Tuple[List[Dict[str, Any]], set]:
        results = self.model_pool.predict(
            self.config["object_model_path"],
            image_rgb,
            conf=self.config["object_confidence"],
            iou=self.config["pose_iou"],
            model_id=self.config.get("object_model_id"),
            revision=self.config.get("inference_revision"),
            enabled_events=self.config.get("enabled_events", []),
            model_ids=self.config.get("scene_model_ids", []),
        )
        objects: List[Dict[str, Any]] = []
        if not results:
            return objects, set()
        result = results[0]
        if isinstance(result, dict):
            objects = [{"class_id": int(item.get("class_id", -1)),
                        "class_name": str(item.get("class") or item.get("class_name") or ""),
                        "bbox": _bbox_from_detection(item),
                        "confidence": float(item.get("confidence", 0.0))}
                       for item in result.get("detections", []) if _bbox_from_detection(item)]
            return objects, {str(name).strip().casefold() for name in result.get("class_names", []) if str(name).strip()}
        names = getattr(result, "names", {}) or {}
        if isinstance(names, dict):
            model_classes = {str(value).strip().casefold() for value in names.values()}
        else:
            model_classes = {str(value).strip().casefold() for value in names}
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return objects, model_classes
        for box in boxes:
            class_id = int(box.cls[0].cpu().numpy())
            class_name = str(names.get(class_id, f"class_{class_id}"))
            objects.append({
                "class_id": class_id,
                "class_name": class_name,
                "bbox": [float(value) for value in box.xyxy[0].cpu().numpy().tolist()],
                "confidence": float(box.conf[0].cpu().numpy()),
            })
        return objects, model_classes

    def _find_owner(
        self,
        obj: Dict[str, Any],
        persons: Dict[int, Dict[str, Any]],
    ) -> Tuple[Optional[int], Optional[float]]:
        object_center = obj.get("center") or bbox_center(obj["bbox"])
        limit = float(self.config["rules"]["phone_use"]["owner_distance_ratio"])
        candidates = []
        for track_id, person in persons.items():
            height = max(float(person["height"]), 1.0)
            wrists = [
                point
                for point in (get_kpt(person, LEFT_WRIST), get_kpt(person, RIGHT_WRIST))
                if point is not None
            ]
            if not wrists:
                continue
            ratio = min(dist(object_center, wrist) for wrist in wrists) / height
            if ratio <= limit:
                candidates.append((ratio, track_id))
        candidates.sort()
        if not candidates:
            return None, None
        if len(candidates) > 1 and candidates[0][0] > candidates[1][0] * 0.85:
            return None, candidates[0][0]
        return candidates[0][1], candidates[0][0]


class BehaviorRuntimeManager:
    def __init__(self):
        self.model_pool = SharedBehaviorModelPool()
        self.sessions: Dict[str, BehaviorSessionRuntime] = {}
        self.lock = threading.RLock()

    def process(
        self,
        session_key: str,
        image_rgb: np.ndarray,
        primary_detections: Optional[List[Dict[str, Any]]],
        config: Optional[Dict[str, Any]],
        *,
        timestamp: Optional[float] = None,
        output_root: Optional[str] = None,
    ) -> Dict[str, Any]:
        normalized = normalize_behavior_config(config)
        if not normalized["enabled"]:
            return {
                "action_events": [],
                "active_events": [],
                "behavior_stats": {"enabled": False},
                "evidence_updates": [],
            }
        closed_result = None
        with self.lock:
            runtime = self.sessions.get(session_key)
            if runtime is None or runtime.config != normalized:
                if runtime is not None:
                    closed_result = runtime.close(timestamp)
                runtime = BehaviorSessionRuntime(session_key, normalized, self.model_pool, output_root)
                self.sessions[session_key] = runtime
        result = runtime.process(image_rgb, primary_detections, timestamp=timestamp)
        if closed_result is not None:
            result["action_events"] = (
                (closed_result.get("action_events") or [])
                + (result.get("action_events") or [])
            )
            result["evidence_updates"] = (
                (closed_result.get("evidence_updates") or [])
                + (result.get("evidence_updates") or [])
            )
        return result

    def close(self, session_key: str, timestamp: Optional[float] = None) -> Dict[str, Any]:
        with self.lock:
            runtime = self.sessions.pop(session_key, None)
        if runtime is None:
            return {"action_events": [], "active_events": [], "behavior_stats": {}, "evidence_updates": []}
        return runtime.close(timestamp)


behavior_runtime_manager = BehaviorRuntimeManager()
