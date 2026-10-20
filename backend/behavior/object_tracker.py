"""Simple object tracker for behavior analysis stage 2.

This tracker is intentionally lightweight. It does not depend on external
tracking libraries, so it can run inside the current backend container using
only OpenCV/NumPy and YOLO detections.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


def bbox_center(box: List[float]) -> Tuple[float, float]:
    x1, y1, x2, y2 = box
    return (float(x1 + x2) / 2.0, float(y1 + y2) / 2.0)


def bbox_area(box: List[float]) -> float:
    x1, y1, x2, y2 = box
    return max(0.0, float(x2) - float(x1)) * max(0.0, float(y2) - float(y1))


def bbox_iou(box_a: List[float], box_b: List[float]) -> float:
    ax1, ay1, ax2, ay2 = box_a
    bx1, by1, bx2, by2 = box_b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    union = bbox_area(box_a) + bbox_area(box_b) - inter
    return float(inter / union) if union > 0 else 0.0


def point_distance(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.sqrt((float(p1[0]) - float(p2[0])) ** 2 + (float(p1[1]) - float(p2[1])) ** 2)


@dataclass
class TrackedObject:
    object_id: int
    class_id: int
    class_name: str
    bbox: List[float]
    confidence: float
    first_frame: int
    last_frame: int
    first_time: float
    last_time: float
    missed_frames: int = 0
    owner_history: List[Dict[str, Any]] = field(default_factory=list)
    transfer_pairs_emitted: set = field(default_factory=set)

    @property
    def center(self) -> Tuple[float, float]:
        return bbox_center(self.bbox)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "object_id": self.object_id,
            "class_id": self.class_id,
            "class_name": self.class_name,
            "bbox": [float(v) for v in self.bbox],
            "center": self.center,
            "confidence": float(self.confidence),
            "first_frame": int(self.first_frame),
            "last_frame": int(self.last_frame),
            "first_time": float(self.first_time),
            "last_time": float(self.last_time),
            "missed_frames": int(self.missed_frames),
            "owner_history": list(self.owner_history),
        }


class SimpleObjectTracker:
    """Class-aware IOU/center-distance tracker for small object interactions."""

    def __init__(
        self,
        iou_threshold: float = 0.20,
        max_center_distance_ratio: float = 1.20,
        max_missed_frames: int = 15,
    ):
        self.iou_threshold = float(iou_threshold)
        self.max_center_distance_ratio = float(max_center_distance_ratio)
        self.max_missed_frames = int(max_missed_frames)
        self.next_id = 1
        self.tracks: Dict[int, TrackedObject] = {}

    def reset(self) -> None:
        self.next_id = 1
        self.tracks.clear()

    def update(
        self,
        detections: List[Dict[str, Any]],
        frame_idx: int,
        timestamp: float,
    ) -> Dict[int, Dict[str, Any]]:
        """Update tracks and return active tracked object dictionaries."""
        unmatched_track_ids = set(self.tracks.keys())
        unmatched_det_indices = set(range(len(detections)))
        matches: List[Tuple[int, int, float]] = []

        # Greedy matching. First prefer IOU; center distance is a fallback for tiny boxes.
        candidates: List[Tuple[float, int, int]] = []
        for tid, track in self.tracks.items():
            for det_idx, det in enumerate(detections):
                if int(det.get("class_id", -1)) != int(track.class_id):
                    continue

                det_box = det.get("bbox") or []
                if len(det_box) < 4:
                    continue

                iou = bbox_iou(track.bbox, det_box)
                track_w = max(1.0, float(track.bbox[2]) - float(track.bbox[0]))
                track_h = max(1.0, float(track.bbox[3]) - float(track.bbox[1]))
                det_w = max(1.0, float(det_box[2]) - float(det_box[0]))
                det_h = max(1.0, float(det_box[3]) - float(det_box[1]))
                scale = max(1.0, (track_w + track_h + det_w + det_h) / 4.0)
                center_dist = point_distance(track.center, bbox_center(det_box))
                center_score = max(0.0, 1.0 - center_dist / (scale * self.max_center_distance_ratio))

                score = max(iou, center_score * 0.75)
                if iou >= self.iou_threshold or center_score > 0.35:
                    candidates.append((score, tid, det_idx))

        candidates.sort(reverse=True, key=lambda item: item[0])
        for score, tid, det_idx in candidates:
            if tid not in unmatched_track_ids or det_idx not in unmatched_det_indices:
                continue
            matches.append((tid, det_idx, score))
            unmatched_track_ids.remove(tid)
            unmatched_det_indices.remove(det_idx)

        for tid, det_idx, _ in matches:
            det = detections[det_idx]
            track = self.tracks[tid]
            track.bbox = [float(v) for v in det["bbox"]]
            track.confidence = float(det.get("confidence", track.confidence))
            track.last_frame = int(frame_idx)
            track.last_time = float(timestamp)
            track.missed_frames = 0

        for det_idx in unmatched_det_indices:
            det = detections[det_idx]
            box = det.get("bbox") or []
            if len(box) < 4:
                continue
            object_id = self.next_id
            self.next_id += 1
            self.tracks[object_id] = TrackedObject(
                object_id=object_id,
                class_id=int(det.get("class_id", -1)),
                class_name=str(det.get("class_name", f"class_{det.get('class_id', -1)}")),
                bbox=[float(v) for v in box],
                confidence=float(det.get("confidence", 0.0)),
                first_frame=int(frame_idx),
                last_frame=int(frame_idx),
                first_time=float(timestamp),
                last_time=float(timestamp),
            )

        expired_ids: List[int] = []
        for tid in unmatched_track_ids:
            track = self.tracks.get(tid)
            if not track:
                continue
            track.missed_frames += 1
            if track.missed_frames > self.max_missed_frames:
                expired_ids.append(tid)

        for tid in expired_ids:
            self.tracks.pop(tid, None)

        return {
            tid: track.to_dict()
            for tid, track in self.tracks.items()
            if track.missed_frames <= self.max_missed_frames
        }

    def add_owner_observation(
        self,
        object_id: int,
        owner_id: Optional[int],
        frame_idx: int,
        timestamp: float,
        distance_ratio: Optional[float] = None,
    ) -> None:
        track = self.tracks.get(object_id)
        if not track:
            return

        observation = {
            "owner_id": owner_id,
            "frame_idx": int(frame_idx),
            "timestamp": float(timestamp),
            "distance_ratio": None if distance_ratio is None else float(distance_ratio),
        }
        track.owner_history.append(observation)
        if len(track.owner_history) > 120:
            track.owner_history = track.owner_history[-120:]

    def mark_transfer_emitted(self, object_id: int, from_person: int, to_person: int) -> None:
        track = self.tracks.get(object_id)
        if track:
            track.transfer_pairs_emitted.add((int(from_person), int(to_person)))

    def transfer_already_emitted(self, object_id: int, from_person: int, to_person: int) -> bool:
        track = self.tracks.get(object_id)
        return bool(track and (int(from_person), int(to_person)) in track.transfer_pairs_emitted)
