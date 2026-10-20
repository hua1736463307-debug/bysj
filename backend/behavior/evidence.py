"""Bounded evidence capture for confirmed behavior events."""

from __future__ import annotations

import os
import threading
from collections import deque
from concurrent.futures import Future, ThreadPoolExecutor, TimeoutError
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np

from .video_output import BrowserCompatibleVideoWriter


@dataclass
class PendingEvidence:
    event_id: str
    confirmed_at: float
    deadline: float
    output_dir: Path
    frames: List[Tuple[float, bytes]] = field(default_factory=list)
    snapshot_path: Optional[str] = None


class BoundedEvidenceEncoder:
    """One shared encoder worker with bounded accepted work."""

    def __init__(self, capacity: int = 8):
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="behavior-evidence")
        self.slots = threading.BoundedSemaphore(max(1, int(capacity)))

    def submit(self, pending: PendingEvidence, behavior_fps: float) -> Optional[Future]:
        if not self.slots.acquire(blocking=False):
            return None
        future = self.executor.submit(_encode_pending_evidence, pending, behavior_fps)
        future.add_done_callback(lambda _: self.slots.release())
        return future


def _encode_pending_evidence(pending: PendingEvidence, behavior_fps: float) -> Dict[str, Any]:
    update = {
        "event_id": pending.event_id,
        "snapshot_path": pending.snapshot_path,
        "clip_path": None,
        "evidence_status": "ready",
        "evidence_error": None,
    }
    unique_frames: List[Tuple[float, bytes]] = []
    seen = set()
    for timestamp, payload in sorted(pending.frames, key=lambda item: item[0]):
        key = (round(timestamp, 6), len(payload))
        if key in seen:
            continue
        seen.add(key)
        unique_frames.append((timestamp, payload))
    if not unique_frames:
        update["evidence_status"] = "failed"
        update["evidence_error"] = "no evidence frames available"
        return update
    first = cv2.imdecode(np.frombuffer(unique_frames[0][1], dtype=np.uint8), cv2.IMREAD_COLOR)
    if first is None:
        update["evidence_status"] = "failed"
        update["evidence_error"] = "failed to decode evidence frame"
        return update
    height, width = first.shape[:2]
    clip_path = pending.output_dir / "clip.mp4"
    try:
        writer = BrowserCompatibleVideoWriter(clip_path, behavior_fps, width, height)
        for _, payload in unique_frames:
            frame = cv2.imdecode(np.frombuffer(payload, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                writer.write(frame)
        ok, format_name = writer.release()
        if not ok:
            raise RuntimeError(f"video writer finalization failed: {format_name}")
        update["clip_path"] = str(clip_path)
    except Exception as exc:
        update["evidence_status"] = "failed"
        update["evidence_error"] = str(exc)
    return update


evidence_encoder = BoundedEvidenceEncoder()


class BehaviorEvidenceRecorder:
    """Store compressed pre-event frames and finalize short clips after confirmation."""

    def __init__(self, source_id: str, config: Dict[str, Any], output_root: Optional[str] = None):
        evidence = config.get("evidence") or {}
        self.source_id = str(source_id)
        # Recorder monitoring passes a grid session key as source_id. Its
        # independent samples still need snapshot-only evidence.
        self.recorder_playback = self.source_id.startswith('recorder:') or bool(config.get('independent_snapshot_alerts'))
        self.behavior_fps = max(1.0, float(config.get("behavior_fps") or 6.0))
        self.pre_seconds = max(0.0, float(evidence.get("clip_pre_seconds") or 0.0))
        self.post_seconds = max(0.0, float(evidence.get("clip_post_seconds") or 0.0))
        self.snapshot_enabled = bool(evidence.get("snapshot", True))
        self.max_width = max(320, int(evidence.get("max_width") or 960))
        self.jpeg_quality = max(40, min(95, int(evidence.get("jpeg_quality") or 70)))
        root = output_root or os.getenv("DEEPVISION_BEHAVIOR_EVENT_DIR", "/app/static/behavior_events")
        self.output_root = Path(root)
        buffer_size = max(1, int(round(self.pre_seconds * self.behavior_fps)) + 2)
        self.buffer: Deque[Tuple[float, bytes]] = deque(maxlen=buffer_size)
        self.pending: Dict[str, PendingEvidence] = {}
        self.encoding: Dict[str, Future] = {}

    def add_frame(self, image_rgb: np.ndarray, timestamp: float) -> List[Dict[str, Any]]:
        if self.recorder_playback:
            return []
        encoded = self._encode_frame(image_rgb)
        if encoded is None:
            return []
        item = (float(timestamp), encoded)
        self.buffer.append(item)
        for pending in self.pending.values():
            if timestamp <= pending.deadline:
                pending.frames.append(item)
        return self._drain_completed() + self._finalize_due(timestamp)

    def confirm(self, event: Dict[str, Any], image_rgb: np.ndarray) -> Dict[str, Any]:
        event_id = str(event["event_id"])
        confirmed_at = float(event.get("confirmed_at") or event.get("started_at") or 0.0)
        day_dir = self.output_root / datetime.fromtimestamp(max(confirmed_at, 0.0)).strftime("%Y%m%d")
        output_dir = day_dir / event_id
        output_dir.mkdir(parents=True, exist_ok=True)
        snapshot_path = None
        if self.snapshot_enabled and image_rgb is not None:
            snapshot_file = output_dir / "snapshot.png"
            bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
            if cv2.imwrite(str(snapshot_file), bgr, [cv2.IMWRITE_PNG_COMPRESSION, 3]):
                snapshot_path = str(snapshot_file)
        if self.recorder_playback:
            return {'event_id':event_id, 'snapshot_path':snapshot_path, 'clip_path':None,
                    'evidence_status':'ready', 'evidence_error':None}
        frames = [
            item
            for item in self.buffer
            if confirmed_at - self.pre_seconds <= item[0] <= confirmed_at
        ]
        self.pending[event_id] = PendingEvidence(
            event_id=event_id,
            confirmed_at=confirmed_at,
            deadline=confirmed_at + self.post_seconds,
            output_dir=output_dir,
            frames=list(frames),
            snapshot_path=snapshot_path,
        )
        return {
            "event_id": event_id,
            "snapshot_path": snapshot_path,
            "clip_path": None,
            "evidence_status": "collecting",
            "evidence_error": None,
        }

    def close(self) -> List[Dict[str, Any]]:
        updates = []
        for event_id in list(self.pending):
            queued = self._queue_finalize(event_id)
            if queued is not None:
                updates.append(queued)
        for event_id, future in list(self.encoding.items()):
            try:
                updates.append(future.result(timeout=15.0))
            except TimeoutError:
                updates.append({
                    "event_id": event_id,
                    "snapshot_path": None,
                    "clip_path": None,
                    "evidence_status": "failed",
                    "evidence_error": "evidence encoding timed out during session cleanup",
                })
            except Exception as exc:
                updates.append({
                    "event_id": event_id,
                    "snapshot_path": None,
                    "clip_path": None,
                    "evidence_status": "failed",
                    "evidence_error": str(exc),
                })
            self.encoding.pop(event_id, None)
        self.buffer.clear()
        return updates

    def _finalize_due(self, timestamp: float) -> List[Dict[str, Any]]:
        updates = []
        for event_id, item in list(self.pending.items()):
            if timestamp >= item.deadline:
                queued = self._queue_finalize(event_id)
                if queued is not None:
                    updates.append(queued)
        return updates

    def _queue_finalize(self, event_id: str) -> Optional[Dict[str, Any]]:
        pending = self.pending.pop(event_id)
        future = evidence_encoder.submit(pending, self.behavior_fps)
        if future is None:
            return {
                "event_id": event_id,
                "snapshot_path": pending.snapshot_path,
                "clip_path": None,
                "evidence_status": "failed",
                "evidence_error": "evidence encoding queue is full",
            }
        self.encoding[event_id] = future
        return None

    def _drain_completed(self) -> List[Dict[str, Any]]:
        updates = []
        for event_id, future in list(self.encoding.items()):
            if not future.done():
                continue
            try:
                updates.append(future.result())
            except Exception as exc:
                updates.append({
                    "event_id": event_id,
                    "snapshot_path": None,
                    "clip_path": None,
                    "evidence_status": "failed",
                    "evidence_error": str(exc),
                })
            self.encoding.pop(event_id, None)
        return updates

    def _encode_frame(self, image_rgb: np.ndarray) -> Optional[bytes]:
        if image_rgb is None:
            return None
        image = image_rgb
        height, width = image.shape[:2]
        if width > self.max_width:
            scale = self.max_width / float(width)
            image = cv2.resize(
                image,
                (self.max_width, max(1, int(round(height * scale)))),
                interpolation=cv2.INTER_AREA,
            )
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        ok, encoded = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, self.jpeg_quality])
        return encoded.tobytes() if ok else None
