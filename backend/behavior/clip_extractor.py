"""Event clip extraction utilities for behavior analysis.

Stage-3 step 1 goal:
- After rule-based event detection, cut short video clips around each event.
- The clips become reviewable samples for later action-classifier training.

This module intentionally uses OpenCV only, because the existing project already
uses cv2 and not every deployment has ffmpeg available in the container.
"""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2

from .video_output import BrowserCompatibleVideoWriter

logger = logging.getLogger(__name__)


_REMOTE_PREFIXES = ("rtsp://", "rtmp://", "http://", "https://")


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _safe_filename_part(value: Any, fallback: str = "event") -> str:
    text = str(value or fallback).strip()
    text = re.sub(r"[^0-9A-Za-z_\-\.\u4e00-\u9fff]+", "_", text)
    text = text.strip("._-")
    return text or fallback


def _to_static_url(path: Any) -> str:
    text = str(path).replace("\\", "/")
    if text.startswith("/app/static/"):
        return text.replace("/app", "", 1)
    marker = "/static/"
    if marker in text:
        return text[text.index(marker):]
    return text


def _normalize_video_path(video_path: str) -> str:
    text = str(video_path or "").strip()
    if text.startswith("/static/"):
        return "/app" + text
    if text.startswith("static/"):
        return "/app/" + text
    return text


def _open_video(video_path: str) -> Tuple[cv2.VideoCapture, float, int, int, int]:
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"无法打开视频用于截取片段: {video_path}")

    fps = _safe_float(cap.get(cv2.CAP_PROP_FPS), 25.0)
    if fps <= 1:
        fps = 25.0

    width = _safe_int(cap.get(cv2.CAP_PROP_FRAME_WIDTH), 0)
    height = _safe_int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT), 0)
    total_frames = _safe_int(cap.get(cv2.CAP_PROP_FRAME_COUNT), 0)

    if width <= 0 or height <= 0:
        ret, frame = cap.read()
        if not ret or frame is None:
            cap.release()
            raise RuntimeError("无法读取视频首帧，不能截取事件片段")
        height, width = frame.shape[:2]
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    return cap, fps, width, height, total_frames


def _compute_clip_range(
    event: Dict[str, Any],
    fps: float,
    total_frames: int,
    pre_seconds: float,
    post_seconds: float,
    min_seconds: float,
    max_seconds: float,
) -> Tuple[int, int, float, float]:
    start_time = max(0.0, _safe_float(event.get("start_time", event.get("timestamp", 0.0)), 0.0) - pre_seconds)
    end_time = max(start_time, _safe_float(event.get("end_time", event.get("timestamp", start_time)), start_time) + post_seconds)

    # Ensure minimum clip length by expanding around the event midpoint.
    current_len = max(0.0, end_time - start_time)
    if current_len < min_seconds:
        mid = (start_time + end_time) / 2.0
        start_time = max(0.0, mid - min_seconds / 2.0)
        end_time = mid + min_seconds / 2.0

    # Limit very long events to keep sample clips compact for review/training.
    current_len = max(0.0, end_time - start_time)
    if max_seconds > 0 and current_len > max_seconds:
        event_start = _safe_float(event.get("start_time", start_time), start_time)
        event_end = _safe_float(event.get("end_time", end_time), end_time)
        mid = (event_start + event_end) / 2.0
        start_time = max(0.0, mid - max_seconds / 2.0)
        end_time = start_time + max_seconds

    start_frame = max(0, int(math.floor(start_time * fps)))
    end_frame = max(start_frame + 1, int(math.ceil(end_time * fps)))

    if total_frames > 0:
        start_frame = min(start_frame, max(0, total_frames - 1))
        end_frame = min(end_frame, total_frames)
        if end_frame <= start_frame:
            end_frame = min(total_frames, start_frame + max(1, int(round(min_seconds * fps))))

    clip_start_time = start_frame / fps
    clip_end_time = end_frame / fps
    return start_frame, end_frame, clip_start_time, clip_end_time


def extract_event_clips(
    video_path: str,
    events: List[Dict[str, Any]],
    output_dir: Any,
    pre_seconds: float = 1.0,
    post_seconds: float = 1.0,
    min_seconds: float = 2.0,
    max_seconds: float = 8.0,
    filename_prefix: str = "event",
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Cut one short MP4 clip for every event and attach `clip` fields.

    Args:
        video_path: Original local video path. Remote streams are skipped.
        events: Formatted event dictionaries returned by BehaviorAnalyzer._format_events.
        output_dir: Directory where MP4 clips will be written.
        pre_seconds/post_seconds: Context before/after the detected event.
        min_seconds/max_seconds: Review-friendly clip length guards.

    Returns:
        (events_with_clip_fields, summary)
    """
    summary = {
        "enabled": True,
        "clips_generated": 0,
        "clips_failed": 0,
        "output_dir": _to_static_url(output_dir),
        "pre_seconds": float(pre_seconds),
        "post_seconds": float(post_seconds),
        "min_seconds": float(min_seconds),
        "max_seconds": float(max_seconds),
        "error": None,
    }

    if not events:
        return events, summary

    normalized_video_path = _normalize_video_path(video_path)
    if normalized_video_path.lower().startswith(_REMOTE_PREFIXES):
        summary["enabled"] = False
        summary["error"] = "远程流/网络地址暂不支持离线片段截取，请先保存为本地视频文件"
        return events, summary

    if not Path(normalized_video_path).exists():
        summary["enabled"] = False
        summary["error"] = f"视频文件不存在，无法截取片段: {normalized_video_path}"
        return events, summary

    clip_dir = Path(output_dir)
    clip_dir.mkdir(parents=True, exist_ok=True)

    try:
        cap, fps, width, height, total_frames = _open_video(normalized_video_path)
    except Exception as exc:
        summary["enabled"] = False
        summary["error"] = str(exc)
        return events, summary

    try:
        for index, event in enumerate(events, start=1):
            event_type = _safe_filename_part(event.get("event_type"), "event")
            event_id = _safe_int(event.get("id"), index)
            persons = event.get("persons") or []
            person_part = "_".join(_safe_filename_part(p, "p") for p in persons) or "none"

            object_part = ""
            if event.get("object_id") is not None:
                object_part = f"_obj{_safe_filename_part(event.get('object_id'), 'x')}"

            filename = f"{filename_prefix}_{event_id:04d}_{event_type}_p{person_part}{object_part}.mp4"
            clip_path = clip_dir / filename

            try:
                start_frame, end_frame, clip_start_time, clip_end_time = _compute_clip_range(
                    event=event,
                    fps=fps,
                    total_frames=total_frames,
                    pre_seconds=max(0.0, float(pre_seconds)),
                    post_seconds=max(0.0, float(post_seconds)),
                    min_seconds=max(0.1, float(min_seconds)),
                    max_seconds=max(0.0, float(max_seconds)),
                )

                writer = BrowserCompatibleVideoWriter(clip_path, fps, width, height)

                cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
                current_frame = start_frame
                written = 0

                while current_frame < end_frame:
                    ret, frame = cap.read()
                    if not ret or frame is None:
                        break
                    writer.write(frame)
                    written += 1
                    current_frame += 1

                video_ok, video_format = writer.release()

                if written <= 0 or not video_ok:
                    try:
                        clip_path.unlink(missing_ok=True)
                    except Exception:
                        pass
                    raise RuntimeError(f"片段没有写入任何帧或视频不可用: {video_format}")

                event["clip"] = _to_static_url(clip_path)
                event["clip_video_format"] = video_format
                event["clip_start_time"] = round(float(clip_start_time), 2)
                event["clip_end_time"] = round(float(clip_end_time), 2)
                event["clip_duration"] = round(float(clip_end_time - clip_start_time), 2)
                event["clip_frame_range"] = [int(start_frame), int(end_frame)]
                summary["clips_generated"] += 1

            except Exception as exc:
                logger.warning("截取事件片段失败: event=%s, error=%s", event.get("id", index), exc)
                event["clip_error"] = str(exc)
                summary["clips_failed"] += 1

    finally:
        cap.release()

    return events, summary
