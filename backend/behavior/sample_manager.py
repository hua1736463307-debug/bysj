"""Behavior review sample manager.

This module only manages review metadata and copies already-generated clips into
a reviewed dataset directory. It does not open, rewrite, transcode, or re-encode
video files, so it will not affect browser playback compatibility.
"""

from __future__ import annotations

import json
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


ALLOWED_REVIEW_LABELS = {
    "phone_use",
    "missing_badge",
    "fall_down",
    "normal",
    "uncertain",
    "delete",
}

IMAGE_FILE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

LABEL_NAME_MAP = {
    "phone_use": "\u4f7f\u7528\u624b\u673a",
    "missing_badge": "\u672a\u4f69\u6234\u5de5\u724c",
    "fall_down": "\u4eba\u5458\u6454\u5012",
    "normal": "普通行为",
    "uncertain": "不确定",
    "delete": "删除样本",
}


def _safe_filename_part(value: Any, fallback: str = "sample") -> str:
    text = str(value or fallback).strip()
    text = re.sub(r"[^0-9A-Za-z_\-\.\u4e00-\u9fff]+", "_", text)
    text = text.strip("._-")
    return text or fallback


def _now() -> str:
    return datetime.now().isoformat()


def _to_static_url(path: Any) -> str:
    text = str(path).replace("\\", "/")
    if text.startswith("/app/static/"):
        return text.replace("/app", "", 1)
    marker = "/static/"
    if marker in text:
        return text[text.index(marker):]
    return text


class BehaviorSampleManager:
    """Manage review manifests and reviewed action clip datasets."""

    def __init__(
        self,
        behavior_results_dir: str | Path = "static/behavior_results",
        samples_root: str | Path = "data/action_samples",
    ):
        self.behavior_results_dir = Path(behavior_results_dir)
        self.samples_root = Path(samples_root)
        self.reviewed_root = self.samples_root / "reviewed"
        self.global_manifest_path = self.samples_root / "review_manifest.json"

        self.samples_root.mkdir(parents=True, exist_ok=True)
        self.reviewed_root.mkdir(parents=True, exist_ok=True)
        for label in ALLOWED_REVIEW_LABELS - {"delete"}:
            (self.reviewed_root / label).mkdir(parents=True, exist_ok=True)

    def _task_dir_candidates(self, task_id: str) -> List[Path]:
        safe_task_id = _safe_filename_part(task_id, "task")
        candidates = [
            self.behavior_results_dir / safe_task_id,
            Path("/app/static/behavior_results") / safe_task_id,
            Path("static/behavior_results") / safe_task_id,
        ]

        # Remove duplicates while preserving order.
        result = []
        seen = set()
        for item in candidates:
            key = str(item)
            if key not in seen:
                result.append(item)
                seen.add(key)
        return result

    def get_task_dir(self, task_id: str) -> Path:
        for candidate in self._task_dir_candidates(task_id):
            if candidate.exists() and candidate.is_dir():
                return candidate
        # Default path used in error messages and for future manifests.
        return self._task_dir_candidates(task_id)[0]

    def get_events_json_path(self, task_id: str) -> Path:
        return self.get_task_dir(task_id) / "events.json"

    def get_task_manifest_path(self, task_id: str) -> Path:
        return self.get_task_dir(task_id) / "review_manifest.json"

    def _read_json(self, path: Path, default: Any):
        if not path.exists():
            return default
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return default

    def _write_json(self, path: Path, data: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp_path = path.with_suffix(path.suffix + ".tmp")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
        tmp_path.replace(path)

    def read_events_payload(self, task_id: str) -> Dict[str, Any]:
        events_path = self.get_events_json_path(task_id)
        if not events_path.exists():
            raise FileNotFoundError(f"events.json 不存在: {events_path}")

        payload = self._read_json(events_path, None)
        if not isinstance(payload, dict):
            raise ValueError(f"events.json 格式异常: {events_path}")

        return payload

    def read_task_manifest(self, task_id: str) -> Dict[str, Any]:
        manifest_path = self.get_task_manifest_path(task_id)
        data = self._read_json(manifest_path, {})
        if not isinstance(data, dict):
            data = {}

        data.setdefault("task_id", task_id)
        data.setdefault("updated_at", None)
        data.setdefault("reviews", {})
        return data

    def write_task_manifest(self, task_id: str, manifest: Dict[str, Any]) -> Dict[str, Any]:
        manifest["task_id"] = task_id
        manifest["updated_at"] = _now()
        manifest.setdefault("reviews", {})
        self._write_json(self.get_task_manifest_path(task_id), manifest)
        return manifest

    def _load_global_manifest(self) -> Dict[str, Any]:
        data = self._read_json(self.global_manifest_path, {})
        if not isinstance(data, dict):
            data = {}
        data.setdefault("updated_at", None)
        data.setdefault("samples", [])
        return data

    def _write_global_manifest(self, data: Dict[str, Any]) -> None:
        data["updated_at"] = _now()
        data.setdefault("samples", [])
        self._write_json(self.global_manifest_path, data)

    def _clip_url_to_path(self, clip_value: Any, task_id: str) -> Optional[Path]:
        if not clip_value:
            return None

        text = str(clip_value).strip().replace("\\", "/")
        if not text:
            return None

        candidates: List[Path] = []
        if text.startswith("/app/"):
            candidates.append(Path(text))
        elif text.startswith("/static/"):
            candidates.append(Path("/app") / text.lstrip("/"))
            candidates.append(Path(text.lstrip("/")))
        elif text.startswith("static/"):
            candidates.append(Path("/app") / text)
            candidates.append(Path(text))
        else:
            candidates.append(self.get_task_dir(task_id) / "clips" / Path(text).name)
            candidates.append(Path(text))

        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate

        return None

    def list_task_clips(self, task_id: str) -> Dict[str, Any]:
        payload = self.read_events_payload(task_id)
        events = payload.get("events") or []
        manifest = self.read_task_manifest(task_id)
        reviews = manifest.get("reviews", {}) or {}

        clips: List[Dict[str, Any]] = []
        for index, event in enumerate(events, start=1):
            if not isinstance(event, dict):
                continue

            event_id = event.get("id", index)
            event_id_str = str(event_id)
            review = reviews.get(event_id_str) or {}

            item = {
                "task_id": task_id,
                "event_id": event_id,
                "event_type": event.get("event_type"),
                "event_name": event.get("event_name"),
                "persons": event.get("persons") or [],
                "object_id": event.get("object_id"),
                "object_class": event.get("object_class"),
                "start_time": event.get("start_time"),
                "end_time": event.get("end_time"),
                "duration": event.get("duration"),
                "confidence": event.get("confidence"),
                "sample_type": event.get("sample_type", "clip" if event.get("clip") else "image"),
                "clip": event.get("clip"),
                "source_image": event.get("source_image"),
                "clip_start_time": event.get("clip_start_time"),
                "clip_end_time": event.get("clip_end_time"),
                "clip_duration": event.get("clip_duration"),
                "snapshot": event.get("snapshot"),
                "reason": event.get("reason", {}),
                "review_status": review.get("review_status", "pending"),
                "review_label": review.get("review_label"),
                "review_label_name": LABEL_NAME_MAP.get(review.get("review_label"), review.get("review_label")),
                "reviewed_at": review.get("reviewed_at"),
                "note": review.get("note", ""),
                "reviewed_clip_path": review.get("reviewed_clip_path"),
            }
            clips.append(item)

        return {
            "task_id": task_id,
            "total": len(clips),
            "reviewed": sum(1 for item in clips if item["review_status"] == "reviewed"),
            "pending": sum(1 for item in clips if item["review_status"] != "reviewed"),
            "clips": clips,
            "events_json": _to_static_url(self.get_events_json_path(task_id)),
            "review_manifest": _to_static_url(self.get_task_manifest_path(task_id)),
        }

    def _find_event(self, task_id: str, event_id: int) -> Dict[str, Any]:
        payload = self.read_events_payload(task_id)
        events = payload.get("events") or []

        for index, event in enumerate(events, start=1):
            if not isinstance(event, dict):
                continue
            current_id = int(event.get("id", index))
            if current_id == int(event_id):
                return event

        raise KeyError(f"事件不存在: task_id={task_id}, event_id={event_id}")

    def _unique_reviewed_clip_path(self, label: str, task_id: str, event_id: int, event: Dict[str, Any], source_path: Path) -> Path:
        label_dir = self.reviewed_root / label
        label_dir.mkdir(parents=True, exist_ok=True)

        event_type = _safe_filename_part(event.get("event_type"), "event")
        suffix = source_path.suffix or ".mp4"
        base_name = f"{_safe_filename_part(task_id)}_event_{int(event_id):04d}_{event_type}{suffix}"
        candidate = label_dir / base_name

        if not candidate.exists():
            return candidate

        stem = candidate.stem
        for i in range(1, 10000):
            next_candidate = label_dir / f"{stem}_{i}{suffix}"
            if not next_candidate.exists():
                return next_candidate

        raise RuntimeError("无法生成唯一样本文件名")

    def _unique_reviewed_image_path(self, label: str, task_id: str, event_id: int, source_path: Path) -> Path:
        label_dir = self.reviewed_root / label
        label_dir.mkdir(parents=True, exist_ok=True)

        suffix = source_path.suffix.lower() if source_path.suffix.lower() in IMAGE_FILE_EXTENSIONS else ".jpg"
        base_name = f"{_safe_filename_part(task_id)}_image_{int(event_id):04d}{suffix}"
        candidate = label_dir / base_name

        if not candidate.exists():
            return candidate

        stem = candidate.stem
        for i in range(1, 10000):
            next_candidate = label_dir / f"{stem}_{i}{suffix}"
            if not next_candidate.exists():
                return next_candidate

        raise RuntimeError("无法生成唯一图片样本文件名")

    def _image_url_to_path(self, image_value: Any, task_id: str) -> Optional[Path]:
        if not image_value:
            return None

        text = str(image_value).strip().replace("\\", "/")
        if not text:
            return None

        candidates: List[Path] = []
        if text.startswith("/app/"):
            candidates.append(Path(text))
        elif text.startswith("/static/"):
            candidates.append(Path("/app") / text.lstrip("/"))
            candidates.append(Path(text.lstrip("/")))
        elif text.startswith("static/"):
            candidates.append(Path("/app") / text)
            candidates.append(Path(text))
        else:
            candidates.append(self.get_task_dir(task_id) / "images" / Path(text).name)
            candidates.append(Path(text))

        for candidate in candidates:
            if candidate.exists() and candidate.is_file() and candidate.suffix.lower() in IMAGE_FILE_EXTENSIONS:
                return candidate

        return None

    def import_image_dataset_as_review_task(self, dataset_dir: str, task_id: str, default_label: str = "uncertain") -> Dict[str, Any]:
        """把图片数据集导入为待人工复核任务。"""
        dataset_path = Path(dataset_dir)
        if not dataset_path.exists() or not dataset_path.is_dir():
            raise FileNotFoundError(f"图片数据集目录不存在: {dataset_path}")

        task_id = _safe_filename_part(task_id, "image_review_task")
        task_dir = self.behavior_results_dir / task_id
        images_dir = task_dir / "images"
        images_dir.mkdir(parents=True, exist_ok=True)

        image_paths: List[Path] = []
        for ext in IMAGE_FILE_EXTENSIONS:
            image_paths.extend(dataset_path.rglob(f"*{ext}"))
            image_paths.extend(dataset_path.rglob(f"*{ext.upper()}"))

        unique_images = []
        seen = set()
        for image_path in image_paths:
            if not image_path.is_file():
                continue
            key = str(image_path.resolve()).lower()
            if key in seen:
                continue
            seen.add(key)
            unique_images.append(image_path)
        unique_images.sort(key=lambda p: str(p).lower())

        if not unique_images:
            raise ValueError("图片数据集中未发现 jpg/jpeg/png/bmp/webp 图片")

        events = []
        for index, source_path in enumerate(unique_images, start=1):
            safe_name = _safe_filename_part(source_path.name, f"image_{index:04d}{source_path.suffix.lower()}")
            suffix = Path(safe_name).suffix or source_path.suffix or ".jpg"
            target_path = images_dir / safe_name
            if target_path.exists():
                target_path = images_dir / f"{Path(safe_name).stem}_{index:04d}{suffix}"
            shutil.copy2(str(source_path), str(target_path))

            snapshot_url = _to_static_url(target_path)
            events.append({
                "id": index,
                "sample_type": "image",
                "event_type": "image_sample",
                "event_name": "图片候选样本",
                "start_time": 0,
                "end_time": 0,
                "duration": 0,
                "confidence": 1.0,
                "clip": None,
                "snapshot": snapshot_url,
                "source_image": snapshot_url,
                "persons": [],
                "reason": {
                    "source": "image_dataset",
                    "default_label": default_label,
                    "original_filename": source_path.name,
                },
            })

        payload = {
            "task_id": task_id,
            "source": "image_dataset",
            "created_at": _now(),
            "summary": {
                "total_events": len(events),
                "image_count": len(events),
                "default_label": default_label,
            },
            "events": events,
        }
        self._write_json(task_dir / "events.json", payload)

        manifest = self.read_task_manifest(task_id)
        manifest.setdefault("reviews", {})
        self.write_task_manifest(task_id, manifest)

        return {
            "success": True,
            "task_id": task_id,
            "image_count": len(events),
            "events_json": _to_static_url(task_dir / "events.json"),
            "review_manifest": _to_static_url(task_dir / "review_manifest.json"),
        }

    def review_clip(self, task_id: str, event_id: int, label: str, note: str = "") -> Dict[str, Any]:
        label = str(label or "").strip()
        if label not in ALLOWED_REVIEW_LABELS:
            raise ValueError(f"不支持的复核标签: {label}")

        event = self._find_event(task_id, int(event_id))
        manifest = self.read_task_manifest(task_id)
        sample_type = str(event.get("sample_type") or ("clip" if event.get("clip") else "image")).strip()

        clip_path = self._clip_url_to_path(event.get("clip"), task_id)
        image_path = self._image_url_to_path(event.get("source_image") or event.get("snapshot"), task_id)
        copied_path: Optional[Path] = None
        reviewed_clip_url = None
        reviewed_image_url = None

        if label != "delete":
            if sample_type == "image" or (clip_path is None and image_path is not None):
                if image_path is None:
                    raise FileNotFoundError(f"图片样本文件不存在: {event.get('source_image') or event.get('snapshot')}")
                copied_path = self._unique_reviewed_image_path(label, task_id, int(event_id), image_path)
                shutil.copy2(str(image_path), str(copied_path))
                sample_type = "image"
                reviewed_image_url = _to_static_url(copied_path)
            else:
                if clip_path is None:
                    raise FileNotFoundError(f"事件 clip 文件不存在: {event.get('clip')}")
                copied_path = self._unique_reviewed_clip_path(label, task_id, int(event_id), event, clip_path)
                shutil.copy2(str(clip_path), str(copied_path))
                sample_type = "clip"
                reviewed_clip_url = _to_static_url(copied_path)

        review_record = {
            "event_id": int(event_id),
            "sample_type": sample_type,
            "source_task_id": task_id,
            "source_event_type": event.get("event_type"),
            "source_event_name": event.get("event_name"),
            "review_label": label,
            "review_label_name": LABEL_NAME_MAP.get(label, label),
            "review_status": "reviewed",
            "reviewed_at": _now(),
            "note": str(note or ""),
            "clip": event.get("clip"),
            "snapshot": event.get("snapshot"),
            "reason": event.get("reason") or {},
            "training_feature_vector": event.get("training_feature_vector") or [],
            "feature_schema_version": event.get("feature_schema_version"),
            "source_image": event.get("source_image"),
            "persons": event.get("persons") or [],
            "object_id": event.get("object_id"),
            "object_class": event.get("object_class"),
            "reviewed_clip_path": str(copied_path) if copied_path and sample_type == "clip" else None,
            "reviewed_clip_url": reviewed_clip_url,
            "reviewed_image_path": str(copied_path) if copied_path and sample_type == "image" else None,
            "reviewed_image_url": reviewed_image_url,
        }

        manifest.setdefault("reviews", {})[str(event_id)] = review_record
        self.write_task_manifest(task_id, manifest)

        global_manifest = self._load_global_manifest()
        samples = [
            item for item in global_manifest.get("samples", [])
            if not (item.get("source_task_id") == task_id and int(item.get("source_event_id", -1)) == int(event_id))
        ]

        if label != "delete":
            samples.append({
                "sample_type": sample_type,
                "clip_path": str(copied_path) if copied_path and sample_type == "clip" else None,
                "image_path": str(copied_path) if copied_path and sample_type == "image" else None,
                "label": label,
                "label_name": LABEL_NAME_MAP.get(label, label),
                "source_task_id": task_id,
                "source_event_id": int(event_id),
                "source_event_type": event.get("event_type"),
                "created_at": review_record["reviewed_at"],
                "note": review_record["note"],
            })

        global_manifest["samples"] = samples
        self._write_global_manifest(global_manifest)

        return {
            "success": True,
            "message": "复核结果已保存",
            "review": review_record,
        }

    def stats(self) -> Dict[str, Any]:
        global_manifest = self._load_global_manifest()
        samples = global_manifest.get("samples", []) or []

        labels: Dict[str, int] = {}
        for item in samples:
            label = str(item.get("label") or "unknown")
            labels[label] = labels.get(label, 0) + 1

        # Also scan directories as fallback.
        scanned_labels: Dict[str, int] = {}
        for label_dir in self.reviewed_root.iterdir() if self.reviewed_root.exists() else []:
            if not label_dir.is_dir():
                continue
            scanned_labels[label_dir.name] = len([p for p in label_dir.iterdir() if p.is_file() and (p.suffix.lower() == ".mp4" or p.suffix.lower() in IMAGE_FILE_EXTENSIONS)])

        return {
            "total_reviewed": len(samples),
            "labels": labels,
            "scanned_labels": scanned_labels,
            "samples_root": str(self.samples_root),
            "reviewed_root": str(self.reviewed_root),
            "manifest": str(self.global_manifest_path),
            "label_names": LABEL_NAME_MAP,
        }

    def dataset(self) -> Dict[str, Any]:
        global_manifest = self._load_global_manifest()
        samples = global_manifest.get("samples", []) or []

        # Keep only files that still exist. Supports clip samples and image samples.
        valid_samples = []
        for item in samples:
            sample_type = str(item.get("sample_type") or "clip")
            path = item.get("image_path") if sample_type == "image" else item.get("clip_path")
            if path and Path(path).exists():
                valid_samples.append(item)

        return {
            "total": len(valid_samples),
            "samples": valid_samples,
            "labels": sorted({item.get("label") for item in valid_samples if item.get("label")}),
            "samples_root": str(self.samples_root),
            "manifest": str(self.global_manifest_path),
        }
