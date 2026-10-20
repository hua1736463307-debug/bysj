"""Authenticated async client for the private inference service."""
from __future__ import annotations

import base64
import asyncio
import os
import threading
import time
from typing import Any

import cv2
import numpy as np
import requests


class InferenceClient:
    def __init__(self) -> None:
        self.base_url = os.getenv("INFERENCE_SERVICE_URL", "http://inference-service:8090").rstrip("/")
        self.token = os.getenv("INFERENCE_SERVICE_TOKEN", "")
        self._revision = 0
        self._ids: dict[str, str] = {}
        self._models: dict[str, dict[str, str]] = {}
        self._generation = 0
        self._switching = False
        self._lock = threading.RLock()

    def configure(self, revision: int, models: list[dict[str, str]]) -> None:
        with self._lock:
            self._revision = revision
            self._generation += 1
            self._switching = False
            self._ids = {item["path"]: item["model_id"] for item in models}
            self._models = {item["model_id"]: item for item in models}

    def begin_switch(self, value=True):
        with self._lock:
            self._switching = bool(value)

    @property
    def revision(self):
        with self._lock:
            return self._revision

    def model_spec(self, model_id: str) -> dict[str, str]:
        with self._lock:
            spec = self._models.get(model_id)
            if not spec:
                raise RuntimeError("所需模型不在已发布的推理配置中")
            return dict(spec)

    def model_id(self, path: str) -> str:
        with self._lock:
            if self._switching:
                raise RuntimeError("推理服务模型配置正在切换")
            if path not in self._ids:
                raise RuntimeError("所需模型不在已发布的推理配置中")
            return self._ids[path]

    def reload(self, revision: int, models: list[dict[str, str]]) -> dict[str, Any]:
        headers = {"X-Inference-Token": self.token}
        response = requests.post(f"{self.base_url}/internal/reload",
                              json={"revision": revision, "models": models},
                              headers=headers, timeout=300)
        if response.status_code >= 400:
            raise RuntimeError("独立推理服务无法加载所选模型")
        status = response.json()
        deadline = time.monotonic() + 300
        while time.monotonic() < deadline:
            if status.get("state") == "ready" or status.get("state") == "failed":
                break
            time.sleep(0.5)
            try:
                health = requests.get(f"{self.base_url}/health", timeout=5)
                health.raise_for_status()
                status = health.json()
            except requests.RequestException:
                # The container briefly closes its socket while Docker restarts it.
                continue
        if status.get("state") != "ready" or status.get("revision") != revision or status.get("error"):
            raise RuntimeError("独立推理服务未能加载所选模型")
        return status

    async def infer(self, image_rgb: np.ndarray, model_ids: list[str], confidence: float,
                    iou: float, *, tracking: bool = False, revision: int | None = None,
                    enabled_events=None, enforce_revision: bool = True) -> dict[str, Any]:
        with self._lock:
            if self._switching:
                raise RuntimeError("推理服务模型配置正在切换")
            revision = self._revision if revision is None else revision
            if revision != self._revision and enforce_revision:
                raise RuntimeError("推理服务模型配置正在切换")
        payload = self._request_payload(image_rgb, model_ids, confidence, iou, tracking)
        payload["revision"] = revision
        payload["enabled_events"] = list(enabled_events or [])
        headers = {"X-Inference-Token": self.token}
        response = await asyncio.to_thread(requests.post, f"{self.base_url}/v1/infer",
                                           json=payload, headers=headers, timeout=30)
        if response.status_code >= 400:
            raise RuntimeError("独立推理服务暂不可用，请检查模型服务状态")
        result = response.json()
        with self._lock:
            if result.get("revision") != revision:
                raise RuntimeError("推理服务模型版本已更新")
        return result

    def infer_sync(self, image_rgb: np.ndarray, model_path: str, **kwargs: Any):
        model_id = str(kwargs.get("model_id") or self.model_id(str(model_path)))
        spec = self.model_spec(model_id)
        payload = self._request_payload(image_rgb, [spec["model_id"]],
                                        kwargs.get("conf", .25), kwargs.get("iou", .7),
                                        kwargs.get("tracking", False))
        payload["revision"] = self.revision
        response = requests.post(f"{self.base_url}/v1/infer", json=payload,
                              headers={"X-Inference-Token": self.token}, timeout=30)
        if response.status_code >= 400:
            raise RuntimeError("独立推理服务暂不可用，请检查模型服务状态")
        result = response.json()
        with self._lock:
            if result.get("revision") != payload["revision"]:
                raise RuntimeError("推理服务模型版本已更新")
        items = result.get("results") or []
        if not items:
            return []
        item = items[0]
        return [{"detections": item.get("detections", []), "names": {},
                 "poses": item.get("poses", [])}]

    def switch_and_configure(self, revision: int, models: list[dict[str, str]]):
        self.begin_switch()
        result = self.reload(revision, models)
        if result.get("state") == "ready":
            self.configure(revision, models)
        else:
            self.begin_switch(False)
        return result

    def _request_payload(self, image_rgb, model_ids, confidence, iou, tracking):
        ok, encoded = cv2.imencode(".png", cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR),
                                   [cv2.IMWRITE_PNG_COMPRESSION, 3])
        if not ok:
            raise RuntimeError("推理图片编码失败")
        with self._lock:
            revision = self._revision
        return {"revision": revision, "model_ids": model_ids,
                "image": base64.b64encode(encoded.tobytes()).decode("ascii"),
                "confidence": confidence, "iou_threshold": iou, "tracking": tracking,
                "enabled_events": []}

    async def infer_monitoring(self, image_rgb: np.ndarray, model_paths: list[str], confidence: float,
                              iou: float, *, tracking: bool = False, enabled_events=None,
                              revision: int | None = None):
        # Monitoring bindings carry registry IDs here; the legacy parameter name is preserved.
        model_ids = list(dict.fromkeys(str(model_id) for model_id in model_paths))
        with self._lock:
            current_revision = self._revision
            generation = self._generation
            revision = current_revision if revision is None else revision
            specs = [self._models.get(model_id) for model_id in model_ids]
        if revision != current_revision or any(spec is None for spec in specs):
            raise RuntimeError("监控算法模型版本已失效，请重新启动本路")
        result = await self.infer(image_rgb, model_ids, confidence, iou,
                                  tracking=tracking, revision=revision,
                                  enabled_events=enabled_events, enforce_revision=False)
        with self._lock:
            if generation != self._generation or result.get("revision") != revision:
                raise RuntimeError("推理服务模型配置已切换，请使用当前配置重试")
        return {"revision": result["revision"], "results": result["results"]}

    def infer_ids_sync(self, image_rgb: np.ndarray, model_ids: list[str], *, revision: int,
                       confidence: float, iou: float, enabled_events=None):
        payload = self._request_payload(image_rgb, model_ids, confidence, iou, False)
        payload["revision"] = revision
        payload["enabled_events"] = list(enabled_events or [])
        response = requests.post(f"{self.base_url}/v1/infer", json=payload,
                              headers={"X-Inference-Token": self.token}, timeout=30)
        if response.status_code >= 400:
            raise RuntimeError("独立推理服务暂不可用，请检查模型服务状态")
        result = response.json()
        with self._lock:
            if result.get("revision") != revision:
                raise RuntimeError("推理服务模型版本已更新")
        return result.get("results", [])

    def infer_flow_sync(self, image_rgb, config, *, revision):
        model_id = config['modelRef']
        with self._lock:
            generation = self._generation
            if revision != self._revision or self._switching or model_id not in self._models:
                raise RuntimeError('流程模型配置未就绪')
        payload = self._request_payload(image_rgb, [model_id], min(config['thresholds'][t] for t in config['targets']), config['postprocess']['iou'], False)
        payload.update(revision=revision, imgsz=config['preprocess']['imgsz'], merge_duplicates=config['postprocess']['mergeDuplicates'])
        response = requests.post(f"{self.base_url}/v1/infer", json=payload, headers={"X-Inference-Token":self.token}, timeout=30)
        if response.status_code >= 400: raise RuntimeError('流程识别失败')
        result = response.json()
        with self._lock:
            if generation != self._generation or result.get('revision') != revision: raise RuntimeError('流程配置版本已更新')
        items = result.get('results', [])
        if len(items) != 1 or items[0].get('model_id') != model_id: raise RuntimeError('流程识别结果不完整')
        return items[0]

    def infer_current_sync(self, image_rgb: np.ndarray, model_ids: list[str], *,
                           confidence: float, iou: float, enabled_events=None):
        with self._lock:
            revision = self._revision
            generation = self._generation
            if self._switching:
                raise RuntimeError("推理服务模型配置正在切换")
        results = self.infer_ids_sync(image_rgb, model_ids, revision=revision,
                                      confidence=confidence, iou=iou, enabled_events=enabled_events)
        with self._lock:
            if generation != self._generation:
                raise RuntimeError("推理服务模型配置已切换")
        return results

    def infer_pinned_sync(self, image_rgb: np.ndarray, model_ids: list[str], *, revision: int,
                          confidence: float, iou: float, enabled_events=None):
        with self._lock:
            if self._switching or revision != self._revision:
                raise RuntimeError("推理服务模型配置正在切换，请重新启动监控")
            generation = self._generation
            if any(model_id not in self._models for model_id in model_ids):
                raise RuntimeError("摄像头绑定的模型已失效，请重新启动本路")
        results = self.infer_ids_sync(image_rgb, model_ids, revision=revision,
                                      confidence=confidence, iou=iou,
                                      enabled_events=enabled_events)
        with self._lock:
            if generation != self._generation:
                raise RuntimeError("推理服务模型配置已切换，请重新启动本路")
        return results


inference_client = InferenceClient()
