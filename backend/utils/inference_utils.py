"""
推理管理工具类
封装YOLO模型推理相关操作
"""

import os
import time
import threading
import cv2
import numpy as np
from typing import Any, Dict, List, Optional
import logging
from yolo_trainer import YOLOTrainer
from gpu_probe_service import require_gpu, report_gpu_fault, WAIT_MESSAGE

logger = logging.getLogger(__name__)


class InferenceManager:
    """推理管理器"""
    
    def __init__(self):
        """初始化推理管理器"""
        self.models_cache = {}  # 模型缓存
        self._cache_lock = threading.RLock()
        self._model_locks = {}
    
    def get_or_load_model(self, model_path: str) -> YOLOTrainer:
        """
        获取或加载模型(带缓存)
        
        Args:
            model_path: 模型路径
            
        Returns:
            YOLOTrainer实例
        """
        with self._cache_lock:
            if model_path not in self.models_cache:
                trainer = YOLOTrainer(model_name=model_path)
                trainer.load_model()
                self.models_cache[model_path] = trainer
                self._model_locks[model_path] = threading.RLock()
                logger.info(f"加载模型: {model_path}")
            return self.models_cache[model_path]
    
    def process_realtime_frame(
        self,
        image: np.ndarray,
        model_path: str,
        confidence: float = 0.25,
        iou_threshold: float = 0.7,
        *,
        render_annotations: bool = True,
    ) -> Dict:
        """
        处理实时推理帧
        
        Args:
            image: 输入图像(RGB格式)
            model_path: 模型路径
            confidence: 置信度阈值
            iou_threshold: IoU阈值
            
        Returns:
            包含检测结果的字典
        """
        try:
            start_time = time.time()
            if self._realtime_device().lower() != "cpu":
                require_gpu()
            
            # 加载模型
            trainer = self.get_or_load_model(model_path)

            if self._realtime_tiling_enabled():
                return self._process_tiled_frame(
                    image,
                    trainer,
                    model_path,
                    confidence,
                    iou_threshold,
                    render_annotations=render_annotations,
                    started_at=start_time,
                )
            
            # Ultralytics accepts numpy arrays directly. Avoid a JPEG encode/write/read
            # round-trip for every realtime frame and serialize access to the cached model.
            model_input = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            imgsz = self._env_int("REALTIME_INFERENCE_IMGSZ", 640, 320, 1920)
            with self._cache_lock:
                model_lock = self._model_locks.setdefault(model_path, threading.RLock())
            with model_lock:
                results = trainer.track(
                    source=model_input,
                    conf=confidence,
                    iou=iou_threshold,
                    imgsz=imgsz,
                )
            
            # 处理结果
            detections = []
            annotated_image = image.copy() if render_annotations else None
            
            if results and len(results) > 0:
                result = results[0]
                if render_annotations:
                    annotated_image = result.plot()
                    annotated_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                
                if hasattr(result, 'boxes') and result.boxes is not None:
                    boxes = result.boxes
                    detections = self._extract_detections(boxes, result)
            
            inference_time = (time.time() - start_time) * 1000  # 转换为毫秒
            
            return {
                "detections": detections,
                "detections_count": len(detections),
                "annotated_image": annotated_image,
                "inference_time": inference_time,
                "timestamp": time.time()
            }
            
        except Exception as e:
            gpu_waiting = report_gpu_fault(e)
            if not gpu_waiting:
                logger.error(f"实时推理处理失败: {e}")
            return {
                "detections": [],
                "detections_count": 0,
                "annotated_image": image,
                "inference_time": 0,
                "error": WAIT_MESSAGE if gpu_waiting else str(e),
                "gpu_waiting": gpu_waiting,
                "timestamp": time.time()
            }

    @staticmethod
    def _env_bool(name: str, default: bool) -> bool:
        value = str(os.getenv(name, "true" if default else "false")).strip().lower()
        return value in {"1", "true", "yes", "on"}

    @staticmethod
    def _env_int(name: str, default: int, minimum: int, maximum: int) -> int:
        try:
            value = int(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            value = default
        return max(minimum, min(maximum, value))

    @staticmethod
    def _env_float(name: str, default: float, minimum: float, maximum: float) -> float:
        try:
            value = float(os.getenv(name, str(default)))
        except (TypeError, ValueError):
            value = default
        return max(minimum, min(maximum, value))

    @staticmethod
    def _realtime_tiling_enabled() -> bool:
        value = str(os.getenv("REALTIME_INFERENCE_TILING", "false") or "false").strip().lower()
        return value in {"1", "true", "yes", "on"}

    @staticmethod
    def _realtime_device() -> str:
        device = str(os.getenv("REALTIME_INFERENCE_DEVICE", "auto") or "auto").strip()
        if device.lower() != "auto":
            return device
        # Auto is GPU-backed; explicit CPU is the only CPU route. Do not touch
        # CUDA inside the API process to decide admission or silently fall back.
        return "0"

    def _format_realtime_result(
        self,
        image: np.ndarray,
        model_result: Any,
        inference_time: float,
        *,
        batch_size: int,
        completed_at: float,
        render_annotations: bool = True,
    ) -> Dict[str, Any]:
        detections = []
        annotated_image = image.copy() if render_annotations else None
        if model_result is not None:
            if render_annotations:
                annotated_image = cv2.cvtColor(model_result.plot(), cv2.COLOR_BGR2RGB)
            boxes = getattr(model_result, "boxes", None)
            if boxes is not None:
                detections = self._extract_detections(boxes, model_result)
        return {
            "detections": detections,
            "detections_count": len(detections),
            "annotated_image": annotated_image,
            "inference_time": inference_time,
            "batch_size": batch_size,
            "timestamp": completed_at,
        }

    def process_realtime_batch(
        self,
        images: List[np.ndarray],
        model_path: str,
        confidence: float = 0.25,
        iou_threshold: float = 0.7,
        *,
        render_annotations: bool = True,
    ) -> List[Dict[str, Any]]:
        """Infer a group of independent stream frames in one GPU-friendly call."""
        if not images:
            return []

        if self._realtime_tiling_enabled():
            return [
                self.process_realtime_frame(
                    image,
                    model_path,
                    confidence,
                    iou_threshold,
                    render_annotations=render_annotations,
                )
                for image in images
            ]

        started_at = time.perf_counter()
        try:
            if self._realtime_device().lower() != "cpu":
                require_gpu()
            trainer = self.get_or_load_model(model_path)
            model_inputs = [cv2.cvtColor(image, cv2.COLOR_RGB2BGR) for image in images]
            device = self._realtime_device()
            half = self._env_bool("REALTIME_INFERENCE_HALF", True)
            if device.lower() == "cpu":
                half = False
            imgsz = self._env_int("REALTIME_INFERENCE_IMGSZ", 640, 320, 1920)

            with self._cache_lock:
                model_lock = self._model_locks.setdefault(model_path, threading.RLock())
            with model_lock:
                model_results = trainer.predict_realtime(
                    source=model_inputs,
                    conf=confidence,
                    iou=iou_threshold,
                    device=device,
                    half=half,
                    imgsz=imgsz,
                )

            completed_at = time.time()
            inference_time = (time.perf_counter() - started_at) * 1000.0
            normalized_results = list(model_results or [])
            output = []
            for index, image in enumerate(images):
                if index >= len(normalized_results):
                    output.append({
                        "detections": [],
                        "detections_count": 0,
                        "annotated_image": image,
                        "inference_time": inference_time,
                        "batch_size": len(images),
                        "error": "批量推理返回结果数量不足",
                        "timestamp": completed_at,
                    })
                    continue
                annotation_started_at = time.perf_counter()
                formatted_result = self._format_realtime_result(
                    image,
                    normalized_results[index],
                    inference_time,
                    batch_size=len(images),
                    completed_at=completed_at,
                    render_annotations=render_annotations,
                )
                formatted_result["annotation_time_ms"] = (
                    time.perf_counter() - annotation_started_at
                ) * 1000.0
                output.append(formatted_result)
            return output
        except Exception as exc:
            gpu_waiting = report_gpu_fault(exc)
            if not gpu_waiting:
                logger.error("批量实时推理失败: %s", exc)
            completed_at = time.time()
            inference_time = (time.perf_counter() - started_at) * 1000.0
            return [{
                "detections": [],
                "detections_count": 0,
                "annotated_image": image,
                "inference_time": inference_time,
                "batch_size": len(images),
                "error": WAIT_MESSAGE if gpu_waiting else str(exc),
                "gpu_waiting": gpu_waiting,
                "timestamp": completed_at,
            } for image in images]
    
    def _process_tiled_frame(
        self,
        image: np.ndarray,
        trainer: YOLOTrainer,
        model_path: str,
        confidence: float,
        iou_threshold: float,
        *,
        render_annotations: bool,
        started_at: float,
    ) -> Dict[str, Any]:
        """Run bounded overlapping tiles and merge detections in source coordinates."""
        tile_specs = self._build_realtime_tiles(image)
        device = self._realtime_device()
        half = self._env_bool("REALTIME_INFERENCE_HALF", True) and device.lower() != "cpu"
        imgsz = self._env_int("REALTIME_INFERENCE_IMGSZ", 640, 320, 1920)
        tile_started_at = time.perf_counter()

        model_inputs = [
            cv2.cvtColor(tile["image"], cv2.COLOR_RGB2BGR)
            for tile in tile_specs
        ]
        with self._cache_lock:
            model_lock = self._model_locks.setdefault(model_path, threading.RLock())
        with model_lock:
            model_results = trainer.predict_realtime(
                source=model_inputs,
                conf=confidence,
                iou=iou_threshold,
                device=device,
                half=half,
                imgsz=imgsz,
            )

        tile_elapsed_ms = (time.perf_counter() - tile_started_at) * 1000.0
        max_tile_ms = self._env_int("REALTIME_INFERENCE_MAX_TILE_MS", 2000, 100, 30000)
        if tile_elapsed_ms > max_tile_ms:
            raise RuntimeError(
                f"受控切片推理超时: {tile_elapsed_ms:.1f}ms > {max_tile_ms}ms"
            )
        normalized_results = list(model_results or [])
        if len(normalized_results) != len(tile_specs):
            raise RuntimeError("受控切片推理返回结果数量与 tile 数不一致")

        detections: List[Dict[str, Any]] = []
        for spec, result in zip(tile_specs, normalized_results):
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            detections.extend(
                self._extract_detections(
                    boxes,
                    result,
                    offset=(spec["left"], spec["top"]),
                )
            )
        nms_iou = self._env_float("REALTIME_INFERENCE_TILE_NMS_IOU", 0.55, 0.1, 0.95)
        detections = self._merge_tiled_detections(detections, iou_threshold=nms_iou)
        for index, detection in enumerate(detections):
            detection["id"] = index

        annotated_image = image.copy() if render_annotations else None
        if annotated_image is not None:
            for detection in detections:
                box = detection["bbox"]
                cv2.rectangle(
                    annotated_image,
                    (box["x1"], box["y1"]),
                    (box["x2"], box["y2"]),
                    (0, 255, 0),
                    2,
                )
        completed_at = time.time()
        return {
            "detections": detections,
            "detections_count": len(detections),
            "annotated_image": annotated_image,
            "inference_time": (completed_at - started_at) * 1000,
            "tile_count": len(tile_specs),
            "tile_inference_time_ms": tile_elapsed_ms,
            "tile_nms_iou": nms_iou,
            "tiling_enabled": True,
            "timestamp": completed_at,
        }

    def _build_realtime_tiles(self, image: np.ndarray) -> List[Dict[str, Any]]:
        height, width = image.shape[:2]
        tile_size = self._env_int("REALTIME_INFERENCE_TILE_SIZE", 640, 320, 1920)
        overlap = self._env_float("REALTIME_INFERENCE_TILE_OVERLAP", 0.25, 0.0, 0.8)
        max_tiles = self._env_int("REALTIME_INFERENCE_MAX_TILES", 8, 1, 32)
        x_starts = self._tile_starts(width, tile_size, overlap)
        y_starts = self._tile_starts(height, tile_size, overlap)
        specs = []
        for top in y_starts:
            for left in x_starts:
                right = min(width, left + tile_size)
                bottom = min(height, top + tile_size)
                specs.append({
                    "image": image[top:bottom, left:right],
                    "left": left,
                    "top": top,
                    "right": right,
                    "bottom": bottom,
                })
        if len(specs) > max_tiles:
            raise ValueError(f"受控切片数量超限: {len(specs)} > {max_tiles}")
        return specs

    @staticmethod
    def _tile_starts(length: int, tile_size: int, overlap: float) -> List[int]:
        if length <= tile_size:
            return [0]
        stride = max(1, int(round(tile_size * (1.0 - overlap))))
        starts = list(range(0, max(1, length - tile_size + 1), stride))
        final_start = length - tile_size
        if starts[-1] != final_start:
            starts.append(final_start)
        return sorted(set(starts))

    @staticmethod
    def _merge_tiled_detections(
        detections: List[Dict[str, Any]],
        *,
        iou_threshold: float,
    ) -> List[Dict[str, Any]]:
        """Apply deterministic class-wise NMS to boxes in source coordinates."""
        kept: List[Dict[str, Any]] = []
        for class_id in sorted({int(item["class_id"]) for item in detections}):
            candidates = sorted(
                (item for item in detections if int(item["class_id"]) == class_id),
                key=lambda item: (
                    -float(item["confidence"]),
                    int(item["bbox"]["x1"]),
                    int(item["bbox"]["y1"]),
                ),
            )
            while candidates:
                current = candidates.pop(0)
                kept.append(current)
                candidates = [
                    item
                    for item in candidates
                    if InferenceManager._box_iou(current["bbox"], item["bbox"]) <= iou_threshold
                ]
        return sorted(
            kept,
            key=lambda item: (-float(item["confidence"]), int(item["class_id"])),
        )

    @staticmethod
    def _box_iou(first: Dict[str, int], second: Dict[str, int]) -> float:
        left = max(first["x1"], second["x1"])
        top = max(first["y1"], second["y1"])
        right = min(first["x2"], second["x2"])
        bottom = min(first["y2"], second["y2"])
        intersection = max(0, right - left) * max(0, bottom - top)
        first_area = max(0, first["x2"] - first["x1"]) * max(0, first["y2"] - first["y1"])
        second_area = max(0, second["x2"] - second["x1"]) * max(0, second["y2"] - second["y1"])
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0

    def _extract_detections(self, boxes, result, offset=(0, 0)) -> List[Dict]:
        """
        从YOLO结果中提取检测信息
        
        Args:
            boxes: YOLO boxes对象
            result: YOLO result对象
            
        Returns:
            检测结果列表
        """
        detections = []
        
        for i in range(len(boxes)):
            # 获取边界框坐标
            box = boxes.xyxy[i].cpu().numpy()
            x1, y1, x2, y2 = box.astype(int)
            x1 += int(offset[0])
            x2 += int(offset[0])
            y1 += int(offset[1])
            y2 += int(offset[1])
            
            # 获取置信度和类别
            confidence = float(boxes.conf[i].cpu().numpy())
            class_id = int(boxes.cls[i].cpu().numpy())
            class_name = result.names[class_id] if hasattr(result, 'names') else f"class_{class_id}"
            track_ids = getattr(boxes, "id", None)
            track_id = int(track_ids[i].cpu().numpy()) if track_ids is not None else -1
            
            detection = {
                "id": i,
                "track_id": track_id,
                "class": class_name,
                "class_id": class_id,
                "confidence": confidence,
                "bbox": {
                    "x1": int(x1), "y1": int(y1),
                    "x2": int(x2), "y2": int(y2),
                    "width": int(x2 - x1),
                    "height": int(y2 - y1),
                    "center_x": int((x1 + x2) / 2),
                    "center_y": int((y1 + y2) / 2)
                }
            }
            detections.append(detection)
        
        return detections
    
    def clear_cache(self):
        """清空模型缓存"""
        with self._cache_lock:
            self.models_cache.clear()
            self._model_locks.clear()
        logger.info("模型缓存已清空")
    
    def get_cached_models(self) -> List[str]:
        """
        获取已缓存的模型列表
        
        Returns:
            模型路径列表
        """
        with self._cache_lock:
            return list(self.models_cache.keys())
