"""
视频处理工具类
封装视频捕获、帧提取等操作
"""

import cv2
import numpy as np
import logging
import os
import random
import threading
import time
from typing import Any, Callable, Dict, Optional, Tuple
from pathlib import Path

logger = logging.getLogger(__name__)


class SupervisedCapture:
    """Continuously drain a live source and expose only its newest frame."""

    def __init__(
        self,
        session_id: str,
        source_type: str,
        source_url: str,
        capture_factory: Callable[[str, str], Any],
        frame_reader: Callable[[Any, bool], Optional[np.ndarray]],
        release_capture: Callable[[Any], None],
        reconnect_initial_seconds: float = 0.5,
        reconnect_max_seconds: float = 15.0,
    ):
        self.session_id = session_id
        self.source_type = source_type
        self.source_url = source_url
        self.capture_factory = capture_factory
        self.frame_reader = frame_reader
        self.release_capture = release_capture
        self.reconnect_initial_seconds = max(0.05, reconnect_initial_seconds)
        self.reconnect_max_seconds = max(self.reconnect_initial_seconds, reconnect_max_seconds)

        self._condition = threading.Condition()
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._capture = None
        self._latest_frame: Optional[np.ndarray] = None
        self._sequence = 0
        self._status = "idle"
        self._last_frame_at = 0.0
        self._last_connect_at = 0.0
        self._last_error = ""
        self._consecutive_failures = 0
        self._reconnect_count = 0

    def matches(self, source_type: str, source_url: str) -> bool:
        return self.source_type == source_type and self.source_url == source_url

    def start(self) -> "SupervisedCapture":
        with self._condition:
            if self._thread is not None and self._thread.is_alive():
                return self
            self._stop_event.clear()
            self._thread = threading.Thread(
                target=self._run,
                name=f"capture-{self.session_id}",
                daemon=True,
            )
            self._thread.start()
        return self

    def stop(self, join_timeout: float = 2.0) -> bool:
        """Request stop and let the reader thread release its own capture."""
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout=max(0.0, join_timeout))
        stopped = not bool(thread and thread.is_alive())
        with self._condition:
            self._status = "stopped" if stopped else "stopping"
            self._condition.notify_all()
        return stopped

    def wait_for_latest(self, after_sequence: int = 0, timeout: float = 1.0):
        deadline = time.monotonic() + max(0.0, timeout)
        with self._condition:
            while (
                not self._stop_event.is_set()
                and self._sequence <= after_sequence
                and time.monotonic() < deadline
            ):
                self._condition.wait(timeout=max(0.0, deadline - time.monotonic()))

            if self._latest_frame is None or self._sequence <= after_sequence:
                return None, after_sequence, self._snapshot_locked()
            return self._latest_frame, self._sequence, self._snapshot_locked()

    def _snapshot_locked(self) -> Dict[str, Any]:
        now = time.time()
        frame_age_ms = None
        if self._last_frame_at > 0:
            frame_age_ms = max(0.0, (now - self._last_frame_at) * 1000.0)
        return {
            "session_id": self.session_id,
            "source_type": self.source_type,
            "status": self._status,
            "sequence": self._sequence,
            "last_frame_at": self._last_frame_at or None,
            "frame_age_ms": round(frame_age_ms, 2) if frame_age_ms is not None else None,
            "last_connect_at": self._last_connect_at or None,
            "last_error": self._last_error or None,
            "consecutive_failures": self._consecutive_failures,
            "reconnect_count": self._reconnect_count,
            "thread_alive": bool(self._thread and self._thread.is_alive()),
        }

    def snapshot(self) -> Dict[str, Any]:
        with self._condition:
            return self._snapshot_locked()

    def _set_status(self, status: str, error: str = ""):
        with self._condition:
            self._status = status
            if error:
                self._last_error = error
            self._condition.notify_all()

    def _run(self):
        backoff = self.reconnect_initial_seconds
        first_attempt = True

        while not self._stop_event.is_set():
            self._set_status("connecting" if first_attempt else "reconnecting")
            first_attempt = False
            capture = self.capture_factory(self.source_type, self.source_url)

            if capture is None:
                with self._condition:
                    self._consecutive_failures += 1
                    self._reconnect_count += 1
                    self._last_error = "open_failed"
                self._wait_before_retry(backoff)
                backoff = min(self.reconnect_max_seconds, backoff * 2)
                continue

            with self._condition:
                self._capture = capture
                self._last_connect_at = time.time()
                self._last_error = ""

            got_stable_frame = False
            try:
                while not self._stop_event.is_set():
                    frame = self.frame_reader(capture, True)
                    if frame is None:
                        raise RuntimeError("read_failed")
                    with self._condition:
                        self._latest_frame = frame
                        self._sequence += 1
                        self._last_frame_at = time.time()
                        self._status = "live"
                        self._last_error = ""
                        self._consecutive_failures = 0
                        self._condition.notify_all()
                    got_stable_frame = True
                    backoff = self.reconnect_initial_seconds
            except Exception as exc:
                if not self._stop_event.is_set():
                    with self._condition:
                        self._consecutive_failures += 1
                        self._reconnect_count += 1
                        self._last_error = str(exc)
                        self._status = "reconnecting"
                        self._condition.notify_all()
            finally:
                should_release = False
                with self._condition:
                    if self._capture is capture:
                        self._capture = None
                        should_release = True
                if should_release:
                    self.release_capture(capture)

            if not self._stop_event.is_set():
                self._wait_before_retry(backoff)
                if not got_stable_frame:
                    backoff = min(self.reconnect_max_seconds, backoff * 2)

        self._set_status("stopped")

    def _wait_before_retry(self, backoff: float):
        jitter = random.uniform(0, min(0.25, backoff * 0.2))
        self._stop_event.wait(backoff + jitter)


class VideoProcessor:
    """视频处理器"""
    
    def __init__(self):
        """初始化视频处理器"""
        self.captures = {}  # 存储视频捕获对象
        self.supervised_captures: Dict[str, SupervisedCapture] = {}
        self._consumer_sequences: Dict[str, int] = {}
        self._capture_lock = threading.RLock()
    
    def create_capture(self, source_type: str, source_url: str = "") -> Optional[cv2.VideoCapture]:
        """
        创建视频捕获对象
        
        Args:
            source_type: 视频源类型(webcam, rtsp, http, file)
            source_url: 视频源URL或文件路径
            
        Returns:
            cv2.VideoCapture对象,失败返回None
        """
        try:
            if source_type == "webcam":
                cap = cv2.VideoCapture(0)
            elif source_type in ["rtsp", "http"]:
                os.environ.setdefault(
                    "OPENCV_FFMPEG_CAPTURE_OPTIONS",
                    "rtsp_transport;tcp|fflags;nobuffer|flags;low_delay|max_delay;0",
                )
                backend = getattr(cv2, "CAP_FFMPEG", 0)
                params = []
                if hasattr(cv2, "CAP_PROP_OPEN_TIMEOUT_MSEC"):
                    params.extend([cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, int(os.getenv("VIDEO_OPEN_TIMEOUT_MS", "5000"))])
                if hasattr(cv2, "CAP_PROP_READ_TIMEOUT_MSEC"):
                    params.extend([cv2.CAP_PROP_READ_TIMEOUT_MSEC, int(os.getenv("VIDEO_READ_TIMEOUT_MS", "5000"))])
                try:
                    cap = cv2.VideoCapture(source_url, backend, params)
                except (TypeError, cv2.error):
                    cap = cv2.VideoCapture(source_url, backend)
                if hasattr(cv2, "CAP_PROP_BUFFERSIZE"):
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            elif source_type == "file":
                cap = cv2.VideoCapture(source_url)
            else:
                raise ValueError(f"不支持的视频源类型: {source_type}")
            
            if not cap.isOpened():
                raise ValueError(f"无法打开视频源: {source_url}")
            
            return cap
        except Exception as e:
            logger.error(f"创建视频捕获失败: {e}")
            return None
    
    def get_video_info(self, cap: cv2.VideoCapture) -> dict:
        """
        获取视频信息
        
        Args:
            cap: 视频捕获对象
            
        Returns:
            包含视频信息的字典
        """
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        return {
            "fps": fps,
            "frame_count": frame_count,
            "width": width,
            "height": height,
            "duration": duration
        }
    
    def read_frame(self, cap: cv2.VideoCapture, convert_to_rgb: bool = True) -> Optional[np.ndarray]:
        """
        从视频捕获对象读取一帧
        
        Args:
            cap: 视频捕获对象
            convert_to_rgb: 是否转换为RGB格式
            
        Returns:
            图像帧,失败返回None
        """
        try:
            ret, frame = cap.read()
            if not ret:
                return None
            
            if convert_to_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            return frame
        except Exception as e:
            logger.error(f"读取视频帧失败: {e}")
            return None
    
    def get_frame_at_position(self, cap: cv2.VideoCapture, frame_index: int, 
                             convert_to_rgb: bool = True) -> Optional[np.ndarray]:
        """
        获取指定位置的帧
        
        Args:
            cap: 视频捕获对象
            frame_index: 帧索引
            convert_to_rgb: 是否转换为RGB格式
            
        Returns:
            图像帧,失败返回None
        """
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            return self.read_frame(cap, convert_to_rgb)
        except Exception as e:
            logger.error(f"获取指定帧失败: {e}")
            return None
    
    def reset_to_start(self, cap: cv2.VideoCapture):
        """
        重置视频到开始位置
        
        Args:
            cap: 视频捕获对象
        """
        cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    
    def release_capture(self, cap: cv2.VideoCapture):
        """
        释放视频捕获对象
        
        Args:
            cap: 视频捕获对象
        """
        try:
            if cap is not None:
                cap.release()
        except Exception as e:
            logger.error(f"释放视频捕获失败: {e}")
    
    def register_capture(self, session_id: str, cap: cv2.VideoCapture):
        """
        注册视频捕获对象到管理器
        
        Args:
            session_id: 会话ID
            cap: 视频捕获对象
        """
        self.captures[session_id] = cap

    def ensure_supervised_capture(self, session_id: str, source_type: str, source_url: str) -> SupervisedCapture:
        """Start or reuse a reconnecting latest-frame worker for a live source."""
        with self._capture_lock:
            existing = self.supervised_captures.get(session_id)
            if existing is not None and existing.matches(source_type, source_url):
                return existing.start()
            if existing is not None:
                existing.stop()

            worker = SupervisedCapture(
                session_id=session_id,
                source_type=source_type,
                source_url=source_url,
                capture_factory=self.create_capture,
                frame_reader=self.read_frame,
                release_capture=self.release_capture,
            )
            self.supervised_captures[session_id] = worker
            return worker.start()

    def wait_for_latest_frame(self, session_id: str, after_sequence: int = 0, timeout: float = 1.0):
        with self._capture_lock:
            worker = self.supervised_captures.get(session_id)
        if worker is None:
            return None, after_sequence, None
        return worker.wait_for_latest(after_sequence=after_sequence, timeout=timeout)

    def read_supervised_frame(self, session_id: str, source_type: str, source_url: str, timeout: float = 1.0):
        self.ensure_supervised_capture(session_id, source_type, source_url)
        with self._capture_lock:
            after_sequence = self._consumer_sequences.get(session_id, 0)
        frame, sequence, health = self.wait_for_latest_frame(session_id, after_sequence, timeout)
        if frame is not None:
            with self._capture_lock:
                self._consumer_sequences[session_id] = sequence
        return frame, health

    def read_snapshot_frame(self, session_id: str, source_type: str, source_url: str):
        """Open a live source for one current frame, then release it immediately."""
        health = {
            "session_id": session_id,
            "source_type": source_type,
            "status": "waiting",
            "last_error": None,
            "reconnect_count": 0,
        }
        capture = self.create_capture(source_type, source_url)
        if capture is None:
            health.update({"status": "error", "last_error": "无法打开视频源"})
            return None, health
        try:
            frame = self.read_frame(capture)
            if frame is None:
                health.update({"status": "error", "last_error": "无法从视频源读取当前帧"})
                return None, health
            health.update({"status": "live", "last_frame_at": time.time()})
            return frame, health
        finally:
            self.release_capture(capture)

    def get_capture_health(self, session_id: str = None) -> Dict[str, Any]:
        with self._capture_lock:
            if session_id is not None:
                worker = self.supervised_captures.get(session_id)
                return worker.snapshot() if worker is not None else {}
            workers = list(self.supervised_captures.items())
        return {key: worker.snapshot() for key, worker in workers}
    
    def get_capture(self, session_id: str) -> Optional[cv2.VideoCapture]:
        """
        获取注册的视频捕获对象
        
        Args:
            session_id: 会话ID
            
        Returns:
            视频捕获对象,不存在返回None
        """
        return self.captures.get(session_id)
    
    def cleanup_capture(self, session_id: str, join_timeout: float = 2.0) -> bool:
        """
        清理指定的视频捕获资源
        
        Args:
            session_id: 会话ID
        """
        with self._capture_lock:
            worker = self.supervised_captures.get(session_id)
        worker_stopped = True
        if worker is not None:
            worker_stopped = worker.stop(join_timeout=join_timeout)
            if worker_stopped:
                with self._capture_lock:
                    if self.supervised_captures.get(session_id) is worker:
                        self.supervised_captures.pop(session_id, None)
                        self._consumer_sequences.pop(session_id, None)

        if worker_stopped and session_id in self.captures:
            try:
                self.captures[session_id].release()
                del self.captures[session_id]
                logger.info(f"清理视频捕获资源: {session_id}")
            except Exception as e:
                logger.error(f"清理视频捕获失败: {e}")
                return False
        return worker_stopped
    
    def cleanup_all(self):
        """清理所有视频捕获资源"""
        session_ids = set(self.captures.keys()) | set(self.supervised_captures.keys())
        for session_id in list(session_ids):
            self.cleanup_capture(session_id)
