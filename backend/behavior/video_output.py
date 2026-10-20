"""Browser-compatible video writing helpers.

OpenCV's default MP4 writer often produces MPEG-4 Part 2 streams (`mp4v`).
Some browsers cannot play those files even though OpenCV and local players can.
This helper writes with OpenCV, then tries to remux/transcode the result with
ffmpeg to H.264/yuv420p + faststart, which is the safest format for HTML5 video.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import Optional, Tuple

import cv2

logger = logging.getLogger(__name__)


def _even(value: int) -> int:
    value = int(value or 0)
    if value <= 0:
        return 2
    return value if value % 2 == 0 else value - 1


class BrowserCompatibleVideoWriter:
    """Small wrapper around cv2.VideoWriter with browser-compatible finalization."""

    def __init__(self, output_path: str | Path, fps: float, width: int, height: int):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        self.work_path = self.output_path.with_name(self.output_path.stem + "__opencv_tmp.mp4")
        self.converted_path = self.output_path.with_name(self.output_path.stem + "__h264_tmp.mp4")
        self.fps = float(fps or 25.0)
        if self.fps <= 1:
            self.fps = 25.0
        self.width = _even(width)
        self.height = _even(height)
        self.codec = None
        self.writer: Optional[cv2.VideoWriter] = None
        self.frames_written = 0

        self._open_writer()

    def _open_writer(self) -> None:
        # mp4v is broadly available in OpenCV builds. We convert it afterward if ffmpeg exists.
        # Try avc1/H264 first in case OpenCV has an H.264 encoder in the deployment image.
        last_error = None
        for codec in ("avc1", "H264", "X264", "mp4v"):
            try:
                fourcc = cv2.VideoWriter_fourcc(*codec)
                writer = cv2.VideoWriter(str(self.work_path), fourcc, self.fps, (self.width, self.height))
                if writer.isOpened():
                    self.writer = writer
                    self.codec = codec
                    logger.info("视频写入器已创建: path=%s codec=%s fps=%s size=%sx%s", self.work_path, codec, self.fps, self.width, self.height)
                    return
                try:
                    writer.release()
                except Exception:
                    pass
            except Exception as exc:
                last_error = exc

        raise RuntimeError(f"无法创建视频输出文件: {self.output_path}; last_error={last_error}")

    def write(self, frame) -> None:
        if self.writer is None:
            raise RuntimeError("视频写入器未初始化")
        if frame is None:
            return

        h, w = frame.shape[:2]
        if w != self.width or h != self.height:
            frame = cv2.resize(frame, (self.width, self.height), interpolation=cv2.INTER_AREA)
        self.writer.write(frame)
        self.frames_written += 1

    def abort(self) -> None:
        """Stop writing and remove partial outputs without running final transcoding."""
        if self.writer is not None:
            try:
                self.writer.release()
            finally:
                self.writer = None
        for path in (self.work_path, self.converted_path, self.output_path):
            try:
                path.unlink(missing_ok=True)
            except Exception:
                logger.warning("清理未完成视频失败: %s", path)

    def release(self) -> Tuple[bool, str]:
        if self.writer is not None:
            self.writer.release()
            self.writer = None

        if self.frames_written <= 0:
            try:
                self.work_path.unlink(missing_ok=True)
            except Exception:
                pass
            return False, "没有写入任何帧"

        if not self.work_path.exists() or self.work_path.stat().st_size <= 0:
            return False, f"OpenCV 输出文件不存在或为空: {self.work_path}"

        # Always try to convert to browser-compatible H.264 when ffmpeg is available.
        if _convert_to_browser_mp4(self.work_path, self.converted_path):
            try:
                if self.output_path.exists():
                    self.output_path.unlink()
                self.converted_path.replace(self.output_path)
                self.work_path.unlink(missing_ok=True)
                return True, "converted_h264"
            except Exception as exc:
                logger.warning("替换 H.264 视频失败，回退 OpenCV 输出: %s", exc)

        # Fallback: keep OpenCV output. Some browsers may still play it; local players usually can.
        try:
            if self.output_path.exists():
                self.output_path.unlink()
            self.work_path.replace(self.output_path)
            return True, f"opencv_{self.codec or 'unknown'}"
        except Exception as exc:
            return False, str(exc)


def _convert_to_browser_mp4(input_path: Path, output_path: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        logger.warning("未检测到 ffmpeg，跳过 H.264 浏览器兼容转码: %s", input_path)
        return False

    output_path.unlink(missing_ok=True)
    cmd = [
        ffmpeg,
        "-y",
        "-loglevel", "error",
        "-i", str(input_path),
        "-an",
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(output_path),
    ]

    try:
        completed = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, timeout=300)
        if completed.returncode == 0 and output_path.exists() and output_path.stat().st_size > 0:
            return True
        logger.warning("ffmpeg H.264 转码失败: returncode=%s stderr=%s", completed.returncode, completed.stderr[-1000:])
    except Exception as exc:
        logger.warning("ffmpeg H.264 转码异常: %s", exc)

    return False
