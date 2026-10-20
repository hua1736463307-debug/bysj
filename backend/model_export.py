"""Safe model artifact conversion helpers."""

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any


def convert_pt_to_onnx(source_path: Path, *, task_id: str, imgsz: int = 640) -> Path:
    """Convert one managed PT artifact and atomically publish its sibling ONNX file.

    Ultralytics writes beside the model it loads. A temporary copy keeps partial
    exporter output away from the source experiment and makes a failed export
    unable to replace an existing valid artifact.
    """
    source = Path(source_path).resolve()
    if source.suffix.lower() != ".pt" or not source.is_file():
        raise ValueError(f"invalid PT source: {source}")
    if int(imgsz) <= 0:
        raise ValueError("imgsz must be positive")

    output = source.with_suffix(".onnx")
    temporary_root = Path(tempfile.mkdtemp(prefix=f".onnx-export-{task_id}-", dir=str(source.parent)))
    temporary_source = temporary_root / source.name
    try:
        shutil.copy2(source, temporary_source)
        from ultralytics import YOLO

        model: Any = YOLO(str(temporary_source))
        exported = model.export(
            format="onnx",
            imgsz=int(imgsz),
            device="cpu",
            half=False,
            simplify=False,
        )
        exported_path = Path(exported).resolve() if exported else temporary_source.with_suffix(".onnx")
        if not exported_path.is_file():
            candidates = sorted(temporary_root.rglob("*.onnx"), key=lambda item: item.stat().st_mtime)
            if len(candidates) != 1:
                raise FileNotFoundError("ONNX 导出完成但未找到唯一输出文件")
            exported_path = candidates[0]

        os.replace(exported_path, output)
        return output
    finally:
        shutil.rmtree(temporary_root, ignore_errors=True)
