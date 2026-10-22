"""受控文件系统边界：路径解析、原子写入、哈希与图像探测。

规则（CONTRACTS C05）：客户端只提供资源 ID；解析出的真实路径必须留在
DATA_ROOT / MODEL_ROOT / ARTIFACT_ROOT 内，并防止 symlink 逃逸。
"""
from __future__ import annotations

import hashlib
import os
import shutil
import tempfile
from pathlib import Path, PurePosixPath

from app.core.errors import ApiError

WINDOWS_RESERVED = {
    "CON",
    "PRN",
    "AUX",
    "NUL",
    *(f"COM{i}" for i in range(1, 10)),
    *(f"LPT{i}" for i in range(1, 10)),
}


def normalize_relative(relative: str) -> PurePosixPath:
    """把客户端/清单中的相对路径规范化为安全的 POSIX 相对路径。"""
    raw = str(relative or "").replace("\\", "/").strip()
    if not raw:
        raise ApiError(422, "invalid_path", "相对路径不能为空")
    if raw.startswith("/") or raw.startswith("//"):
        raise ApiError(422, "invalid_path", f"不允许绝对路径: {relative}")
    if len(raw) >= 2 and raw[1] == ":":
        raise ApiError(422, "invalid_path", f"不允许盘符路径: {relative}")
    path = PurePosixPath(raw)
    parts = []
    for part in path.parts:
        if part in ("", "."):
            continue
        if part == "..":
            raise ApiError(422, "invalid_path", f"不允许上级目录: {relative}")
        if part.split(".")[0].upper() in WINDOWS_RESERVED:
            raise ApiError(422, "invalid_path", f"保留文件名: {relative}")
        parts.append(part)
    if not parts:
        raise ApiError(422, "invalid_path", f"路径无效: {relative}")
    return PurePosixPath(*parts)


def resolve_under(root: Path, relative: str, *, must_exist: bool = False) -> Path:
    """解析相对路径并确认结果仍在 root 内（含 symlink 解析后）。"""
    root_resolved = root.resolve()
    candidate = (root_resolved / normalize_relative(relative)).resolve()
    try:
        candidate.relative_to(root_resolved)
    except ValueError as exc:
        raise ApiError(422, "invalid_path", "路径越界") from exc
    if must_exist and not candidate.exists():
        raise ApiError(404, "not_found", "文件不存在")
    return candidate


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def atomic_write_bytes(path: Path, data: bytes) -> str:
    """在目标目录内写临时文件后原子替换，返回 sha256。"""
    ensure_dir(path.parent)
    digest = hashlib.sha256(data).hexdigest()
    fd, tmp_name = tempfile.mkstemp(dir=str(path.parent), prefix=".tmp-", suffix=".part")
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp_name, path)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return digest


def atomic_move(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    shutil.move(str(src), str(dst))


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def canonical_json_bytes(payload) -> bytes:
    import json

    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def probe_image(path: Path) -> tuple[int, int, str]:
    from PIL import Image

    with Image.open(path) as image:
        image.verify()
    with Image.open(path) as image:
        return int(image.width), int(image.height), (image.format or "").upper()


def remove_tree(path: Path, *, allowed_root: Path) -> None:
    """只删除位于受控 root 之下的目录，用于失败 staging 清理。"""
    resolved = path.resolve()
    root_resolved = allowed_root.resolve()
    try:
        resolved.relative_to(root_resolved)
    except ValueError as exc:
        raise ApiError(500, "unsafe_cleanup", "拒绝删除受控根之外的目录") from exc
    if resolved == root_resolved:
        raise ApiError(500, "unsafe_cleanup", "拒绝删除根目录本身")
    shutil.rmtree(resolved, ignore_errors=True)


def walk_files(root: Path) -> list[Path]:
    return sorted((p for p in root.rglob("*") if p.is_file()), key=lambda p: p.as_posix())
