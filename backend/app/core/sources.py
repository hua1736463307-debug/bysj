"""受控只读数据来源解析（多方向共用）。

优先级：
1. `DATASET_SOURCES` 显式映射（source_id -> 绝对目录），管理员配置；
2. `<DATASET_IMPORT_ROOT>/<source_id>`；
3. 方向专用的历史键（CXR 的 `CXR_SOURCES` / `CXR_IMPORT_ROOT`），仅为兼容保留。

API 只接受已配置的 source_id，不接受任意服务器路径、网络 URL 或 pickle。
"""
from __future__ import annotations

from pathlib import Path

from app.core.config import Settings
from app.core.errors import ApiError


def configured_sources(settings: Settings) -> dict[str, str]:
    return dict(getattr(settings, "dataset_sources", None) or {})


def resolve_source_root(
    settings: Settings,
    source_id: str,
    *,
    legacy_sources: dict[str, str] | None = None,
    legacy_root: Path | None = None,
    direction: str = "dataset",
    code_prefix: str = "",
) -> Path:
    text = str(source_id or "").strip()
    if not text:
        raise ApiError(422, "source_id_required", "必须提供 source_id")

    configured = configured_sources(settings).get(text)
    if configured:
        root = Path(configured).expanduser()
    else:
        base = getattr(settings, "dataset_import_root", None)
        if base:
            root = Path(base).expanduser() / text
        elif legacy_sources and text in legacy_sources:
            root = Path(legacy_sources[text]).expanduser()
        elif legacy_root:
            root = Path(legacy_root).expanduser() / text
        else:
            raise ApiError(
                409,
                f"{code_prefix}source_not_configured",
                f"未配置 {direction} 数据来源；管理员需设置 DATASET_SOURCES 或 DATASET_IMPORT_ROOT",
            )
    root = root.resolve()
    if not root.is_dir():
        raise ApiError(404, f"{code_prefix}source_missing", "配置的数据来源目录不存在或不可读")
    return root


def resolve_inside(root: Path, relative: str) -> Path:
    """在受控根内解析相对路径，拒绝绝对路径、盘符与上级目录。"""
    text = str(relative or "").replace("\\", "/").strip()
    if not text:
        raise ApiError(422, "invalid_path", "路径不能为空")
    if text.startswith("/") or (len(text) >= 2 and text[1] == ":"):
        raise ApiError(422, "invalid_path", f"不允许绝对路径: {relative}")
    if any(part == ".." for part in Path(text).parts):
        raise ApiError(422, "invalid_path", f"不允许上级目录: {relative}")
    candidate = (Path(root) / text).resolve()
    try:
        candidate.relative_to(Path(root).resolve())
    except ValueError as exc:
        raise ApiError(422, "invalid_path", f"路径越界: {relative}") from exc
    return candidate
