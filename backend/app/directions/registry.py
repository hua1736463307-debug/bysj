"""方向注册表：唯一入口，进程内只读。

注册在导入时完成，不在请求期动态增删；未知方向一律显式报错，不回退到默认标签集。
"""
from __future__ import annotations

from app.directions.base import DirectionError, DirectionSpec

_REGISTRY: dict[str, DirectionSpec] = {}


def register(spec: DirectionSpec) -> DirectionSpec:
    spec.validate()
    if spec.key in _REGISTRY:
        raise DirectionError(f"方向重复注册: {spec.key}")
    _REGISTRY[spec.key] = spec
    return spec


def get(key: str) -> DirectionSpec | None:
    return _REGISTRY.get(key)


def require(key: str) -> DirectionSpec:
    spec = _REGISTRY.get(key)
    if spec is None:
        raise DirectionError(f"未知方向: {key}")
    return spec


def all_directions() -> list[DirectionSpec]:
    return [_REGISTRY[key] for key in sorted(_REGISTRY)]


def by_task_kind(task_kind: str) -> list[DirectionSpec]:
    return [spec for spec in all_directions() if spec.task_kind == task_kind]


def registered_keys() -> tuple[str, ...]:
    return tuple(sorted(_REGISTRY))


# 导入即注册；顺序无关，重复注册会在导入期报错。
from app.directions import cxr as _cxr  # noqa: E402,F401
from app.directions import detect as _detect  # noqa: E402,F401
from app.directions import isic as _isic  # noqa: E402,F401
