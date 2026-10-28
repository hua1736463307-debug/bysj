"""v1 方向注册接口：只读，暴露任务契约而不暴露数据路径或凭据。"""
from __future__ import annotations

from fastapi import APIRouter, Depends, Query

from app.api.deps import CurrentUser, get_current_user
from app.core.errors import ApiError
from app.directions import registry
from app.directions.base import DirectionError
from app.core.errors import ok

router = APIRouter(prefix="/directions", tags=["directions"])


@router.get("", response_model=None)
def list_directions(
    task_kind: str | None = Query(default=None, description="按任务语义过滤"),
    user: CurrentUser = Depends(get_current_user),
):
    specs = registry.by_task_kind(task_kind) if task_kind else registry.all_directions()
    return ok(
        {
            "items": [
                {
                    "direction_key": spec.key,
                    "display_name": spec.display_name,
                    "modality": spec.modality,
                    "task_kind": spec.task_kind,
                    "label_schema": spec.label_schema,
                    "label_count": len(spec.labels),
                    "status": spec.status,
                    "data_status": spec.data_status,
                }
                for spec in specs
            ],
            "total": len(specs),
            "note": "data_status 表示真实数据取得情况；synthetic/dummy 不改变该字段。",
        }
    )


@router.get("/{direction_key}", response_model=None)
def get_direction(direction_key: str, user: CurrentUser = Depends(get_current_user)):
    try:
        spec = registry.require(direction_key)
    except DirectionError as exc:
        raise ApiError(404, "direction_not_found", str(exc)) from exc
    return ok(spec.to_public())
