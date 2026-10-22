"""v1 健康检查：live 不依赖 DB/GPU；ready 分项报告依赖状态。"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from fastapi.responses import JSONResponse
from sqlalchemy import text
from sqlalchemy.orm import Session

from app.core.errors import envelope
from app.db.session import get_db

router = APIRouter(prefix="/health", tags=["health"])


@router.get("/live", response_model=None)
def live():
    from app.version import build_info

    return envelope({"status": "live", **build_info()}, message="alive")


@router.get("/ready", response_model=None)
def ready(db: Session = Depends(get_db)):
    components: dict[str, dict] = {}
    try:
        db.execute(text("SELECT 1"))
        components["database"] = {"status": "ok"}
    except Exception as exc:  # pragma: no cover - 取决于运行环境
        components["database"] = {"status": "error", "detail": type(exc).__name__}

    try:
        from app.domains.lease_service import worker_roles
        from app.core.config import get_settings

        roles = worker_roles(db)
        missing = [role for role in get_settings().required_worker_roles if not roles.get(role)]
        components["workers"] = {"status": "missing" if missing else "ok", "roles": roles,
                                  "active_leases": sum(roles.values()), "missing": missing}
    except Exception as exc:  # pragma: no cover
        components["workers"] = {"status": "unknown", "detail": type(exc).__name__}

    healthy = all(components.get(key, {}).get("status") == "ok" for key in ("database", "workers"))
    payload = {"status": "ready" if healthy else "degraded", "components": components}
    return JSONResponse(
        status_code=200 if healthy else 503,
        content=envelope(payload, code="ok" if healthy else "not_ready", message="ready"),
    )
