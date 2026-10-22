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
        from app.domains.lease_service import active_worker_leases

        leases = active_worker_leases(db)
        components["workers"] = {"status": "ok", "active_leases": leases}
    except Exception as exc:  # pragma: no cover
        components["workers"] = {"status": "unknown", "detail": type(exc).__name__}

    healthy = components.get("database", {}).get("status") == "ok"
    payload = {"status": "ready" if healthy else "degraded", "components": components}
    return JSONResponse(
        status_code=200 if healthy else 503,
        content=envelope(payload, code="ok" if healthy else "not_ready", message="ready"),
    )
