"""Authenticated inference service status and reload orchestration routes."""
from fastapi import APIRouter, Depends, HTTPException

from auth_utils import require_admin
from inference_service.catalog import enqueue_configuration, published_models, inference_revision
from database import AlgorithmCatalog

router = APIRouter(tags=["inference-service"])


@router.get("/system/inference-service/status")
@router.get("/api/system/inference-service/status")
def status(admin=Depends(require_admin)):
    from inference_service.client import inference_client
    import requests
    try:
        value = requests.get(f"{inference_client.base_url}/health", timeout=2).json()
        with inference_client._lock:
            busy = inference_client._switching
            current_revision = inference_client._revision
        return {"state": "loading" if busy else value.get("state"),
                "revision": value.get("revision"), "model_count": value.get("model_count"),
                "active_revision": current_revision}
    except Exception:
        raise HTTPException(503, "独立推理服务暂不可用") from None


@router.post("/system/inference-service/reload")
@router.post("/api/system/inference-service/reload")
def reload(admin=Depends(require_admin)):
    from database import SessionLocal
    with SessionLocal() as db:
        row = db.get(AlgorithmCatalog, "global")
        if row is None:
            raise HTTPException(409, "请先保存告警算法配置")
        revision = inference_revision(row)
        models = published_models(db)
    return {"operation_id": enqueue_configuration(revision, models), "state": "loading"}
