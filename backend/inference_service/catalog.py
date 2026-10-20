"""Resolve administrator-published registry entries for the isolated inference worker."""
from __future__ import annotations

import asyncio
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from fastapi import HTTPException

from database import AlgorithmCatalog, ModelRegistry
from alarm_flow import flow_model_ids, flow_model_specs
from model_tasks import registry_task


def inference_revision(row):
    """Metadata-only edits advance the catalog without changing live model routing."""
    if row is None:
        return 0
    revision = (row.config_json or {}).get('_inference_revision', row.revision)
    if type(revision) is not int or not 0 <= revision <= row.revision:
        return row.revision
    return revision


def published_models(db, cfg=None):
    from algorithm_catalog import configuration
    import continuous_monitoring as cm
    if cm.runtime is None:
        raise HTTPException(503, "监控服务尚未初始化")
    cfg = cfg or configuration(db)
    requested = [spec for item in cfg["algorithms"] if item.get("enabled") for spec in flow_model_specs(item)]
    if cfg.get("pose_model_id"):
        requested.append((cfg["pose_model_id"], "pose"))
    models, seen = [], set()
    for model_id, kind in requested:
        if not model_id or model_id in seen:
            continue
        seen.add(model_id)
        record = db.get(ModelRegistry, model_id)
        if not record or record.status != "active" or record.source not in ("pretrained", "trained") or record.format != "pt":
            raise HTTPException(422, "请选择有效的已注册 PT 模型")
        task = registry_task(record)
        if task and task not in (('detect', 'segment') if kind=='detect' else (kind,)):
            raise HTTPException(422, "模型任务类型与配置用途不匹配")
        path = str(cm.runtime.api.resolve_managed_model_path(record.path, record))
        models.append({"model_id": str(model_id), "path": path, "kind": kind})
    if len(models)>64:
        raise HTTPException(422, "启用流程的模型总数不能超过 64 个")
    return models


def registered_model_by_id(db, model_id: str, kind="detect"):
    import continuous_monitoring as cm
    record = db.get(ModelRegistry, model_id)
    if not record or record.status != "active" or record.source not in ("pretrained", "trained") or record.format != "pt":
        raise HTTPException(422, "请选择有效的已注册 PT 模型")
    task = registry_task(record)
    if task and task not in (('detect', 'segment') if kind=='detect' else (kind,)):
        raise HTTPException(422, "模型任务类型与配置用途不匹配")
    path = str(cm.runtime.api.resolve_managed_model_path(record.path, record))
    return {"model_id": str(model_id), "path": path, "kind": kind}


def ensure_current_models(db, requested_ids):
    from inference_service.client import inference_client
    from algorithm_catalog import configuration
    row = db.get(AlgorithmCatalog, "global")
    revision = inference_revision(row)
    cfg = configuration(db)
    required = {model_id for item in cfg["algorithms"] if item.get("enabled") for model_id in flow_model_ids(item)}
    if cfg.get("pose_model_id"):
        required.add(cfg["pose_model_id"])
    required.update(item for item in requested_ids if item)
    if inference_switch_lock.locked():
        raise HTTPException(503, "推理服务模型正在切换，请稍后启动监控")
    with inference_client._lock:
        ready = inference_client.revision == revision and not inference_client._switching
        loaded = set(inference_client._models)
    missing = required - loaded
    if ready and not missing:
        return revision
    kinds = {model_id: kind for item in cfg["algorithms"] for model_id,kind in flow_model_specs(item)}
    if cfg.get("pose_model_id"):
        kinds[cfg["pose_model_id"]] = "pose"
    models = published_models(db, cfg)
    if required - {item["model_id"] for item in models}:
        raise HTTPException(422, "所选告警或姿态模型未配置有效模型")
    previous_revision = inference_client.revision
    with inference_client._lock:
        previous_models = list(inference_client._models.values())
    if not inference_switch_lock.acquire(blocking=False):
        raise HTTPException(503, "推理服务模型正在切换，请稍后启动监控")
    try:
        inference_client.begin_switch()
        response = inference_client.reload(revision, models)
        if response.get("state") == "ready":
            inference_client.configure(revision, models)
        elif previous_models:
            inference_client.configure(previous_revision, previous_models)
        else:
            inference_client.begin_switch(False)
    finally:
        inference_switch_lock.release()
    if response.get("state") != "ready":
        raise HTTPException(503, "所选模型尚未加载，请稍后重试")
    return revision


def activate_configuration(db, revision: int):
    from inference_service.client import inference_client
    models = published_models(db)
    response = inference_client.reload(revision, models)
    if response.get("state") != "ready":
        raise HTTPException(503, "模型切换失败，旧推理配置仍在运行")
    inference_client.configure(revision, models)
    return response


def reload_current_configuration():
    from database import SessionLocal, AlgorithmCatalog
    from inference_service.client import inference_client
    with SessionLocal() as db:
        row = db.get(AlgorithmCatalog, "global")
        if row is None:
            return {"state": "empty", "revision": 0}
        revision = inference_revision(row)
        models = published_models(db)
    if not inference_switch_lock.acquire(blocking=False):
        raise RuntimeError("A model switch is already running")
    try:
        result = reuse_loaded_configuration(revision, models)
        if result is None:
            result = inference_client.switch_and_configure(revision, models)
    finally:
        inference_switch_lock.release()
    return result


def reuse_loaded_configuration(revision, models):
    """An API restart can reconnect to an identical healthy worker without reloading it."""
    import json
    import os
    from pathlib import Path
    import requests
    from inference_service.client import inference_client
    try:
        saved = json.loads(Path(os.getenv('INFERENCE_MODEL_STATE_PATH', '/app/temp/inference_service/current-models.json')).read_text(encoding='utf-8'))
        if saved.get('revision') != revision or saved.get('models') != models:
            return None
        response = requests.get(f'{inference_client.base_url}/health', timeout=3)
        response.raise_for_status()
        health = response.json()
        if health.get('state') != 'ready' or health.get('revision') != revision or health.get('error') or health.get('model_count') != len(models):
            return None
    except (OSError, ValueError, TypeError, requests.RequestException):
        return None
    inference_client.configure(revision, models)
    return health


def model_aliases(db):
    return {model["path"]: model["model_id"] for model in published_models(db)}


def current_switch(db):
    row = db.get(AlgorithmCatalog, "global")
    return {"revision": inference_revision(row)}


_switch_lock = threading.RLock()
inference_switch_lock = threading.Lock()
_switches: dict[str, dict] = {}
_switch_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="model-switch")


def enqueue_configuration(revision: int, models: list[dict[str, str]]) -> str:
    from inference_service.client import inference_client
    operation_id = uuid.uuid4().hex
    with _switch_lock:
        for old_id in [key for key, value in _switches.items() if value.get("state") != "loading"]:
            _switches.pop(old_id, None)
        if len(_switches) >= 128:
            raise HTTPException(503, "模型切换队列繁忙，请稍后重试")
        _switches[operation_id] = {"state": "loading", "revision": revision,
                                   "models": models}
    if not inference_switch_lock.acquire(blocking=False):
        with _switch_lock:
            _switches[operation_id] = {"state": "failed", "revision": revision,
                                       "message": "另一个模型切换仍在执行"}
        return operation_id
    with inference_client._lock:
        previous_revision = inference_client._revision
        previous_models = list(inference_client._models.values())
    inference_client.begin_switch()

    def run():
        try:
            from database import SessionLocal
            with SessionLocal() as db:
                current = db.get(AlgorithmCatalog, "global")
                if current is None or inference_revision(current) != revision:
                    if previous_models:
                        inference_client.configure(previous_revision, previous_models)
                    else:
                        inference_client.begin_switch(False)
                    with _switch_lock:
                        _switches[operation_id] = {"state": "failed", "revision": revision,
                                                   "message": "存在更新的模型配置，本次切换已跳过"}
                    return
                with _switch_lock:
                    current = _switches.get(operation_id)
                    if current and current.get("state") == "failed":
                        return
            result = inference_client.reload(revision, models)
            if result.get("state") == "ready":
                if db_revision_is_current(revision):
                    inference_client.configure(revision, models)
                else:
                    with SessionLocal() as db:
                        latest = db.get(AlgorithmCatalog, "global")
                        latest_models = published_models(db) if latest else []
                        latest_revision = inference_revision(latest) if latest else revision
                    if latest and latest_models:
                        inference_client.configure(latest_revision, latest_models)
                    elif previous_models:
                        inference_client.configure(previous_revision, previous_models)
                with _switch_lock:
                    _switches[operation_id] = {"state": "ready", "revision": revision}
            else:
                inference_client.begin_switch(False)
                with _switch_lock:
                    _switches[operation_id] = {"state": "failed", "revision": revision,
                                               "message": "模型加载失败，推理服务保留原配置"}
        except Exception:
            # Restore routing to the last successfully loaded registry mapping.
            try:
                import requests
                health = requests.get(f"{inference_client.base_url}/health", timeout=2).json()
                if health.get("state") == "ready":
                    if previous_models:
                        if health.get("revision") != previous_revision:
                            inference_client.reload(previous_revision, previous_models)
                        inference_client.configure(previous_revision, previous_models)
                    else:
                        inference_client.begin_switch(False)
                else:
                    inference_client.begin_switch(False)
            except Exception:
                pass
            with _switch_lock:
                _switches[operation_id] = {"state": "failed", "revision": revision,
                                           "message": "推理服务未响应，配置已保存，请检查服务状态"}
        finally:
            with inference_client._lock:
                still_switching = inference_client._switching
            if still_switching:
                inference_client.begin_switch(False)
            inference_switch_lock.release()
    _switch_executor.submit(run)
    return operation_id


def db_revision_is_current(revision: int) -> bool:
    from database import SessionLocal
    with SessionLocal() as db:
        row = db.get(AlgorithmCatalog, "global")
        return bool(row and inference_revision(row) == revision)


def switch_status(operation_id: str) -> dict:
    from inference_service.client import inference_client
    with _switch_lock:
        result = _switches.get(operation_id)
    if result is None:
        raise HTTPException(404, "模型切换任务不存在")
    # The worker owns completion and updates routing only after its revision guard.
    if result["state"] == "failed":
        try:
            import requests
            health = requests.get(f"{inference_client.base_url}/health", timeout=2).json()
            if health.get("state") == "ready":
                result = {"state": "failed", "revision": result["revision"],
                          "message": result.get("message", "模型切换失败，仍使用旧模型")}
        except Exception:
            pass
    return {key: value for key, value in result.items() if key != "models"}
