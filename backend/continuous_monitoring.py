"""Persistent monitoring control; browser requests never own a stream task."""
import asyncio
import logging
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, Field
from sqlalchemy import text

from auth_utils import require_any_menu
from database import ContinuousMonitor, JobTask, SessionLocal, engine, get_db
from inference_service.client import inference_client

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/continuous-monitoring", tags=["continuous-monitoring"])
access = require_any_menu("/inference-service", "/inference")
GPU_LOCK = 4924218313798924623
SERVICE_LOCK = 4924218313798924687
ACTIVE_JOBS = {"training", "behavior_analysis", "behavior_temporal_training", "annotation", "model_export", "pose_training", "pose_evaluation"}
runtime = None


def configure_session(db):
    if db.bind.dialect.name == "postgresql":
        db.execute(text("SET LOCAL lock_timeout = '3s'"))
        db.execute(text("SET LOCAL statement_timeout = '5s'"))


def admission_lock(db):
    configure_session(db)
    if db.bind.dialect.name == "postgresql":
        db.execute(text("select pg_advisory_xact_lock(:key)"), {"key": GPU_LOCK})


def visible(row, user):
    if row is None or (row.owner_id != str(user["id"]) and user.get("role") != "admin"):
        raise HTTPException(404, "持续监控任务不存在")
    return row


def snapshot(row):
    now = datetime.now()
    stale = not row.heartbeat_at or (now - row.heartbeat_at).total_seconds() > 20
    config = row.config_json or {}
    # URLs may carry camera credentials. Never return persisted input or raw errors.
    return {
        "id": row.id, "name": row.name, "owner_id": row.owner_id,
        "desired_state": row.desired_state,
        "status": "recovering" if row.desired_state == "running" and stale else row.status,
        "heartbeat_at": row.heartbeat_at.isoformat() if row.heartbeat_at else None,
        "source_count": len(config.get("video_sources", [])),
        "max_fps": config.get("max_fps"),
        "events": (config.get("shared_behavior_config") or {}).get("enabled_events", []),
        "result": row.result_json or {},
    }


class StartRequest(BaseModel):
    name: str = Field(default="持续监控", min_length=1, max_length=100)
    config: dict


@router.get("")
def list_monitors(db=Depends(get_db), user=Depends(access)):
    # Each account has one reusable plan. Admins may address another plan by ID.
    rows = db.query(ContinuousMonitor).filter(ContinuousMonitor.owner_id == str(user["id"])).all()
    return {"items": [snapshot(row) for row in rows]}


def check_admission(db, row, config=None):
    other = db.query(ContinuousMonitor.id).filter(
        ContinuousMonitor.id != row.id,
        ((ContinuousMonitor.desired_state == "running") | (ContinuousMonitor.status == "stopping")),
    ).first()
    busy = db.query(JobTask).filter(
        JobTask.type.in_(ACTIVE_JOBS), JobTask.status.in_({"running", "cancelling"}),
    ).all()
    if other:
        raise HTTPException(409, "已有持续监控运行，请先停止或等待完成")
    if busy:
        if config is None or any(job.type != 'training' or job.status != 'running' for job in busy):
            raise HTTPException(409, "已有 GPU 作业运行，请先停止或等待完成")
        from monitoring_gpu_admission import admit_training
        config['training_gpu_guard'] = admit_training(config, db, row.owner_id)
    elif config is not None:
        config.pop('training_gpu_guard', None)


def reserve(db, row, config=None):
    check_admission(db, row, config)
    row.desired_state, row.status = "running", "starting"
    row.heartbeat_at = None


@router.post("/start")
def start_monitor(body: StartRequest, db=Depends(get_db), user=Depends(access)):
    admission_lock(db)
    row = db.query(ContinuousMonitor).filter(ContinuousMonitor.owner_id == str(user["id"])).with_for_update().first()
    if row and row.desired_state == "running":
        return snapshot(row)  # Retry after an uncertain HTTP result cannot duplicate work.
    if row and row.status == "stopping":
        raise HTTPException(409, "任务正在释放资源，请稍后启动")
    config = runtime.validate(body.config, db, user)
    if row is None:
        row = ContinuousMonitor(owner_id=str(user["id"]), owner_username=user["username"])
        db.add(row)
        db.flush()
    row.name, row.config_json, row.result_json = body.name, config, {}
    reserve(db, row)
    db.commit()
    return snapshot(row)


@router.post("/{task_id}/stop")
def stop_monitor(task_id: str, db=Depends(get_db), user=Depends(access)):
    admission_lock(db)
    row = visible(db.query(ContinuousMonitor).filter(ContinuousMonitor.id == task_id).with_for_update().first(), user)
    if row.desired_state == "running":
        row.desired_state, row.status = "stopped", "stopping"
        db.commit()
    return snapshot(row)


@router.post("/{task_id}/resume")
def resume_monitor(task_id: str, db=Depends(get_db), user=Depends(access)):
    admission_lock(db)
    row = visible(db.query(ContinuousMonitor).filter(ContinuousMonitor.id == task_id).with_for_update().first(), user)
    if (row.config_json or {}).get('per_camera'):
        raise HTTPException(422, '请在算法配置页逐路启动摄像头')
    if row.desired_state == "running":
        return snapshot(row)
    if row.status == "stopping":
        raise HTTPException(409, "任务正在释放资源，请稍后继续")
    # Recheck model and source access when explicitly resuming saved settings.
    row.config_json = runtime.validate(row.config_json, db, user)
    reserve(db, row)
    db.commit()
    return snapshot(row)


class MonitorSupervisor:
    def __init__(self, adapter, session_factory=SessionLocal, db_engine=engine):
        self.adapter, self.session_factory, self.engine = adapter, session_factory, db_engine
        self.stop_event = asyncio.Event()
        self.active = set()
        self.lease = None
        self.lease_pid = None

    def lease_ok(self):
        if self.lease is None:
            connection = self.engine.connect()
            try:
                connection.execute(text("SET statement_timeout = '5s'"))
                acquired = connection.execute(text("select pg_try_advisory_lock(:key)"), {"key": SERVICE_LOCK}).scalar()
                self.lease_pid = connection.execute(text("select pg_backend_pid()")).scalar()
                connection.commit()
                if not acquired:
                    connection.close()
                    return False
                self.lease = connection
            except Exception:
                connection.close()
                raise
        pid = self.lease.execute(text("select pg_backend_pid()")).scalar()
        self.lease.commit()
        if pid != self.lease_pid:
            raise RuntimeError("monitor supervisor lease connection changed")
        return True

    def release(self):
        if self.lease is not None:
            try:
                if not self.lease.invalidated:
                    self.lease.execute(text("select pg_advisory_unlock(:key)"), {"key": SERVICE_LOCK})
                    self.lease.commit()
            except Exception:
                self.lease.invalidate()
            finally:
                self.lease.close()
                self.lease = None

    async def stop_all(self):
        # Never release singleton ownership while native inference/decoding is alive.
        while self.active:
            for task_id in list(self.active):
                if await self.adapter.stop(task_id):
                    self.active.remove(task_id)
            if self.active:
                await asyncio.sleep(1)

    async def tick(self):
        with self.session_factory() as db:
            configure_session(db)
            rows = db.query(ContinuousMonitor).filter(
                (ContinuousMonitor.desired_state == "running") | (ContinuousMonitor.status == "stopping")
            ).all()
            # Check memory outside the DB transaction; CUDA is never initialized here.
            guarded = [r.id for r in rows if r.desired_state == 'running' and (r.config_json or {}).get('training_gpu_guard')]
        if guarded:
            from monitoring_gpu_admission import guard_reason
            reason = await asyncio.to_thread(guard_reason, inference_client)
            if reason:
                with self.session_factory() as db:
                    configure_session(db)
                    for task_id in guarded:
                        row = db.query(ContinuousMonitor).filter(ContinuousMonitor.id == task_id).with_for_update().first()
                        if row and row.desired_state == 'running' and (row.config_json or {}).get('training_gpu_guard'):
                            from copy import deepcopy
                            cfg = deepcopy(row.config_json)
                            for source in cfg.get('video_sources', []):
                                source['desired_state'] = 'stopped'
                            row.config_json = cfg
                            row.desired_state, row.status = 'stopped', 'stopping'
                            row.result_json = {**(row.result_json or {}), 'gpu_guard_reason': reason}
                    db.commit()
        with self.session_factory() as db:
            configure_session(db)
            rows = db.query(ContinuousMonitor).filter(
                (ContinuousMonitor.desired_state == "running") | (ContinuousMonitor.status == "stopping")
            ).all()
            plans = [(r.id, r.desired_state, r.config_json, r.owner_id, r.owner_username) for r in rows]
        wanted = {p[0] for p in plans if p[1] == "running"}
        for task_id in list(self.active - wanted):
            if await self.adapter.stop(task_id):
                self.active.remove(task_id)
        for task_id, desired, config, owner_id, username in plans:
            if desired == "running":
                if task_id in self.active and self.adapter.finished(task_id):
                    if not await self.adapter.stop(task_id):
                        continue
                    self.active.remove(task_id)
                if task_id not in self.active:
                    self.active.add(task_id)  # Includes partially started resources in cleanup.
                    await self.adapter.start(task_id, config, {"id": owner_id, "username": username})
                elif hasattr(self.adapter, 'reconcile'):
                    await self.adapter.reconcile(task_id, config, {"id": owner_id, "username": username})
            result = self.adapter.snapshot(task_id) if task_id in self.active else {}
            with self.session_factory() as db:
                configure_session(db)
                row = db.query(ContinuousMonitor).filter(ContinuousMonitor.id == task_id).with_for_update().first()
                if row is None:
                    continue
                # An HTTP stop racing this tick remains authoritative.
                if row.desired_state == "stopped":
                    row.status = "stopping" if task_id in self.active else "stopped"
                else:
                    active_positions = [i for i, s in enumerate(config['video_sources']) if s.get('desired_state', 'running') == 'running']
                    cells = [(result.get('cells') or {}).get(i, {}) for i in active_positions]
                    healthy = bool(cells) and all(
                        cell.get("status") == "live" and cell.get("updated_at")
                        and (datetime.now() - datetime.fromisoformat(cell["updated_at"])).total_seconds() < 30
                        for cell in cells
                    )
                    row.status = "running" if healthy else "degraded"
                    row.result_json = result
                row.heartbeat_at = datetime.now()
                db.commit()

    async def run(self):
        failures = 0
        try:
            while not self.stop_event.is_set():
                try:
                    if self.lease_ok():
                        await self.tick()
                        failures = 0
                except Exception:
                    failures = min(failures + 1, 4)
                    logger.exception("Continuous monitoring supervisor recovery")
                    await self.stop_all()
                    self.release()
                try:
                    await asyncio.wait_for(self.stop_event.wait(), timeout=min(60, 5 * 2 ** failures))
                except asyncio.TimeoutError:
                    pass
        finally:
            await self.stop_all()
            self.release()


def install(app, adapter):
    global runtime
    runtime = adapter
    supervisor = MonitorSupervisor(adapter)
    app.include_router(router)
    app.include_router(router, prefix="/api")
    from monitoring_workspace import router as workspace_router
    app.include_router(workspace_router)
    app.include_router(workspace_router, prefix='/api')
    from monitoring_webrtc import router as webrtc_router
    app.include_router(webrtc_router)
    app.include_router(webrtc_router, prefix='/api')
    from algorithm_catalog import router as catalog_router
    app.include_router(catalog_router)
    app.include_router(catalog_router, prefix='/api')
    from inference_service.router import router as inference_router
    app.include_router(inference_router)

    @app.on_event("startup")
    async def start():
        try:
            from inference_service.catalog import reload_current_configuration
            await asyncio.to_thread(reload_current_configuration)
        except Exception:
            logger.exception("Initial isolated inference model load failed")
        app.state.inference_service_client = inference_client
        app.state.monitor_supervisor_task = asyncio.create_task(supervisor.run())

    @app.on_event("shutdown")
    async def stop():
        supervisor.stop_event.set()
        await app.state.monitor_supervisor_task
