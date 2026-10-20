#!/usr/bin/env python3
"""
YOLO训练和推理API服务
提供完整的YOLO模型训练、推理、导出等功能的RESTful API接口

主要功能模块:
- 模型训练管理: 提交、监控、停止训练任务
- 数据集管理: 上传、解析、统计数据集
- 实时推理: 支持 WebCam、RTSP、视频文件等多种源
- 九宫格监控: 支持多路视频并发推理
- 结果管理: 保存、导出、清理推理结果
"""

# --- 核心依赖导入 ---
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, WebSocket, WebSocketDisconnect, Form, Request, Depends
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from starlette.websockets import WebSocketState
from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Mapping, Tuple
import os
import json
import uuid
import shutil
import subprocess
from pathlib import Path
import asyncio
import logging
from logging.handlers import RotatingFileHandler
from datetime import datetime
import zipfile
import tempfile
import time
import base64
import copy
import re
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageOps
import io
import mimetypes
import ipaddress
from urllib.parse import urlparse
from sqlalchemy import func, or_, text

try:
    import psutil
except ImportError:  # psutil 未安装时使用 /proc 兜底
    psutil = None

# --- 业务逻辑组件导入 ---
from yolo_trainer import YOLOTrainer
from training_worker_health import read_worker_heartbeat, training_worker_is_available
from training_validation import can_reuse_prepared_segmentation_dataset, serialize_training_submission
from gpu_runtime import gpu_runtime_is_ready, probe_gpu, training_device_requires_gpu
from gpu_probe_service import WAIT_MESSAGE, GPUUnavailable, report_gpu_fault, require_gpu

# --- 内部工具类导入 ---
from utils import (
    ImageProcessor,
    VideoProcessor,
    TaskManager,
    DatasetManager,
    InferenceManager,
    FileManager,
    ResultManager,
    constants
)
from utils.archive_utils import ArchiveLimits, ArchiveSafetyError, safe_extract_zip, save_upload_file_limited
from http_access import is_public_http_route
from grid_batch_inference import GridBatchInferenceScheduler

# --- 系统配置与日志初始化 ---
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# --- 模型目录兼容配置 ---
# 说明：前端 /models 下拉框需要同时读取 data/models/pretrained、data/models 和 data/runs。
# 这几个变量必须在 list_models() 使用前定义，避免 PRETRAINED_MODELS_DIR 未定义导致 /models 500。
BACKEND_DIR = Path(__file__).resolve().parent
if BACKEND_DIR.name.lower() == "backend":
    PROJECT_ROOT = BACKEND_DIR.parent.resolve()
else:
    PROJECT_ROOT = BACKEND_DIR.resolve()

DATA_DIR = Path(os.getenv("DEEPVISION_DATA_DIR", str(PROJECT_ROOT / "data"))).resolve()
REALTIME_STREAM_EVENT_LOG_PATH = Path(
    os.getenv(
        "REALTIME_STREAM_EVENT_LOG_FILE",
        str(DATA_DIR / "logs" / "realtime_stream_events.log"),
    )
).resolve()
MODELS_DIR = DATA_DIR / "models"
PRETRAINED_MODELS_DIR = MODELS_DIR / "pretrained"
BEHAVIOR_MODELS_DIR = MODELS_DIR / "behavior"
RUNS_DIR = DATA_DIR / "runs"
TRAINING_DATASETS_DIR = Path(
    os.getenv("DEEPVISION_TRAINING_DATASETS_DIR", str(DATA_DIR / "training_datasets"))
).resolve()
TRAINING_WORKSPACES_DIR = Path(
    os.getenv("DEEPVISION_TRAINING_WORKSPACES_DIR", str(DATA_DIR / "training_workspaces"))
).resolve()
TRAINING_LOG_ROOT = Path(os.getenv("TRAINING_LOG_ROOT", str(PROJECT_ROOT / "temp" / "training_logs"))).resolve()


def configure_realtime_stream_event_logger(log_path: Optional[Path] = None) -> logging.Logger:
    """Configure the durable, independently rotated realtime-stream event log."""
    target_path = Path(log_path or REALTIME_STREAM_EVENT_LOG_PATH).resolve()
    target_path.parent.mkdir(parents=True, exist_ok=True)
    event_logger = logging.getLogger("deepvision.realtime_stream_events")
    event_logger.setLevel(logging.INFO)
    event_logger.propagate = False

    for handler in event_logger.handlers:
        if isinstance(handler, RotatingFileHandler) and Path(handler.baseFilename).resolve() == target_path:
            return event_logger

    handler = RotatingFileHandler(
        target_path,
        maxBytes=10 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    event_logger.addHandler(handler)
    return event_logger


realtime_stream_event_logger = configure_realtime_stream_event_logger()


def write_realtime_stream_event_log(event: Dict[str, Any]) -> None:
    """Write a structured source outage/recovery event without source credentials."""
    record = {
        "event": event.get("event"),
        "monitor_type": event.get("monitor_type", "realtime"),
        "session_id": event.get("session_id"),
        "grid_id": event.get("grid_id"),
        "grid_position": event.get("grid_position"),
        "source_type": event.get("source_type"),
        "reason": event.get("reason"),
        "reconnect_count": event.get("reconnect_count", 0),
        "timestamp": event.get("timestamp", time.time()),
    }
    message = json.dumps(record, ensure_ascii=False, separators=(",", ":"))
    try:
        if event.get("level") == "error":
            realtime_stream_event_logger.error(message)
        else:
            realtime_stream_event_logger.info(message)
    except Exception as exc:
        logger.error("写入实时视频流独立日志失败: %s", exc)

PRETRAINED_MODEL_NAMES = (
    "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
    "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
    "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt",
    "yolov8n-pose.pt", "yolov8s-pose.pt", "yolov8m-pose.pt", "yolov8l-pose.pt", "yolov8x-pose.pt",
    "yolo11n-pose.pt", "yolo11s-pose.pt", "yolo11m-pose.pt", "yolo11l-pose.pt", "yolo11x-pose.pt",
    "yolo26n-pose.pt", "yolo26s-pose.pt", "yolo26m-pose.pt", "yolo26l-pose.pt", "yolo26x-pose.pt",
    "yoloe-26n-seg.pt", "yoloe-26s-seg.pt", "yoloe-26m-seg.pt", "yoloe-26l-seg.pt", "yoloe-26x-seg.pt",
)
PRETRAINED_MODEL_NAME_SET = {name.lower() for name in PRETRAINED_MODEL_NAMES}

# 确保目录存在；同时把 constants 中的旧目录同步到 data/*，让旧代码也能访问统一目录。
for _dir in [
    DATA_DIR,
    MODELS_DIR,
    PRETRAINED_MODELS_DIR,
    BEHAVIOR_MODELS_DIR,
    RUNS_DIR,
    TRAINING_DATASETS_DIR,
    TRAINING_WORKSPACES_DIR,
]:
    try:
        _dir.mkdir(parents=True, exist_ok=True)
    except Exception:
        pass

try:
    constants.MODELS_DIR = str(MODELS_DIR)
    constants.RUNS_DIR = str(RUNS_DIR)
    constants.TRAINING_DATASETS_DIR = str(TRAINING_DATASETS_DIR)
except Exception:
    pass


def _safe_relative_to(path: Path, root: Path) -> bool:
    try:
        Path(path).resolve().relative_to(Path(root).resolve())
        return True
    except Exception:
        return False


def is_protected_pretrained_model(model_path: Any, resolved_path: Optional[Path] = None) -> bool:
    """判断模型是否属于预训练模型，避免前端绕过后直接删除基础权重。"""
    raw = str(model_path or "").strip().replace("\\", "/")
    filename = Path(raw).name.lower()

    if filename in PRETRAINED_MODEL_NAME_SET:
        return True

    raw_parts = [part.lower() for part in raw.split("/") if part]
    if "pretrained" in raw_parts:
        return True

    path_candidates = []
    if resolved_path is not None:
        path_candidates.append(Path(resolved_path))
    if raw:
        path_candidates.append(Path(raw))

    pretrained_roots = [
        PRETRAINED_MODELS_DIR,
        Path("/app/data/models/pretrained"),
        Path("/app/models/pretrained"),
    ]

    for candidate in path_candidates:
        for root in pretrained_roots:
            if _safe_relative_to(candidate, root):
                return True

    return False

# 创建FastAPI应用
app = FastAPI(
    title="YOLO Training & Inference API",
    description="完整的YOLO模型训练和推理API服务",
    version="1.0.0"
)

# 允许前端页面从 localhost:3000 读取后端图片和接口。
# 没有这个配置时，浏览器直接 fetch http://localhost:8000/image/xxx 会报 Failed to fetch。
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- 权限管理系统：PostgreSQL + JWT 登录鉴权 ---
# 说明：
# 1. 登录、健康检查、文档、静态图片资源保持公开；
# 2. 其它 HTTP API 默认要求携带 Authorization: Bearer <token>；
# 3. WebSocket 端点暂不强制鉴权，避免影响现有实时监控/九宫格连接。
from annotation_draft_batches import import_draft_batch, latest_draft_batch
from database import AnnotationDraft, AnnotationDraftAsset, BehaviorEvent, BehaviorReviewSample, DatasetAsset, DatasetResource, DatasetVersion, JobEvent, JobTask, ModelRegistry, Role, SessionLocal, ensure_system_seed_data, get_db, init_db
from training_model_test import describe_training_test_model, resolve_training_test_model
from dataset_access import (
    backfill_dataset_resources,
    can_read_dataset,
    ensure_dataset_resource,
    get_dataset_resource,
    is_admin_user,
    readable_dataset_ids,
    require_dataset_manage,
    require_dataset_read,
    select_upload_role,
)
from dataset_versioning import (
    analyze_training_version,
    class_subset_plan,
    create_annotation_version,
    create_initial_version,
    delete_dataset_version_records,
    derive_class_subset,
    export_training_version,
    resolve_fixed_training_class_map,
    summarize_annotation_assets,
)
from annotation_class_aliases import annotation_class_key, class_name_ids, resolve_model_page_classes, routing_page_class_ids
from image_size_processing import SUPPORTED_IMAGE_SIZES, optimal_image_size_from_counts, TILING_POLICY_VERSION, resolve_tiling_profile
from dataset_versioning import summarize_mobilephone_export_quality, filter_annotations_by_geometry
from training_optimization import (
    normalize_training_config,
    recommended_training_config,
    resolve_training_profile,
)
from training_acceptance import evaluate_yoloe_candidate
from model_capabilities import model_annotation_geometry
from annotation_session_classes import (
    checkpoint_class_names, draft_dataset_view, isolated_draft, isolated_version,
    normalize_class_context,
    validate_session_annotation_ids,
    session_base_annotations,
    manifest_frontend_annotations,
)
from sqlalchemy.exc import IntegrityError
from training_workspace import cleanup_training_workspace
from annotation_export import (
    build_asset_path_indexes,
    frontend_annotation_to_manifest,
    normalize_class_names,
    ordered_class_names,
    resolve_ai_class_id,
    resolve_result_relative_path,
)
from dataset_deletion import purge_staged_directory, restore_staged_directory, stage_directory_for_deletion
from dataset_merge import DatasetMergeError, merge_original_dataset_assets
from image_augmentation import normalize_augmentation_operations as normalize_fine_augmentation_operations
from job_store import (
    ACTIVE_STATUSES,
    append_job_event,
    create_job,
    delete_terminal_job,
    get_job_health_snapshot,
    is_cancel_requested,
    recover_active_jobs,
    request_job_cancel,
    update_job,
)
from model_registry import (
    build_model_lineage,
    filter_available_model_records,
    hydrate_trained_model_payload,
    merge_preserved_model_metadata,
    model_supports_text_prompts,
    normalize_visible_model_payload,
    reconcile_models,
    sha256_file,
    trained_model_metadata,
)
from model_registry import register_model_file
from model_export import convert_pt_to_onnx
from model_lifecycle import (
    ModelDeletionRecoveryError,
    delete_model_and_linked_training_task,
    delete_training_task_and_models,
)
from model_usage import is_model_in_active_use
from annotation_drafts import (
    ANNOTATION_KINDS,
    delete_draft,
    delete_drafts_for_dataset,
    get_draft,
    get_draft_by_id,
    get_draft_for_update,
    normalize_annotation_kind,
    sanitize_draft_payload,
    serialize_draft,
    upsert_draft,
    user_id as draft_user_id,
)
from auth_utils import ensure_default_admin, get_current_user_from_authorization, require_admin, require_any_menu
from auth_routes import router as auth_router
from admin_routes import router as admin_router
from system_log_routes import router as system_log_router
from system_fault_routes import router as system_fault_router
from system_admin_routes import router as system_admin_router
from alarm_push import router as alarm_push_router
from behavior.event_routes import router as behavior_event_router
from behavior.event_store import (
    cleanup_expired_behavior_events,
    persist_behavior_transitions,
    persist_behavior_transitions_resilient,
    replay_spooled_behavior_events,
    resolve_behavior_profile,
    update_behavior_evidence_resilient,
)
from behavior.verifier_runtime import apply_realtime_verifiers, preload_behavior_verifier, prepare_realtime_feature_vectors
from behavior.feedback_mapping import select_binary_behavior_samples
from behavior.batch_jobs import run_behavior_analysis_payload
from behavior.realtime_engine import normalize_behavior_config
from behavior.realtime_queue import (
    BehaviorFifoFrameScheduler,
    resolve_behavior_max_frame_age_ms,
    resolve_behavior_scheduler_interval_ms,
)
from behavior.runtime import behavior_runtime_manager
from behavior.temporal_fusion import (
    DEFAULT_CLIP_LENGTH as TEMPORAL_DEFAULT_CLIP_LENGTH,
    DEFAULT_IMAGE_SIZE as TEMPORAL_DEFAULT_IMAGE_SIZE,
    DEFAULT_THRESHOLDS as TEMPORAL_DEFAULT_THRESHOLDS,
    MODEL_FAMILY as TEMPORAL_MODEL_FAMILY,
    TEMPORAL_EVENT_TYPES,
    load_temporal_model,
)
from behavior_worker_health import read_behavior_worker_heartbeat

behavior_retention_task: Optional[asyncio.Task] = None
behavior_event_routes_registered = False
model_export_task_handles: Dict[str, asyncio.Task] = {}


def ensure_behavior_event_routes_registered() -> None:
    """Register a router that may still have been initializing during import."""
    global behavior_event_routes_registered
    if behavior_event_routes_registered:
        return
    app.include_router(behavior_event_router)
    behavior_event_routes_registered = True


async def run_behavior_retention_loop() -> None:
    last_cleanup = time.monotonic()
    while True:
        await asyncio.sleep(60)
        try:
            await asyncio.to_thread(replay_spooled_behavior_events)
            if time.monotonic() - last_cleanup >= 24 * 60 * 60:
                await asyncio.to_thread(cleanup_expired_behavior_events, retention_days=90)
                last_cleanup = time.monotonic()
        except Exception as exc:
            logger.warning("行为告警补偿重放或保留期清理失败，将在下次周期重试: %s", exc)


@app.middleware("http")
async def jwt_auth_middleware(request: Request, call_next):
    # This one internal receiver verifies an HMAC over timestamp + full request body.
    if request.method.upper() == 'POST' and request.url.path == '/integration/face-test-receiver':
        return await call_next(request)
    if request.method.upper() == "OPTIONS" or is_public_http_route(request.method, request.url.path):
        return await call_next(request)

    try:
        current_user = get_current_user_from_authorization(request.headers.get("Authorization"))
        request.state.current_user = current_user
    except HTTPException as exc:
        return JSONResponse(status_code=exc.status_code, content={"detail": exc.detail})
    except Exception as exc:
        logger.exception("权限校验失败")
        return JSONResponse(status_code=401, content={"detail": f"权限校验失败: {str(exc)}"})

    return await call_next(request)


@app.middleware("http")
async def record_unhandled_http_fault(request: Request, call_next):
    try:
        response = await call_next(request)
        if response.status_code >= 500:
            from system_faults import record_fault
            try:
                record_fault(service="backend", severity="ERROR", category="http_5xx",
                             message=f"{request.method} {request.url.path}: HTTP {response.status_code}")
            except Exception:
                logger.exception("系统故障日志写入失败")
        return response
    except Exception as exc:
        from system_faults import record_fault
        import traceback
        try:
            record_fault(service="backend", severity="ERROR", category="http_exception",
                         message=f"{request.method} {request.url.path}: {type(exc).__name__}",
                         details="".join(traceback.format_exception(exc)))
        except Exception:
            logger.exception("系统故障日志写入失败")
        raise


@app.on_event("startup")
async def startup_auth_system():
    global behavior_retention_task
    ensure_behavior_event_routes_registered()
    init_db()
    ensure_default_admin()
    ensure_system_seed_data()
    db = SessionLocal()
    try:
        backfill_dataset_resources(db)
        migrated_behavior_reviews = migrate_legacy_behavior_review_samples(db)
        recovery = recover_active_jobs(db)
    finally:
        db.close()
    if migrated_behavior_reviews:
        logger.info("已迁移 %s 条历史行为复核样本到 PostgreSQL", migrated_behavior_reviews)
    if recovery["annotation_interrupted"]:
        await resume_interrupted_annotation_jobs(recovery["annotation_interrupted"])
    try:
        await asyncio.to_thread(replay_spooled_behavior_events)
        await asyncio.to_thread(cleanup_expired_behavior_events, retention_days=90)
    except Exception as exc:
        logger.warning("行为事件保留期清理失败，将在下次启动重试: %s", exc)
    if behavior_retention_task is None or behavior_retention_task.done():
        behavior_retention_task = asyncio.create_task(run_behavior_retention_loop())
    logger.info("权限管理系统初始化完成，默认管理员账号已检查")


@app.on_event("shutdown")
async def shutdown_behavior_retention() -> None:
    global behavior_retention_task
    if behavior_retention_task is not None:
        behavior_retention_task.cancel()
        try:
            await behavior_retention_task
        except asyncio.CancelledError:
            pass
        behavior_retention_task = None


app.include_router(auth_router)
from pose_training.routes import router as pose_training_router
app.include_router(pose_training_router)
app.include_router(pose_training_router, prefix="/api")
app.include_router(admin_router)
app.include_router(system_log_router)
app.include_router(system_fault_router)
app.include_router(system_admin_router)
app.include_router(alarm_push_router)
from monitoring_policy import filter_detections as filter_monitor_detections, filter_events as filter_monitor_events
from recorder_config import router as recorder_config_router
app.include_router(recorder_config_router)
from face_capture.routes import register as register_face_capture
register_face_capture(app, live_context=lambda: (video_processor, behavior_session_contexts))

# --- 全局工具类实例初始化 ---
task_manager = TaskManager()
dataset_manager = DatasetManager()
training_dataset_manager = DatasetManager(str(TRAINING_DATASETS_DIR))
video_processor = VideoProcessor()
inference_manager = InferenceManager()
image_processor = ImageProcessor()
file_manager = FileManager()
result_manager = ResultManager()

# --- 静态文件与目录配置 ---
file_manager.ensure_dir(constants.STATIC_DIR)
file_manager.ensure_dir(constants.RESULTS_DIR)

# 统一静态目录。
# 行为分析结果、上传视频、截图等都写入 constants.STATIC_DIR，
# 因此 StaticFiles 也必须挂载同一个目录，不能硬编码为 "static"，
# 否则服务器/容器工作目录变化时会出现“磁盘有文件，但浏览器 404”。
STATIC_ROOT = Path(constants.STATIC_DIR).resolve()
STATIC_ROOT.mkdir(parents=True, exist_ok=True)

# 图片上传/标注页独立图片目录。
# 容器内路径为 /app/data/image；配合 docker-compose 挂载后，对应宿主机 ./data/image。
IMAGE_UPLOAD_DIR = Path("data") / "image"
IMAGE_UPLOAD_URL_PREFIX = "/image"
IMAGE_FILE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
DATASETS_ROOT = Path(constants.DATASETS_DIR).resolve()
DATASETS_ROOT.mkdir(parents=True, exist_ok=True)
TRAINING_DATASETS_DIR.mkdir(parents=True, exist_ok=True)

app.mount("/static", StaticFiles(directory=str(STATIC_ROOT)), name="static")

# 兼容前端通过 /api/static/... 访问行为分析结果、视频、截图。
app.mount("/api/static", StaticFiles(directory=str(STATIC_ROOT)), name="api_static")

app.mount(IMAGE_UPLOAD_URL_PREFIX, StaticFiles(directory=str(IMAGE_UPLOAD_DIR)), name="uploaded_images")
app.mount("/api/image", StaticFiles(directory=str(IMAGE_UPLOAD_DIR)), name="api_uploaded_images")
# --- 全局状态管理变量 ---
uploaded_datasets = {}       # 上传的数据集信息(向后兼容)
IMAGE_UPLOAD_MAX_BYTES = int(os.getenv("IMAGE_UPLOAD_MAX_BYTES", str(20 * 1024**2)))
BATCH_INFERENCE_MAX_TOTAL_BYTES = int(os.getenv("BATCH_INFERENCE_MAX_TOTAL_BYTES", str(100 * 1024**2)))
IMAGE_MAX_PIXELS = int(os.getenv("IMAGE_MAX_PIXELS", str(50_000_000)))
realtime_connections = {}    # WebSocket连接管理 (session_id -> websocket)
realtime_stats = {}          # 实时推理统计 (session_id -> stats)
video_sources = {}           # 视频源管理 (source_id -> config)
grid_sessions = {}           # 九宫格会话管理 (grid_id -> session_info)
grid_connections = {}        # 九宫格WebSocket连接 (grid_id -> websocket)
grid_stream_tasks = {}       # 九宫格后台任务 (grid_id -> {session_id -> asyncio.Task})
grid_send_locks = {}         # 九宫格WebSocket发送锁 (grid_id -> asyncio.Lock)
grid_stop_events = {}        # 九宫格停止信号 (grid_id -> asyncio.Event)
grid_inference_schedulers = {}  # grid_id -> GridBatchInferenceScheduler
grid_cleanup_tasks = {}      # grid_id -> 延迟回收 asyncio.Task

# 实时监控配置运行态
# 这些状态用于保证前端配置变更后，后端实时推理循环能读取并使用最新配置，
# 而不是只在启动视频流时读取一次旧配置。
realtime_session_configs = {}      # session_id -> 当前 RealtimeConfig 字典
realtime_frame_counters = {}       # session_id -> 已接收/读取帧计数，用于 frame_skip
realtime_last_process_time = {}    # session_id -> 上一帧实际推理时间，用于 max_fps 限流
realtime_stream_tasks = {}         # session_id -> 服务器端视频流 asyncio.Task
realtime_send_locks = {}           # session_id -> asyncio.Lock，串行化控制消息与推理结果发送
realtime_frame_delivery = {}       # session_id -> 浏览器图像帧确认/背压状态
realtime_stop_events = {}          # session_id -> asyncio.Event，统一停止信号
realtime_cleanup_locks = {}        # session_id -> asyncio.Lock，保证清理幂等
realtime_cleanup_tasks = {}        # session_id -> 延迟回收 asyncio.Task
realtime_session_states = {}       # session_id -> active/stopping/stopped
behavior_session_contexts = {}     # session_key -> owner/mode/source context
behavior_resolved_configs = {}     # session_key -> profile/request merge cache
behavior_frame_schedulers = {}     # session_key -> bounded FIFO behavior scheduler

# 后端批量 AI 标注任务状态。
# 目的：图片数量较多时，不让前端逐张上传 base64 调用 AI 标注接口，
# 而是在后端直接扫描 ./data/image 并顺序推理，前端只轮询进度。
annotation_batch_task_handles: Dict[str, asyncio.Task] = {}
annotation_export_task_handles: Dict[str, asyncio.Task] = {}
dataset_augmentation_task_handles: Dict[str, asyncio.Task] = {}
annotation_batch_inference_lock = asyncio.Lock()



# --- 辅助功能函数 ---


# --- 上传文件命名工具 ---
def sanitize_original_filename(filename: Any, fallback: str = "uploaded_file") -> str:
    """
    保留用户上传时的文件名，同时去掉路径和系统不安全字符。

    说明：
    1. 不再主动改成 video_<uuid>.mp4、image_0.jpg 这类名称；
    2. 只做必要的安全清理，防止路径穿越和 Windows/Linux 非法字符；
    3. 保留中文、英文、数字、空格等用户可识别内容。
    """
    raw_name = Path(str(filename or fallback)).name.strip()
    if not raw_name or raw_name in {".", ".."}:
        raw_name = fallback

    # Windows/Linux 常见非法或危险字符，统一替换为下划线。
    for ch in ['<', '>', ':', '"', '/', '\\', '|', '?', '*', '\x00']:
        raw_name = raw_name.replace(ch, '_')

    raw_name = raw_name.strip().strip('.')
    if not raw_name:
        raw_name = fallback

    # 避免极长文件名导致文件系统报错，保留扩展名。
    path = Path(raw_name)
    stem = path.stem or Path(fallback).stem or 'uploaded_file'
    suffix = path.suffix
    max_len = 180
    if len(raw_name) > max_len:
        keep_stem_len = max(1, max_len - len(suffix))
        raw_name = f"{stem[:keep_stem_len]}{suffix}"

    return raw_name


def sanitize_original_stem(filename: Any, fallback: str = "uploaded") -> str:
    """从原始文件名中提取安全的目录名/资源名。"""
    safe_name = sanitize_original_filename(filename, fallback=f"{fallback}.zip")
    stem = Path(safe_name).stem.strip().strip('.')
    return stem or fallback


def build_unique_upload_path(directory: Any, filename: Any, fallback: str = "uploaded_file") -> Path:
    """
    在目标目录下构造“尽量保持原文件名”的保存路径。

    如果同名文件已存在，为了避免覆盖历史文件，会追加 _1、_2、_3。
    例如：demo.mp4 -> demo_1.mp4。
    """
    target_dir = Path(directory)
    target_dir.mkdir(parents=True, exist_ok=True)

    safe_name = sanitize_original_filename(filename, fallback=fallback)
    candidate = target_dir / safe_name

    if not candidate.exists():
        return candidate

    path = Path(safe_name)
    stem = path.stem or 'uploaded_file'
    suffix = path.suffix

    index = 1
    while True:
        candidate = target_dir / f"{stem}_{index}{suffix}"
        if not candidate.exists():
            return candidate
        index += 1


def build_unique_upload_dir(parent_directory: Any, dirname: Any, fallback: str = "uploaded") -> Path:
    """在目标父目录下构造“尽量保持原名称”的目录路径，重名时追加 _1、_2。"""
    parent = Path(parent_directory)
    parent.mkdir(parents=True, exist_ok=True)

    safe_dirname = sanitize_original_stem(dirname, fallback=fallback)
    candidate = parent / safe_dirname

    if not candidate.exists():
        return candidate

    index = 1
    while True:
        candidate = parent / f"{safe_dirname}_{index}"
        if not candidate.exists():
            return candidate
        index += 1


def save_upload_file_keep_name(
    file: UploadFile,
    directory: Any,
    fallback: str = "uploaded_file",
    max_bytes: Optional[int] = None,
) -> Path:
    """保存 UploadFile，文件名优先使用用户上传时的原始文件名。"""
    target_path = build_unique_upload_path(directory, file.filename, fallback=fallback)

    if max_bytes is not None:
        save_upload_file_limited(file, target_path, max_bytes)
    else:
        with open(target_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

    return target_path


# --- 图片目录读取工具：供标注页从 ./data/image 读取图片 ---
def get_image_upload_dir() -> Path:
    """获取图片保存目录，并确保目录存在。"""
    IMAGE_UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    return IMAGE_UPLOAD_DIR


def is_allowed_image_upload(file: UploadFile) -> bool:
    """校验上传文件是否为允许的图片类型。"""
    filename = file.filename or ""
    suffix = Path(filename).suffix.lower()
    content_type = (file.content_type or "").lower()
    return content_type.startswith("image/") or suffix in IMAGE_FILE_EXTENSIONS


def validate_image_pixel_limit(image_path: Path) -> None:
    try:
        with Image.open(image_path) as image:
            pixels = int(image.width) * int(image.height)
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"无法读取图片: {exc}")
    if pixels > IMAGE_MAX_PIXELS:
        raise HTTPException(status_code=413, detail=f"图片像素数超过限制 {IMAGE_MAX_PIXELS}")


def get_uploaded_image_size(image_path: Path) -> tuple:
    """读取图片宽高，读取失败时返回 0,0。"""
    try:
        return image_processor.get_image_size_from_file(str(image_path))
    except Exception as e:
        logger.warning(f"读取图片尺寸失败: {image_path}, error={e}")
        return 0, 0


def build_uploaded_image_url(filename: str) -> str:
    """构造浏览器可访问的图片 URL，支持中文、空格等文件名。"""
    from urllib.parse import quote
    return f"{IMAGE_UPLOAD_URL_PREFIX}/{quote(filename)}"


def build_uploaded_image_record(
    image_path: Path,
    original_filename: str = None,
    include_dimensions: bool = True,
    stat_result: Any = None
) -> Dict[str, Any]:
    """
    把磁盘图片文件转换成前端图片列表记录。

    include_dimensions=True 时会用 PIL 打开图片读取宽高，适合单张上传/点击图片；
    include_dimensions=False 时只读文件系统元信息，适合大量图片列表分页返回。
    """
    image_path = Path(image_path)
    stat = stat_result or image_path.stat()

    if include_dimensions:
        width, height = get_uploaded_image_size(image_path)
    else:
        width, height = 0, 0

    created_at = datetime.fromtimestamp(stat.st_ctime).isoformat()
    updated_at = datetime.fromtimestamp(stat.st_mtime).isoformat()

    return {
        "id": image_path.name,
        "filename": image_path.name,
        "name": image_path.name,
        "original_name": original_filename or image_path.name,
        "url": build_uploaded_image_url(image_path.name),
        "path": str(image_path),
        "saved_path": f"./data/image/{image_path.name}",
        "size": stat.st_size,
        "width": width,
        "height": height,
        "coordinate_space": "raw",
        "type": f"image/{image_path.suffix.lower().lstrip('.') or '*'}",
        "created_at": created_at,
        "createdAt": created_at,
        "updated_at": updated_at,
        "updatedAt": updated_at,
        "annotations": []
    }


def resolve_uploaded_image_path_from_value(value: Any) -> Optional[Path]:
    """
    将前端传入的图片 URL / 磁盘路径 / 文件名解析为 ./data/image 下的真实文件。

    支持：
    - /image/xxx.jpg
    - /api/image/xxx.jpg
    - http://localhost:8000/image/xxx.jpg
    - ./data/image/xxx.jpg
    - data/image/xxx.jpg
    - 直接文件名 xxx.jpg
    """
    if not value:
        return None

    text = str(value).strip()
    if not text or text.startswith('data:image/'):
        return None

    try:
        from urllib.parse import urlparse, unquote
        parsed = urlparse(text)
        path_text = unquote(parsed.path if parsed.scheme else text)
    except Exception:
        path_text = text

    normalized = path_text.replace('\\', '/')
    filename = None

    if normalized.startswith('/api/image/'):
        filename = normalized.split('/api/image/', 1)[1]
    elif normalized.startswith('/image/'):
        filename = normalized.split('/image/', 1)[1]
    elif './data/image/' in normalized:
        filename = normalized.split('./data/image/', 1)[1]
    elif 'data/image/' in normalized:
        filename = normalized.split('data/image/', 1)[1]
    else:
        # 最后兜底：把它当作文件名或路径，只取 basename。
        filename = Path(normalized).name

    filename = Path(str(filename or '')).name
    if not filename:
        return None

    safe_filename = sanitize_original_filename(filename, fallback='image.jpg')
    image_dir = get_image_upload_dir().resolve()
    image_path = (image_dir / safe_filename).resolve()

    try:
        image_path.relative_to(image_dir)
    except ValueError:
        return None

    if not image_path.exists() or not image_path.is_file():
        return None

    if image_path.suffix.lower() not in IMAGE_FILE_EXTENSIONS:
        return None

    return image_path


def resolve_managed_dataset_root(dataset_id: Any, root_dir: Any = None) -> Path:
    """Resolve a dataset id to a directory under the requested managed root."""
    root = Path(root_dir or constants.DATASETS_DIR).resolve()
    safe_id = sanitize_original_stem(dataset_id, fallback="dataset")
    dataset_path = (root / safe_id).resolve()

    try:
        dataset_path.relative_to(root)
    except ValueError:
        raise HTTPException(status_code=400, detail="非法数据集ID")

    if not dataset_path.exists() or not dataset_path.is_dir():
        raise HTTPException(status_code=404, detail="数据集不存在")

    return dataset_path


def resolve_relative_file_under_root(root: Path, relative_path: Any) -> Path:
    """Resolve a user supplied relative file path under a dataset root."""
    from urllib.parse import unquote

    rel_text = unquote(str(relative_path or "").replace("\\", "/")).strip().lstrip("/")
    if not rel_text:
        raise HTTPException(status_code=400, detail="缺少图片相对路径")

    candidate = (Path(root).resolve() / rel_text).resolve()
    try:
        candidate.relative_to(Path(root).resolve())
    except ValueError:
        raise HTTPException(status_code=400, detail="非法图片路径")

    if not candidate.exists() or not candidate.is_file():
        raise HTTPException(status_code=404, detail="图片文件不存在")

    if candidate.suffix.lower() not in IMAGE_FILE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="只支持图片文件")

    return candidate


def get_dataset_image_paths(dataset_path: Path) -> List[Path]:
    """Return image files from an original dataset."""
    dataset_path = Path(dataset_path)
    image_paths: List[Path] = []

    for split in ("train", "val", "test"):
        image_dir = dataset_path / split / "images"
        if not image_dir.exists():
            continue
        for image_path in image_dir.rglob("*"):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_FILE_EXTENSIONS:
                image_paths.append(image_path)

    if not image_paths:
        skip_parts = {"labels", "__MACOSX"}
        for image_path in dataset_path.rglob("*"):
            if not image_path.is_file() or image_path.suffix.lower() not in IMAGE_FILE_EXTENSIONS:
                continue
            lower_parts = {part.lower() for part in image_path.relative_to(dataset_path).parts}
            if lower_parts & skip_parts:
                continue
            image_paths.append(image_path)

    split_order = {"train": 0, "val": 1, "test": 2}

    def sort_key(path: Path):
        try:
            rel_parts = path.relative_to(dataset_path).parts
        except Exception:
            rel_parts = path.parts
        return (split_order.get(rel_parts[0], 9) if rel_parts else 9, str(path).lower())

    return sorted(image_paths, key=sort_key)


def find_yolo_label_for_image(dataset_path: Path, image_path: Path) -> Optional[Path]:
    """Find the YOLO label file that corresponds to an image in a dataset."""
    dataset_path = Path(dataset_path)
    image_path = Path(image_path)

    try:
        rel_parts = list(image_path.relative_to(dataset_path).parts)
    except ValueError:
        return None

    candidates = []
    if len(rel_parts) >= 3 and rel_parts[1].lower() == "images":
        label_parts = list(rel_parts)
        label_parts[1] = "labels"
        label_parts[-1] = f"{Path(label_parts[-1]).stem}.txt"
        candidates.append(dataset_path.joinpath(*label_parts))

    candidates.append(image_path.with_suffix(".txt"))
    candidates.append(image_path.parent.parent / "labels" / f"{image_path.stem}.txt")

    for candidate in candidates:
        try:
            resolved = candidate.resolve()
            resolved.relative_to(dataset_path.resolve())
        except Exception:
            continue
        if resolved.exists() and resolved.is_file():
            return resolved

    return None


def read_yolo_label_annotations(label_path: Optional[Path]) -> List[Dict[str, Any]]:
    """Read a YOLO txt label file into the frontend's normalized annotation format."""
    if not label_path or not Path(label_path).exists():
        return []

    annotations: List[Dict[str, Any]] = []
    try:
        with open(label_path, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 5:
                    continue
                class_id = int(float(parts[0]))

                if len(parts) >= 7 and (len(parts) - 1) % 2 == 0:
                    points = normalize_annotation_points([float(value) for value in parts[1:]])
                    if points:
                        xs = [point["x"] for point in points]
                        ys = [point["y"] for point in points]
                        annotations.append({
                            "class": class_id,
                            "type": "polygon",
                            "points": points,
                            "x": max(0.0, min(xs)),
                            "y": max(0.0, min(ys)),
                            "width": max(0.0, max(xs) - min(xs)),
                            "height": max(0.0, max(ys) - min(ys))
                        })
                        continue

                center_x = float(parts[1])
                center_y = float(parts[2])
                width = float(parts[3])
                height = float(parts[4])
                annotations.append({
                    "class": class_id,
                    "x": max(0.0, center_x - width / 2),
                    "y": max(0.0, center_y - height / 2),
                    "width": max(0.0, width),
                    "height": max(0.0, height),
                    "type": "rect"
                })
    except Exception as e:
        logger.warning(f"读取 YOLO 标签失败: {label_path}, error={e}")

    return annotations


def build_dataset_image_url(dataset_id: str, relative_path: str) -> str:
    """Build a public static URL for an image inside an original dataset."""
    from urllib.parse import quote
    return f"/dataset-files/{quote(dataset_id)}/{quote(relative_path, safe='/')}"


@app.get("/dataset-files/{dataset_id}/{relative_path:path}")
@app.get("/api/dataset-files/{dataset_id}/{relative_path:path}")
async def get_protected_dataset_file(
    dataset_id: str,
    relative_path: str,
    request: Request,
    db=Depends(get_db),
):
    require_dataset_read(db, get_request_current_user(request), dataset_id, "original")
    root = resolve_managed_dataset_root(dataset_id, constants.DATASETS_DIR)
    target = resolve_relative_file_under_root(root, relative_path)
    return FileResponse(target, media_type=mimetypes.guess_type(target.name)[0] or "application/octet-stream")


@app.get("/training-dataset-files/{dataset_id}/{relative_path:path}")
@app.get("/api/training-dataset-files/{dataset_id}/{relative_path:path}")
async def get_protected_training_dataset_file(
    dataset_id: str,
    relative_path: str,
    request: Request,
    db=Depends(get_db),
):
    require_dataset_read(db, get_request_current_user(request), dataset_id, "training")
    root = resolve_managed_dataset_root(dataset_id, TRAINING_DATASETS_DIR)
    target = resolve_relative_file_under_root(root, relative_path)
    return FileResponse(target, media_type=mimetypes.guess_type(target.name)[0] or "application/octet-stream")


def build_dataset_image_record(
    dataset_id: str,
    dataset_name: str,
    dataset_path: Path,
    image_path: Path,
    include_dimensions: bool = False
) -> Dict[str, Any]:
    """Build the frontend image record for an image inside a dataset."""
    image_path = Path(image_path)
    stat = image_path.stat()
    relative_path = image_path.relative_to(dataset_path).as_posix()
    width, height = get_uploaded_image_size(image_path) if include_dimensions else (0, 0)
    label_path = find_yolo_label_for_image(dataset_path, image_path)
    annotations = read_yolo_label_annotations(label_path)

    return {
        "id": f"{dataset_id}:{relative_path}",
        "filename": image_path.name,
        "name": image_path.name,
        "dataset_id": dataset_id,
        "dataset_name": dataset_name,
        "relative_path": relative_path,
        "url": build_dataset_image_url(dataset_id, relative_path),
        "path": str(image_path),
        "saved_path": relative_path,
        "source": dataset_name,
        "sourcePath": str(dataset_path),
        "size": stat.st_size,
        "width": width,
        "height": height,
        "coordinate_space": "raw",
        "type": f"image/{image_path.suffix.lower().lstrip('.') or '*'}",
        "created_at": datetime.fromtimestamp(stat.st_ctime).isoformat(),
        "updated_at": datetime.fromtimestamp(stat.st_mtime).isoformat(),
        "annotations": annotations
    }


def flatten_single_yolo_root(dataset_path: Path) -> Path:
    """If a ZIP extracts to one top-level folder containing YOLO data, move it to dataset root."""
    dataset_path = Path(dataset_path)
    if (dataset_path / "train" / "images").exists() or (dataset_path / "val" / "images").exists():
        return dataset_path

    children = [p for p in dataset_path.iterdir() if p.is_dir()]
    if len(children) != 1:
        return dataset_path

    child = children[0]
    if not ((child / "train" / "images").exists() or (child / "val" / "images").exists()):
        return dataset_path

    for item in child.iterdir():
        target = dataset_path / item.name
        if target.exists():
            continue
        shutil.move(str(item), str(target))
    try:
        child.rmdir()
    except OSError:
        pass
    return dataset_path


def write_training_dataset_metadata(dataset_path: Path, dataset_info: Dict[str, Any]) -> None:
    """Write metadata for a training dataset."""
    write_dataset_metadata(Path(dataset_path), dataset_info)


def build_training_dataset_info_from_metadata(dataset_path: Path) -> Dict[str, Any]:
    """Build training dataset info, falling back to a lightweight scan when metadata is missing."""
    dataset_path = Path(dataset_path)
    metadata = read_dataset_metadata(dataset_path)
    if metadata:
        metadata.setdefault("id", dataset_path.name)
        metadata.setdefault("name", dataset_path.name)
        metadata.setdefault("path", str(dataset_path))
        metadata["dataset_role"] = "training"
        return metadata

    stats = training_dataset_manager.analyze_dataset(dataset_path)
    created_at = datetime.fromtimestamp(dataset_path.stat().st_ctime).isoformat()
    return {
        "id": dataset_path.name,
        "name": dataset_path.name,
        "path": str(dataset_path),
        "created_at": created_at,
        "updated_at": datetime.fromtimestamp(dataset_path.stat().st_mtime).isoformat(),
        "status": constants.DATASET_STATUS_READY if stats.get("image_count", 0) > 0 else constants.DATASET_STATUS_EMPTY,
        "dataset_role": "training",
        **stats
    }


def normalize_yolo_annotation_line(annotation: Dict[str, Any], image_width: int = 0, image_height: int = 0) -> Optional[str]:
    """Convert a frontend annotation to one YOLO detect/segment label line."""
    try:
        class_id = int(annotation.get("class", 0))
        points = normalize_annotation_points(
            annotation.get("points") or annotation.get("polygon") or annotation.get("segmentation"),
            image_width=image_width,
            image_height=image_height
        )
        if points:
            values = []
            for point in points:
                values.extend([point["x"], point["y"]])
            return f"{class_id} " + " ".join(f"{value:.6f}" for value in values)

        x = float(annotation.get("x", 0))
        y = float(annotation.get("y", 0))
        width = float(annotation.get("width", 0))
        height = float(annotation.get("height", 0))

        if max(abs(x), abs(y), abs(width), abs(height)) > 1 and image_width and image_height:
            center_x = (x + width / 2) / image_width
            center_y = (y + height / 2) / image_height
            rel_width = width / image_width
            rel_height = height / image_height
        else:
            center_x = x + width / 2
            center_y = y + height / 2
            rel_width = width
            rel_height = height

        values = [center_x, center_y, rel_width, rel_height]
        values = [min(1.0, max(0.0, value)) for value in values]
        if values[2] <= 0 or values[3] <= 0:
            return None
        return f"{class_id} {values[0]:.6f} {values[1]:.6f} {values[2]:.6f} {values[3]:.6f}"
    except Exception:
        return None


def build_unique_training_image_name(original_name: Any, used_names: set) -> str:
    """Return a safe image file name unique within an exported training dataset."""
    safe_name = sanitize_original_filename(original_name, fallback="image.jpg")
    stem = Path(safe_name).stem or "image"
    suffix = Path(safe_name).suffix or ".jpg"
    candidate = f"{stem}{suffix}"
    index = 1
    while candidate.lower() in used_names:
        candidate = f"{stem}_{index}{suffix}"
        index += 1
    used_names.add(candidate.lower())
    return candidate


def infer_class_names_from_yolo_dataset(dataset_path: Path) -> List[str]:
    """Infer class names for a YOLO dataset when dataset.yaml is missing."""
    dataset_path = Path(dataset_path)
    classes_file = dataset_path / "classes.txt"
    if classes_file.exists():
        try:
            names = [
                line.strip()
                for line in classes_file.read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if names:
                return names
        except Exception:
            pass

    max_class_id = -1
    for label_file in dataset_path.rglob("*.txt"):
        if label_file.name.lower() == "classes.txt" or label_file.name.endswith(".cache"):
            continue
        try:
            for line in label_file.read_text(encoding="utf-8").splitlines():
                parts = line.strip().split()
                if not parts:
                    continue
                max_class_id = max(max_class_id, int(float(parts[0])))
        except Exception:
            continue

    if max_class_id >= 0:
        return [f"class_{index}" for index in range(max_class_id + 1)]

    return ["object"]


def ensure_training_dataset_yaml(dataset_path: Path) -> None:
    """Create a minimal dataset.yaml for a YOLO training dataset if it is missing."""
    dataset_path = Path(dataset_path)
    yaml_path = dataset_path / "dataset.yaml"
    if yaml_path.exists():
        return

    class_names = infer_class_names_from_yolo_dataset(dataset_path)
    content = (
        "names:\n"
        + "\n".join([f"  {index}: {name}" for index, name in enumerate(class_names)])
        + f"\nnc: {len(class_names)}\n"
        + f"path: {dataset_path}\n"
        + "train: train/images\n"
        + "val: val/images\n"
        + "test: test/images\n"
    )
    yaml_path.write_text(content, encoding="utf-8")


def load_annotation_image_from_request_value(image_value: Any) -> Optional[np.ndarray]:
    """
    AI 辅助标注图片加载函数。

    原版本只支持 base64；现在同时支持 data:image、/image/xxx、/api/image/xxx、
    ./data/image/xxx 这几种前端常见传值，避免“400: 无效的图片数据”。
    返回 RGB numpy.ndarray。
    """
    if not image_value:
        return None

    text = str(image_value).strip()

    if text.startswith('data:image/') or text.startswith('data:'):
        return image_processor.decode_base64_image(text)

    image_path = resolve_uploaded_image_path_from_value(text)
    if image_path is None:
        # 兼容纯 base64 字符串，没有 data:image 前缀的情况。
        return image_processor.decode_base64_image(text)

    try:
        image = image_processor.decode_image_file(str(image_path))
        if image is None:
            raise ValueError("图片解码失败")
        return image
    except Exception as e:
        logger.warning(f'读取 AI 辅助标注图片失败: {image_path}, error={e}')
        return None



# --- 后端批量 AI 标注任务：稳定处理大量 ./data/image 图片 ---
def list_uploaded_image_paths_for_batch(search: str = "") -> List[Path]:
    """扫描 ./data/image，返回需要批量 AI 标注的图片路径列表。"""
    image_dir = get_image_upload_dir()
    search_text = str(search or "").strip().lower()
    image_paths: List[Path] = []

    try:
        with os.scandir(image_dir) as iterator:
            for entry in iterator:
                if not entry.is_file():
                    continue

                suffix = Path(entry.name).suffix.lower()
                if suffix not in IMAGE_FILE_EXTENSIONS:
                    continue

                if search_text and search_text not in entry.name.lower():
                    continue

                image_paths.append(Path(entry.path))
    except FileNotFoundError:
        return []

    # 与 /images 列表保持一致：最近修改的图片在前。
    image_paths.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
    return image_paths


def list_annotation_image_paths_for_batch(dataset_id: str = "", search: str = "") -> List[Path]:
    """Return image paths for backend batch annotation from a dataset or legacy image store."""
    search_text = str(search or "").strip().lower()
    dataset_text = str(dataset_id or "").strip()

    if dataset_text:
        db = SessionLocal()
        try:
            version = (
                db.query(DatasetVersion)
                .filter(DatasetVersion.dataset_id == dataset_text, DatasetVersion.dataset_role == "original", DatasetVersion.status == "ready")
                .order_by(DatasetVersion.created_at.desc())
                .first()
            )
            if not version:
                raise HTTPException(status_code=409, detail="数据集尚未建立可用资产索引")
            query = db.query(DatasetAsset).filter(DatasetAsset.dataset_version_id == version.id)
            if search_text:
                query = query.filter(DatasetAsset.relative_path.ilike(f"%{search_text}%"))
            root = Path(version.root_path).resolve()
            return [resolve_relative_file_under_root(root, asset.relative_path) for asset in query.order_by(DatasetAsset.relative_path).all()]
        finally:
            db.close()

    return list_uploaded_image_paths_for_batch(search=search_text)


def normalize_ai_model_path(model_path: str) -> str:
    """沿用单图 AI 标注接口的模型路径查找规则。"""
    selected_model = str(model_path or "yolo11n.pt").strip() or "yolo11n.pt"

    candidates = [
        Path(selected_model),
        Path("models") / selected_model,
        MODELS_DIR / selected_model,
        PRETRAINED_MODELS_DIR / selected_model,
        Path("/app/data/models/pretrained") / selected_model,
        Path("/app/data/models") / selected_model,
        Path("/app/models/pretrained") / selected_model,
        Path("/app/models") / selected_model,
    ]

    for candidate in candidates:
        try:
            if candidate.exists() and candidate.is_file():
                return str(candidate)
        except OSError:
            continue

    selected_name = Path(selected_model).name
    db = SessionLocal()
    try:
        registry_item = (
            db.query(ModelRegistry)
            .filter(ModelRegistry.status == "active", ModelRegistry.name == selected_name)
            .order_by(ModelRegistry.updated_at.desc())
            .first()
        )
        if registry_item and Path(registry_item.path).is_file():
            return registry_item.path
    finally:
        db.close()

    raise HTTPException(status_code=404, detail=f"模型文件不存在: {selected_model}")


def normalize_annotation_prompt_classes(value: Any) -> List[str]:
    """Normalize YOLOE text prompts/classes from request payload."""
    collected: List[str] = []

    def add_item(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, dict):
            add_item(item.get("name") or item.get("label") or item.get("class_name") or item.get("value"))
            return
        if isinstance(item, (list, tuple, set)):
            for sub_item in item:
                add_item(sub_item)
            return

        text = str(item or "").strip()
        if not text:
            return

        for part in re.split(r"[\n,，;；]+", text):
            name = part.strip()
            if name:
                collected.append(name[:120])

    add_item(value)

    normalized: List[str] = []
    seen = set()
    for name in collected:
        key = name.lower()
        if key in seen:
            continue
        seen.add(key)
        normalized.append(name)
        if len(normalized) >= 100:
            break

    return normalized


def get_annotation_prompt_classes_from_request(request: Dict[str, Any]) -> List[str]:
    """Read YOLOE prompt classes from multiple compatible request field names."""
    for key in (
        "prompt_classes",
        "text_prompts",
        "prompts",
        "yoloe_prompts",
        "class_names",
        "classes",
    ):
        if key in request:
            prompts = normalize_annotation_prompt_classes(request.get(key))
            if prompts:
                return prompts
    return []


ANNOTATION_CLASS_BINDINGS_METADATA_KEY = "annotation_class_bindings"


def _annotation_class_key(value: Any) -> str:
    return annotation_class_key(value)


def _coerce_annotation_class_id(value: Any) -> Optional[int]:
    try:
        class_id = int(value)
    except (TypeError, ValueError):
        return None
    return class_id if class_id >= 0 else None


def normalize_annotation_class_bindings(raw_bindings: Any, page_classes: Optional[List[str]] = None) -> List[Dict[str, Any]]:
    """Normalize page-to-native-model bindings without guessing a semantic class."""
    if isinstance(raw_bindings, dict):
        raw_bindings = [
            {"page_index": key, **(value if isinstance(value, dict) else {"model_class_id": value})}
            for key, value in raw_bindings.items()
        ]
    if not isinstance(raw_bindings, (list, tuple)):
        return []

    normalized: List[Dict[str, Any]] = []
    seen_pages = set()
    for item in raw_bindings:
        if not isinstance(item, dict):
            continue
        try:
            page_index = int(item.get("page_index", item.get("pageIndex")))
        except (TypeError, ValueError):
            continue
        if page_index < 0 or (page_classes is not None and page_index >= len(page_classes)) or page_index in seen_pages:
            continue
        model_class_id = _coerce_annotation_class_id(
            item.get("model_class_id", item.get("modelClassId", item.get("class_id")))
        )
        model_class_name = str(
            item.get("model_class_name", item.get("modelClassName", item.get("class_name"))) or ""
        ).strip()
        page_name = (
            str(page_classes[page_index]).strip()
            if page_classes is not None and page_index < len(page_classes)
            else str(item.get("name") or item.get("page_class_name") or "").strip()
        )
        if model_class_id is None and not model_class_name:
            continue
        normalized.append({
            "page_index": page_index,
            "name": page_name,
            "model_class_id": model_class_id,
            "model_class_name": model_class_name,
        })
        seen_pages.add(page_index)
    return sorted(normalized, key=lambda item: item["page_index"])


def get_model_class_names(model: Any, result: Any = None) -> Dict[int, str]:
    """Return the immutable native class-id/name space exposed by a YOLO model."""
    names = getattr(model, "names", None) or getattr(result, "names", None) or {}
    if isinstance(names, dict):
        items = names.items()
    elif isinstance(names, (list, tuple)):
        items = enumerate(names)
    else:
        return {}

    resolved: Dict[int, str] = {}
    for raw_id, raw_name in items:
        class_id = _coerce_annotation_class_id(raw_id)
        name = str(raw_name or "").strip()
        if class_id is not None and name:
            resolved[class_id] = name
    return dict(sorted(resolved.items()))


def build_annotation_class_bindings(page_classes: List[str], model_names: Any, inactive_ids: Optional[List[int]] = None) -> List[Dict[str, Any]]:
    """Bind dataset slots to native model IDs using unambiguous class aliases."""
    classes = [str(value or "").strip() for value in (page_classes or [])]
    if isinstance(model_names, dict):
        native_names = {
            int(key): str(value or "").strip()
            for key, value in model_names.items()
            if _coerce_annotation_class_id(key) is not None and str(value or "").strip()
        }
    else:
        native_names = {
            index: str(value or "").strip()
            for index, value in enumerate(model_names or [])
            if str(value or "").strip()
        }
    try:
        class_name_ids(classes, inactive_ids or [])
        by_name = class_name_ids(native_names)
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    bindings = []
    for page_index, page_name in enumerate(classes):
        if page_index in (inactive_ids or []):
            continue
        matched = by_name.get(_annotation_class_key(page_name))
        if matched is None:
            continue
        model_class_id, model_class_name = matched, native_names[matched]
        bindings.append({
            "page_index": page_index,
            "name": page_name,
            "model_class_id": model_class_id,
            "model_class_name": model_class_name,
        })
    return bindings


def annotation_model_supports_text_prompts(
    model_path: Any = "",
    model_metadata: Optional[Dict[str, Any]] = None,
    model: Any = None,
) -> bool:
    """Gate prompt mutation by explicit model capability, never by method presence."""
    references = [model_metadata or {}, model_path]
    if model is not None:
        references.extend([
            getattr(model, "model_family", ""),
            getattr(model, "task", ""),
        ])
    return bool(model_supports_text_prompts(*references))


def annotation_model_supports_dynamic_text_prompts(
    model_path: Any = "",
    model_metadata: Optional[Dict[str, Any]] = None,
) -> bool:
    """Distinguish open-vocabulary YOLOE from fixed-head trained YOLOE checkpoints."""
    metadata = model_metadata if isinstance(model_metadata, dict) else {}
    explicit = metadata.get("supports_dynamic_text_prompts")
    if explicit is not None:
        return bool(explicit)
    source = str(metadata.get("source") or metadata.get("type") or "").strip().lower()
    path_text = str(model_path or "").strip().lower().replace("\\", "/")
    if source == "trained" or "/runs/train/" in path_text:
        return False
    return bool(annotation_model_supports_text_prompts(model_path, metadata))


def load_annotation_inference_model(
    model_path: str,
    prompt_classes: Any = None,
    model_metadata: Optional[Dict[str, Any]] = None,
) -> Any:
    """Load prompted PT weights through YOLOE; keep ordinary YOLO unchanged."""
    prompts = normalize_annotation_prompt_classes(prompt_classes or [])
    if not prompts:
        from ultralytics import YOLO

        return YOLO(model_path)

    if Path(str(model_path or "")).suffix.lower() != ".pt":
        raise HTTPException(
            status_code=400,
            detail="导出模型不支持动态YOLOE提示词，请选择原始PT模型",
        )
    if not annotation_model_supports_text_prompts(model_path, model_metadata):
        raise HTTPException(
            status_code=400,
            detail="当前模型不支持动态YOLOE文本提示词",
        )

    try:
        from ultralytics import YOLOE
    except ImportError as exc:
        raise HTTPException(
            status_code=500,
            detail=f"当前Ultralytics版本不支持YOLOE文本提示词: {str(exc)}",
        )

    try:
        return YOLOE(model_path)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"YOLOE模型加载失败: {str(exc)}")


def build_annotation_class_routing(
    model: Any,
    page_classes: Optional[List[str]] = None,
    page_bindings: Any = None,
    *,
    prompt_classes: Optional[List[str]] = None,
    prompt_page_indices: Optional[List[Any]] = None,
    target_page_index: Any = None,
    text_prompt_model: bool = False,
    dynamic_text_prompt_model: Optional[bool] = None,
) -> Dict[str, Any]:
    """Build native filtering and page-index mapping for one inference snapshot."""
    if dynamic_text_prompt_model is None:
        dynamic_text_prompt_model = bool(text_prompt_model)
    native_names = get_model_class_names(model)
    normalized_bindings = normalize_annotation_class_bindings(page_bindings, page_classes)
    page_to_model: Dict[int, int] = {}
    eligible_pages = set(range(len(page_classes))) if page_classes is not None else set()
    try:
        native_by_name = class_name_ids(native_names)
        if page_classes is not None:
            eligible_pages = set(routing_page_class_ids(page_classes, normalized_bindings).values())
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))

    if page_classes is not None:
        for binding in normalized_bindings:
            page_index = binding["page_index"]
            if page_index not in eligible_pages:
                continue
            model_class_id = binding.get("model_class_id")
            model_class_name = str(binding.get("model_class_name") or "").strip()
            current_name = native_names.get(model_class_id)
            if model_class_id is not None and current_name and (
                not model_class_name or _annotation_class_key(current_name) == _annotation_class_key(model_class_name)
            ):
                page_to_model[page_index] = model_class_id
                continue
            page_name = page_classes[page_index] if page_index < len(page_classes) else ""
            fallback_id = native_by_name.get(_annotation_class_key(page_name))
            if fallback_id is not None:
                page_to_model[page_index] = fallback_id

        for page_index, page_name in enumerate(page_classes):
            if page_index not in eligible_pages:
                continue
            if page_index not in page_to_model:
                fallback_id = native_by_name.get(_annotation_class_key(page_name))
                if fallback_id is not None:
                    page_to_model[page_index] = fallback_id

    target_index = _coerce_annotation_class_id(target_page_index)
    normalized_prompts = normalize_annotation_prompt_classes(prompt_classes or [])
    if target_index is not None and page_classes is not None:
        allowed_model_ids = {
            page_to_model[target_index]
        } if target_index in page_to_model else set()
    elif not text_prompt_model or not dynamic_text_prompt_model:
        if normalized_prompts:
            allowed_model_ids = {
                native_by_name[_annotation_class_key(prompt)]
                for prompt in normalized_prompts
                if _annotation_class_key(prompt) in native_by_name
            }
        elif page_classes is not None:
            allowed_model_ids = set(page_to_model.values())
        else:
            allowed_model_ids = None
    else:
        allowed_model_ids = None

    model_to_page = {model_id: page_index for page_index, model_id in page_to_model.items()}
    if text_prompt_model and dynamic_text_prompt_model and normalized_prompts:
        model_to_page = {}
        prompt_mapping = []
        supplied_indices = prompt_page_indices if isinstance(prompt_page_indices, (list, tuple)) else []
        for index, prompt in enumerate(normalized_prompts):
            page_index = _coerce_annotation_class_id(supplied_indices[index]) if index < len(supplied_indices) else None
            if page_index is None and page_classes is not None:
                for candidate_index, page_name in enumerate(page_classes):
                    if candidate_index not in eligible_pages:
                        continue
                    if _annotation_class_key(page_name) == _annotation_class_key(prompt):
                        page_index = candidate_index
                        break
            prompt_mapping.append(page_index)
            if page_index is not None:
                model_to_page[index] = page_index
        model_to_page = {model_id: page_index for model_id, page_index in model_to_page.items() if page_index is not None}

    return {
        "page_classes": page_classes,
        "bindings": normalized_bindings,
        "native_names": native_names,
        "page_to_model": page_to_model,
        "model_to_page": model_to_page,
        "allowed_model_ids": allowed_model_ids,
        "target_page_index": target_index,
        "prompt_classes": normalized_prompts,
        "unmapped_page_classes": (
            [index for index in range(len(page_classes)) if index not in page_to_model]
            if page_classes is not None else []
        ),
        "unmapped_page_class_names": (
            [str(page_classes[index]) for index in range(len(page_classes)) if index not in page_to_model]
            if page_classes is not None else []
        ),
    }


def apply_annotation_prompt_classes(
    model: Any,
    prompt_classes: List[str],
    *,
    supports_text_prompts: Optional[bool] = None,
    supports_dynamic_text_prompts: Optional[bool] = None,
) -> List[str]:
    """Apply page-name prompts only after an explicit YOLOE capability check."""
    prompts = normalize_annotation_prompt_classes(prompt_classes)
    if not prompts:
        return []

    if supports_text_prompts is False:
        raise HTTPException(status_code=400, detail="所选模型不支持YOLOE文本提示词")

    if supports_dynamic_text_prompts is False:
        configured_names = list(get_model_class_names(model).values())
        configured_keys = {str(name).strip().casefold() for name in configured_names}
        if any(name.casefold() not in configured_keys for name in prompts):
            raise HTTPException(
                status_code=400,
                detail=(
                    "训练型YOLOE模型仅支持筛选训练类别，不能新增动态文本类别；"
                    f"可用类别={configured_names or '无法读取'}，请求类别={prompts}"
                ),
            )
        # PE/fine-tuned YOLOE heads have a fixed class dimension. Keep their
        # native vocabulary and let the common class routing filter requested names.
        return prompts

    set_classes = getattr(model, "set_classes", None)
    if not callable(set_classes):
        raise HTTPException(status_code=400, detail="所选模型未提供YOLOE文本提示词能力")

    try:
        get_text_pe = getattr(model, "get_text_pe", None)
        if callable(get_text_pe):
            embeddings = get_text_pe(prompts)
            set_classes(prompts, embeddings)
        else:
            set_classes(prompts)

        configured_names = list(get_model_class_names(model).values())
        if [name.casefold() for name in configured_names] != [name.casefold() for name in prompts]:
            raise HTTPException(
                status_code=500,
                detail=(
                    "YOLOE提示词未真正写入模型类别表："
                    f"期望 {prompts}，实际 {configured_names or '无法读取'}"
                ),
            )
        return prompts
    except ModuleNotFoundError as e:
        missing_name = getattr(e, "name", "") or str(e)
        if "clip" in missing_name.lower():
            raise HTTPException(
                status_code=500,
                detail=(
                    "YOLOE文本提示词需要CLIP依赖，但后端环境缺少clip模块。"
                    "请在后端镜像中安装 git 和 git+https://github.com/ultralytics/CLIP.git 后重试。"
                )
            )
        raise
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"YOLOE文本提示词配置失败: {str(e)}")


def enforce_annotation_prompt_results(
    annotations: List[Dict[str, Any]],
    applied_prompt_classes: Any,
) -> List[Dict[str, Any]]:
    """Reject annotations whose native model label is outside the applied prompt vocabulary."""
    prompts = normalize_annotation_prompt_classes(applied_prompt_classes or [])
    if not prompts:
        return annotations

    allowed = {name.casefold() for name in prompts}
    unexpected = sorted({
        str(annotation.get("model_class_name") or annotation.get("class_name") or "").strip()
        for annotation in annotations
        if str(annotation.get("model_class_name") or annotation.get("class_name") or "").strip().casefold()
        not in allowed
    })
    if unexpected:
        raise HTTPException(
            status_code=500,
            detail=f"YOLOE返回了提示词范围外的类别，已拒绝写入标注: {unexpected}",
        )
    return annotations


def build_annotation_batch_request_signature(
    dataset_id: Any,
    search: Any,
    model_path: Any,
    confidence: Any,
    prompt_classes: Any,
    prompt_page_indices: Any,
    target_page_index: Any,
    page_classes: Any,
    page_bindings: Any,
    model_metadata: Any,
) -> Dict[str, Any]:
    """Build the complete inference signature used for safe active-job reuse."""
    try:
        normalized_confidence = round(float(confidence), 6)
    except (TypeError, ValueError):
        normalized_confidence = round(float(constants.DEFAULT_CONFIDENCE), 6)
    metadata = dict(model_metadata or {}) if isinstance(model_metadata, dict) else {}
    return {
        "dataset_id": str(dataset_id or "").strip(),
        "search": str(search or "").strip(),
        "model_path": str(model_path or "").strip(),
        "confidence": normalized_confidence,
        "prompt_classes": normalize_annotation_prompt_classes(prompt_classes or []),
        "prompt_page_indices": list(prompt_page_indices or []) if isinstance(prompt_page_indices, (list, tuple)) else [],
        "target_page_index": _coerce_annotation_class_id(target_page_index),
        "annotation_classes": list(page_classes) if isinstance(page_classes, (list, tuple)) else page_classes,
        "annotation_class_bindings": normalize_annotation_class_bindings(page_bindings, page_classes),
        "model_prompt_metadata": {
            key: metadata.get(key)
            for key in (
                "id",
                "sha256",
                "model_family",
                "supports_text_prompts",
                "base_model",
                "lineage_root_model",
            )
        },
    }


def annotation_batch_job_matches_signature(task_input: Dict[str, Any], signature: Dict[str, Any]) -> bool:
    """Return whether an active task has the exact same annotation inference inputs."""
    current = dict(task_input or {})
    return build_annotation_batch_request_signature(
        current.get("dataset_id"),
        current.get("search"),
        current.get("model_path"),
        current.get("confidence"),
        current.get("prompt_classes"),
        current.get("prompt_page_indices"),
        current.get("target_page_index"),
        current.get("annotation_classes"),
        current.get("annotation_class_bindings"),
        current.get("model_metadata"),
    ) == signature


def resolve_model_class_name(model: Any, result: Any, class_id: int) -> str:
    """Resolve a class id from model/result names with COCO fallback."""
    names = getattr(model, "names", None) or getattr(result, "names", None) or {}
    if isinstance(names, dict):
        return str(names.get(class_id, f"class_{class_id}"))
    if isinstance(names, (list, tuple)) and class_id < len(names):
        return str(names[class_id])
    return constants.COCO_CLASSES[class_id] if class_id < len(constants.COCO_CLASSES) else f"class_{class_id}"


def get_request_current_username(request: Optional[Request]) -> str:
    """Return the authenticated username stored by the JWT middleware."""
    current_user = getattr(getattr(request, "state", None), "current_user", None) if request else None
    if current_user:
        for key in ("username", "name", "display_name", "real_name", "email"):
            value = current_user.get(key) if isinstance(current_user, dict) else getattr(current_user, key, None)
            if value:
                return str(value)
    return "管理员"


def get_request_current_user(request: Optional[Request]) -> Dict[str, Any]:
    current_user = getattr(getattr(request, "state", None), "current_user", None) if request else None
    return current_user if isinstance(current_user, dict) else {}


def require_job_visibility(task: JobTask, request: Request) -> None:
    current_user = get_request_current_user(request)
    if current_user.get("role") == "admin":
        return
    username = str(current_user.get("username") or "")
    if not username or task.owner_username != username:
        raise HTTPException(status_code=404, detail="任务不存在")


def normalize_annotation_points(
    points: Any,
    image_width: int = 0,
    image_height: int = 0,
    assume_normalized: bool = True
) -> List[Dict[str, float]]:
    """Normalize polygon points to frontend/YOLO relative coordinates."""
    if points is None:
        return []

    try:
        if hasattr(points, "cpu"):
            points = points.cpu().numpy()
        if hasattr(points, "tolist"):
            points = points.tolist()
    except Exception:
        pass

    if isinstance(points, tuple):
        points = list(points)

    if not isinstance(points, list) or not points:
        return []

    if all(not isinstance(item, (list, tuple, dict)) for item in points):
        if len(points) < 6 or len(points) % 2 != 0:
            return []
        points = [[points[index], points[index + 1]] for index in range(0, len(points), 2)]

    normalized: List[Dict[str, float]] = []
    for point in points:
        try:
            if isinstance(point, dict):
                x_value = point.get("x")
                y_value = point.get("y")
            elif isinstance(point, (list, tuple)) and len(point) >= 2:
                x_value = point[0]
                y_value = point[1]
            else:
                continue

            x = float(x_value)
            y = float(y_value)
            if not np.isfinite(x) or not np.isfinite(y):
                continue

            if (not assume_normalized or abs(x) > 1 or abs(y) > 1) and image_width and image_height:
                x = x / image_width
                y = y / image_height

            normalized.append({
                "x": float(min(1.0, max(0.0, x))),
                "y": float(min(1.0, max(0.0, y)))
            })
        except Exception:
            continue

    if len(normalized) < 3:
        return []
    return normalized


def extract_segmentation_points(result: Any, index: int, image_width: int, image_height: int) -> List[Dict[str, float]]:
    """Extract one instance mask contour from an Ultralytics result."""
    masks = getattr(result, "masks", None)
    if masks is None:
        return []

    for attr_name, assume_normalized in (("xyn", True), ("xy", False)):
        try:
            mask_polygons = getattr(masks, attr_name, None)
            if mask_polygons is None or index >= len(mask_polygons):
                continue
            points = normalize_annotation_points(
                mask_polygons[index],
                image_width=image_width,
                image_height=image_height,
                assume_normalized=assume_normalized
            )
            if points:
                return points
        except Exception:
            continue

    return []


def build_ai_annotation_from_box(
    model: Any,
    result: Any,
    box: Any,
    index: int,
    image_width: int,
    image_height: int,
    routing: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Convert one Ultralytics detection into the frontend annotation schema."""
    try:
        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
        center_x = ((x1 + x2) / 2) / image_width
        center_y = ((y1 + y2) / 2) / image_height
        width = (x2 - x1) / image_width
        height = (y2 - y1) / image_height

        class_id = int(box.cls[0].cpu().numpy())
        confidence_score = float(box.conf[0].cpu().numpy())
        class_name = resolve_model_class_name(model, result, class_id)

        routing = routing or {}
        page_classes = routing.get("page_classes")
        model_to_page = routing.get("model_to_page") or {}
        target_page_index = routing.get("target_page_index")
        page_class_id = target_page_index if target_page_index is not None else model_to_page.get(class_id)
        if page_classes is not None:
            if page_class_id is None or page_class_id < 0 or page_class_id >= len(page_classes):
                return None
            page_class_name = str(page_classes[page_class_id])
        else:
            page_class_id = class_id
            page_class_name = str(class_name)

        annotation: Dict[str, Any] = {
            "class": page_class_id,
            "class_name": page_class_name,
            "page_class_id": page_class_id,
            "page_class_name": page_class_name,
            "model_class_id": class_id,
            "model_class_name": str(class_name),
            "x": float(center_x - width / 2),
            "y": float(center_y - height / 2),
            "width": float(width),
            "height": float(height),
            "coordinate_space": "raw",
            "image_width": int(image_width),
            "image_height": int(image_height),
            "image_orientation": 1,
            "confidence": confidence_score,
            "type": "rect"
        }

        points = extract_segmentation_points(result, index, image_width, image_height)
        if points:
            annotation["type"] = "polygon"
            annotation["points"] = points

        return annotation
    except Exception as e:
        logger.warning(f"转换AI标注结果失败: {e}")
        return None


def _run_annotation_model(
    model: Any,
    source: Any,
    confidence: float,
    allowed_model_ids: Optional[set] = None,
) -> Any:
    kwargs = {"conf": confidence, "verbose": False}
    if allowed_model_ids is not None:
        kwargs["classes"] = sorted(allowed_model_ids)
    try:
        return model(source, **kwargs)
    except TypeError:
        if "classes" not in kwargs:
            raise
        kwargs.pop("classes", None)
        return model(source, **kwargs)


def detect_annotations_for_image_path(
    model: Any,
    image_path: Path,
    confidence: float,
    routing: Optional[Dict[str, Any]] = None,
) -> List[Dict[str, Any]]:
    """对单张磁盘图片执行 YOLO 推理，并返回前端已有格式的相对坐标标注。"""
    image_path = Path(image_path)

    image_rgb = image_processor.decode_image_file(str(image_path))
    if image_rgb is None:
        return []
    img_height, img_width = image_processor.get_image_size(image_rgb)

    routing = routing or {}
    allowed_model_ids = routing.get("allowed_model_ids")
    if allowed_model_ids is not None and not allowed_model_ids:
        return []
    # 使用上传时的原始像素，确保模型输出与画布和 YOLO 坐标同一尺寸。
    image_bgr = image_processor.convert_rgb_to_bgr(image_rgb)
    results = _run_annotation_model(model, image_bgr, confidence, allowed_model_ids)

    if (not results) or len(results) == 0:
        return []

    result = results[0]
    if (img_width <= 0 or img_height <= 0) and hasattr(result, "orig_shape"):
        try:
            img_height, img_width = result.orig_shape[:2]
        except Exception:
            img_width, img_height = 0, 0

    if img_width <= 0 or img_height <= 0:
        return []

    boxes = getattr(result, "boxes", None)
    if boxes is None:
        return []

    annotations: List[Dict[str, Any]] = []
    for index, box in enumerate(boxes):
        annotation = build_ai_annotation_from_box(model, result, box, index, img_width, img_height, routing)
        if annotation:
            annotations.append(annotation)

    return annotations


def gpu_http_guard(function):
    from functools import wraps
    @wraps(function)
    async def guarded(*args, **kwargs):
        try:
            require_gpu()
            return await function(*args, **kwargs)
        except Exception as exc:
            cause = exc.detail if isinstance(exc, HTTPException) else exc
            if isinstance(exc, GPUUnavailable) or report_gpu_fault(cause):
                raise HTTPException(status_code=503, headers={"Retry-After": "3"},
                                    detail={"code": "GPU_RUNTIME_UNAVAILABLE", "message": WAIT_MESSAGE}) from exc
            raise
    return guarded


async def _gpu_job_call(task_id: str, function, *args, requires_gpu=True):
    """Keep persistent GPU work cancellable while waiting; retry only CUDA faults."""
    from job_store import is_cancel_requested
    if not requires_gpu:
        return await asyncio.to_thread(function, *args)
    waiting = False
    failures = 0
    while True:
        if is_cancel_requested(task_id):
            db = SessionLocal()
            try:
                update_job(db, task_id, status="cancelled", stage="cancelled", message="任务已取消")
            finally:
                db.close()
            raise asyncio.CancelledError()
        if not gpu_runtime_is_ready(probe_gpu()):
            if not waiting:
                db = SessionLocal()
                try:
                    update_job(db, task_id, stage="waiting_gpu", message=WAIT_MESSAGE)
                finally:
                    db.close()
                waiting = True
            await asyncio.sleep(2)
            continue
        if waiting:
            db = SessionLocal()
            try:
                update_job(db, task_id, stage="resuming", message="运算力已恢复，任务继续执行")
            finally:
                db.close()
            waiting = False
        try:
            return await asyncio.to_thread(function, *args)
        except Exception as exc:
            if not report_gpu_fault(exc):
                raise
            failures += 1
            # Back off repeated workload OOM; check cancellation during the wait.
            for _ in range(min(150, 2 ** min(failures, 7))):
                if is_cancel_requested(task_id):
                    break
                await asyncio.sleep(2)


async def _run_annotation_batch_task(
    task_id: str,
    image_paths: List[Path],
    model_path: str,
    confidence: float,
    prompt_classes: Optional[List[str]] = None,
    page_classes: Optional[List[str]] = None,
    page_bindings: Any = None,
    prompt_page_indices: Optional[List[Any]] = None,
    target_page_index: Any = None,
    model_metadata: Optional[Dict[str, Any]] = None,
):
    """后台批量 AI 标注任务；状态和逐资产结果均持久化到 PostgreSQL。"""
    try:
        db = SessionLocal()
        try:
            update_job(db, task_id, status="running", stage="loading_model", progress=0, message="正在加载模型...")
        finally:
            db.close()

        persisted_model_id = str(
            (model_metadata or {}).get("id")
            or (model_metadata or {}).get("model_id")
            or ""
        ).strip()
        resolution_db = SessionLocal()
        try:
            normalized_model_path, _, model_metadata = _resolve_annotation_request_model(
                resolution_db,
                model_path,
                persisted_model_id,
                model_metadata,
            )
        finally:
            resolution_db.close()
        supports_text_prompts = annotation_model_supports_text_prompts(
            normalized_model_path,
            model_metadata,
        )
        supports_dynamic_text_prompts = annotation_model_supports_dynamic_text_prompts(
            normalized_model_path,
            model_metadata,
        )
        effective_prompt_classes = prompt_classes or (page_classes if supports_text_prompts else []) or []
        model = await _gpu_job_call(
            task_id, load_annotation_inference_model,
            normalized_model_path,
            effective_prompt_classes,
            model_metadata,
        )
        supports_text_prompts = annotation_model_supports_text_prompts(
            normalized_model_path,
            model_metadata,
            model,
        )
        applied_prompt_classes = apply_annotation_prompt_classes(
            model,
            effective_prompt_classes,
            supports_text_prompts=supports_text_prompts,
            supports_dynamic_text_prompts=supports_dynamic_text_prompts,
        )
        routing = build_annotation_class_routing(
            model,
            page_classes,
            page_bindings,
            prompt_classes=applied_prompt_classes,
            prompt_page_indices=prompt_page_indices,
            target_page_index=target_page_index,
            text_prompt_model=supports_text_prompts,
            dynamic_text_prompt_model=supports_dynamic_text_prompts,
        )

        db = SessionLocal()
        try:
            update_job(
                db,
                task_id,
                status="running",
                stage="annotating",
                progress=0,
                message=(
                    f"模型加载完成，已应用 {len(applied_prompt_classes)} 个YOLOE提示词，开始批量标注"
                if applied_prompt_classes
                else "模型加载完成，开始批量标注"
                ),
                event_payload={
                    "model_path": normalized_model_path,
                    "prompt_classes": applied_prompt_classes,
                    "supports_dynamic_text_prompts": supports_dynamic_text_prompts,
                    "annotation_classes": page_classes,
                    "annotation_class_bindings": routing["bindings"],
                    "unmapped_page_classes": routing["unmapped_page_classes"],
                    "unmapped_page_class_names": routing["unmapped_page_class_names"],
                },
            )
        finally:
            db.close()

        db = SessionLocal()
        try:
            persisted_task = db.query(JobTask).filter(JobTask.id == task_id).first()
            persisted_result = dict(persisted_task.result_json or {}) if persisted_task else {}
            persisted_input = dict(persisted_task.input_json or {}) if persisted_task else {}
        finally:
            db.close()
        total = int(persisted_input.get("total") or len(image_paths))
        processed_before = int(persisted_result.get("processed") or 0)
        success = int(persisted_result.get("success") or 0)
        failed = int(persisted_result.get("failed") or 0)
        added = int(persisted_result.get("added") or 0)
        for offset, image_path in enumerate(image_paths, start=1):
            index = processed_before + offset
            db = SessionLocal()
            try:
                current = db.query(JobTask).filter(JobTask.id == task_id).first()
                cancel_requested = bool(current and current.cancel_requested)
            finally:
                db.close()
            if cancel_requested:
                db = SessionLocal()
                try:
                    update_job(db, task_id, status="cancelled", stage="cancelled", message="任务已取消")
                finally:
                    db.close()
                return

            filename = image_path.name
            try:
                annotations = await _gpu_job_call(
                    task_id,
                    detect_annotations_for_image_path,
                    model,
                    image_path,
                    confidence,
                    routing,
                )
                annotations = enforce_annotation_prompt_results(annotations, applied_prompt_classes)

                item = {
                    "asset_key": str(image_path.resolve()),
                    "filename": filename,
                    "name": filename,
                    "path": str(image_path),
                    "saved_path": f"./data/image/{filename}",
                    "url": build_uploaded_image_url(filename),
                    "annotations": annotations,
                    "count": len(annotations),
                    "status": "success"
                }
                success += 1
                added += len(annotations)

            except HTTPException:
                raise
            except Exception as e:
                logger.warning(f"批量 AI 标注单图失败: task={task_id}, image={image_path}, error={e}")
                item = {
                    "asset_key": str(image_path.resolve()),
                    "filename": filename,
                    "name": filename,
                    "path": str(image_path),
                    "saved_path": f"./data/image/{filename}",
                    "url": build_uploaded_image_url(filename),
                    "annotations": [],
                    "count": 0,
                    "status": "failed",
                    "error": str(e)
                }
                failed += 1

            summary = {
                "total": total,
                "processed": index,
                "success": success,
                "failed": failed,
                "added": added,
                "current_filename": filename,
                "model_path": normalized_model_path,
                "confidence": confidence,
                "prompt_classes": applied_prompt_classes,
                "annotation_classes": page_classes,
                "annotation_class_bindings": routing["bindings"],
                "unmapped_page_classes": routing["unmapped_page_classes"],
                "unmapped_page_class_names": routing["unmapped_page_class_names"],
            }
            db = SessionLocal()
            try:
                append_job_event(
                    db,
                    task_id,
                    event_type="annotation_result",
                    message=f"已处理 {filename}",
                    level="warning" if item["status"] == "failed" else "info",
                    payload={"item": item},
                )
                update_job(
                    db,
                    task_id,
                    status="running",
                    stage="annotating",
                    progress=round(index / max(total, 1) * 100, 2),
                    message=f"正在标注 {index}/{total}：{filename}",
                    result=summary,
                    event_type="progress",
                )
            finally:
                db.close()

        summary["current_filename"] = ""
        db = SessionLocal()
        try:
            update_job(
                db,
                task_id,
                status="completed",
                stage="completed",
                progress=100,
                message="批量 AI 标注完成",
                result=summary,
            )
        finally:
            db.close()

    except HTTPException as e:
        db = SessionLocal()
        try:
            update_job(db, task_id, status="failed", stage="failed", message=e.detail, error_code="annotation_failed", error_detail=e.detail)
        finally:
            db.close()
    except Exception as e:
        logger.error(f"批量 AI 标注任务失败: task={task_id}, error={e}")
        db = SessionLocal()
        try:
            update_job(db, task_id, status="failed", stage="failed", message=f"批量 AI 标注失败: {str(e)}", error_code="annotation_failed", error_detail=str(e))
        finally:
            db.close()


async def run_annotation_batch_task(
    task_id: str,
    image_paths: List[Path],
    model_path: str,
    confidence: float,
    prompt_classes: Optional[List[str]] = None,
    page_classes: Optional[List[str]] = None,
    page_bindings: Any = None,
    prompt_page_indices: Optional[List[Any]] = None,
    target_page_index: Any = None,
    model_metadata: Optional[Dict[str, Any]] = None,
):
    """串行执行批量标注，避免多个 YOLO/CUDA 任务并发导致 API 进程崩溃。"""
    try:
        if annotation_batch_inference_lock.locked():
            db = SessionLocal()
            try:
                update_job(
                    db,
                    task_id,
                    status="pending",
                    stage="queued",
                    progress=0,
                    message="已有批量标注任务运行，当前任务正在排队",
                )
            finally:
                db.close()

        async with annotation_batch_inference_lock:
            db = SessionLocal()
            try:
                task = db.query(JobTask).filter(JobTask.id == task_id).first()
                if task and task.cancel_requested:
                    update_job(db, task_id, status="cancelled", stage="cancelled", message="任务已取消")
                    return
            finally:
                db.close()

            await _run_annotation_batch_task(
                task_id,
                image_paths,
                model_path,
                confidence,
                prompt_classes,
                page_classes,
                page_bindings,
                prompt_page_indices,
                target_page_index,
                model_metadata,
            )
    finally:
        annotation_batch_task_handles.pop(task_id, None)


def build_annotation_batch_status_payload(task_id: str, request: Optional[Request] = None) -> Dict[str, Any]:
    """从 PostgreSQL 构造批量 AI 标注状态，完整结果由分页接口提供。"""
    db = SessionLocal()
    try:
        task = db.query(JobTask).filter(JobTask.id == task_id, JobTask.type == "annotation").first()
        if not task:
            raise HTTPException(status_code=404, detail="批量标注任务不存在")
        if request:
            require_job_visibility(task, request)
        task_info = {**(task.input_json or {}), **(task.result_json or {})}
        task_payload = task.to_dict()
    finally:
        db.close()

    total = int(task_info.get("total", 0) or 0)
    processed = int(task_info.get("processed", 0) or 0)
    percentage = round((processed / total * 100), 2) if total > 0 else 0.0

    payload = {
        "task_id": task_id,
        "status": task_payload["status"],
        "message": task_payload["message"],
        "total": total,
        "processed": processed,
        "success": int(task_info.get("success", 0) or 0),
        "failed": int(task_info.get("failed", 0) or 0),
        "added": int(task_info.get("added", 0) or 0),
        "percentage": percentage,
        "current_filename": task_info.get("current_filename", ""),
        "model_path": task_info.get("model_path", ""),
        "model_id": task_info.get("model_id", ""),
        "confidence": task_info.get("confidence", 0.25),
        "prompt_classes": task_info.get("prompt_classes", []),
        "annotation_classes": task_info.get("annotation_classes"),
        "annotation_class_bindings": task_info.get("annotation_class_bindings", []),
        "unmapped_page_classes": task_info.get("unmapped_page_classes", []),
        "unmapped_page_class_names": task_info.get("unmapped_page_class_names", []),
        "created_at": task_payload.get("created_at"),
        "updated_at": task_payload.get("updated_at"),
        "completed_at": task_payload.get("finished_at"),
        "error": task_payload.get("error_detail")
    }
    return payload


@app.post("/annotation/ai-assist-batch/start")
@app.post("/api/annotation/ai-assist-batch/start")
async def start_ai_assist_batch_annotation(request: Dict[str, Any], http_request: Request, db=Depends(get_db)):
    """
    启动后端批量 AI 标注任务。

    与前端逐张请求 /annotation/ai-assist 不同，这个接口只创建任务，立即返回 task_id；
    后端随后直接读取指定数据集或 ./data/image 中的图片顺序推理，前端轮询状态即可。
    """
    model_path = str(request.get("model_path") or request.get("model") or "yolo11n.pt").strip() or "yolo11n.pt"
    confidence = _safe_ratio(request.get("confidence", constants.DEFAULT_CONFIDENCE), constants.DEFAULT_CONFIDENCE)
    search = str(request.get("search") or "").strip()
    dataset_id = str(request.get("dataset_id") or request.get("datasetId") or "").strip()
    prompt_classes = get_annotation_prompt_classes_from_request(request)
    raw_page_classes = request.get("annotation_classes", request.get("page_classes"))
    try:
        page_classes = normalize_class_names(raw_page_classes) if raw_page_classes is not None else None
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    page_bindings = request.get("annotation_class_bindings", request.get("class_bindings"))
    prompt_page_indices = request.get("prompt_page_indices")
    target_page_index = request.get("target_page_index")
    model_metadata = request.get("model_metadata") if isinstance(request.get("model_metadata"), dict) else {}
    requested_model_id = str(
        request.get("model_id")
        or model_metadata.get("id")
        or model_metadata.get("model_id")
        or ""
    ).strip()
    normalized_model_path, registry_model, model_metadata = _resolve_annotation_request_model(
        db,
        model_path,
        requested_model_id,
        model_metadata,
    )

    source_version = None
    if dataset_id:
        source_version = (
            db.query(DatasetVersion)
            .filter(
                DatasetVersion.dataset_id == dataset_id,
                DatasetVersion.dataset_role == "original",
                DatasetVersion.status == "ready",
            )
            .order_by(DatasetVersion.created_at.desc())
            .first()
        )
        if source_version and page_classes is None:
            page_classes, _ = get_dataset_annotation_classes(source_version)
            metadata = dict(source_version.metadata_json or {})
            page_bindings = metadata.get(ANNOTATION_CLASS_BINDINGS_METADATA_KEY) or []
    draft = None
    annotation_draft_id = str(request.get("annotation_draft_id") or "").strip()
    if annotation_draft_id:
        current_user = get_request_current_user(http_request)
        draft = _annotation_request_draft(db, dataset_id, annotation_draft_id, current_user, normalized_model_path)
        if isolated_draft(draft):
            source_version = _draft_working_version(db, dataset_id, draft, source_version)
            if page_classes is not None and page_classes != list(draft.class_names_json or []):
                raise HTTPException(status_code=409, detail="标注会话类别已变化，请先保存草稿")
            page_classes = list(draft.class_names_json or [])
            page_bindings = list(draft.class_bindings_json or [])
    normalized_request_bindings = normalize_annotation_class_bindings(page_bindings, page_classes)
    request_signature = build_annotation_batch_request_signature(
        dataset_id,
        search,
        normalized_model_path,
        confidence,
        prompt_classes,
        prompt_page_indices,
        target_page_index,
        page_classes,
        normalized_request_bindings,
        model_metadata,
    )

    current_user = get_request_current_user(http_request)
    owner_username = str(current_user.get("username") or get_request_current_username(http_request))
    active_jobs = (
        db.query(JobTask)
        .filter(
            JobTask.type == "annotation",
            JobTask.owner_username == owner_username,
            JobTask.status.in_({"pending", "running"}),
        )
        .all()
    )
    matching_jobs = [
        task
        for task in active_jobs
        if str((task.input_json or {}).get("annotation_draft_id") or "") == annotation_draft_id
        and annotation_batch_job_matches_signature(task.input_json or {}, request_signature)
        and (task.input_json or {}).get("annotation_mode", "replace") == request.get("annotation_mode", "replace")
    ]
    if matching_jobs:
        existing_job = max(matching_jobs, key=lambda task: float(task.progress or 0))
        return build_annotation_batch_status_payload(existing_job.id, http_request)

    image_paths = list_annotation_image_paths_for_batch(dataset_id=source_version.dataset_id if source_version else dataset_id, search=search)
    if not image_paths:
        if dataset_id:
            raise HTTPException(status_code=400, detail="当前数据集下没有可标注图片")
        raise HTTPException(status_code=400, detail="./data/image 下没有可标注图片")

    # 启动前先验证模型是否存在，避免创建任务后立即失败但前端误以为开始了。
    source_version_id = source_version.id if source_version else ""
    input_data = {
        "total": len(image_paths),
        "annotation_draft_id": annotation_draft_id,
        "annotation_draft_source_version_id": draft.source_version_id if draft else "",
        "annotation_draft_model_path": draft.model_path if draft else "",
        "annotation_draft_class_bindings": draft.class_bindings_json if draft else [],
        "annotation_mode": "append" if request.get("annotation_mode") == "append" else "replace",
        "model_path": normalized_model_path,
        "model_id": str(registry_model.id) if registry_model is not None else "",
        "confidence": confidence,
        "prompt_classes": prompt_classes,
        "prompt_page_indices": prompt_page_indices or [],
        "target_page_index": target_page_index,
        "annotation_classes": page_classes,
        "annotation_class_bindings": normalized_request_bindings,
        "model_metadata": model_metadata,
        "search": search,
        "dataset_id": dataset_id,
        "source_version_id": source_version_id,
    }
    job = create_job(
        db,
        job_type="annotation",
        owner_username=owner_username,
        owner_id=current_user.get("id"),
        input_data=input_data,
        message="批量 AI 标注任务已创建",
    )
    task_id = job.id

    task = asyncio.create_task(
        run_annotation_batch_task(
            task_id,
            image_paths,
            normalized_model_path,
            confidence,
            prompt_classes,
            page_classes,
            input_data["annotation_class_bindings"],
            input_data["prompt_page_indices"],
            target_page_index,
            model_metadata,
        )
    )
    annotation_batch_task_handles[task_id] = task

    return build_annotation_batch_status_payload(task_id, http_request)


@app.get("/annotation/ai-assist-batch/status/{task_id}")
@app.get("/api/annotation/ai-assist-batch/status/{task_id}")
async def get_ai_assist_batch_annotation_status(task_id: str, request: Request):
    """获取后端批量 AI 标注任务进度。"""
    return build_annotation_batch_status_payload(task_id, request)


@app.get("/annotation/ai-assist-batch/results/{task_id}")
@app.get("/api/annotation/ai-assist-batch/results/{task_id}")
async def get_ai_assist_batch_annotation_results(
    task_id: str,
    request: Request,
    page: int = 1,
    page_size: int = 500,
    db=Depends(get_db),
):
    """分页获取后端批量 AI 标注结果，避免一次返回上万张图片的标注结果。"""
    task = db.query(JobTask).filter(JobTask.id == task_id, JobTask.type == "annotation").first()
    if not task:
        raise HTTPException(status_code=404, detail="批量标注任务不存在")
    require_job_visibility(task, request)

    page = max(1, int(page or 1))
    page_size = max(1, min(int(page_size or 500), 2000))

    result_query = db.query(JobEvent).filter(JobEvent.task_id == task_id, JobEvent.event_type == "annotation_result")
    total = result_query.count()
    start = (page - 1) * page_size
    events = result_query.order_by(JobEvent.sequence.asc()).offset(start).limit(page_size).all()
    page_items = [(event.payload_json or {}).get("item", {}) for event in events]
    task_dataset_id = str((task.input_json or {}).get("dataset_id") or "")
    if task_dataset_id:
        source_version = _get_latest_original_dataset_version(db, task_dataset_id)
        inactive_ids = get_inactive_annotation_class_ids(source_version)
        stable_classes, _ = get_dataset_annotation_classes(source_version)
        page_items = [
            {**item, "annotations": without_inactive_annotations(
                item.get("annotations"), inactive_ids, stable_classes,
            )} if isinstance(item, dict) else item
            for item in page_items
        ]

    return {
        "task_id": task_id,
        "status": task.status,
        "total": total,
        "count": len(page_items),
        "page": page,
        "page_size": page_size,
        "has_more": start + len(page_items) < total,
        "items": page_items
    }


@app.post("/annotation/ai-assist-batch/cancel/{task_id}")
@app.post("/api/annotation/ai-assist-batch/cancel/{task_id}")
async def cancel_ai_assist_batch_annotation(task_id: str, request: Request, db=Depends(get_db)):
    """请求取消后端批量 AI 标注任务。"""
    task = db.query(JobTask).filter(JobTask.id == task_id, JobTask.type == "annotation").first()
    if not task:
        raise HTTPException(status_code=404, detail="批量标注任务不存在")
    require_job_visibility(task, request)
    request_job_cancel(db, task_id)
    return build_annotation_batch_status_payload(task_id, request)


async def resume_interrupted_annotation_jobs(task_ids: List[str]) -> None:
    """服务启动时恢复可重放的批量标注任务。"""
    for task_id in task_ids:
        db = SessionLocal()
        try:
            task = db.query(JobTask).filter(JobTask.id == task_id, JobTask.type == "annotation").first()
            if not task:
                continue
            input_data = task.input_json or {}
            completed_events = db.query(JobEvent).filter(
                JobEvent.task_id == task_id,
                JobEvent.event_type == "annotation_result",
            ).all()
            completed_asset_keys = {
                str(((event.payload_json or {}).get("item") or {}).get("asset_key") or "")
                for event in completed_events
            }
            image_paths = [
                item
                for item in list_annotation_image_paths_for_batch(
                    dataset_id=str(input_data.get("dataset_id") or ""),
                    search=str(input_data.get("search") or ""),
                )
                if str(item.resolve()) not in completed_asset_keys
            ]
            if not image_paths and completed_asset_keys:
                update_job(db, task_id, status="completed", stage="completed", progress=100, message="服务重启后确认批量标注已完成")
                continue
            if not image_paths:
                update_job(db, task_id, status="failed", stage="failed", message="服务重启后无法恢复：输入图片不存在", error_code="resume_inputs_missing")
                continue
            task.cancel_requested = False
            db.commit()
            update_job(db, task_id, status="pending", stage="queued", progress=0, message="服务重启后重新排队")
            handle = asyncio.create_task(
                run_annotation_batch_task(
                    task_id,
                    image_paths,
                    str(input_data.get("model_path") or "yolo11n.pt"),
                    float(input_data.get("confidence") or constants.DEFAULT_CONFIDENCE),
                    input_data.get("prompt_classes") or [],
                    input_data.get("annotation_classes"),
                    input_data.get("annotation_class_bindings") or [],
                    input_data.get("prompt_page_indices") or [],
                    input_data.get("target_page_index"),
                    input_data.get("model_metadata") or {},
                )
            )
            annotation_batch_task_handles[task_id] = handle
        finally:
            db.close()

def get_dataset_metadata_path(dataset_path: Path) -> Path:
    """
    获取数据集元信息文件路径
    """
    return dataset_path / "metadata.json"


def read_dataset_metadata(dataset_path: Path) -> Optional[Dict[str, Any]]:
    """
    读取数据集 metadata.json。
    这个函数很轻量，不扫描图片和标注文件。
    """
    metadata_path = get_dataset_metadata_path(dataset_path)

    if not metadata_path.exists():
        return None

    try:
        with open(metadata_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"读取数据集 metadata 失败: {metadata_path}, error={e}")
        return None


def write_dataset_metadata(dataset_path: Path, dataset_info: Dict[str, Any]) -> None:
    """
    写入数据集 metadata.json。
    上传、导入、刷新统计后调用。
    """
    metadata_path = get_dataset_metadata_path(dataset_path)

    data = dict(dataset_info)
    data["updated_at"] = datetime.now().isoformat()

    if "created_at" not in data or not data["created_at"]:
        data["created_at"] = data["updated_at"]

    try:
        with open(metadata_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f"写入数据集 metadata 失败: {metadata_path}, error={e}")


def get_dataset_storage_path(dataset_path: Path) -> str:
    """Return a portable data-volume-relative dataset path for database metadata."""
    resolved_path = Path(dataset_path).resolve()
    try:
        return resolved_path.relative_to(DATA_DIR.resolve()).as_posix()
    except ValueError:
        return resolved_path.name


def sync_dataset_metadata_to_database(
    db,
    dataset_path: Path,
    dataset_info: Dict[str, Any],
    dataset_role: str,
    source_type: str,
) -> DatasetVersion:
    """Create or reuse the filesystem manifest version, then persist UI metadata in PostgreSQL."""
    version, _report, _created = create_initial_version(
        db,
        dataset_id=str(dataset_info["id"]),
        root_path=dataset_path,
        dataset_role=dataset_role,
        manifest_root=DATA_DIR / "manifests",
        created_by=str(dataset_info.get("created_by") or "system"),
    )
    persist_dataset_business_metadata(db, version, dataset_path, dataset_info, source_type)
    return version


def persist_dataset_business_metadata(
    db,
    version: DatasetVersion,
    dataset_path: Path,
    dataset_info: Dict[str, Any],
    source_type: str,
) -> None:
    """Persist list-page metadata for an existing manifest version."""
    metadata = dict(version.metadata_json or {})
    metadata.update(dataset_info)
    metadata.update({
        "version_id": version.id,
        "storage_path": get_dataset_storage_path(dataset_path),
    })
    metadata.setdefault("source_type", source_type)
    version.metadata_json = metadata
    if version.status not in {"deleting", "deleted"}:
        version.status = str(dataset_info.get("status") or version.status)
    db.commit()
    db.refresh(version)
    dataset_info["version_id"] = version.id
    dataset_info["storage_path"] = metadata["storage_path"]


def get_dataset_annotation_model(
    db,
    version: DatasetVersion,
    allow_source_fallback: bool = True,
) -> Optional[Dict[str, Any]]:
    """Return the model selected for annotation, including training-source inheritance."""
    metadata = dict(version.metadata_json or {})
    binding = metadata.get("annotation_model")
    if isinstance(binding, dict):
        path = str(binding.get("path") or binding.get("model_path") or "").strip()
        if path:
            return {
                **binding,
                "path": path,
                "name": str(binding.get("name") or binding.get("model_name") or path).strip() or path,
                "geometry": str(binding.get("geometry") or metadata.get("annotation_geometry") or "").strip().lower(),
            }

    path = str(metadata.get("annotation_model_path") or "").strip()
    if path:
        return {
            "path": path,
            "name": str(metadata.get("annotation_model_name") or path).strip() or path,
            "geometry": str(metadata.get("annotation_geometry") or "").strip().lower(),
        }

    if allow_source_fallback and version.dataset_role == "training":
        source_dataset_id = str(metadata.get("source_dataset_id") or "").strip()
        if source_dataset_id:
            source_version = (
                db.query(DatasetVersion)
                .filter(
                    DatasetVersion.dataset_id == source_dataset_id,
                    DatasetVersion.dataset_role == "original",
                    DatasetVersion.status == "ready",
                )
                .order_by(DatasetVersion.created_at.desc(), DatasetVersion.id.desc())
                .first()
            )
            if source_version:
                return get_dataset_annotation_model(db, source_version, allow_source_fallback=False)
    return None


def build_annotation_model_binding(
    model_path: Any,
    model_name: Any = None,
    geometry: Any = None,
    *,
    selected_by: Optional[str] = None,
    selected_at: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    """Normalize a model selected for one annotation/export session."""
    path = str(model_path or "").strip()
    if not path:
        return None
    normalized_geometry = str(geometry or "").strip().lower()
    if normalized_geometry in {"rect", "bbox"}:
        normalized_geometry = "rect"
    elif normalized_geometry in {"polygon", "segmentation", "segment"}:
        normalized_geometry = "polygon"
    else:
        normalized_geometry = model_annotation_geometry(path)
    binding = {
        "path": path,
        "name": str(model_name or path).strip() or path,
        "geometry": normalized_geometry,
    }
    if selected_by:
        binding["selected_by"] = str(selected_by)
    if selected_at:
        binding["selected_at"] = str(selected_at)
    return binding


def annotation_model_matches(requested_model: str, binding: Optional[Dict[str, Any]]) -> bool:
    if not binding:
        return True
    requested = str(requested_model or "").strip().lower()
    requested_name = Path(requested).name
    candidates = {
        str(binding.get("path") or "").strip().lower(),
        str(binding.get("name") or "").strip().lower(),
    }
    return bool(
        requested
        and any(
            requested == item or requested_name == Path(item).name
            for item in candidates
            if item
        )
    )


def build_dataset_info_from_database_version(
    version: DatasetVersion,
    asset_stats: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Return the legacy dataset-list payload from persisted database metadata."""
    metadata = dict(version.metadata_json or {})
    asset_stats = asset_stats or {}
    image_count = int(asset_stats.get("image_count") or 0)
    label_count = int(asset_stats.get("label_count") or 0)
    size = int(asset_stats.get("size") or 0)
    cover_asset_id = asset_stats.get("cover_asset_id")
    metadata.setdefault("id", version.dataset_id)
    metadata.setdefault("name", version.dataset_id)
    metadata.setdefault("path", version.root_path)
    metadata.setdefault("dataset_role", version.dataset_role)
    metadata.setdefault("status", version.status)
    metadata.setdefault("created_by", version.created_by)
    metadata.setdefault("uploader", version.created_by)
    metadata.setdefault("upload_user", version.created_by)
    metadata.setdefault("created_at", version.created_at.isoformat() if version.created_at else None)
    metadata["updated_at"] = version.created_at.isoformat() if version.created_at else metadata.get("updated_at")
    metadata["image_count"] = image_count
    metadata["label_count"] = label_count
    metadata["size"] = size
    metadata["version_id"] = version.id
    metadata["class_map"] = version.class_map_json or {}
    metadata["split_config"] = version.split_config_json or {}
    metadata["cover_url"] = (
        f"/api/datasets/{version.dataset_id}/assets/{cover_asset_id}/thumbnail?max_size=320"
        if cover_asset_id is not None
        else ""
    )
    return metadata


DEFAULT_ANNOTATION_CLASSES = ["object"]
ANNOTATION_CLASSES_METADATA_KEY = "annotation_classes"
ANNOTATION_CLASS_LABELS_METADATA_KEY = "annotation_class_labels_zh"
ANNOTATION_INACTIVE_CLASS_IDS_METADATA_KEY = "annotation_inactive_class_ids"


def get_inactive_annotation_class_ids(version: DatasetVersion) -> List[int]:
    classes, _ = get_dataset_annotation_classes(version)
    raw = (version.metadata_json or {}).get(ANNOTATION_INACTIVE_CLASS_IDS_METADATA_KEY) or []
    if not isinstance(raw, list):
        return []
    return sorted({value for item in raw if (value := _coerce_annotation_class_id(item)) is not None
                   and value < len(classes)})


def without_inactive_annotations(annotations: Any, inactive_ids: List[int], classes: List[str]) -> List[Any]:
    if not isinstance(annotations, list) or not inactive_ids:
        return annotations if isinstance(annotations, list) else []
    def is_retired(item: Any) -> bool:
        if not isinstance(item, dict):
            return False
        page_id = _coerce_annotation_class_id(
            item.get("page_class_id", item.get("class", item.get("class_id")))
        )
        if page_id is not None:
            return page_id in inactive_ids
        name = item.get("page_class_name", item.get("class_name"))
        exact_ids = [index for index, value in enumerate(classes) if str(value) == str(name)]
        if exact_ids:
            return all(index in inactive_ids for index in exact_ids)
        alias_ids = [index for index, value in enumerate(classes)
                     if _annotation_class_key(value) == _annotation_class_key(name)]
        return bool(alias_ids) and all(index in inactive_ids for index in alias_ids)
    return [item for item in annotations if not is_retired(item)]




def get_dataset_annotation_classes(version: DatasetVersion) -> Tuple[List[str], str]:
    """Return dataset-scoped annotation classes, never the legacy global list."""
    metadata = dict(version.metadata_json or {})
    classes = ordered_class_names(metadata.get(ANNOTATION_CLASSES_METADATA_KEY))
    if classes:
        return classes, str(metadata.get("annotation_classes_source") or "dataset")
    return list(DEFAULT_ANNOTATION_CLASSES), "default_new"


def get_dataset_annotation_class_context(version: DatasetVersion) -> Dict[str, Any]:
    """Return persisted page classes and optional native-model bindings."""
    classes, source = get_dataset_annotation_classes(version)
    metadata = dict(version.metadata_json or {})
    return {
        "classes": classes,
        "inactive_class_ids": get_inactive_annotation_class_ids(version),
        "labels_zh": {name: label for name, label in (metadata.get(ANNOTATION_CLASS_LABELS_METADATA_KEY) or {}).items()
                      if name in classes and isinstance(label, str) and label.strip()}
        if isinstance(metadata.get(ANNOTATION_CLASS_LABELS_METADATA_KEY), dict) else {},
        "bindings": normalize_annotation_class_bindings(
            metadata.get(ANNOTATION_CLASS_BINDINGS_METADATA_KEY),
            classes,
        ),
        "source": source,
    }


ANNOTATION_MODEL_ELIGIBLE_STATUSES = ("active", "candidate")


def _normalized_model_reference_keys(value: Any) -> set:
    """Return canonical path aliases without collapsing unrelated same-name models."""
    raw = str(value or "").strip().lower().replace("\\", "/")
    if not raw:
        return set()
    keys = {raw}
    try:
        keys.add(str(Path(raw).resolve()).lower().replace("\\", "/"))
    except (OSError, RuntimeError, ValueError):
        pass
    aliases = {
        "/app/data/runs/": "/app/runs/",
        "/app/runs/": "/app/data/runs/",
        "/app/data/models/": "/app/models/",
        "/app/models/": "/app/data/models/",
    }
    for key in tuple(keys):
        for source, target in aliases.items():
            if key.startswith(source):
                keys.add(target + key[len(source):])
    return keys


def _model_registry_record_path_matches(record: Any, requested_path: str) -> bool:
    requested = str(requested_path or "").strip().replace("\\", "/")
    if not requested or "/" not in requested:
        return False
    requested_keys = _normalized_model_reference_keys(requested)
    record_keys = _normalized_model_reference_keys(getattr(record, "path", ""))
    return bool(requested_keys.intersection(record_keys))


def _model_registry_record_matches(record: Any, requested_path: str) -> bool:
    """Match an exact path, or a basename only when the caller supplied only a basename."""
    requested = str(requested_path or "").strip().lower().replace("\\", "/")
    if not requested:
        return False
    if "/" in requested:
        return _model_registry_record_path_matches(record, requested)
    return requested == str(getattr(record, "name", "") or "").strip().lower()


def _resolve_annotation_model_record(
    db,
    requested_path: str,
    requested_model_id: Optional[str] = None,
    *,
    current_user: Optional[Dict[str, Any]] = None,
) -> Optional[Any]:
    """Resolve one eligible model by ID/path, rejecting ambiguous shared filenames."""
    raw_path = str(requested_path or "").strip().replace("\\", "/")
    model_id = str(requested_model_id or "").strip()
    eligible = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.status.in_(ANNOTATION_MODEL_ELIGIBLE_STATUSES))
        .all()
    )
    record = None

    if model_id:
        record = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
        if record is None:
            raise HTTPException(status_code=404, detail="所选模型注册记录不存在")
        if str(getattr(record, "status", "") or "") not in ANNOTATION_MODEL_ELIGIBLE_STATUSES:
            raise HTTPException(status_code=409, detail="所选模型当前不可用于标注")
        if raw_path and "/" in raw_path and not _model_registry_record_path_matches(record, raw_path):
            raise HTTPException(status_code=409, detail="模型ID与模型路径不一致")
        if raw_path and "/" not in raw_path and raw_path.lower() != str(record.name or "").lower():
            raise HTTPException(status_code=409, detail="模型ID与模型名称不一致")
    elif raw_path:
        exact_matches = [
            item for item in eligible
            if _model_registry_record_path_matches(item, raw_path)
        ] if "/" in raw_path else []
        if len(exact_matches) > 1:
            raise HTTPException(status_code=409, detail="模型路径对应多个注册记录")
        if exact_matches:
            record = exact_matches[0]
        elif "/" not in raw_path:
            name_matches = [
                item for item in eligible
                if str(getattr(item, "name", "") or "").strip().lower() == raw_path.lower()
            ]
            if len(name_matches) > 1:
                raise HTTPException(
                    status_code=409,
                    detail="模型名称不唯一，请使用模型ID或完整模型路径",
                )
            if name_matches:
                record = name_matches[0]

    if record is not None:
        if current_user is not None and not _model_registry_record_visible_to_user(db, record, current_user):
            raise HTTPException(status_code=403, detail="当前用户无权使用该训练模型")
        if not Path(record.path).is_file():
            raise HTTPException(status_code=409, detail="所选模型文件已不可用")
    return record


def _resolve_annotation_request_model(
    db,
    requested_path: str,
    requested_model_id: Optional[str] = None,
    request_metadata: Optional[Dict[str, Any]] = None,
    *,
    current_user: Optional[Dict[str, Any]] = None,
) -> Tuple[str, Optional[Any], Dict[str, Any]]:
    """Resolve the model once and return a server-authoritative path and metadata."""
    raw_path = str(requested_path or "").strip() or "yolo11n.pt"
    registry_record = _resolve_annotation_model_record(
        db,
        raw_path,
        requested_model_id,
        current_user=current_user,
    )
    if registry_record is not None:
        resolved_path = str(resolve_registered_model_path(registry_record))
        return resolved_path, registry_record, dict(registry_record.to_dict())

    # Unregistered references are limited to explicitly known built-in weights.
    if "/" in raw_path and Path(raw_path).name.lower() not in PRETRAINED_MODEL_NAME_SET:
        raise HTTPException(status_code=409, detail="所选模型不在当前模型注册表中")
    resolved_path = normalize_ai_model_path(raw_path)
    metadata = dict(request_metadata or {}) if isinstance(request_metadata, dict) else {}
    # Client metadata is not allowed to elevate an ordinary model to YOLOE.
    for key in (
        "id",
        "sha256",
        "model_family",
        "supports_text_prompts",
        "base_model",
        "lineage_root_model",
        "parent_model",
    ):
        metadata.pop(key, None)
    metadata["supports_text_prompts"] = model_supports_text_prompts(resolved_path)
    metadata["model_family"] = "yoloe" if metadata["supports_text_prompts"] else "yolo"
    return resolved_path, None, metadata


def _model_registry_record_visible_to_user(db, record: Any, current_user: Dict[str, Any]) -> bool:
    """Apply the server-side model visibility rule used by both /models and selection."""
    if str(getattr(record, "status", "") or "") == "missing":
        return False
    if str(getattr(record, "source", "") or "").lower() != "trained":
        return True
    if is_admin_user(current_user):
        return True
    linked_version_id = str(getattr(record, "dataset_version_id", "") or "").strip()
    if not linked_version_id:
        return False
    linked_version = db.query(DatasetVersion).filter(DatasetVersion.id == linked_version_id).first()
    if linked_version is None:
        return False
    resource = get_dataset_resource(db, linked_version.dataset_id, linked_version.dataset_role)
    return can_read_dataset(db, current_user, resource)


def _find_visible_model_record(db, model_path: str, current_user: Dict[str, Any]) -> Optional[Any]:
    return _resolve_annotation_model_record(
        db,
        model_path,
        current_user=current_user,
    )


def _validate_annotation_model_selection(
    db,
    current_user: Dict[str, Any],
    model_path: str,
    annotation_kind: str = "",
) -> Optional[Any]:
    """Validate PT-only model selection and the multimodal/YOLOE capability gate."""
    path = str(model_path or "").strip().replace("\\", "/")
    if not path or Path(path).suffix.lower() != ".pt":
        raise HTTPException(status_code=400, detail="标注模型仅支持 PT 格式")
    record = _find_visible_model_record(db, path, current_user)
    filename = Path(path).name.lower()
    if record is None and filename not in PRETRAINED_MODEL_NAME_SET:
        raise HTTPException(status_code=409, detail="所选模型不在当前用户可见模型列表中")
    if record is not None and str(getattr(record, "source", "") or "").lower() not in {"pretrained", "trained"}:
        raise HTTPException(status_code=409, detail="标注模型仅支持预训练模型或训练模型")
    model_payload = record.to_dict() if record is not None else {"name": filename, "path": path}
    supports_text_prompts = annotation_model_supports_text_prompts(path, model_payload)
    if annotation_kind == "multimodal" and not supports_text_prompts:
        raise HTTPException(status_code=409, detail="多模态标注仅支持 YOLOE 模型")
    if annotation_kind == "yolo" and supports_text_prompts:
        raise HTTPException(status_code=409, detail="普通 YOLO 标注不能选择 YOLOE 模型")
    return record


def get_training_model_class_names(
    db,
    model_path: str,
    request: Optional[Dict[str, Any]] = None,
) -> List[str]:
    """Resolve trained class order from the request, registry metadata, or linked version."""
    request = request or {}
    requested = (
        request.get("trained_class_names")
        or request.get("model_class_names")
        or request.get("class_names")
    )
    names = ordered_class_names(requested)
    if names:
        return names

    records = db.query(ModelRegistry).filter(ModelRegistry.status.in_(ANNOTATION_MODEL_ELIGIBLE_STATUSES)).all()
    for record in records:
        if not _model_registry_record_matches(record, model_path):
            continue
        payload = record.to_dict()
        names = ordered_class_names(
            payload.get("trained_class_names")
            or payload.get("class_names")
            or payload.get("names")
        )
        if names:
            return names
        linked_version_id = str(getattr(record, "dataset_version_id", "") or "")
        if linked_version_id:
            linked_version = db.query(DatasetVersion).filter(DatasetVersion.id == linked_version_id).first()
            names = ordered_class_names(linked_version.class_map_json if linked_version else None)
            if names:
                return names
    return []


def _query_latest_visible_dataset_versions(
    db,
    dataset_role: str,
    current_user: Dict[str, Any],
    name: str = "",
    creator: str = "",
    include_temporary: bool = False,
):
    visible_ids = readable_dataset_ids(db, current_user, dataset_role)
    if not visible_ids:
        return None

    filters = [
        DatasetVersion.dataset_role == dataset_role,
        DatasetVersion.status != "deleted",
        DatasetVersion.dataset_id.in_(visible_ids),
    ]
    if dataset_role == "original" and not include_temporary:
        source_type = DatasetVersion.metadata_json["source_type"].as_string()
        filters.append(or_(source_type.is_(None), source_type != "annotation_temporary"))
    normalized_creator = str(creator or "").strip()
    if normalized_creator:
        filters.append(DatasetVersion.created_by == normalized_creator)

    normalized_name = str(name or "").strip()
    if normalized_name:
        pattern = f"%{normalized_name}%"
        filters.append(or_(
            DatasetVersion.dataset_id.ilike(pattern),
            DatasetVersion.created_by.ilike(pattern),
            DatasetVersion.metadata_json["name"].as_string().ilike(pattern),
        ))

    ranked = (
        db.query(
            DatasetVersion.id.label("version_id"),
            func.row_number().over(
                partition_by=DatasetVersion.dataset_id,
                order_by=(DatasetVersion.created_at.desc(), DatasetVersion.id.desc()),
            ).label("version_rank"),
        )
        .filter(*filters)
        .subquery()
    )
    return (
        db.query(DatasetVersion)
        .join(ranked, DatasetVersion.id == ranked.c.version_id)
        .filter(ranked.c.version_rank == 1)
    )


def list_persisted_dataset_infos(
    db,
    dataset_role: str,
    page: int,
    page_size: int,
    current_user: Dict[str, Any],
    name: str = "",
    creator: str = "",
    include_temporary: bool = False,
) -> Tuple[int, List[Dict[str, Any]]]:
    """List one active latest version per dataset without scanning dataset directories."""
    latest_versions = _query_latest_visible_dataset_versions(
        db,
        dataset_role,
        current_user,
        name=name,
        creator=creator,
        include_temporary=include_temporary,
    )
    if latest_versions is None:
        return 0, []

    total = latest_versions.order_by(None).count()
    versions = (
        latest_versions
        .order_by(DatasetVersion.created_at.desc(), DatasetVersion.id.desc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    if not versions:
        return total, []

    version_ids = [version.id for version in versions]
    asset_rows = (
        db.query(
            DatasetAsset.dataset_version_id.label("version_id"),
            func.count(DatasetAsset.id).label("image_count"),
            func.count(DatasetAsset.label_path).label("label_count"),
            func.coalesce(func.sum(DatasetAsset.size), 0).label("size"),
            func.min(DatasetAsset.id).label("cover_asset_id"),
        )
        .filter(DatasetAsset.dataset_version_id.in_(version_ids))
        .group_by(DatasetAsset.dataset_version_id)
        .all()
    )
    asset_stats_by_version = {
        row.version_id: {
            "image_count": row.image_count,
            "label_count": row.label_count,
            "size": row.size,
            "cover_asset_id": row.cover_asset_id,
        }
        for row in asset_rows
    }
    dataset_ids = [version.dataset_id for version in versions]
    resources_by_dataset = {
        resource.dataset_id: resource
        for resource in db.query(DatasetResource).filter(
            DatasetResource.dataset_role == dataset_role,
            DatasetResource.dataset_id.in_(dataset_ids),
        ).all()
    }

    datasets = []
    for version in versions:
        item = build_dataset_info_from_database_version(
            version, asset_stats_by_version.get(version.id)
        )
        resource = resources_by_dataset.get(version.dataset_id)
        item.update({
            "owner_user_id": resource.owner_user_id if resource else None,
            "owner_username": resource.owner_username if resource else version.created_by,
            "owner_role_id": resource.owner_role_id if resource else None,
            "can_delete": bool(
                resource
                and (
                    get_request_current_user_id(current_user) == resource.owner_user_id
                    or current_user.get("role") == "admin"
                    or "admin" in (current_user.get("roleKeys") or [])
                )
            ),
        })
        datasets.append(item)
    return total, datasets


def list_dataset_creators(db, dataset_role: str, current_user: Dict[str, Any]) -> List[str]:
    latest_versions = _query_latest_visible_dataset_versions(db, dataset_role, current_user)
    if latest_versions is None:
        return []
    return [
        row[0]
        for row in (
            latest_versions.with_entities(DatasetVersion.created_by)
            .filter(DatasetVersion.created_by.isnot(None), DatasetVersion.created_by != "")
            .distinct()
            .order_by(DatasetVersion.created_by.asc())
            .all()
        )
    ]


def get_request_current_user_id(current_user: Dict[str, Any]) -> Optional[int]:
    try:
        return int(current_user.get("id"))
    except (TypeError, ValueError):
        return None


def delete_persisted_dataset(db, dataset_id: str, dataset_role: str, dataset_path: Path) -> Dict[str, int]:
    """Delete a managed dataset after its files and database rows can change together."""
    staged_path = None
    temporary_dataset_ids = [
        str(row[0])
        for row in db.query(AnnotationDraft.temporary_dataset_id)
        .filter(
            AnnotationDraft.dataset_id == dataset_id,
            AnnotationDraft.temporary_dataset_id.isnot(None),
            AnnotationDraft.temporary_dataset_id != "",
        )
        .distinct()
        .all()
    ] if dataset_role == "original" else []
    try:
        staged_path = stage_directory_for_deletion(dataset_path)
        deleted_versions = delete_dataset_version_records(db, dataset_id, dataset_role)
        deleted_resources = int(db.query(DatasetResource).filter(
            DatasetResource.dataset_id == dataset_id,
            DatasetResource.dataset_role == dataset_role,
        ).delete(synchronize_session=False) or 0)
        deleted_drafts = delete_drafts_for_dataset(db, dataset_id)
        db.commit()
    except Exception:
        db.rollback()
        restore_staged_directory(staged_path, dataset_path)
        raise

    try:
        purge_staged_directory(staged_path)
    except OSError:
        logger.exception("数据集隔离目录清理失败，将由后续维护清理: %s", staged_path)

    deleted_temporary_datasets = 0
    for temporary_dataset_id in temporary_dataset_ids:
        try:
            deleted_temporary_datasets += int(_delete_annotation_temporary_dataset(
                db,
                temporary_dataset_id,
                dataset_id,
            ))
        except Exception:
            logger.exception(
                "原始数据集删除后清理草稿临时数据集失败: dataset=%s temporary=%s",
                dataset_id,
                temporary_dataset_id,
            )

    return {
        "deleted_version_count": deleted_versions,
        "deleted_resource_count": deleted_resources,
        "deleted_draft_count": deleted_drafts,
        "deleted_temporary_dataset_count": deleted_temporary_datasets,
    }


def analyze_original_dataset(dataset_path: Path) -> Dict[str, int]:
    """
    递归统计原始数据集。

    原始数据集只按图片目录处理，不要求也不修复 YOLO train/val 结构。
    DatasetManager.analyze_dataset() 在存在空 train/val 目录时只统计这些目录，
    会漏掉 ZIP 中真实的原始图片目录，所以这里单独用于原始数据集列表统计。
    """
    dataset_path = Path(dataset_path)
    image_count = 0
    dataset_size = 0

    if not dataset_path.exists() or not dataset_path.is_dir():
        return {
            "image_count": 0,
            "label_count": 0,
            "size": 0
        }

    for file_path in dataset_path.rglob("*"):
        if not file_path.is_file():
            continue

        suffix = file_path.suffix.lower()
        if suffix not in IMAGE_FILE_EXTENSIONS:
            continue

        image_count += 1
        try:
            dataset_size += file_path.stat().st_size
        except OSError:
            pass

    return {
        "image_count": image_count,
        "label_count": 0,
        "size": dataset_size
    }


def build_dataset_info_from_metadata(dataset_path: Path) -> Dict[str, Any]:
    """
    从 metadata.json 快速构造数据集信息。
    如果 metadata 不存在，则返回轻量占位信息，不扫描大目录。
    """
    dataset_id = dataset_path.name
    metadata = read_dataset_metadata(dataset_path)

    if metadata:
        metadata.setdefault("id", dataset_id)
        metadata.setdefault("name", f"数据集_{dataset_id[:8]}")
        metadata.setdefault("path", str(dataset_path))
        metadata.setdefault("image_count", 0)
        metadata.setdefault("label_count", 0)
        metadata.setdefault("size", 0)
        metadata.setdefault("status", constants.DATASET_STATUS_READY)
        metadata.setdefault("dataset_role", "original")

        if (
            int(metadata.get("image_count") or 0) == 0
            and int(metadata.get("size") or 0) == 0
        ):
            stats = analyze_original_dataset(dataset_path)
            if stats.get("image_count", 0) > 0 or stats.get("size", 0) > 0:
                metadata.update(stats)
                metadata["status"] = (
                    constants.DATASET_STATUS_READY
                    if stats.get("image_count", 0) > 0
                    else constants.DATASET_STATUS_EMPTY
                )
                write_dataset_metadata(dataset_path, metadata)

        return metadata

    created_at = datetime.fromtimestamp(dataset_path.stat().st_ctime).isoformat()

    return {
        "id": dataset_id,
        "name": uploaded_datasets.get(dataset_id, {}).get("name", f"数据集_{dataset_id[:8]}"),
        "path": str(dataset_path),
        "created_at": uploaded_datasets.get(dataset_id, {}).get("created_at", created_at),
        "updated_at": None,
        "status": "unknown",
        "dataset_role": "original",
        "image_count": 0,
        "label_count": 0,
        "size": 0,
        "need_refresh": True
    }


def refresh_dataset_metadata(dataset_id: str, name: Optional[str] = None) -> Dict[str, Any]:
    """
    主动刷新某个数据集统计信息。
    注意：这个函数会扫描数据集目录，只能在上传完成或用户手动刷新时调用。
    不要在 GET /datasets 中默认调用。
    """
    dataset_path = Path(constants.DATASETS_DIR) / dataset_id

    if not dataset_path.exists() or not dataset_path.is_dir():
        raise HTTPException(status_code=404, detail="数据集不存在")

    old_info = read_dataset_metadata(dataset_path) or uploaded_datasets.get(dataset_id, {})

    stats = analyze_original_dataset(dataset_path)

    dataset_info = {
        "id": dataset_id,
        "name": name or old_info.get("name", f"数据集_{dataset_id[:8]}"),
        "path": str(dataset_path),
        "created_at": old_info.get(
            "created_at",
            datetime.fromtimestamp(dataset_path.stat().st_ctime).isoformat()
        ),
        "updated_at": datetime.now().isoformat(),
        "status": constants.DATASET_STATUS_READY if stats.get("image_count", 0) > 0 else constants.DATASET_STATUS_EMPTY,
        "dataset_role": "original",
        **stats
    }

    uploaded_datasets[dataset_id] = dataset_info
    write_dataset_metadata(dataset_path, dataset_info)

    return dataset_info

def load_datasets_from_disk():
    """
    从磁盘快速加载已存在的数据集信息。
    只读取 metadata.json，不实时扫描大数据集目录。
    """
    datasets_dir = Path(constants.DATASETS_DIR)
    if not datasets_dir.exists():
        return

    loaded_count = 0
    missing_metadata_count = 0

    for dataset_id_dir in datasets_dir.iterdir():
        if not dataset_id_dir.is_dir():
            continue

        dataset_id = dataset_id_dir.name
        dataset_info = build_dataset_info_from_metadata(dataset_id_dir)

        if dataset_info.get("need_refresh"):
            missing_metadata_count += 1

        uploaded_datasets[dataset_id] = dataset_info
        loaded_count += 1

    logger.info(
        f"从磁盘快速加载了 {loaded_count} 个数据集，"
        f"{missing_metadata_count} 个缺少 metadata，需要手动刷新统计"
    )

# 启动时加载已存在的数据集
load_datasets_from_disk()



# --- 视频源持久化与磁盘恢复 ---
VIDEO_FILE_EXTENSIONS = {
    '.mp4', '.avi', '.mov', '.mkv', '.flv', '.wmv', '.webm', '.m4v', '.mpeg', '.mpg'
}
VIDEO_SOURCES_METADATA_NAME = 'video_sources.json'
VIDEO_UPLOAD_MAX_BYTES = int(os.getenv("VIDEO_UPLOAD_MAX_BYTES", str(2 * 1024**3)))
VIDEO_UPLOAD_MAX_DURATION_SECONDS = float(os.getenv("VIDEO_UPLOAD_MAX_DURATION_SECONDS", str(6 * 60 * 60)))
VIDEO_UPLOAD_CHUNK_BYTES = int(os.getenv("VIDEO_UPLOAD_CHUNK_BYTES", str(8 * 1024**2)))
VIDEO_UPLOAD_MAX_CHUNKS = int(os.getenv("VIDEO_UPLOAD_MAX_CHUNKS", "4096"))
VIDEO_UPLOAD_SESSION_TTL_SECONDS = int(os.getenv("VIDEO_UPLOAD_SESSION_TTL_SECONDS", "86400"))
VIDEO_UPLOAD_SESSION_ROOT = DATA_DIR / "temp" / "video_uploads"
ANNOTATION_VIDEO_MAX_FRAMES = int(os.getenv("ANNOTATION_VIDEO_MAX_FRAMES", "500"))
ANNOTATION_VIDEO_MAX_INTERVAL_SECONDS = float(os.getenv("ANNOTATION_VIDEO_MAX_INTERVAL_SECONDS", "3600"))
ANNOTATION_FRAME_MAX_WIDTH = int(os.getenv("ANNOTATION_FRAME_MAX_WIDTH", "1920"))
ANNOTATION_EXTRACT_MAX_RESPONSE_BYTES = int(os.getenv("ANNOTATION_EXTRACT_MAX_RESPONSE_BYTES", str(20 * 1024**2)))
GENERIC_UPLOAD_CONTENT_TYPES = {"", "application/octet-stream", "binary/octet-stream"}


def get_video_upload_dir() -> Path:
    """获取上传视频保存目录，并确保目录存在。"""
    video_dir = Path(constants.STATIC_DIR) / 'videos'
    video_dir.mkdir(parents=True, exist_ok=True)
    return video_dir


def get_video_sources_metadata_path() -> Path:
    """获取视频源持久化 metadata 文件路径。"""
    return get_video_upload_dir() / VIDEO_SOURCES_METADATA_NAME


def validate_video_upload_metadata(filename: Any, content_type: Any, size: Optional[int] = None) -> Dict[str, Any]:
    safe_name = sanitize_original_filename(filename, fallback="video.mp4")
    suffix = Path(safe_name).suffix.lower()
    normalized_content_type = str(content_type or "").strip().lower()
    if suffix not in VIDEO_FILE_EXTENSIONS:
        raise HTTPException(status_code=400, detail=f"不支持的视频扩展名: {suffix or '无'}")
    if normalized_content_type not in GENERIC_UPLOAD_CONTENT_TYPES and not normalized_content_type.startswith("video/"):
        raise HTTPException(status_code=400, detail=f"不支持的视频 MIME 类型: {normalized_content_type}")
    if size is not None:
        normalized_size = max(0, int(size))
        if normalized_size <= 0:
            raise HTTPException(status_code=400, detail="视频文件为空")
        if normalized_size > VIDEO_UPLOAD_MAX_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"视频超过上传限制 {VIDEO_UPLOAD_MAX_BYTES} bytes",
            )
    return {"filename": safe_name, "suffix": suffix, "content_type": normalized_content_type}


def _parse_ffprobe_rate(value: Any) -> float:
    text_value = str(value or "0").strip()
    try:
        if "/" in text_value:
            numerator, denominator = text_value.split("/", 1)
            return float(numerator) / float(denominator) if float(denominator) else 0.0
        return float(text_value)
    except (TypeError, ValueError, ZeroDivisionError):
        return 0.0


def probe_uploaded_video(video_path: Path) -> Dict[str, Any]:
    """Require ffprobe metadata and an actually decodable first frame."""
    video_path = Path(video_path)
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        raise RuntimeError("服务器未安装 ffprobe，无法执行视频上传预检")
    command = [
        ffprobe,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=codec_name,width,height,avg_frame_rate,nb_frames,duration:format=duration",
        "-of", "json",
        str(video_path),
    ]
    completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
    if completed.returncode != 0:
        raise ValueError(f"ffprobe 无法解析视频: {completed.stderr[-500:]}")
    probe = json.loads(completed.stdout or "{}")
    streams = probe.get("streams") or []
    if not streams:
        raise ValueError("视频文件不包含可用的视频流")
    stream = streams[0]
    duration = float((probe.get("format") or {}).get("duration") or stream.get("duration") or 0.0)
    if duration <= 0:
        raise ValueError("无法获取有效视频时长")
    if duration > VIDEO_UPLOAD_MAX_DURATION_SECONDS:
        raise HTTPException(
            status_code=413,
            detail=f"视频时长超过限制 {int(VIDEO_UPLOAD_MAX_DURATION_SECONDS)} 秒",
        )

    cap = cv2.VideoCapture(str(video_path))
    try:
        if not cap.isOpened():
            raise ValueError("OpenCV 无法打开视频")
        readable, frame = cap.read()
        if not readable or frame is None or frame.size <= 0:
            raise ValueError("视频首帧无法解码")
    finally:
        cap.release()

    fps = _parse_ffprobe_rate(stream.get("avg_frame_rate"))
    frame_count = int(stream.get("nb_frames") or round(duration * fps) or 0)
    return {
        "codec": str(stream.get("codec_name") or "unknown"),
        "fps": round(fps, 3),
        "frame_count": frame_count,
        "width": int(stream.get("width") or 0),
        "height": int(stream.get("height") or 0),
        "duration": round(duration, 3),
        "preflight": "ffprobe_and_first_frame",
    }


def register_uploaded_video(video_path: Path, original_filename: str) -> Dict[str, Any]:
    video_path = Path(video_path)
    try:
        video_info_data = probe_uploaded_video(video_path)
    except Exception:
        video_path.unlink(missing_ok=True)
        raise
    video_id = video_path.stem
    video_info = {
        "id": video_id,
        "name": original_filename,
        "original_filename": original_filename,
        "filename": video_path.name,
        "path": str(video_path),
        "url": f"/static/videos/{video_path.name}",
        "type": "file",
        "size": int(video_path.stat().st_size),
        "created_at": datetime.now().isoformat(),
        "status": "available",
        **video_info_data,
    }
    video_sources[video_id] = video_info
    save_video_sources_metadata()
    return video_info


def _video_upload_session_dir(upload_id: str) -> Path:
    normalized = str(upload_id or "").strip().lower()
    if not re.fullmatch(r"[0-9a-f]{32}", normalized):
        raise HTTPException(status_code=400, detail="无效的分片上传 ID")
    root = VIDEO_UPLOAD_SESSION_ROOT.resolve()
    session_dir = (root / normalized).resolve()
    if session_dir.parent != root:
        raise HTTPException(status_code=400, detail="无效的分片上传目录")
    return session_dir


def _read_video_upload_session(upload_id: str) -> Tuple[Path, Dict[str, Any]]:
    session_dir = _video_upload_session_dir(upload_id)
    metadata_path = session_dir / "metadata.json"
    if not metadata_path.exists():
        raise HTTPException(status_code=404, detail="分片上传会话不存在")
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, json.JSONDecodeError) as exc:
        raise HTTPException(status_code=409, detail=f"分片上传元数据损坏: {exc}")
    return session_dir, metadata


def _write_video_upload_session(session_dir: Path, metadata: Dict[str, Any]) -> None:
    metadata_path = session_dir / "metadata.json"
    temporary = session_dir / ".metadata.tmp"
    temporary.write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    os.replace(temporary, metadata_path)


def cleanup_stale_video_upload_sessions(now: Optional[float] = None) -> int:
    root = VIDEO_UPLOAD_SESSION_ROOT.resolve()
    root.mkdir(parents=True, exist_ok=True)
    current_time = float(now if now is not None else time.time())
    removed = 0
    for candidate in root.iterdir():
        try:
            if candidate.is_symlink() or not candidate.is_dir() or candidate.resolve().parent != root:
                continue
            age = current_time - candidate.stat().st_mtime
            if age <= VIDEO_UPLOAD_SESSION_TTL_SECONDS:
                continue
            shutil.rmtree(candidate)
            removed += 1
        except OSError as exc:
            logger.warning("清理过期视频分片会话失败: path=%s error=%s", candidate, exc)
    return removed


def _complete_video_chunk_upload_sync(upload_id: str) -> Dict[str, Any]:
    session_dir, metadata = _read_video_upload_session(upload_id)
    total_chunks = int(metadata["total_chunks"])
    chunk_paths = [session_dir / f"{index}.part" for index in range(total_chunks)]
    missing = [index for index, path in enumerate(chunk_paths) if not path.exists()]
    if missing:
        raise HTTPException(status_code=409, detail={"message": "仍有分片未上传", "missing_chunks": missing[:100]})
    video_path = build_unique_upload_path(
        get_video_upload_dir(),
        metadata["filename"],
        fallback="video.mp4",
    )
    written = 0
    try:
        with video_path.open("wb") as output:
            for chunk_path in chunk_paths:
                with chunk_path.open("rb") as source:
                    while True:
                        block = source.read(1024 * 1024)
                        if not block:
                            break
                        written += len(block)
                        if written > VIDEO_UPLOAD_MAX_BYTES:
                            raise HTTPException(status_code=413, detail="合并后视频超过上传限制")
                        output.write(block)
        if written != int(metadata["size"]):
            raise HTTPException(status_code=409, detail=f"合并文件大小错误: expected={metadata['size']}, actual={written}")
        video_info = register_uploaded_video(video_path, metadata.get("original_filename") or metadata["filename"])
    except Exception:
        video_path.unlink(missing_ok=True)
        raise
    try:
        shutil.rmtree(session_dir)
    except OSError as exc:
        logger.warning("清理已完成分片上传目录失败: %s", exc)
    return video_info


def _normalize_video_file_path(path_value: Any) -> Path:
    """把 metadata/url/path 中的视频地址统一转换为容器内文件路径。"""
    text = str(path_value or '').strip()

    if text.startswith('/static/videos/'):
        return Path(constants.STATIC_DIR) / 'videos' / text.split('/static/videos/', 1)[1]

    if text.startswith('static/videos/') or text.startswith('static\\videos\\'):
        return Path(text)

    candidate = Path(text)
    if candidate.is_absolute() or text:
        return candidate

    return get_video_upload_dir()




def restore_video_file_original_name_if_possible(video_path: Path, metadata: Dict[str, Any] = None) -> Path:
    """
    兼容旧版本已经保存成 video_<uuid>.mp4 的历史文件。

    如果 video_sources.json 中仍保存着 original_filename，则在加载视频源时把磁盘文件
    重命名回用户上传时的文件名。没有 original_filename 的历史文件无法可靠还原。
    """
    metadata = metadata or {}
    video_path = Path(video_path)

    original_filename = metadata.get('original_filename') or metadata.get('original_name')
    if not original_filename:
        return video_path

    expected_name = sanitize_original_filename(original_filename, fallback=video_path.name)
    if not expected_name or expected_name == video_path.name:
        return video_path

    if not video_path.exists() or not video_path.is_file():
        return video_path

    try:
        target_path = build_unique_upload_path(video_path.parent, expected_name, fallback=video_path.name)
        if target_path.resolve() == video_path.resolve():
            return video_path

        video_path.rename(target_path)
        metadata['filename'] = target_path.name
        metadata['path'] = str(target_path)
        metadata['url'] = f'/static/videos/{target_path.name}'
        logger.info(f'已将历史视频文件恢复为原始文件名: {video_path.name} -> {target_path.name}')
        return target_path
    except Exception as e:
        logger.warning(f'恢复历史视频原始文件名失败: {video_path}, error={e}')
        return video_path


def _build_video_source_from_file(video_path: Path, metadata: Dict[str, Any] = None) -> Optional[Dict[str, Any]]:
    """从 static/videos 中的实际视频文件构造一个可用于前端的视频源。"""
    metadata = metadata or {}
    video_path = Path(video_path)

    if not video_path.exists() or not video_path.is_file():
        return None

    if video_path.suffix.lower() not in VIDEO_FILE_EXTENSIONS:
        return None

    # 兼容旧版本：如果 metadata 中保存了 original_filename，则把 video_<uuid>.mp4 恢复成原始文件名。
    video_path = restore_video_file_original_name_if_possible(video_path, metadata)

    # 文件名现在优先保留上传原名，source_id 默认使用文件名 stem。
    stem = video_path.stem
    if stem.startswith('video_') and len(stem) > len('video_'):
        source_id = metadata.get('id') or stem.replace('video_', '', 1)
    else:
        source_id = metadata.get('id') or stem

    try:
        stat = video_path.stat()
    except Exception:
        stat = None

    video_info_data = {}
    try:
        cap = video_processor.create_capture('file', str(video_path))
        if cap:
            video_info_data = video_processor.get_video_info(cap) or {}
            video_processor.release_capture(cap)
    except Exception as e:
        logger.warning(f'读取已上传视频信息失败: {video_path}, error={e}')

    file_name = video_path.name
    source_info = {
        'id': str(source_id),
        'name': metadata.get('name') or metadata.get('original_filename') or file_name,
        'original_filename': metadata.get('original_filename') or metadata.get('name') or file_name,
        'filename': file_name,
        'path': str(video_path),
        'url': f'/static/videos/{file_name}',
        'type': 'file',
        'size': metadata.get('size') or (stat.st_size if stat else 0),
        'created_at': metadata.get('created_at') or (
            datetime.fromtimestamp(stat.st_ctime).isoformat() if stat else datetime.now().isoformat()
        ),
        'status': 'available',
        **video_info_data
    }

    # 保留 metadata 中其它非空字段，但不覆盖上面根据真实文件校准的 path/url/type/status。
    for key, value in metadata.items():
        if key not in source_info and value not in (None, ''):
            source_info[key] = value

    return source_info


def save_video_sources_metadata() -> None:
    """把当前视频源列表写入 static/videos/video_sources.json，避免刷新/重启后丢失。"""
    metadata_path = get_video_sources_metadata_path()

    serializable_sources = {}
    for source_id, source_info in video_sources.items():
        if not isinstance(source_info, dict):
            continue

        data = dict(source_info)
        data['id'] = str(source_id)
        serializable_sources[str(source_id)] = data

    payload = {
        'updated_at': datetime.now().isoformat(),
        'sources': serializable_sources
    }

    try:
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.warning(f'保存视频源 metadata 失败: {metadata_path}, error={e}')


def load_video_sources_from_disk() -> None:
    """
    从 static/videos/video_sources.json 和 static/videos 实际文件恢复视频源。

    作用：
    1. 页面刷新后前端重新请求 /video/sources 时仍能看到历史上传视频；
    2. 后端热重载、容器重启后，内存 video_sources 自动从磁盘恢复；
    3. 即使 metadata 丢失，也能扫描 static/videos 下已有视频文件并注册。
    """
    video_dir = get_video_upload_dir()
    metadata_path = get_video_sources_metadata_path()

    loaded_sources: Dict[str, Dict[str, Any]] = {}

    # 先读取 metadata，保留原始文件名、上传时间等信息。
    if metadata_path.exists():
        try:
            with open(metadata_path, 'r', encoding='utf-8') as f:
                payload = json.load(f) or {}

            raw_sources = payload.get('sources', payload)
            if isinstance(raw_sources, list):
                raw_sources = {
                    str(item.get('id') or item.get('source_id') or idx): item
                    for idx, item in enumerate(raw_sources)
                    if isinstance(item, dict)
                }

            if isinstance(raw_sources, dict):
                for source_id, source_info in raw_sources.items():
                    if not isinstance(source_info, dict):
                        continue

                    source_type = source_info.get('type') or source_info.get('source_type')

                    if source_type == 'file':
                        video_path = _normalize_video_file_path(
                            source_info.get('path') or source_info.get('source_url') or source_info.get('url')
                        )
                        restored = _build_video_source_from_file(video_path, {**source_info, 'id': source_id})
                        if restored:
                            loaded_sources[restored['id']] = restored
                    else:
                        # RTSP/HTTP 源也一起持久化，页面刷新后不需要重新添加。
                        source_url = source_info.get('url') or source_info.get('source_url') or ''
                        restored = {
                            **source_info,
                            'id': str(source_info.get('id') or source_id),
                            'type': source_type or source_info.get('type') or 'rtsp',
                            'url': source_url,
                            'status': source_info.get('status', 'available')
                        }
                        loaded_sources[restored['id']] = restored
        except Exception as e:
            logger.warning(f'读取视频源 metadata 失败: {metadata_path}, error={e}')

    # 再扫描实际视频目录，补齐 metadata 没记录但文件仍存在的视频。
    try:
        for video_path in video_dir.iterdir():
            if not video_path.is_file():
                continue
            if video_path.name == VIDEO_SOURCES_METADATA_NAME:
                continue
            if video_path.suffix.lower() not in VIDEO_FILE_EXTENSIONS:
                continue

            restored = _build_video_source_from_file(video_path)
            if restored and restored['id'] not in loaded_sources:
                loaded_sources[restored['id']] = restored
    except Exception as e:
        logger.warning(f'扫描已上传视频目录失败: {video_dir}, error={e}')

    # 合并：保留内存里当前刚添加的源，同时用磁盘中存在的源补齐。
    video_sources.update(loaded_sources)

    # 写回一次，保证 orphan 文件被纳入 metadata。
    if loaded_sources:
        save_video_sources_metadata()

    logger.info(f'已从磁盘恢复 {len(loaded_sources)} 个视频源')


# 启动时恢复 static/videos 中已存在的视频源
load_video_sources_from_disk()


# --- Pydantic 数据模型定义 ---

class DatasetMergeRequest(BaseModel):
    dataset_name: str
    source_dataset_ids: List[str]
    owner_role_id: Optional[int] = None


class TrainingConfig(BaseModel):
    """
    模型训练配置参数模型
    """
    model_name: str = "yolov8n.pt"  # 基础模型名称
    training_profile: str = "auto"
    epochs: Optional[int] = None
    batch_size: Optional[int] = None
    img_size: Optional[int] = None
    device: str = "auto"           # 训练设备 (cpu/0/1/auto)
    learning_rate: Optional[float] = None
    optimizer: Optional[str] = None
    patience: Optional[int] = None
    major_class_threshold: int = 20
    prepare_latest_annotations: bool = True
    class_names: List[str] = []    # 训练类别列表
    project_name: str = "training_project" # 项目保存名称


class CandidateActivationRequest(BaseModel):
    pretrained_metrics: Dict[str, Any] = {}
    current_best_metrics: Dict[str, Any]
    candidate_metrics: Dict[str, Any]
    smoke_tests: Dict[str, Any]



class InferenceConfig(BaseModel):
    """
    图片/视频推理配置参数模型
    """
    confidence: float = 0.25      # 置信度阈值
    iou_threshold: float = 0.7    # NMS IoU 阈值
    max_detections: int = 100     # 最大检测目标数
    save_results: bool = True     # 是否保存推理结果图


def normalize_inference_config(
    confidence: float = 0.25,
    iou_threshold: float = 0.7,
    max_detections: int = 100,
    max_det: Optional[int] = None,
    save_results: bool = True
) -> InferenceConfig:
    """
    统一清洗推理参数。

    前端推理接口通过 query string 传参，例如：
    /predict?confidence=0.25&iou_threshold=0.7

    这里显式接收 query 参数后构造 InferenceConfig，确保置信度阈值
    和 IoU 阈值可以真正传入 YOLO 推理流程。
    """
    try:
        confidence_value = float(confidence)
    except (TypeError, ValueError):
        confidence_value = 0.25

    try:
        iou_threshold_value = float(iou_threshold)
    except (TypeError, ValueError):
        iou_threshold_value = 0.7

    selected_max_detections = max_det if max_det is not None else max_detections
    try:
        max_detections_value = int(selected_max_detections)
    except (TypeError, ValueError):
        max_detections_value = 100

    # 限制到合理范围，避免异常参数直接进入模型推理。
    confidence_value = max(0.0, min(1.0, confidence_value))
    iou_threshold_value = max(0.0, min(1.0, iou_threshold_value))
    max_detections_value = max(1, max_detections_value)

    return InferenceConfig(
        confidence=confidence_value,
        iou_threshold=iou_threshold_value,
        max_detections=max_detections_value,
        save_results=bool(save_results)
    )



# --- 推理结果图片 URL 兜底工具 ---
IMAGE_RESULT_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def _is_http_or_static_url(value: Any) -> bool:
    """判断一个值是否已经是浏览器可访问的图片 URL。"""
    if not value:
        return False

    text = str(value).strip()
    return (
        text.startswith('/static/')
        or text.startswith('http://')
        or text.startswith('https://')
        or text.startswith('data:image/')
        or text.startswith('blob:')
    )


def _static_url_from_path(path: Path) -> Optional[str]:
    """
    将 static 目录下的文件路径转换为 /static/... URL。
    """
    try:
        static_root = Path(constants.STATIC_DIR).resolve()
        resolved_path = Path(path).resolve()
        relative_path = resolved_path.relative_to(static_root)
        return '/static/' + '/'.join(relative_path.parts)
    except Exception:
        return None


def _safe_result_file_name(original_name: str, fallback_name: str = 'image.jpg') -> str:
    """
    为标注图生成稳定文件名，避免出现 xxx.jpg_annotated 这种没有真实图片扩展名的名称。
    """
    source_name = original_name or fallback_name
    source_name = os.path.basename(str(source_name)) or fallback_name

    path = Path(source_name)
    stem = path.stem or 'image'
    suffix = path.suffix.lower()

    if suffix not in IMAGE_RESULT_EXTENSIONS:
        suffix = Path(fallback_name).suffix.lower()

    if suffix not in IMAGE_RESULT_EXTENSIONS:
        suffix = '.jpg'

    # 只清理路径分隔符，尽量保留原始文件名，方便用户辨识。
    safe_stem = stem.replace('/', '_').replace('\\', '_').strip() or 'image'
    return f'{safe_stem}_annotated{suffix}'


def _resolve_yolo_output_dir(result_type: str, run_name: str) -> Optional[Path]:
    """
    查找 Ultralytics 实际保存推理结果的目录。
    当前项目中调用形如：
    trainer.predict(save_dir='/app/runs/batch', name='batch_xxx')
    因此常见输出目录是 /app/runs/batch/batch_xxx。
    """
    candidates = [
        Path('/app/runs') / result_type / run_name,
        Path('runs') / result_type / run_name,
        Path('/app') / 'runs' / result_type / run_name,
    ]

    for candidate in candidates:
        if candidate.exists() and candidate.is_dir():
            return candidate

    return None


def _find_saved_annotated_image(output_dir: Optional[Path], input_file_name: str) -> Optional[Path]:
    """
    在 YOLO 输出目录中，根据输入图片名寻找保存后的标注图。
    """
    if not output_dir or not output_dir.exists():
        return None

    input_name = os.path.basename(str(input_file_name or ''))
    input_stem = Path(input_name).stem

    if input_name:
        exact_path = output_dir / input_name
        if exact_path.exists() and exact_path.suffix.lower() in IMAGE_RESULT_EXTENSIONS:
            return exact_path

    image_files = [
        path for path in output_dir.rglob('*')
        if path.is_file() and path.suffix.lower() in IMAGE_RESULT_EXTENSIONS
    ]

    if input_stem:
        for path in image_files:
            if path.stem == input_stem or path.name == input_name:
                return path

    return image_files[0] if len(image_files) == 1 else None


def _copy_annotated_image_to_static(
    source_path: Optional[Path],
    group_id: str,
    original_name: str,
    fallback_name: str = 'image.jpg'
) -> Optional[str]:
    """
    将 YOLO 输出的标注图复制到 static/predictions/<group_id>/ 下，并返回 /static/... URL。
    """
    if not source_path or not Path(source_path).exists():
        return None

    target_dir = Path(constants.STATIC_DIR) / 'predictions' / str(group_id)
    target_dir.mkdir(parents=True, exist_ok=True)

    target_name = _safe_result_file_name(original_name, fallback_name)
    target_path = target_dir / target_name

    try:
        shutil.copy2(str(source_path), str(target_path))
    except Exception as e:
        logger.warning(f'复制标注图失败: source={source_path}, target={target_path}, error={e}')
        return None

    return _static_url_from_path(target_path)


def _extract_valid_url_list(values: Any) -> List[str]:
    """
    从旧 ResultManager 返回值中提取可用 URL，过滤掉 2010_006324.jpg_annotated 这类不可访问字符串。
    """
    if not values:
        return []

    if isinstance(values, dict):
        iterable = values.values()
    elif isinstance(values, list):
        iterable = values
    else:
        iterable = [values]

    valid_urls = []
    for item in iterable:
        if _is_http_or_static_url(item):
            valid_urls.append(str(item))

    return valid_urls


def _build_single_annotated_image_url(
    prediction_id: str,
    results: Any,
    original_name: str
) -> Optional[str]:
    """
    为单图推理构造可靠的标注图 URL。
    """
    output_dir = _resolve_yolo_output_dir('predict', f'prediction_{prediction_id}')

    input_file_name = ''
    if results and len(results) > 0:
        input_path = getattr(results[0], 'path', '') or ''
        input_file_name = os.path.basename(input_path)

    annotated_path = _find_saved_annotated_image(output_dir, input_file_name)
    return _copy_annotated_image_to_static(
        annotated_path,
        f'prediction_{prediction_id}',
        original_name,
        input_file_name or original_name
    )


def _build_batch_annotated_image_map(
    batch_id: str,
    results: Any,
    file_name_map: Dict[str, str]
) -> Dict[str, str]:
    """
    为批量推理构造可靠的 “输入文件名 -> 标注图 URL” 映射。
    同时兼容临时文件名 image_0.jpg 和原始上传文件名 2010_006324.jpg。
    """
    output_dir = _resolve_yolo_output_dir('batch', f'batch_{batch_id}')
    url_map: Dict[str, str] = {}

    if not results:
        return url_map

    for idx, result in enumerate(results):
        input_path = getattr(result, 'path', '') or ''
        file_name = os.path.basename(input_path) if input_path else f'image_{idx}.jpg'
        original_name = file_name_map.get(file_name, file_name)

        annotated_path = _find_saved_annotated_image(output_dir, file_name)
        annotated_url = _copy_annotated_image_to_static(
            annotated_path,
            f'batch_{batch_id}',
            original_name,
            file_name
        )

        if annotated_url:
            url_map[file_name] = annotated_url
            url_map[original_name] = annotated_url
            url_map[Path(file_name).stem] = annotated_url
            url_map[Path(original_name).stem] = annotated_url

    return url_map

class OptimizedInferenceConfig(BaseModel):
    """
    针对特定场景优化的推理配置 (例如二维码识别)
    """
    confidence: float = 0.015     # 极低置信度以捕获小目标
    iou_threshold: float = 0.1     # 极低 IoU 以避免重叠过滤
    max_detections: int = 100
    save_results: bool = True
    model_type: str = "qr_code"    # 模型类型标识


class ExportConfig(BaseModel):
    """
    模型导出配置参数模型
    """
    format: str = "onnx"          # 导出格式: onnx, torchscript, tflite, coreml
    optimize: bool = True         # 是否进行导出优化
    half_precision: bool = False  # 是否使用半精度 (FP16)


class TaskStatus(BaseModel):
    """
    异步任务状态模型
    """
    task_id: str                  # 任务唯一标识
    status: str                   # 状态: pending, running, completed, failed
    progress: float = 0.0         # 进度 (0.0 - 100.0)
    message: str = ""             # 状态描述信息
    created_at: Optional[str] = None
    updated_at: Optional[str] = None
    results: Optional[Dict[str, Any]] = None
    test_model: Optional[Dict[str, Any]] = None


class RealtimeConfig(BaseModel):
    """
    实时推理会话配置模型
    """
    model_path: str = "yolov8n.pt" # 使用的模型路径
    confidence: float = 0.25       # 推理置信度
    iou_threshold: float = 0.7     # 推理 IoU
    max_fps: int = 10             # 最大处理帧率 (限制带宽和计算压力)
    frame_skip: int = 1           # 跳帧处理频率
    save_frames: bool = False     # 是否保存推理帧
    show_labels: bool = True      # 是否在返回图中显示标签
    show_confidence: bool = True  # 是否在返回图中显示置信度
    client_side_overlay: bool = False  # Browser draws detections after decoding raw preview.
    source_type: str = "webcam"    # 源类型: webcam, rtsp, http, file
    source_url: str = ""           # 源地址 (URL或路径)
    loop_video: bool = True       # 视频文件是否循环播放
    behavior_config: Dict[str, Any] = Field(default_factory=dict)


class VideoSourceConfig(BaseModel):
    """
    视频源配置模型
    """
    source_type: str              # webcam, rtsp, http, file
    source_url: str = ""          # 源地址
    name: str = ""                # 友好名称
    resolution: str = "640x480"   # 期望分辨率
    fps: int = 30                 # 期望帧率
    behavior_config: Dict[str, Any] = Field(default_factory=dict)


class RealtimeStats(BaseModel):
    """
    实时推理性能统计模型
    """
    session_id: str               # 会话ID
    frames_processed: int = 0     # 已处理总帧数
    total_detections: int = 0     # 已检测目标总数
    avg_inference_time: float = 0.0 # 平均单帧推理时间 (ms)
    current_fps: float = 0.0      # 当前实际处理帧率
    start_time: str
    last_update: str


class GridSession(BaseModel):
    """
    九宫格会话配置模型
    """
    grid_id: str
    grid_name: str = "九宫格监控"
    grid_size: str = "3x3"        # 宫格布局 (2x2, 3x3 等)
    sessions: List[str] = []      # 包含的各个子会话 ID 列表
    created_at: str
    is_active: bool = True


class GridConfig(BaseModel):
    """
    九宫格全局配置模型
    """
    grid_name: str = "九宫格监控"
    grid_size: str = "3x3"
    video_sources: List[Dict[str, Any]] = [] # 各宫格对应的视频源配置
    shared_model: str = "yolov8n.pt"         # 共享使用的模型
    shared_confidence: float = 0.25
    shared_iou_threshold: float = 0.7
    max_fps: int = 60
    client_side_overlay: bool = False
    shared_behavior_config: Dict[str, Any] = Field(default_factory=dict)


# --- 辅助逻辑函数 ---

def generate_task_id():
    """
    生成全局唯一的任务/资源 ID
    """
    return task_manager.generate_id()


def decode_base64_image(base64_string: str) -> np.ndarray:
    """
    将 Base64 编码的图片字符串解码为 OpenCV 格式 (numpy 数组)
    """
    return image_processor.decode_base64_image(base64_string)


def encode_image_to_base64(image: np.ndarray) -> str:
    """
    将 OpenCV 格式的图片 (numpy 数组) 编码为 Base64 字符串
    """
    return image_processor.encode_image_to_base64(image)


def _bounded_env_int(name: str, default: int, minimum: int, maximum: int) -> int:
    try:
        value = int(os.getenv(name, str(default)))
    except (TypeError, ValueError):
        value = default
    return max(minimum, min(maximum, value))


def encode_realtime_preview(
    image: np.ndarray,
    *,
    max_width: Optional[int] = None,
    quality: Optional[int] = None,
) -> Tuple[Optional[str], Dict[str, int]]:
    """Encode a bandwidth-bounded preview instead of the full RTSP source frame."""
    if image is None:
        return None, {}
    height, width = image.shape[:2]
    max_width = (
        _bounded_env_int("REALTIME_OUTPUT_MAX_WIDTH", 960, 320, 1920)
        if max_width is None
        else max(320, min(1920, int(max_width)))
    )
    preview = image
    if width > max_width:
        scale = max_width / float(width)
        preview = cv2.resize(
            image,
            (max_width, max(1, int(round(height * scale)))),
            interpolation=cv2.INTER_AREA,
        )
    quality = (
        _bounded_env_int("REALTIME_JPEG_QUALITY", 70, 40, 90)
        if quality is None
        else max(40, min(100, int(quality)))
    )
    payload = image_processor.encode_image_to_base64(preview, quality=quality)
    preview_height, preview_width = preview.shape[:2]
    return payload, {
        "width": int(preview_width),
        "height": int(preview_height),
        "quality": quality,
        "payload_bytes": len(payload.encode("utf-8")) if payload else 0,
    }


def configure_realtime_frame_delivery(session_id: str, supports_ack: bool = False) -> Dict[str, Any]:
    state = realtime_frame_delivery.setdefault(session_id, {})
    state.setdefault("images_sent", 0)
    state.setdefault("images_suppressed", 0)
    state.setdefault("last_acked_sequence", 0)
    state.setdefault("last_image_sent_at", None)
    state.setdefault("last_result_sent_at", None)
    state["supports_ack"] = bool(supports_ack)
    if not state["supports_ack"]:
        state["pending_sequence"] = None
        state["pending_since"] = None
    return state


def should_send_realtime_image(
    session_id: str,
    now: Optional[float] = None,
    max_fps: Optional[float] = None,
) -> bool:
    state = configure_realtime_frame_delivery(
        session_id,
        realtime_frame_delivery.get(session_id, {}).get("supports_ack", False),
    )
    now = time.monotonic() if now is None else now
    pending_sequence = state.get("pending_sequence")
    if state["supports_ack"] and pending_sequence is not None:
        pending_since = float(state.get("pending_since") or now)
        ack_timeout = _bounded_env_int("REALTIME_FRAME_ACK_TIMEOUT_MS", 2000, 500, 10000) / 1000.0
        if now - pending_since >= ack_timeout:
            state["pending_sequence"] = None
            state["pending_since"] = None
        else:
            state["images_suppressed"] = int(state.get("images_suppressed") or 0) + 1
            return False

    if max_fps is not None and float(max_fps) > 0:
        last_sent_at = state.get("last_image_sent_at")
        min_interval = 1.0 / float(max_fps)
        if last_sent_at is not None and now - float(last_sent_at) < min_interval:
            state["images_suppressed"] = int(state.get("images_suppressed") or 0) + 1
            return False
    return True


def mark_realtime_image_sent(
    session_id: str,
    frame_sequence: Optional[int],
    preview_meta: Dict[str, int],
) -> None:
    state = configure_realtime_frame_delivery(
        session_id,
        realtime_frame_delivery.get(session_id, {}).get("supports_ack", False),
    )
    state["images_sent"] = int(state.get("images_sent") or 0) + 1
    state["last_preview"] = dict(preview_meta or {})
    state["last_image_sent_at"] = time.monotonic()
    if state["supports_ack"] and frame_sequence is not None:
        state["pending_sequence"] = int(frame_sequence)
        state["pending_since"] = time.monotonic()


def should_send_grid_result(
    session_id: str,
    now: Optional[float] = None,
    max_fps: float = 60.0,
) -> bool:
    state = configure_realtime_frame_delivery(
        session_id,
        realtime_frame_delivery.get(session_id, {}).get("supports_ack", False),
    )
    now = time.monotonic() if now is None else now
    last_sent_at = state.get("last_result_sent_at")
    if last_sent_at is None:
        return True
    return now - float(last_sent_at) >= 1.0 / max(1.0, float(max_fps))


def mark_grid_result_sent(session_id: str, now: Optional[float] = None) -> None:
    state = configure_realtime_frame_delivery(
        session_id,
        realtime_frame_delivery.get(session_id, {}).get("supports_ack", False),
    )
    state["last_result_sent_at"] = time.monotonic() if now is None else now


def acknowledge_realtime_image(session_id: str, frame_sequence: Any) -> None:
    state = realtime_frame_delivery.get(session_id)
    if not state:
        return
    try:
        acknowledged = int(frame_sequence)
    except (TypeError, ValueError):
        return
    state["last_acked_sequence"] = max(int(state.get("last_acked_sequence") or 0), acknowledged)
    pending_sequence = state.get("pending_sequence")
    if pending_sequence is not None and acknowledged >= int(pending_sequence):
        state["pending_sequence"] = None
        state["pending_since"] = None


def get_realtime_delivery_snapshot(session_id: str) -> Dict[str, Any]:
    state = realtime_frame_delivery.get(session_id) or {}
    snapshot = {
        key: value
        for key, value in state.items()
        if key not in {"pending_since", "last_image_sent_at", "last_result_sent_at"}
    }
    pending_since = state.get("pending_since")
    snapshot["pending_age_ms"] = (
        round(max(0.0, time.monotonic() - float(pending_since)) * 1000.0, 2)
        if pending_since is not None
        else None
    )
    return snapshot


def create_video_capture(source_type: str, source_url: str = "") -> cv2.VideoCapture:
    """
    根据源类型和地址创建 OpenCV 视频捕获对象
    """
    return video_processor.create_capture(source_type, source_url)



def _to_bool(value: Any, default: bool = False) -> bool:
    """把前端传入的 bool/string/int 安全转换为布尔值。"""
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on", "y"}
    return default


def _safe_int(value: Any, default: int = 1, min_value: int = None, max_value: int = None) -> int:
    """安全转换整数并限制范围。"""
    try:
        number = int(float(value))
    except (TypeError, ValueError):
        number = default

    if min_value is not None:
        number = max(min_value, number)
    if max_value is not None:
        number = min(max_value, number)
    return number


def _safe_ratio(value: Any, default: float = 0.25) -> float:
    """安全转换 0~1 浮点比例。"""
    try:
        number = float(value)
    except (TypeError, ValueError):
        number = default
    return max(0.0, min(1.0, number))


def normalize_realtime_config_data(config_data: Dict[str, Any] = None, base_config: Dict[str, Any] = None) -> Dict[str, Any]:
    """
    统一清洗实时监控配置。

    目的：
    1. 前端开始监控、逐帧推理、运行中更新配置时都走同一套参数；
    2. 避免字段名差异导致配置看似调整，实际后端仍使用默认值；
    3. 限制参数范围，防止异常配置拖垮实时监控循环。
    """
    data = {}
    if base_config:
        data.update(base_config)
    if config_data:
        data.update(config_data)

    model_path = data.get("model_path") or data.get("modelPath") or data.get("model") or "yolov8n.pt"
    confidence = data.get("confidence", data.get("conf", 0.25))
    iou_threshold = data.get("iou_threshold", data.get("iouThreshold", data.get("iou", 0.7)))
    max_fps = data.get("max_fps", data.get("maxFps", 10))
    frame_skip = data.get("frame_skip", data.get("frameSkip", 1))
    save_frames = data.get("save_frames", data.get("saveFrames", False))
    show_labels = data.get("show_labels", data.get("showLabels", True))
    show_confidence = data.get("show_confidence", data.get("showConfidence", True))
    client_side_overlay = data.get("client_side_overlay", data.get("clientSideOverlay", False))
    source_type = data.get("source_type", data.get("sourceType", "webcam"))
    source_url = data.get("source_url", data.get("sourceUrl", ""))
    loop_video = data.get("loop_video", data.get("loopVideo", True))
    base_behavior_config = {}
    if base_config:
        base_behavior_config = base_config.get(
            "behavior_config",
            base_config.get("behaviorConfig", {}),
        )
    request_behavior_config = {}
    if config_data:
        request_behavior_config = config_data.get(
            "behavior_config",
            config_data.get("behaviorConfig", {}),
        )

    return {
        "model_path": str(model_path).strip() or "yolov8n.pt",
        "confidence": _safe_ratio(confidence, 0.25),
        "iou_threshold": _safe_ratio(iou_threshold, 0.7),
        "max_fps": _safe_int(max_fps, default=10, min_value=1, max_value=60),
        "frame_skip": _safe_int(frame_skip, default=1, min_value=1, max_value=300),
        "save_frames": _to_bool(save_frames, False),
        "show_labels": _to_bool(show_labels, True),
        "show_confidence": _to_bool(show_confidence, True),
        "client_side_overlay": _to_bool(client_side_overlay, False),
        "source_type": str(source_type or "webcam").strip(),
        "source_url": str(source_url or "").strip(),
        "loop_video": _to_bool(loop_video, True),
        "behavior_config": normalize_behavior_config(
            base_behavior_config,
            request_behavior_config,
        ),
    }


def get_websocket_behavior_owner(websocket: WebSocket) -> Dict[str, Any]:
    """Resolve a WebSocket owner before accept; anonymous sockets are rejected."""
    authorization = str(websocket.headers.get("authorization") or "").strip()
    token = str(websocket.query_params.get("token") or "").strip()
    if not authorization and token:
        authorization = f"Bearer {token}"
    if not authorization:
        raise HTTPException(status_code=401, detail="WebSocket 需要登录认证")
    return get_current_user_from_authorization(authorization, request=None)


async def authenticate_websocket(websocket: WebSocket) -> Optional[Dict[str, Any]]:
    try:
        owner = get_websocket_behavior_owner(websocket)
    except HTTPException as exc:
        code = 4403 if exc.status_code == 403 else 4401
        await websocket.close(code=code, reason="authentication required")
        return None
    except Exception:
        logger.exception("WebSocket owner validation failed")
        await websocket.close(code=4401, reason="authentication required")
        return None
    return owner


def _same_websocket_owner(existing: Dict[str, Any], owner: Dict[str, Any]) -> bool:
    existing_id = str(existing.get("owner_id") or "").strip()
    owner_id = str(owner.get("id") or "").strip()
    return bool(existing_id and owner_id and existing_id == owner_id)


def _grid_allowed_file_path(source_url: str) -> Optional[str]:
    candidate = Path(source_url)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    try:
        resolved = candidate.resolve(strict=False)
    except OSError:
        return None
    raw_roots = os.getenv("GRID_ALLOWED_FILE_ROOTS") or ",".join(
        str(path) for path in (STATIC_ROOT, TRAINING_DATASETS_DIR, IMAGE_UPLOAD_DIR.resolve())
    )
    roots = []
    for value in raw_roots.split(","):
        value = value.strip()
        if not value:
            continue
        try:
            roots.append(Path(value).resolve(strict=False))
        except OSError:
            continue
    if not any(resolved == root or root in resolved.parents for root in roots):
        return None
    return str(resolved)


def behavior_monitoring_requires_login(config: Any, owner: Dict[str, Any]) -> bool:
    behavior_config = getattr(config, "behavior_config", None)
    if behavior_config is None and isinstance(config, dict):
        behavior_config = config.get("behavior_config") or config.get("shared_behavior_config")
    if hasattr(behavior_config, "dict"):
        behavior_config = behavior_config.dict()
    if not isinstance(behavior_config, dict):
        return False
    return bool(behavior_config.get("enabled")) and not owner.get("id")


def process_behavior_frame_sync(
    session_key: str,
    image: np.ndarray,
    result: Dict[str, Any],
    behavior_config: Dict[str, Any],
    *,
    context: Dict[str, Any],
    timestamp: Optional[float] = None,
) -> Dict[str, Any]:
    # The CPU face consumer must never receive a later preview or annotated frame.
    face_sampling_image = image.copy() if 'monitoring_algorithms' in context else None
    face_sampling_detections = list(result.get('detections') or [])
    owner_id = context.get("owner_id")
    source_id = str(context.get("source_id") or session_key)
    request_config = behavior_config or {}
    cached = behavior_resolved_configs.get(session_key)
    if (
        cached is None
        or cached.get("owner_id") != owner_id
        or cached.get("source_id") != source_id
        or cached.get("request_config") != request_config
        or time.monotonic() - float(cached.get("resolved_at") or 0.0) >= 5.0
    ):
        resolved = resolve_behavior_profile(owner_id, source_id, request_config)
        behavior_resolved_configs[session_key] = {
            "owner_id": owner_id,
            "source_id": source_id,
            "request_config": copy.deepcopy(request_config),
            "resolved": resolved,
            "resolved_at": time.monotonic(),
        }
    else:
        resolved = cached["resolved"]

    if 'monitoring_algorithms' in context:
        resolved = copy.deepcopy(resolved)
        flow_codes = {item['code'] for item in context.get('alarm_flows', [])}
        resolved['enabled_events'] = [code for code in context['monitoring_algorithms'] if code not in flow_codes]
        resolved['enabled'] = bool(resolved['enabled_events'])
        resolved['process_every_sampled_frame'] = True
        resolved['face_required_events'] = [code for code in context.get('face_required_algorithms', []) if code in resolved['enabled_events']]
    behavior_result = behavior_runtime_manager.process(
        session_key,
        image,
        result.get("detections") or [],
        resolved,
        timestamp=timestamp,
    )
    # A successful synchronous pose pass exposes only this frame's persons.
    # Never use skipped/failed results or the later preview cache as face ROIs.
    if not behavior_result.get('skipped') and not behavior_result.get('error'):
        face_sampling_detections.extend(behavior_result.get('_monitor_detections') or [])
    prepare_realtime_feature_vectors(behavior_result.get("action_events") or [], image)
    apply_realtime_verifiers(behavior_result.get("action_events") or [])
    if context.get('alarm_flows'):
        from alarm_flow import process_flows
        flow_result = process_flows(context['alarm_flows'], image, session_key=session_key, context=context, timestamp=timestamp, face_image=face_sampling_image)
        if flow_result.get('face_gate'):
            behavior_result.setdefault('behavior_stats', {})['alarm_face_gate'] = flow_result['face_gate']
        for key in ('action_events', 'active_events'):
            behavior_result.setdefault(key, []).extend(flow_result[key])
        behavior_result.setdefault('_monitor_detections', []).extend(flow_result['detections'])
        face_sampling_detections.extend(flow_result['detections'])
    if face_sampling_image is not None:
        from face_capture.sampling import submit_alarm_frame
        try:
            signal = submit_alarm_frame(
                face_sampling_image,
                face_sampling_detections,
                context, session_key=session_key,
            )
        except Exception as exc:
            logger.warning('Alarm-frame face signal unavailable: %s', type(exc).__name__)
            signal = {'status': 'unavailable'}
        behavior_result.setdefault('behavior_stats', {})['face_sampling'] = signal
        if signal.get('status') in {'queued', 'duplicate'}:
            for collection in ('action_events', 'active_events'):
                for event in behavior_result.get(collection) or []:
                    event['reason'] = {**(event.get('reason') or {}), 'face_frame_id': signal['frame_id']}
    temporal_shadow_events = behavior_result.get("temporal_shadow_events") or []
    temporal_score_snapshots = behavior_result.get("temporal_score_snapshots") or []
    temporal_model = behavior_result.get("temporal_model") if isinstance(behavior_result.get("temporal_model"), dict) else {}
    scheduler_timing = {
        key: context[key]
        for key in ("capture_timestamp", "dequeue_timestamp", "dequeue_age_ms", "sample_reason")
        if context.get(key) is not None
    }
    if scheduler_timing:
        behavior_result.setdefault("behavior_stats", {})["scheduler_timing"] = scheduler_timing
    for event in behavior_result.get("action_events") or []:
        final_decision = event.get("final_decision") if isinstance(event.get("final_decision"), dict) else {}
        final_decision["accepted"] = bool(event.get("should_alert", True))
        event["final_decision"] = final_decision
        decision = event.get("decision") if isinstance(event.get("decision"), dict) else {}
        decision["final"] = final_decision
        event_type = str(event.get("event_type") or "")
        tracks = set(event.get("persons") or event.get("track_ids") or [])
        timestamp_value = float(event.get("confirmed_at") or event.get("start_time") or 0.0)
        shadow = next(
            (
                candidate for candidate in temporal_shadow_events
                if str(candidate.get("event_type") or "") == event_type
                and (
                    (tracks and set(candidate.get("persons") or []) and tracks.intersection(set(candidate.get("persons") or [])))
                    or (not tracks and not set(candidate.get("persons") or []))
                )
                and abs(float(candidate.get("confirmed_at") or candidate.get("start_time") or 0.0) - timestamp_value) <= 3.0
            ),
            None,
        )
        snapshot = next(
            (
                candidate for candidate in temporal_score_snapshots
                if str(candidate.get("event_type") or "") == event_type
                and (
                    (tracks and set(candidate.get("persons") or []) and tracks.intersection(set(candidate.get("persons") or [])))
                    or (not tracks and not set(candidate.get("persons") or []))
                )
                and abs(float(candidate.get("timestamp") or candidate.get("start_time") or 0.0) - timestamp_value) <= 3.0
            ),
            None,
        )
        if snapshot and str(temporal_model.get("mode") or "") == "shadow":
            shadow_reason = shadow.get("reason") if isinstance(shadow, dict) and isinstance(shadow.get("reason"), dict) else {}
            model_threshold = float(snapshot.get("threshold") or 0.35)
            decision["temporal_shadow"] = {
                "model_id": temporal_model.get("model_id") or snapshot.get("model_id") or shadow_reason.get("model_id"),
                "model_sha256": temporal_model.get("model_sha256"),
                "model_version": temporal_model.get("model_version") or snapshot.get("model_version"),
                "confidence": snapshot.get("confidence"),
                "score": snapshot.get("score"),
                "would_add_alert": bool(not event.get("should_alert") and snapshot.get("above_threshold")),
                "would_suppress_alert": bool(event.get("should_alert") and not snapshot.get("above_threshold")),
                "threshold": model_threshold,
                "input_quality": snapshot.get("input_quality") or shadow_reason.get("input_quality") or {},
                "temporal_span": {
                    "start": snapshot.get("start_time"),
                    "end": snapshot.get("timestamp"),
                },
                "cooldown_remaining_sec": snapshot.get("cooldown_remaining_sec"),
                "would_emit": snapshot.get("would_emit"),
            }
        if scheduler_timing:
            decision["scheduler_timing"] = dict(scheduler_timing)
        if (
            str(temporal_model.get("mode") or "") == "filter"
            and temporal_model.get("available")
            and decision.get("temporal", {}).get("accepted")
        ):
            event["decision_model_id"] = temporal_model.get("model_id") or event.get("model_id")
            event["decision_model_version"] = temporal_model.get("model_version") or event.get("model_version")
            decision["temporal_model_sha256"] = temporal_model.get("model_sha256")
        event["decision"] = decision
    confirmed_by_id = {
        str(item.get("event_id")): item
        for item in (behavior_result.get("action_events") or [])
        if item.get("event_id") and item.get("state") == "confirmed"
    }
    for collection_name in ("action_events", "active_events"):
        for event in behavior_result.get(collection_name) or []:
            confirmed = confirmed_by_id.get(str(event.get("event_id") or ""))
            if confirmed is not None and collection_name == "active_events":
                for key in (
                    "should_alert", "alerted_at", "decision_model_id",
                    "decision_model_version", "decision", "rule_decision",
                    "temporal_decision", "final_decision", "model_id",
                    "model_version", "input_quality", "temporal_span", "fallback",
                    "scheduler_timing", "feature_schema_version",
                ):
                    if key in confirmed:
                        event[key] = copy.deepcopy(confirmed[key])
            event.update({
                "mode": context.get("mode") or "realtime",
                "source_id": source_id,
                "session_id": context.get("session_id"),
                "grid_id": context.get("grid_id"),
                "grid_position": context.get("grid_position"),
            })
    if 'monitoring_algorithms' in context:
        behavior_result = filter_monitor_events(behavior_result, context['monitoring_algorithms'])
    if owner_id is not None:
        persistence = persist_behavior_transitions_resilient(
            behavior_result.get("action_events") or [],
            owner_id=int(owner_id),
            owner_username=str(context.get("owner_username") or ""),
            mode=str(context.get("mode") or "realtime"),
            source_id=source_id,
            session_id=context.get("session_id"),
            grid_id=context.get("grid_id"),
            grid_position=context.get("grid_position"),
        )
        behavior_result["persistence_status"] = persistence["status"]
        if persistence["status"] != "persisted":
            behavior_result["persistence_warning"] = (
                "告警记录已进入本地补偿队列，数据库恢复后自动同步"
                if persistence["status"] == "spooled"
                else "告警已发出，但数据库与本地补偿队列均写入失败"
            )
            for event in behavior_result.get("action_events") or []:
                event["persistence_status"] = persistence["status"]
        evidence_persistence = update_behavior_evidence_resilient(behavior_result.get("evidence_updates") or [])
        if evidence_persistence["status"] != "persisted":
            behavior_result["persistence_warning"] = (
                behavior_result.get("persistence_warning")
                or "告警已发出，证据状态已进入补偿队列"
            )
    elif resolved.get("enabled"):
        behavior_result["persistence_warning"] = "WebSocket 未携带登录令牌，事件未写入数据库"
    for collection_name in ("action_events", "active_events"):
        for event in behavior_result.get(collection_name) or []:
            decision = event.get("decision")
            if isinstance(decision, dict) and "training_feature_vector" in decision:
                event["decision"] = {key: value for key, value in decision.items() if key != "training_feature_vector"}
    monitor_detections = behavior_result.pop('_monitor_detections', None)
    # Recorder monitoring renders client-side overlays.  Its behavior models may
    # contribute people, badges, or phones that the primary detector did not
    # return, so preserve the exact input image with the corresponding filtered
    # detections.  The bounded scheduler keeps only completed results until the
    # next drain; the grid sender consumes this as a latest-only preview.
    if 'monitoring_algorithms' in context and monitor_detections is not None:
        behavior_result['_monitor_preview'] = {
            'image': image,
            'detections': monitor_detections,
            'source_timestamp': timestamp,
            'source_sequence': context.get('_monitor_source_sequence'),
        }
    return behavior_result


def close_behavior_session_sync(session_key: str, timestamp: Optional[float] = None) -> Dict[str, Any]:
    context = behavior_session_contexts.pop(session_key, {})
    behavior_resolved_configs.pop(session_key, None)
    behavior_result = behavior_runtime_manager.close(session_key, timestamp)
    owner_id = context.get("owner_id")
    if 'monitoring_algorithms' in context:
        behavior_result = filter_monitor_events(behavior_result, context['monitoring_algorithms'])
    if owner_id is not None:
        persistence = persist_behavior_transitions_resilient(
            behavior_result.get("action_events") or [],
            owner_id=int(owner_id),
            owner_username=str(context.get("owner_username") or ""),
            mode=str(context.get("mode") or "realtime"),
            source_id=str(context.get("source_id") or session_key),
            session_id=context.get("session_id"),
            grid_id=context.get("grid_id"),
            grid_position=context.get("grid_position"),
        )
        behavior_result["persistence_status"] = persistence["status"]
        update_behavior_evidence_resilient(behavior_result.get("evidence_updates") or [])
    return behavior_result


def submit_behavior_frame(
    session_key: str,
    image: np.ndarray,
    result: Dict[str, Any],
    config: Dict[str, Any],
    *,
    context: Dict[str, Any],
    timestamp: float,
) -> Dict[str, Any]:
    scheduler = behavior_frame_schedulers.get(session_key)
    if scheduler is None or not scheduler.accepting:
        scheduler = BehaviorFifoFrameScheduler(
            session_key,
            process_behavior_frame_sync,
            maxsize=_bounded_env_int("REALTIME_BEHAVIOR_QUEUE_MAX_SIZE", 8, 1, 120),
            process_interval_ms=resolve_behavior_scheduler_interval_ms(),
            max_frame_age_ms=resolve_behavior_max_frame_age_ms(),
        )
        behavior_frame_schedulers[session_key] = scheduler
    return scheduler.submit(image, result, config, context, timestamp)


async def stop_behavior_frame_scheduler(session_key: str, timeout: float = 15.0) -> bool:
    scheduler = behavior_frame_schedulers.get(session_key)
    if scheduler is None:
        return True
    complete = await scheduler.close(timeout)
    if complete:
        behavior_frame_schedulers.pop(session_key, None)
    return complete


def get_default_realtime_config_dict() -> Dict[str, Any]:
    """获取默认实时监控配置字典。"""
    return normalize_realtime_config_data({})


def get_current_realtime_config(session_id: str, fallback_config: RealtimeConfig = None) -> RealtimeConfig:
    """
    获取某个实时监控会话当前生效配置。
    服务器端视频流循环每一轮都会读取它，因此 config_update 能即时生效。
    """
    fallback_data = None
    if fallback_config is not None:
        try:
            fallback_data = fallback_config.dict()
        except Exception:
            fallback_data = getattr(fallback_config, "__dict__", {})

    config_data = realtime_session_configs.get(session_id) or fallback_data or get_default_realtime_config_dict()
    normalized = normalize_realtime_config_data(config_data)
    realtime_session_configs[session_id] = normalized
    return RealtimeConfig(**normalized)


def update_realtime_session_config(session_id: str, config_data: Dict[str, Any] = None) -> tuple:
    """
    更新实时监控会话配置。
    返回：(新配置对象, 视频源是否变化)
    """
    old_config = realtime_session_configs.get(session_id) or get_default_realtime_config_dict()
    new_config = normalize_realtime_config_data(config_data or {}, old_config)

    source_changed = (
        old_config.get("source_type") != new_config.get("source_type")
        or old_config.get("source_url") != new_config.get("source_url")
        or old_config.get("loop_video") != new_config.get("loop_video")
    )

    realtime_session_configs[session_id] = new_config

    if source_changed:
        # 实际释放由流任务在当前 read()/推理返回后执行，禁止跨线程释放。
        realtime_frame_counters[session_id] = 0
        realtime_last_process_time[session_id] = 0

    return RealtimeConfig(**new_config), source_changed


def should_process_realtime_frame(session_id: str, config: RealtimeConfig) -> bool:
    """
    根据 max_fps 和 frame_skip 判断当前帧是否需要真正推理。

    max_fps：限制后端实际推理频率，前端发帧过快时自动丢帧；
    frame_skip：每 N 帧推理 1 帧，降低 GPU/CPU 压力。
    """
    frame_count = realtime_frame_counters.get(session_id, 0) + 1
    realtime_frame_counters[session_id] = frame_count

    if config.frame_skip > 1 and frame_count % config.frame_skip != 0:
        return False

    now = time.time()
    last_process_time = realtime_last_process_time.get(session_id, 0.0)
    min_interval = 1.0 / max(1, int(config.max_fps))

    if last_process_time > 0 and now - last_process_time < min_interval:
        return False

    realtime_last_process_time[session_id] = now
    return True


def _extract_detection_bbox_xyxy(detection: Dict[str, Any]):
    """从不同格式 detection 中提取 x1,y1,x2,y2。"""
    bbox = detection.get("bbox") or detection.get("box") or detection.get("xyxy") or detection

    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return [float(bbox[0]), float(bbox[1]), float(bbox[2]), float(bbox[3])]

    if isinstance(bbox, dict):
        x1 = bbox.get("x1", bbox.get("xmin", bbox.get("left", bbox.get("x", 0))))
        y1 = bbox.get("y1", bbox.get("ymin", bbox.get("top", bbox.get("y", 0))))

        if "x2" in bbox or "xmax" in bbox or "right" in bbox:
            x2 = bbox.get("x2", bbox.get("xmax", bbox.get("right", 0)))
        else:
            x2 = float(x1) + float(bbox.get("width", bbox.get("w", 0)))

        if "y2" in bbox or "ymax" in bbox or "bottom" in bbox:
            y2 = bbox.get("y2", bbox.get("ymax", bbox.get("bottom", 0)))
        else:
            y2 = float(y1) + float(bbox.get("height", bbox.get("h", 0)))

        return [float(x1), float(y1), float(x2), float(y2)]

    return None


def render_realtime_image_without_confidence(image: np.ndarray, detections: List[Dict[str, Any]]) -> np.ndarray:
    """
    当 show_confidence=False 时，重新绘制标注图，只显示类别，不显示置信度。
    这样“显示置信度”配置可以真实影响返回给前端的画面。
    """
    if image is None:
        return None

    annotated = image.copy()

    for detection in detections or []:
        bbox = _extract_detection_bbox_xyxy(detection)
        if not bbox:
            continue

        x1, y1, x2, y2 = [int(round(v)) for v in bbox]
        class_name = str(
            detection.get("class")
            or detection.get("class_name")
            or detection.get("name")
            or detection.get("label")
            or "object"
        )

        cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 255, 0), 2)

        label_y = max(y1 - 6, 12)
        cv2.putText(
            annotated,
            class_name,
            (x1, label_y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (0, 255, 0),
            2,
            cv2.LINE_AA
        )

    return annotated


def get_realtime_annotated_image(image: np.ndarray, result: Dict[str, Any], config: RealtimeConfig) -> Optional[np.ndarray]:
    """根据 show_labels/show_confidence 返回最终要推送给前端的标注图。"""
    if image is None:
        return None

    if config.client_side_overlay or not config.show_labels:
        return image

    if config.show_confidence:
        return result.get("annotated_image")

    return render_realtime_image_without_confidence(image, result.get("detections", []))


def save_realtime_frame_if_needed(session_id: str, image: np.ndarray, config: RealtimeConfig) -> Optional[str]:
    """按配置保存实时监控帧，并返回 /static/... URL。"""
    if not config.save_frames or image is None:
        return None

    try:
        save_dir = Path(constants.STATIC_DIR) / "realtime_frames" / session_id
        save_dir.mkdir(parents=True, exist_ok=True)
        filename = f"frame_{int(time.time() * 1000)}.jpg"
        save_path = save_dir / filename

        # image_processor.encode_image_to_base64 通常处理 RGB/BGR 均可显示；这里保存时用 cv2.imwrite。
        cv2.imwrite(str(save_path), image)
        return _static_url_from_path(save_path)
    except Exception as e:
        logger.warning(f"保存实时监控帧失败: session={session_id}, error={e}")
        return None


def update_realtime_stats_and_build_response(
    session_id: str,
    result: Dict[str, Any],
    annotated_image_base64: Optional[str],
    config: RealtimeConfig,
    object_counts: Dict[str, int] = None,
    saved_frame_url: Optional[str] = None,
    stream_health: Optional[Dict[str, Any]] = None,
    frame_sequence: Optional[int] = None,
    source_timestamp: Optional[float] = None,
    behavior_result: Optional[Dict[str, Any]] = None,
    preview_meta: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """更新实时监控统计并构造统一 WebSocket 响应。"""
    stats = realtime_stats.setdefault(session_id, {
        "session_id": session_id,
        "frames_processed": 0,
        "total_detections": 0,
        "inference_times": [],
        "start_time": datetime.now().isoformat(),
        "last_update": datetime.now().isoformat(),
        "is_active": True
    })

    stats["frames_processed"] += 1
    stats["total_detections"] += int(result.get("detections_count", 0))
    stats.setdefault("inference_times", []).append(float(result.get("inference_time", 0)))
    stats.setdefault("result_times", []).append(time.time())

    if len(stats["inference_times"]) > 30:
        stats["inference_times"] = stats["inference_times"][-30:]
    if len(stats["result_times"]) > 30:
        stats["result_times"] = stats["result_times"][-30:]

    avg_inference_time = sum(stats["inference_times"]) / len(stats["inference_times"]) if stats["inference_times"] else 0
    if len(stats["result_times"]) >= 2:
        elapsed = stats["result_times"][-1] - stats["result_times"][0]
        current_fps = (len(stats["result_times"]) - 1) / elapsed if elapsed > 0 else 0
    else:
        current_fps = 0

    stats["avg_inference_time"] = avg_inference_time
    stats["current_fps"] = min(current_fps, config.max_fps)
    stats["last_update"] = datetime.now().isoformat()
    stats["config"] = normalize_realtime_config_data(config.dict())

    source_to_result_ms = None
    if source_timestamp:
        source_to_result_ms = max(0.0, (time.time() - float(source_timestamp)) * 1000.0)
    if stream_health is not None:
        stats["stream_health"] = stream_health
    if frame_sequence is not None:
        previous_sequence = int(stats.get("last_frame_sequence") or 0)
        stats["dropped_source_frames"] = int(stats.get("dropped_source_frames") or 0) + max(
            0, int(frame_sequence) - previous_sequence - 1
        )
        stats["last_frame_sequence"] = int(frame_sequence)
    if source_to_result_ms is not None:
        stats["source_to_result_ms"] = source_to_result_ms

    response_stats = {
        "frames_processed": stats["frames_processed"],
        "total_detections": stats["total_detections"],
        "avg_inference_time": round(avg_inference_time, 2),
        "current_fps": round(min(current_fps, config.max_fps), 1),
        "max_fps": config.max_fps,
        "frame_skip": config.frame_skip,
        "dropped_source_frames": int(stats.get("dropped_source_frames") or 0),
    }

    if stream_health is not None:
        response_stats["stream_health"] = stream_health
        response_stats["capture_frame_age_ms"] = stream_health.get("frame_age_ms")
        response_stats["capture_reconnect_count"] = stream_health.get("reconnect_count", 0)
    if source_to_result_ms is not None:
        response_stats["source_to_result_ms"] = round(source_to_result_ms, 2)

    delivery_snapshot = get_realtime_delivery_snapshot(session_id)
    if delivery_snapshot:
        stats["frame_delivery"] = delivery_snapshot
        response_stats["frame_delivery"] = delivery_snapshot

    if object_counts is not None:
        response_stats["unique_object_counts"] = object_counts
    behavior_result = behavior_result or {}
    action_events = behavior_result.get("action_events") or []
    alarm_events = [
        item for item in action_events
        if item.get("state") == "confirmed" and item.get("should_alert")
    ]
    confirmed_count = sum(1 for item in action_events if item.get("state") == "confirmed")
    stats["total_action_events"] = int(stats.get("total_action_events") or 0) + confirmed_count
    response_stats["total_action_events"] = stats["total_action_events"]
    stats["total_alarm_events"] = int(stats.get("total_alarm_events") or 0) + len(alarm_events)
    response_stats["total_alarm_events"] = stats["total_alarm_events"]

    response = {
        "type": "inference_result",
        "session_id": session_id,
        "detections": result.get("detections", []),
        "detections_count": result.get("detections_count", 0),
        "inference_time": result.get("inference_time", 0),
        "annotated_image": annotated_image_base64,
        "preview": preview_meta or {},
        "client_side_overlay": bool(config.client_side_overlay),
        "saved_frame_url": saved_frame_url,
        "action_events": action_events,
        "alarm_events": alarm_events,
        "active_events": behavior_result.get("active_events") or [],
        "temporal_shadow_events": behavior_result.get("temporal_shadow_events") or [],
        "temporal_score_snapshots": behavior_result.get("temporal_score_snapshots") or [],
        "temporal_model": behavior_result.get("temporal_model") or {},
        "behavior_stats": behavior_result.get("behavior_stats") or {"enabled": False},
        "behavior_persistence_status": behavior_result.get("persistence_status", "persisted"),
        "stats": response_stats,
        "config_applied": normalize_realtime_config_data(config.dict()),
        "timestamp": result.get("timestamp", datetime.now().isoformat())
    }

    if frame_sequence is not None:
        response["frame_sequence"] = int(frame_sequence)
    if source_timestamp is not None:
        response["source_timestamp"] = source_timestamp

    if "error" in result:
        response["error"] = result["error"]
    if behavior_result.get("error"):
        response["behavior_error"] = behavior_result["error"]
    if behavior_result.get("persistence_warning"):
        response["behavior_persistence_warning"] = behavior_result["persistence_warning"]

    return response

def get_frame_from_source(session_id: str, config: RealtimeConfig) -> np.ndarray:
    """
    从指定会话的视频源中获取当前帧
    处理视频重连、循环播放等逻辑
    """
    try:
        if config.source_type in ["rtsp", "http"]:
            frame, _health = video_processor.read_supervised_frame(
                session_id,
                config.source_type,
                config.source_url,
                timeout=1.0,
            )
            return frame

        # 获取现有的 Capture 对象，若不存在则创建
        cap = video_processor.get_capture(session_id)
        if cap is None:
            cap = video_processor.create_capture(config.source_type, config.source_url)
            if cap is None:
                return None
            video_processor.register_capture(session_id, cap)
        
        # 读取一帧
        frame = video_processor.read_frame(cap, convert_to_rgb=True)
        if frame is None:
            # 如果是视频文件且开启了循环播放，重置到起始位置
            if config.source_type == "file" and config.loop_video:
                video_processor.reset_to_start(cap)
                frame = video_processor.read_frame(cap, convert_to_rgb=True)
            
            if frame is None:
                logger.warning(f"无法从源读取帧: {session_id}")
                return None
        
        return frame
        
    except Exception as e:
        logger.error(f"从源获取帧失败: {e}")
        return None


def cleanup_video_capture(session_id: str, join_timeout: float = 2.0) -> bool:
    """
    释放指定会话占用的视频捕获资源
    """
    return video_processor.cleanup_capture(session_id, join_timeout=join_timeout)


def realtime_session_should_stop(session_id: str) -> bool:
    event = realtime_stop_events.get(session_id)
    return realtime_session_states.get(session_id) in {"stopping", "stopped"} or bool(
        event and event.is_set()
    )


def _remaining_cleanup_seconds(deadline: float) -> float:
    return max(0.0, deadline - time.monotonic())


async def _attempt_realtime_cleanup(
    session_id: str,
    *,
    websocket: WebSocket = None,
    timeout_seconds: float = 15.0,
) -> bool:
    """Cooperatively stop one session without cancelling native work."""
    lock = realtime_cleanup_locks.setdefault(session_id, asyncio.Lock())
    async with lock:
        if realtime_session_states.get(session_id) == "stopped":
            return True

        deadline = time.monotonic() + max(0.0, timeout_seconds)
        realtime_session_states[session_id] = "stopping"
        stop_event = realtime_stop_events.setdefault(session_id, asyncio.Event())
        stop_event.set()
        if session_id in realtime_stats:
            realtime_stats[session_id]["is_active"] = False

        stream_task = realtime_stream_tasks.get(session_id)
        if stream_task is not None and stream_task is not asyncio.current_task() and not stream_task.done():
            try:
                await asyncio.wait_for(
                    asyncio.shield(stream_task),
                    timeout=_remaining_cleanup_seconds(deadline),
                )
            except asyncio.TimeoutError:
                return False

        capture_stopped = await asyncio.to_thread(
            cleanup_video_capture,
            session_id,
            _remaining_cleanup_seconds(deadline),
        )
        if not capture_stopped:
            return False
        behavior_stopped = await stop_behavior_frame_scheduler(
            session_id,
            _remaining_cleanup_seconds(deadline),
        )
        if not behavior_stopped:
            return False
        await asyncio.to_thread(close_behavior_session_sync, session_id, time.time())

        current_websocket = realtime_connections.get(session_id)
        target_websocket = websocket or current_websocket
        if target_websocket is not None:
            try:
                await target_websocket.close(code=1000, reason="session stopped")
            except Exception:
                pass
        if current_websocket is target_websocket:
            realtime_connections.pop(session_id, None)

        realtime_stream_tasks.pop(session_id, None)
        realtime_send_locks.pop(session_id, None)
        realtime_frame_delivery.pop(session_id, None)
        realtime_session_configs.pop(session_id, None)
        realtime_frame_counters.pop(session_id, None)
        realtime_last_process_time.pop(session_id, None)
        realtime_session_states[session_id] = "stopped"
        return True


async def _reap_realtime_session(session_id: str, websocket: WebSocket = None):
    try:
        while not await _attempt_realtime_cleanup(
            session_id,
            websocket=websocket,
            timeout_seconds=2.0,
        ):
            await asyncio.sleep(2.0)
    finally:
        task = realtime_cleanup_tasks.get(session_id)
        if task is asyncio.current_task():
            realtime_cleanup_tasks.pop(session_id, None)


async def request_realtime_cleanup(
    session_id: str,
    *,
    websocket: WebSocket = None,
    timeout_seconds: float = 15.0,
) -> bool:
    cleanup_complete = await _attempt_realtime_cleanup(
        session_id,
        websocket=websocket,
        timeout_seconds=timeout_seconds,
    )
    if not cleanup_complete:
        reaper = realtime_cleanup_tasks.get(session_id)
        if reaper is None or reaper.done():
            realtime_cleanup_tasks[session_id] = asyncio.create_task(
                _reap_realtime_session(session_id, websocket)
            )
    return cleanup_complete


async def send_realtime_message_safe(
    session_id: str,
    websocket: WebSocket,
    payload: Dict[str, Any],
    timeout_seconds: float = 3.0,
) -> bool:
    """Serialize sends and disconnect slow clients before frames can queue forever."""
    if (
        realtime_session_should_stop(session_id)
        or
        realtime_connections.get(session_id) is not websocket
        or websocket.application_state != WebSocketState.CONNECTED
        or websocket.client_state != WebSocketState.CONNECTED
    ):
        return False
    lock = realtime_send_locks.setdefault(session_id, asyncio.Lock())
    try:
        async with lock:
            if (
                realtime_session_should_stop(session_id)
                or
                realtime_connections.get(session_id) is not websocket
                or websocket.application_state != WebSocketState.CONNECTED
                or websocket.client_state != WebSocketState.CONNECTED
            ):
                return False
            await asyncio.wait_for(websocket.send_json(payload), timeout=timeout_seconds)
        return True
    except (WebSocketDisconnect, RuntimeError):
        return False
    except Exception as exc:
        logger.warning(f"实时推理消息发送失败: session={session_id}, error={exc}")
        try:
            if websocket.application_state == WebSocketState.CONNECTED:
                await websocket.close(code=1013, reason="client too slow or unavailable")
        except Exception:
            pass
        return False


def build_realtime_stream_transition_event(
    session_id: str,
    previous_status: Optional[str],
    status: str,
    health: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Create one observable event for a source outage or recovery transition."""
    if status == "reconnecting" and previous_status != "reconnecting":
        health = health or {}
        reason = str(health.get("last_error") or "source_unavailable")
        return {
            "type": "stream_event",
            "event": "source_disconnected",
            "level": "error",
            "session_id": session_id,
            "source_type": health.get("source_type"),
            "message": "视频流断开，正在自动重连",
            "reason": reason,
            "reconnect_count": health.get("reconnect_count", 0),
            "timestamp": time.time(),
        }

    if status == "live" and previous_status == "reconnecting":
        return {
            "type": "stream_event",
            "event": "source_recovered",
            "level": "info",
            "session_id": session_id,
            "source_type": (health or {}).get("source_type"),
            "message": "视频流已恢复",
            "reconnect_count": (health or {}).get("reconnect_count", 0),
            "timestamp": time.time(),
        }

    return None


def build_grid_stream_transition_event(
    grid_id: str,
    grid_position: int,
    session_id: str,
    previous_status: Optional[str],
    status: str,
    health: Optional[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Annotate a source outage/recovery event with its grid location."""
    event = build_realtime_stream_transition_event(
        session_id,
        previous_status,
        status,
        health,
    )
    if event is None:
        return None

    event.update({
        "monitor_type": "grid",
        "grid_id": grid_id,
        "grid_position": grid_position,
        "event": "grid_source_disconnected"
        if event["event"] == "source_disconnected"
        else "grid_source_recovered",
        "message": f"九宫格第 {grid_position + 1} 路视频流断开，正在自动重连"
        if event["level"] == "error"
        else f"九宫格第 {grid_position + 1} 路视频流已恢复",
    })
    return event


async def process_server_stream(websocket, session_id: str, config: RealtimeConfig):
    """Process the newest available frame without blocking the ASGI event loop."""
    counted_ids = set()
    object_counts = {}
    last_sequence = 0
    last_source_key = None
    last_status = None
    last_status_sent_at = 0.0

    try:
        update_realtime_session_config(session_id, config.dict())
        logger.info(f"启动服务器端视频流处理: {session_id}, 源类型: {config.source_type}")

        while (
            realtime_connections.get(session_id) is websocket
            and not realtime_session_should_stop(session_id)
        ):
            loop_start_time = time.time()
            current_config = get_current_realtime_config(session_id, config)

            if current_config.source_type not in ["rtsp", "http", "file"]:
                await asyncio.to_thread(cleanup_video_capture, session_id, 2.0)
                break

            stream_health = None
            frame_sequence = None
            source_timestamp = None

            if current_config.source_type in ["rtsp", "http"]:
                source_key = (current_config.source_type, current_config.source_url)
                if source_key != last_source_key:
                    if last_source_key is not None:
                        capture_stopped = await asyncio.to_thread(
                            cleanup_video_capture,
                            session_id,
                            6.0,
                        )
                        if not capture_stopped or realtime_session_should_stop(session_id):
                            break
                    await asyncio.to_thread(
                        video_processor.ensure_supervised_capture,
                        session_id,
                        current_config.source_type,
                        current_config.source_url,
                    )
                    if realtime_session_should_stop(session_id):
                        break
                    last_source_key = source_key
                    last_sequence = 0

                image, frame_sequence, stream_health = await asyncio.to_thread(
                    video_processor.wait_for_latest_frame,
                    session_id,
                    last_sequence,
                    1.0,
                )
                if realtime_session_should_stop(session_id):
                    break
                if frame_sequence is not None:
                    last_sequence = frame_sequence
                if stream_health:
                    source_timestamp = stream_health.get("last_frame_at")
            else:
                image = await asyncio.to_thread(get_frame_from_source, session_id, current_config)
                if realtime_session_should_stop(session_id):
                    break
                source_timestamp = time.time() if image is not None else None
                stream_health = {
                    "session_id": session_id,
                    "source_type": current_config.source_type,
                    "status": "live" if image is not None else "waiting",
                    "frame_age_ms": 0 if image is not None else None,
                    "reconnect_count": 0,
                    "consecutive_failures": 0 if image is not None else 1,
                }

            status = (stream_health or {}).get("status", "waiting")
            transition_event = build_realtime_stream_transition_event(
                session_id,
                last_status,
                status,
                stream_health,
            )
            if status != last_status or time.time() - last_status_sent_at >= 5.0:
                if not await send_realtime_message_safe(session_id, websocket, {
                    "type": "stream_status",
                    "session_id": session_id,
                    "status": status,
                    "health": stream_health or {},
                    "timestamp": time.time(),
                }):
                    break
                last_status = status
                last_status_sent_at = time.time()

            if transition_event:
                write_realtime_stream_event_log(transition_event)
                if transition_event["level"] == "error":
                    logger.error(
                        "实时视频流断开: session=%s, source_type=%s, reason=%s, reconnect_count=%s",
                        session_id,
                        transition_event.get("source_type"),
                        transition_event.get("reason"),
                        transition_event.get("reconnect_count"),
                    )
                else:
                    logger.info(
                        "实时视频流恢复: session=%s, source_type=%s, reconnect_count=%s",
                        session_id,
                        transition_event.get("source_type"),
                        transition_event.get("reconnect_count"),
                    )
                if not await send_realtime_message_safe(session_id, websocket, transition_event):
                    break

            if image is None:
                await asyncio.sleep(0.1)
                continue

            if not should_process_realtime_frame(session_id, current_config):
                await asyncio.sleep(0)
                continue

            result = await asyncio.to_thread(
                process_realtime_frame,
                image,
                current_config.model_path,
                current_config,
            )
            if result.get("gpu_waiting"):
                await send_realtime_message_safe(session_id, websocket, {
                    "type": "gpu_waiting", "message": WAIT_MESSAGE,
                })
                await asyncio.sleep(2)
                continue
            if realtime_session_should_stop(session_id):
                break
            behavior_context = behavior_session_contexts.get(session_id, {})
            behavior_context.update({
                "mode": "realtime",
            "source_id": redact_stream_url(current_config.source_url) or session_id,
                "session_id": session_id,
            })
            behavior_session_contexts[session_id] = behavior_context
            behavior_result = submit_behavior_frame(
                session_id,
                image,
                result,
                current_config.behavior_config,
                context=behavior_context,
                timestamp=source_timestamp or time.time(),
            )

            # 更新唯一对象计数
            if "detections" in result:
                for detection in result["detections"]:
                    track_id = detection.get("track_id")
                    if track_id is not None and track_id != -1 and track_id not in counted_ids:
                        counted_ids.add(track_id)
                        class_name = detection.get("class") or detection.get("class_name") or "object"
                        object_counts[class_name] = object_counts.get(class_name, 0) + 1
                        logger.info(f"发现新跟踪对象: track_id={track_id}, class_name='{class_name}'")

            annotated_image = get_realtime_annotated_image(image, result, current_config)
            annotated_image_base64 = None
            preview_meta = {}
            if annotated_image is not None and should_send_realtime_image(session_id):
                annotated_image_base64, preview_meta = await asyncio.to_thread(
                    encode_realtime_preview,
                    annotated_image,
                )
                preview_meta["source_width"] = int(image.shape[1])
                preview_meta["source_height"] = int(image.shape[0])
                if realtime_session_should_stop(session_id):
                    break
                if annotated_image_base64:
                    mark_realtime_image_sent(session_id, frame_sequence, preview_meta)
            saved_frame_url = await asyncio.to_thread(
                save_realtime_frame_if_needed,
                session_id,
                annotated_image if annotated_image is not None else image,
                current_config,
            )
            if realtime_session_should_stop(session_id):
                break

            response = update_realtime_stats_and_build_response(
                session_id,
                result,
                annotated_image_base64,
                current_config,
                object_counts=object_counts,
                saved_frame_url=saved_frame_url,
                stream_health=stream_health,
                frame_sequence=frame_sequence,
                source_timestamp=source_timestamp,
                behavior_result=behavior_result,
                preview_meta=preview_meta,
            )

            for alarm in response.get("alarm_events") or []:
                if not await send_realtime_message_safe(session_id, websocket, {
                    "type": "alarm_event",
                    "session_id": session_id,
                    "event": alarm,
                    "timestamp": alarm.get("alerted_at") or response.get("timestamp"),
                }):
                    break
            if not await send_realtime_message_safe(session_id, websocket, response):
                break

            # 服务器端视频流仍保留节流，max_fps 配置变更后下一轮立即生效。
            elapsed = time.time() - loop_start_time
            sleep_time = max(0, (1.0 / max(1, current_config.max_fps)) - elapsed)
            if sleep_time > 0:
                await asyncio.sleep(sleep_time)

    except asyncio.CancelledError:
        logger.info(f"服务器端视频流任务已取消: {session_id}")
    except Exception as e:
        logger.error(f"服务器端视频流处理错误 {session_id}: {e}")
    finally:
        logger.info(f"服务器端视频流处理结束: {session_id}")
        task = realtime_stream_tasks.get(session_id)
        if task is asyncio.current_task():
            realtime_stream_tasks.pop(session_id, None)


def process_realtime_frame(image: np.ndarray, model_path: str, config: RealtimeConfig) -> Dict:
    """处理实时推理帧(包装工具类)"""
    return inference_manager.process_realtime_frame(
        image, 
        model_path, 
        config.confidence, 
        config.iou_threshold,
        render_annotations=not (config.client_side_overlay or not config.show_labels),
    )


def process_realtime_batch(
    images: List[np.ndarray],
    model_path: str,
    confidence: float,
    iou_threshold: float,
    *,
    render_annotations: bool = True,
) -> List[Dict[str, Any]]:
    """Run one model call for compatible frames from multiple grid cells."""
    return inference_manager.process_realtime_batch(
        images,
        model_path,
        confidence,
        iou_threshold,
        render_annotations=render_annotations,
    )


async def train_model_background_deprecated(task_id: str, config: TrainingConfig, dataset_path: str):
    """已废弃的后台训练任务（保留用于兼容性）"""
    logger.warning("使用已废弃的训练方法，建议使用独立进程训练")
    # 这个函数现在只是一个占位符，实际训练由独立进程处理


# API端点

@app.get("/")
async def root():
    """API根端点"""
    return {
        "message": "YOLO Training & Inference API",
        "version": "1.0.0",
        "endpoints": {
            "training": "/train",
            "inference": "/predict",
            "export": "/export",
            "status": "/status/{task_id}",
            "models": "/models",
            "docs": "/docs"
        }
    }



@app.post("/upload/image")
@app.post("/api/upload/image")
async def upload_image_file(file: UploadFile = File(...)):
    """上传图片并保存到 ./data/image。"""
    if not is_allowed_image_upload(file):
        raise HTTPException(status_code=400, detail="只支持 JPG、JPEG、PNG、BMP、WEBP 图片文件")

    saved_path = None

    try:
        image_dir = get_image_upload_dir()
        original_filename = file.filename or "image.jpg"

        saved_path = await asyncio.to_thread(
            save_upload_file_keep_name,
            file,
            image_dir,
            fallback=sanitize_original_filename(original_filename, "image.jpg"),
            max_bytes=IMAGE_UPLOAD_MAX_BYTES,
        )

        # 二次校验：防止伪装成图片的文件被保存到 data/image。
        try:
            with Image.open(saved_path) as img:
                img.verify()
        except Exception:
            file_manager.cleanup_file(str(saved_path))
            raise HTTPException(status_code=400, detail="上传文件不是有效图片")
        validate_image_pixel_limit(saved_path)

        image_record = build_uploaded_image_record(saved_path, original_filename=original_filename)
        image_record["message"] = "图片上传成功"

        logger.info(f"图片上传成功: {saved_path}")
        return image_record

    except HTTPException:
        if saved_path:
            try:
                file_manager.cleanup_file(str(saved_path))
            except Exception:
                pass
        raise
    except Exception as e:
        if saved_path:
            try:
                file_manager.cleanup_file(str(saved_path))
            except Exception:
                pass
        logger.error(f"图片上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"图片上传失败: {str(e)}")


@app.get("/images")
@app.get("/api/images")
async def list_uploaded_images(
    page: int = 1,
    page_size: int = 100,
    search: str = "",
    include_dimensions: bool = False
):
    """
    分页列出 ./data/image 中已保存的图片，供上传页/标注页读取。

    大目录优化：
    1. 默认只返回第一页，避免一次性返回几千/几万张图片导致接口超时；
    2. 默认不逐张打开图片读取宽高，只读取文件名、大小、时间和 URL；
    3. 需要宽高时，前端点击单张图片后调用 /images/{filename}/data 获取。
    """
    started_at = time.time()

    try:
        image_dir = get_image_upload_dir()
        page = max(1, int(page or 1))
        page_size = max(1, min(int(page_size or 100), 500))
        search_text = str(search or "").strip().lower()

        image_entries = []
        try:
            with os.scandir(image_dir) as iterator:
                for entry in iterator:
                    if not entry.is_file():
                        continue

                    suffix = Path(entry.name).suffix.lower()
                    if suffix not in IMAGE_FILE_EXTENSIONS:
                        continue

                    if search_text and search_text not in entry.name.lower():
                        continue

                    stat = entry.stat()
                    image_entries.append((Path(entry.path), stat))
        except FileNotFoundError:
            image_entries = []

        image_entries.sort(key=lambda item: item[1].st_mtime, reverse=True)

        total = len(image_entries)
        start = (page - 1) * page_size
        end = start + page_size
        page_entries = image_entries[start:end]

        images = [
            build_uploaded_image_record(
                path,
                include_dimensions=include_dimensions,
                stat_result=stat
            )
            for path, stat in page_entries
        ]

        return {
            "total": total,
            "count": len(images),
            "page": page,
            "page_size": page_size,
            "has_more": end < total,
            "include_dimensions": include_dimensions,
            "search": search,
            "save_dir": str(image_dir),
            "save_path": "./data/image",
            "elapsed_ms": round((time.time() - started_at) * 1000, 2),
            "images": images
        }

    except Exception as e:
        logger.error(f"获取图片列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取图片列表失败: {str(e)}")


@app.get("/images/{filename}/data")
@app.get("/api/images/{filename}/data")
async def get_uploaded_image_data(filename: str):
    """
    读取 ./data/image 中指定图片并返回 base64 数据。

    用途：标注页点击左侧图片时，不再让浏览器跨域 fetch 静态图片，
    而是通过普通 API 返回 data:image/...;base64,...，避免 Failed to fetch。
    """
    try:
        image_path = resolve_uploaded_image_path_from_value(filename)
        if image_path is None:
            raise HTTPException(status_code=404, detail="图片文件不存在")

        encoded_data = image_processor.encode_image_file_to_base64(str(image_path))
        if not encoded_data:
            raise HTTPException(status_code=422, detail="图片 raw 编码失败")
        data_url, mime_type = encoded_data

        image_record = build_uploaded_image_record(image_path)
        image_record["data"] = data_url
        image_record["mime_type"] = mime_type
        return image_record

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"读取图片 base64 数据失败: {filename}, error={e}")
        raise HTTPException(status_code=500, detail=f"读取图片数据失败: {str(e)}")


@app.delete("/images/{filename}")
@app.delete("/api/images/{filename}")
async def delete_uploaded_image(filename: str):
    """删除 ./data/image 中指定图片。"""
    try:
        image_dir = get_image_upload_dir().resolve()
        safe_filename = sanitize_original_filename(filename, fallback="image.jpg")
        image_path = (image_dir / safe_filename).resolve()

        try:
            image_path.relative_to(image_dir)
        except ValueError:
            raise HTTPException(status_code=400, detail="非法文件名")

        if not image_path.exists() or not image_path.is_file():
            raise HTTPException(status_code=404, detail="图片文件不存在")

        if image_path.suffix.lower() not in IMAGE_FILE_EXTENSIONS:
            raise HTTPException(status_code=400, detail="只允许删除图片文件")

        deleted_name = image_path.name
        image_path.unlink()

        logger.info(f"图片已删除: {image_path}")
        return {
            "message": "图片已删除",
            "filename": deleted_name,
            "deleted_path": f"./data/image/{deleted_name}"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"删除图片失败: {e}")
        raise HTTPException(status_code=500, detail=f"删除图片失败: {str(e)}")


@app.post("/upload/dataset")
async def upload_dataset(
    request: Request,
    file: UploadFile = File(...),
    dataset_name: str = Form(""),
    owner_role_id: Optional[int] = Form(None),
    db=Depends(get_db),
):
    """上传数据集压缩包，优先使用用户填写的数据集名称。"""
    original_filename = file.filename or "dataset.zip"
    if not original_filename.lower().endswith('.zip'):
        raise HTTPException(status_code=400, detail="只支持ZIP格式的数据集")

    requested_name = str(dataset_name or "").strip()
    dataset_stem = sanitize_original_stem(requested_name or original_filename, fallback="dataset")
    dataset_path = build_unique_upload_dir(
        constants.DATASETS_DIR,
        dataset_stem,
        fallback="dataset"
    )
    dataset_id = dataset_path.name
    uploader_name = get_request_current_username(request)
    current_user = get_request_current_user(request)
    owner_role = select_upload_role(db, current_user, owner_role_id)
    archive_limits = ArchiveLimits.from_env()

    try:
        dataset_path.mkdir(parents=True, exist_ok=True)

        # 保存 ZIP 时保留上传原始文件名，不再固定为 dataset.zip。
        zip_path = save_upload_file_keep_name(
            file,
            dataset_path,
            fallback=sanitize_original_filename(original_filename, "dataset.zip"),
            max_bytes=archive_limits.max_upload_bytes,
        )

        safe_extract_zip(zip_path, dataset_path, archive_limits)

        # 删除ZIP文件
        file_manager.cleanup_file(str(zip_path))

        # 原始数据集只存放待标注图片，不在这里修复/生成 YOLO train/val/test 结构。
        stats = analyze_original_dataset(dataset_path)

        if stats.get("image_count", 0) <= 0:
            raise HTTPException(status_code=400, detail="原始数据集ZIP中没有可识别图片")

        # 构建数据集信息
        dataset_info = {
            "id": dataset_id,
            "name": requested_name or dataset_stem,
            "original_filename": original_filename,
            "path": str(dataset_path),
            "created_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
            "status": constants.DATASET_STATUS_READY if stats.get("image_count", 0) > 0 else constants.DATASET_STATUS_EMPTY,
            "dataset_role": "original",
            "uploader": uploader_name,
            "upload_user": uploader_name,
            "created_by": uploader_name,
            **stats
        }

        uploaded_datasets[dataset_id] = dataset_info

        # 新增：保存数据集元信息，之后 /datasets 直接读它，不再扫描大目录
        write_dataset_metadata(dataset_path, dataset_info)
        sync_dataset_metadata_to_database(
            db,
            dataset_path,
            dataset_info,
            dataset_role="original",
            source_type="upload",
        )
        ensure_dataset_resource(db, dataset_id, "original", current_user, owner_role)

        return {
            "dataset_id": dataset_id,
            "dataset_path": str(dataset_path),
            "original_filename": original_filename,
            "message": "数据集上传成功",
            "dataset_info": dataset_info
        }

    except ArchiveSafetyError as exc:
        file_manager.cleanup_dir(str(dataset_path))
        raise HTTPException(status_code=413, detail=str(exc))
    except (zipfile.BadZipFile, RuntimeError) as exc:
        file_manager.cleanup_dir(str(dataset_path))
        raise HTTPException(status_code=400, detail=f"无效的 ZIP 文件: {exc}")
    except HTTPException:
        file_manager.cleanup_dir(str(dataset_path))
        raise
    except Exception as e:
        # 清理失败的上传
        file_manager.cleanup_dir(str(dataset_path))
        logger.error(f"数据集上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"数据集上传失败: {str(e)}")


@app.post("/datasets/merge")
@app.post("/api/datasets/merge")
def merge_datasets(payload: DatasetMergeRequest, request: Request, db=Depends(get_db)):
    """Create a new original dataset from indexed images in multiple readable datasets."""
    dataset_name = str(payload.dataset_name or "").strip()
    source_dataset_ids = list(dict.fromkeys(
        str(value or "").strip() for value in (payload.source_dataset_ids or []) if str(value or "").strip()
    ))
    if not dataset_name:
        raise HTTPException(status_code=400, detail="请输入新数据集名称")
    if len(dataset_name) > 128:
        raise HTTPException(status_code=400, detail="新数据集名称不能超过128个字符")
    if len(source_dataset_ids) < 2:
        raise HTTPException(status_code=400, detail="至少选择两个源数据集")
    if len(source_dataset_ids) > 50:
        raise HTTPException(status_code=400, detail="单次最多合并50个数据集")

    current_user = get_request_current_user(request)
    owner_role = select_upload_role(db, current_user, payload.owner_role_id)
    dataset_path = build_unique_upload_dir(constants.DATASETS_DIR, dataset_name, fallback="merged-dataset")
    dataset_id = dataset_path.name
    merge_completed = False

    try:
        source_specs = []
        source_lineage = []
        required_bytes = 0
        for source_id in source_dataset_ids:
            source_resource = require_dataset_read(db, current_user, source_id, "original")
            source_owner_role = db.query(Role).filter(
                Role.id == source_resource.owner_role_id,
                Role.status == "active",
            ).first()
            if (
                source_owner_role is not None
                and int(owner_role.authority_level or 0) < int(source_owner_role.authority_level or 0)
            ):
                raise HTTPException(
                    status_code=403,
                    detail=f"新数据集归属角色级别不能低于源数据集: {source_id}",
                )
            version = (
                db.query(DatasetVersion)
                .filter(
                    DatasetVersion.dataset_id == source_id,
                    DatasetVersion.dataset_role == "original",
                    DatasetVersion.status == "ready",
                )
                .order_by(DatasetVersion.created_at.desc())
                .first()
            )
            if not version:
                raise HTTPException(status_code=409, detail=f"源数据集尚未建立可用资产索引: {source_id}")
            assets = (
                db.query(DatasetAsset)
                .filter(DatasetAsset.dataset_version_id == version.id)
                .order_by(DatasetAsset.relative_path.asc())
                .all()
            )
            required_bytes += sum(max(0, int(asset.size or 0)) for asset in assets)
            source_root = resolve_managed_dataset_root(source_id, constants.DATASETS_DIR)
            source_metadata = dict(version.metadata_json or {})
            source_specs.append({
                "dataset_id": source_id,
                "root_path": source_root,
                "relative_paths": [asset.relative_path for asset in assets],
            })
            source_lineage.append({
                "dataset_id": source_id,
                "dataset_name": source_metadata.get("name") or source_id,
                "version_id": version.id,
                "image_count": len(assets),
            })

        available_bytes = shutil.disk_usage(Path(constants.DATASETS_DIR)).free
        if required_bytes > available_bytes:
            raise HTTPException(status_code=507, detail="磁盘可用空间不足，无法创建合并数据集")
        merge_report = merge_original_dataset_assets(source_specs, dataset_path)
        stats = analyze_original_dataset(dataset_path)
        now = datetime.now().isoformat()
        uploader_name = get_request_current_username(request)
        dataset_info = {
            "id": dataset_id,
            "name": dataset_name,
            "path": str(dataset_path),
            "created_at": now,
            "updated_at": now,
            "status": constants.DATASET_STATUS_READY,
            "dataset_role": "original",
            "uploader": uploader_name,
            "upload_user": uploader_name,
            "created_by": uploader_name,
            "source_type": "merge",
            "merged_from_dataset_ids": source_dataset_ids,
            "merged_sources": source_lineage,
            "merge_strategy": "copy_indexed_images_preserve_duplicates",
            **stats,
        }
        if stats.get("image_count") != merge_report.get("image_count"):
            raise DatasetMergeError("合并后的图片统计与复制数量不一致")

        uploaded_datasets[dataset_id] = dataset_info
        write_dataset_metadata(dataset_path, dataset_info)
        sync_dataset_metadata_to_database(
            db,
            dataset_path,
            dataset_info,
            dataset_role="original",
            source_type="merge",
        )
        ensure_dataset_resource(db, dataset_id, "original", current_user, owner_role)
        merge_completed = True
        return {
            "message": "数据集合并成功",
            "dataset_id": dataset_id,
            "dataset_path": str(dataset_path),
            "dataset_info": dataset_info,
            "merge_report": merge_report,
        }
    except HTTPException:
        raise
    except DatasetMergeError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        logger.exception("数据集合并失败: %s", exc)
        raise HTTPException(status_code=500, detail=f"数据集合并失败: {exc}")
    finally:
        if not merge_completed:
            uploaded_datasets.pop(dataset_id, None)
            try:
                db.rollback()
                delete_dataset_version_records(db, dataset_id, "original")
                db.query(DatasetResource).filter(
                    DatasetResource.dataset_id == dataset_id,
                    DatasetResource.dataset_role == "original",
                ).delete(synchronize_session=False)
                db.commit()
            except Exception:
                db.rollback()
                logger.exception("清理失败的数据集合并数据库记录失败: %s", dataset_id)
            shutil.rmtree(dataset_path, ignore_errors=True)
            shutil.rmtree(DATA_DIR / "manifests" / dataset_id, ignore_errors=True)


@app.post("/datasets/{dataset_id}/augment/start")
@app.post("/api/datasets/{dataset_id}/augment/start")
async def start_dataset_augmentation(dataset_id: str, request: Dict[str, Any], http_request: Request, db=Depends(get_db)):
    """Start a durable job that creates a new original dataset with image variants."""
    dataset_id = str(dataset_id or "").strip()
    operations = normalize_augmentation_operations(request.get("operations"))
    if not dataset_id:
        raise HTTPException(status_code=400, detail="缺少源数据集 ID")
    if not operations:
        raise HTTPException(status_code=400, detail="至少选择一种数据增强操作")
    if len(operations) > 5:
        raise HTTPException(status_code=400, detail="数据增强操作数量超过限制")
    try:
        copies_per_image = max(1, min(int(request.get("copies_per_image") or 1), 3))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="每张图片生成数量无效")
    output_name = str(request.get("name") or f"{dataset_id}_增强").strip()
    if not output_name or len(output_name) > 128 or any(ord(char) < 32 for char in output_name):
        raise HTTPException(status_code=400, detail="增强数据集名称格式不正确")

    current_user = get_request_current_user(http_request)
    source_resource = require_dataset_manage(db, current_user, dataset_id, "original")
    source_version = (
        db.query(DatasetVersion)
        .filter(
            DatasetVersion.dataset_id == dataset_id,
            DatasetVersion.dataset_role == "original",
            DatasetVersion.status == "ready",
        )
        .order_by(DatasetVersion.created_at.desc())
        .first()
    )
    if not source_version:
        raise HTTPException(status_code=409, detail="数据集尚未建立可用资产索引")

    owner_role_id = request.get("owner_role_id")
    owner_role = None
    if owner_role_id is None and source_resource.owner_role_id is not None:
        owner_role = db.query(Role).filter(Role.id == source_resource.owner_role_id, Role.status == "active").first()
    if owner_role is None:
        owner_role = select_upload_role(db, current_user, owner_role_id)

    owner_username = str(current_user.get("username") or get_request_current_username(http_request))
    include_original = bool(request.get("include_original", True))
    normalized_input = {
        "source_dataset_id": dataset_id,
        "source_version_id": source_version.id,
        "operations": operations,
        "include_original": include_original,
        "copies_per_image": copies_per_image,
        "name": output_name,
        "owner_role_id": owner_role.id,
        "owner_id": current_user.get("id"),
        "owner_username": owner_username,
    }
    active_tasks = db.query(JobTask).filter(
        JobTask.type == "dataset_augmentation",
        JobTask.owner_username == owner_username,
        JobTask.status.in_(ACTIVE_STATUSES),
    ).order_by(JobTask.created_at.desc()).limit(20).all()
    for active_task in active_tasks:
        if dict(active_task.input_json or {}) == normalized_input:
            return {
                "task_id": active_task.id,
                "status": active_task.status,
                "stage": active_task.stage,
                "progress": active_task.progress,
                "message": active_task.message,
                "deduplicated": True,
            }

    job = create_job(
        db,
        job_type="dataset_augmentation",
        owner_username=owner_username,
        owner_id=current_user.get("id"),
        input_data=normalized_input,
        message="数据增强任务已创建",
    )
    handle = asyncio.create_task(asyncio.to_thread(run_dataset_augmentation_task, job.id))
    dataset_augmentation_task_handles[job.id] = handle
    handle.add_done_callback(lambda _task, augmentation_id=job.id: dataset_augmentation_task_handles.pop(augmentation_id, None))
    return {
        "task_id": job.id,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "message": job.message,
    }


@app.get("/datasets")
async def list_datasets(
    request: Request,
    page: int = 1,
    page_size: int = 20,
    include_temporary: bool = False,
    db=Depends(get_db),
):
    """
    获取已上传的数据集列表。

    优化点：
    1. 不再每次调用 analyze_dataset()
    2. 只读取 metadata.json
    3. 支持分页
    4. 对缺少 metadata 的旧数据集返回占位信息，并标记 need_refresh=True
    """
    try:
        page = max(int(page or 1), 1)
        page_size = max(min(int(page_size or 20), 100), 1)
        total, datasets_list = list_persisted_dataset_infos(
            db,
            "original",
            page,
            page_size,
            get_request_current_user(request),
            include_temporary=include_temporary,
        )
        for dataset_info in datasets_list:
            uploaded_datasets[dataset_info["id"]] = dataset_info

        return {
            "total_datasets": total,
            "total": total,
            "page": page,
            "page_size": page_size,
            "datasets": datasets_list
        }

    except Exception as e:
        logger.error(f"获取数据集列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取数据集列表失败: {str(e)}")


@app.post("/datasets/{dataset_id}/annotation-model")
@app.post("/api/datasets/{dataset_id}/annotation-model")
async def save_dataset_annotation_model(
    dataset_id: str,
    request: Dict[str, Any],
    http_request: Request,
    db=Depends(get_db),
):
    """Persist the latest model selected before entering annotation.

    The value is informational/default metadata only. A later annotation
    session may select a different model and geometry.
    """
    dataset_id = str(dataset_id or "").strip()
    model_path = str(request.get("model_path") or request.get("model") or "").strip()
    model_name = str(request.get("model_name") or request.get("modelName") or model_path).strip()
    geometry = str(request.get("annotation_geometry") or request.get("geometry") or "").strip().lower()
    annotation_kind = str(request.get("annotation_kind") or request.get("draft_kind") or "").strip().lower()
    if annotation_kind and annotation_kind not in ANNOTATION_KINDS:
        raise HTTPException(status_code=400, detail="annotation_kind 仅支持 multimodal 或 yolo")
    if not dataset_id or not model_path:
        raise HTTPException(status_code=400, detail="缂哄皯鏁版嵁闆嗘垨鏍囨敞妯″瀷")
    if geometry not in {"bbox", "rect", "polygon"}:
        raise HTTPException(status_code=400, detail="annotation_geometry 仅支持 rect、bbox 或 polygon")
    geometry = "polygon" if geometry == "polygon" else "rect"

    current_user = get_request_current_user(http_request)
    require_dataset_manage(db, current_user, dataset_id, "original")
    matched_registry_model = _validate_annotation_model_selection(
        db,
        current_user,
        model_path,
        annotation_kind,
    )
    version = (
        db.query(DatasetVersion)
        .filter(
            DatasetVersion.dataset_id == dataset_id,
            DatasetVersion.dataset_role == "original",
            DatasetVersion.status == "ready",
        )
        .order_by(DatasetVersion.created_at.desc(), DatasetVersion.id.desc())
        .first()
    )
    if not version:
        raise HTTPException(status_code=409, detail="鏁版嵁闆嗗皻鏈缓绔嬪彲鐢ㄧ増鏈储寮?")

    model_source = str(
        request.get("model_source") or request.get("source") or request.get("model_type") or ""
    ).strip().lower()
    is_trained_model = model_source == "trained" or bool(
        matched_registry_model and str(getattr(matched_registry_model, "source", "")).lower() == "trained"
    )
    metadata = dict(version.metadata_json or {})
    imported_classes: List[str] = []
    import_warning = ""
    if is_trained_model:
        imported_classes = get_training_model_class_names(db, model_path, request)
        if imported_classes:
            try:
                page_classes = resolve_model_page_classes(
                    get_dataset_annotation_classes(version)[0],
                    imported_classes,
                    get_inactive_annotation_class_ids(version),
                )
            except ValueError as exc:
                raise HTTPException(status_code=409, detail=str(exc))
            metadata[ANNOTATION_CLASSES_METADATA_KEY] = page_classes
            metadata[ANNOTATION_CLASS_BINDINGS_METADATA_KEY] = build_annotation_class_bindings(
                page_classes,
                imported_classes,
                get_inactive_annotation_class_ids(version),
            )
            metadata["annotation_classes_source"] = "trained_model"
            metadata["annotation_classes_updated_at"] = datetime.now().isoformat()
            metadata["annotation_model_native_classes"] = imported_classes
        else:
            import_warning = "所选训练模型缺少类别元数据，已保留当前页面类别；请手动确认类别映射。"

    binding = {
        "path": model_path,
        "name": model_name or model_path,
        "geometry": geometry,
        "selected_by": str(current_user.get("username") or "system"),
        "selected_at": datetime.utcnow().isoformat() + "Z",
    }
    existing = get_dataset_annotation_model(db, version, allow_source_fallback=False)
    history = metadata.get("annotation_model_history")
    if not isinstance(history, list):
        history = []
    if existing:
        existing_key = (
            str(existing.get("path") or "").strip().lower(),
            str(existing.get("geometry") or "").strip().lower(),
        )
        binding_key = (model_path.lower(), geometry)
        if existing_key != binding_key:
            history.append(existing)
    metadata["annotation_model_history"] = history[-50:]
    metadata["annotation_model_selection_mode"] = "reselectable"
    metadata["annotation_model"] = binding
    metadata["annotation_model_path"] = model_path
    metadata["annotation_model_name"] = model_name or model_path
    metadata["annotation_geometry"] = geometry
    version.metadata_json = metadata
    dataset_path = Path(version.root_path)
    try:
        file_metadata = read_dataset_metadata(dataset_path) or {}
        file_metadata.update({
            "annotation_model_history": metadata["annotation_model_history"],
            "annotation_model_selection_mode": "reselectable",
            "annotation_model": binding,
            "annotation_model_path": model_path,
            "annotation_model_name": model_name or model_path,
            "annotation_geometry": geometry,
            **(
                {
                    ANNOTATION_CLASSES_METADATA_KEY: metadata.get(ANNOTATION_CLASSES_METADATA_KEY),
                    ANNOTATION_CLASS_BINDINGS_METADATA_KEY: metadata.get(ANNOTATION_CLASS_BINDINGS_METADATA_KEY, []),
                    "annotation_classes_source": metadata.get("annotation_classes_source"),
                    "annotation_model_native_classes": metadata.get("annotation_model_native_classes"),
                }
                if imported_classes
                else {}
            ),
        })
        write_dataset_metadata(dataset_path, file_metadata)
        db.commit()
        db.refresh(version)
    except Exception:
        db.rollback()
        raise

    return {
        "dataset_id": dataset_id,
        "version_id": version.id,
        "annotation_model": binding,
        "annotation_classes": metadata.get(ANNOTATION_CLASSES_METADATA_KEY),
        "annotation_class_bindings": normalize_annotation_class_bindings(
            metadata.get(ANNOTATION_CLASS_BINDINGS_METADATA_KEY),
            metadata.get(ANNOTATION_CLASSES_METADATA_KEY),
        ),
        "annotation_classes_source": metadata.get("annotation_classes_source"),
        "warning": import_warning,
    }


def _draft_current_version_or_conflict(db, dataset_id: str, draft: AnnotationDraft) -> DatasetVersion:
    version = _get_latest_original_dataset_version(db, dataset_id)
    if str(draft.source_version_id or "") != str(version.id):
        raise HTTPException(status_code=409, detail="标注草稿对应的数据集版本已变化，请重新开始标注")
    return version


def _draft_working_version(db, dataset_id: str, draft: AnnotationDraft, source_version: Optional[DatasetVersion] = None) -> DatasetVersion:
    """Resolve the draft-only dataset copy without weakening source-version drift checks."""
    source_version = source_version or _draft_current_version_or_conflict(db, dataset_id, draft)
    temporary_dataset_id = str(draft.temporary_dataset_id or "").strip()
    if not temporary_dataset_id:
        return draft_dataset_view(source_version, draft)
    working_version = (
        db.query(DatasetVersion)
        .filter(
            DatasetVersion.dataset_id == temporary_dataset_id,
            DatasetVersion.dataset_role == "original",
            DatasetVersion.status == "ready",
        )
        .order_by(DatasetVersion.created_at.desc(), DatasetVersion.id.desc())
        .first()
    )
    metadata = dict(working_version.metadata_json or {}) if working_version else {}
    if (
        working_version is None
        or metadata.get("source_type") != "annotation_temporary"
        or str(metadata.get("source_dataset_id") or "") != str(dataset_id)
    ):
        raise HTTPException(status_code=409, detail="标注草稿的临时增强数据集已失效，请重新开始标注")
    return draft_dataset_view(working_version, draft)


def _delete_annotation_temporary_dataset(db, temporary_dataset_id: str, source_dataset_id: str) -> bool:
    temporary_dataset_id = str(temporary_dataset_id or "").strip()
    if not temporary_dataset_id:
        return False
    version = (
        db.query(DatasetVersion)
        .filter(
            DatasetVersion.dataset_id == temporary_dataset_id,
            DatasetVersion.dataset_role == "original",
            DatasetVersion.status == "ready",
        )
        .order_by(DatasetVersion.created_at.desc(), DatasetVersion.id.desc())
        .first()
    )
    if version is None:
        return False
    metadata = dict(version.metadata_json or {})
    if (
        metadata.get("source_type") != "annotation_temporary"
        or str(metadata.get("source_dataset_id") or "") != str(source_dataset_id)
    ):
        raise HTTPException(status_code=409, detail="拒绝清理不属于当前标注草稿的临时数据集")
    temporary_path = resolve_managed_dataset_root(temporary_dataset_id, constants.DATASETS_DIR)
    staged_path = None
    try:
        staged_path = stage_directory_for_deletion(temporary_path)
        delete_dataset_version_records(db, temporary_dataset_id, "annotation")
        delete_dataset_version_records(db, temporary_dataset_id, "original")
        db.query(DatasetResource).filter(
            DatasetResource.dataset_id == temporary_dataset_id,
        ).delete(synchronize_session=False)
        delete_drafts_for_dataset(db, temporary_dataset_id)
        db.commit()
    except Exception:
        db.rollback()
        restore_staged_directory(staged_path, temporary_path)
        raise
    purge_staged_directory(staged_path)
    uploaded_datasets.pop(temporary_dataset_id, None)
    shutil.rmtree(DATA_DIR / "manifests" / temporary_dataset_id, ignore_errors=True)
    return True


def _delete_draft_temporary_dataset(db, draft: Optional[AnnotationDraft]) -> bool:
    temporary_dataset_id = str(draft.temporary_dataset_id or "").strip() if draft else ""
    deleted = _delete_annotation_temporary_dataset(db, temporary_dataset_id, draft.dataset_id) if draft else False
    if draft:
        draft.temporary_dataset_id = None
    return deleted


def _annotation_request_draft(db, dataset_id, draft_id, current_user, model_path):
    require_dataset_manage(db, current_user, dataset_id, "original")
    draft = get_draft_by_id(db, _draft_http_user_id(current_user), draft_id)
    if draft is None or str(draft.dataset_id) != str(dataset_id):
        raise HTTPException(status_code=404, detail="标注草稿不存在或不属于当前数据集")
    _draft_current_version_or_conflict(db, dataset_id, draft)
    if not annotation_model_matches(model_path, {"path": draft.model_path}):
        raise HTTPException(status_code=409, detail="请求模型与标注会话模型不一致")
    return draft


def _draft_http_user_id(current_user: Dict[str, Any]) -> int:
    try:
        return draft_user_id(current_user)
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


@app.get("/datasets/{dataset_id}/annotation-drafts")
@app.get("/api/datasets/{dataset_id}/annotation-drafts")
async def get_annotation_draft_summary(
    dataset_id: str,
    request: Request,
    db=Depends(get_db),
):
    current_user = get_request_current_user(request)
    require_dataset_read(db, current_user, dataset_id, "original")
    version = _get_latest_original_dataset_version(db, dataset_id)
    owner_id = _draft_http_user_id(current_user)
    drafts = {}
    for kind in sorted(ANNOTATION_KINDS):
        draft = get_draft(db, owner_id, dataset_id, kind)
        if draft is None:
            drafts[kind] = {"exists": False}
            continue
        drafts[kind] = {
            "exists": True,
            "stale": str(draft.source_version_id or "") != str(version.id),
            "id": draft.id,
            "annotation_kind": kind,
            "model_path": draft.model_path,
            "model_name": draft.model_name,
            "annotation_geometry": draft.annotation_geometry,
            "revision": int(draft.revision or 0),
            "last_page": int(draft.last_page or 1),
            "last_image_path": draft.last_image_path or "",
            "temporary_dataset_id": draft.temporary_dataset_id,
            "updated_at": draft.updated_at.isoformat() if draft.updated_at else None,
            "saved_image_count": serialize_draft(db, draft)["saved_image_count"],
        }
    return {"dataset_id": dataset_id, "source_version_id": version.id, "drafts": drafts}


@app.get("/datasets/{dataset_id}/annotation-drafts/{kind}")
@app.get("/api/datasets/{dataset_id}/annotation-drafts/{kind}")
async def get_annotation_draft_metadata(
    dataset_id: str,
    kind: str,
    request: Request,
    db=Depends(get_db),
    draft_id: str = "",
):
    try:
        kind = normalize_annotation_kind(kind)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    current_user = get_request_current_user(request)
    require_dataset_read(db, current_user, dataset_id, "original")
    owner_id = _draft_http_user_id(current_user)
    draft = get_draft(db, owner_id, dataset_id, kind)
    if draft is None:
        raise HTTPException(status_code=404, detail="标注草稿不存在")
    if draft_id and str(draft.id) != draft_id:
        raise HTTPException(status_code=409, detail="当前标注草稿已被新建会话替换，请重新进入")
    _draft_current_version_or_conflict(db, dataset_id, draft)
    _validate_annotation_model_selection(db, current_user, draft.model_path, kind)
    return {**serialize_draft(db, draft), "annotation_batch": latest_draft_batch(db, draft)}


@app.post("/datasets/{dataset_id}/annotation-drafts/{kind}")
@app.post("/api/datasets/{dataset_id}/annotation-drafts/{kind}")
async def create_annotation_draft(
    dataset_id: str, kind: str, request: Dict[str, Any], http_request: Request,
    db=Depends(get_db),
):
    try:
        kind = normalize_annotation_kind(kind)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    current_user = get_request_current_user(http_request)
    require_dataset_manage(db, current_user, dataset_id, "original")
    owner_id = _draft_http_user_id(current_user)
    version = _get_latest_original_dataset_version(db, dataset_id)
    existing = get_draft_for_update(db, owner_id, dataset_id, kind)
    if existing is not None:
        if str(request.get("replace_draft_id") or "") != str(existing.id) or request.get("replace_revision") != int(existing.revision or 0):
            raise HTTPException(status_code=409, detail="同类型已有标注草稿，请恢复或重新确认覆盖")
        active = db.query(JobTask.id).filter(JobTask.type.in_(['annotation','dataset_export','dataset_augmentation']),
            JobTask.status.in_(ACTIVE_STATUSES), JobTask.input_json['annotation_draft_id'].as_string() == existing.id).first()
        if active:
            raise HTTPException(status_code=409, detail="原草稿仍有进行中的标注、增强或导出任务，请等待结束后新建")
    elif request.get("replace_draft_id"):
        raise HTTPException(status_code=409, detail="原草稿已变化，请重新选择新建标注")
    model_path = str(request.get("model_path") or "").strip()
    record = _validate_annotation_model_selection(db, current_user, model_path, kind)
    if record is None:
        raise HTTPException(status_code=409, detail="新建标注必须选择已注册的模型")
    model_path = str(resolve_registered_model_path(record))
    try:
        classes = await asyncio.to_thread(checkpoint_class_names, model_path)
        dynamic = annotation_model_supports_dynamic_text_prompts(model_path, record.to_dict())
        placeholders = dynamic and all(name == str(index) for index, name in enumerate(classes))
        if placeholders:
            classes = ["object"]
        model_inactive = []
        linked_id = str(getattr(record, "dataset_version_id", "") or "")
        if linked_id and not placeholders:
            linked = db.query(DatasetVersion).filter(DatasetVersion.id == linked_id).first()
            if linked and ordered_class_names(linked.class_map_json) == classes:
                model_inactive = get_inactive_annotation_class_ids(linked)
        context = normalize_class_context(classes, {"inactive_class_ids": model_inactive})
        if placeholders:
            context.update({"initialization_mode": "dynamic_prompt",
                "initialization_note": "所选开放词汇模型未提供固定语义类别；本次以 object 初始化，请编辑当前标注类别。"})
    except Exception as exc:
        raise HTTPException(status_code=409, detail=f"无法初始化所选模型的类别表: {exc}") from exc
    bindings = [{"page_index": index, "name": name, "model_class_id": index,
                 "model_class_name": name} for index, name in enumerate(classes)]
    geometry = str(request.get("annotation_geometry") or model_annotation_geometry(record.to_dict(), model_path)).lower()
    geometry = "polygon" if geometry in {"polygon", "segment", "segmentation"} else "rect"
    old_temporary = str(existing.temporary_dataset_id or "") if existing else ""
    try:
        if existing is not None:
            delete_draft(db, existing)
            db.flush()
        draft = upsert_draft(db, draft=None, owner_id=owner_id,
            owner_username=str(current_user.get("username") or "system"), dataset_id=dataset_id,
            source_version_id=version.id, annotation_kind=kind, model_path=model_path,
            model_name=str(record.to_dict().get("display_name") or record.name),
            annotation_geometry=geometry, class_names=classes, class_bindings=bindings,
            last_page=1, last_image_path="", assets=[])
        draft.class_context_json = context
        db.commit()
    except IntegrityError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail="其他标签页已新建标注，请恢复最新草稿") from exc
    db.refresh(draft)
    result = serialize_draft(db, draft)
    if old_temporary:
        try:
            _delete_annotation_temporary_dataset(db, old_temporary, dataset_id)
        except Exception:
            db.rollback()
            logger.exception("新标注已建立，旧增强副本清理失败: %s", old_temporary)
            result['warning'] = "新标注已建立，旧增强副本尚未清理；原始数据和新草稿不受影响。"
    return result


@app.put("/datasets/{dataset_id}/annotation-drafts/{kind}")
@app.put("/api/datasets/{dataset_id}/annotation-drafts/{kind}")
async def put_annotation_draft(
    dataset_id: str,
    kind: str,
    request: Dict[str, Any],
    http_request: Request,
    db=Depends(get_db),
):
    try:
        kind = normalize_annotation_kind(kind)
        payload = sanitize_draft_payload(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    current_user = get_request_current_user(http_request)
    require_dataset_manage(db, current_user, dataset_id, "original")
    version = _get_latest_original_dataset_version(db, dataset_id)
    owner_id = _draft_http_user_id(current_user)
    draft = get_draft_for_update(db, owner_id, dataset_id, kind)
    requested_draft_id = str(request.get("draft_id") or "").strip()
    if requested_draft_id and (draft is None or str(draft.id) != requested_draft_id):
        raise HTTPException(status_code=409, detail="当前标注草稿已被替换，请重新进入")
    inactive_ids = [] if isolated_draft(draft) else get_inactive_annotation_class_ids(version)
    if inactive_ids:
        stable_classes, _ = get_dataset_annotation_classes(version)
        if payload.get("class_names") != stable_classes:
            raise HTTPException(status_code=409, detail="训练类别表已更新，请刷新标注页面")
        if any(len(without_inactive_annotations(item.get("annotations"), inactive_ids, stable_classes))
               != len(item.get("annotations") or []) for item in payload.get("assets") or []):
            raise HTTPException(status_code=409, detail="已停用类别的标注不能重新保存，请刷新标注页面")
    if draft is not None and str(draft.source_version_id or "") != str(version.id):
        raise HTTPException(status_code=409, detail="标注草稿对应的数据集版本已变化")
    expected_revision = request.get("revision")
    if draft is not None:
        try:
            if expected_revision is None or int(expected_revision) != int(draft.revision or 0):
                raise HTTPException(status_code=409, detail="标注草稿已在其他标签页更新，请刷新后确认覆盖")
        except (TypeError, ValueError):
            raise HTTPException(status_code=409, detail="标注草稿修订号无效")
    elif expected_revision not in (None, "", 0, "0"):
        raise HTTPException(status_code=409, detail="新建标注草稿修订号无效")

    model_path = str(request.get("model_path") or request.get("model") or (draft.model_path if draft else "")).strip()
    model_name = str(request.get("model_name") or request.get("modelName") or (draft.model_name if draft else model_path)).strip()
    geometry = str(
        request.get("annotation_geometry")
        or request.get("geometry")
        or (draft.annotation_geometry if draft else "rect")
    ).strip().lower()
    geometry = "polygon" if geometry in {"polygon", "segmentation", "segment"} else "rect"
    _validate_annotation_model_selection(db, current_user, model_path, kind)
    if isolated_draft(draft) and not annotation_model_matches(model_path, {"path": draft.model_path}):
        raise HTTPException(status_code=409, detail="本次标注模型已锁定，更换模型请新建标注")
    source_version_id = str(request.get("source_version_id") or version.id).strip()
    if source_version_id != str(version.id):
        raise HTTPException(status_code=409, detail="标注草稿来源版本已变化")

    working_version = _draft_working_version(db, dataset_id, draft, version) if draft is not None else version
    requested_paths = {item["relative_path"] for item in payload["assets"]}
    if requested_paths:
        valid_paths = {
            row[0]
            for row in db.query(DatasetAsset.relative_path)
            .filter(
                DatasetAsset.dataset_version_id == working_version.id,
                DatasetAsset.relative_path.in_(requested_paths),
            )
            .all()
        }
        unknown_paths = requested_paths - valid_paths
        if unknown_paths:
            raise HTTPException(status_code=400, detail="标注草稿包含不属于当前数据集的图片")
    if bool(request.get("clear_all_annotations", False)):
        explicit_assets = {item["relative_path"]: item for item in payload["assets"]}
        payload["assets"] = [
            explicit_assets.get(asset.relative_path) or {
                "relative_path": asset.relative_path,
                "width": int(asset.width or 0),
                "height": int(asset.height or 0),
                "annotations": [],
            }
            for asset in db.query(DatasetAsset)
            .filter(DatasetAsset.dataset_version_id == working_version.id)
            .order_by(DatasetAsset.relative_path.asc())
            .all()
        ]

    class_names = (
        payload["class_names"]
        if "class_names" in request or "classes" in request
        else (draft.class_names_json if draft else [])
    )
    class_bindings = (
        payload["class_bindings"]
        if "class_bindings" in request or "annotation_class_bindings" in request
        else (draft.class_bindings_json if draft else [])
    )
    if isolated_draft(draft):
        old_classes = list(draft.class_names_json or [])
        previous_context = draft.class_context_json or {}
        try:
            class_names = normalize_class_names(class_names)
            if len(class_names) < len(old_classes):
                raise ValueError("当前会话的类别编号必须保持稳定，请使用停用类别")
            if previous_context.get("inactive_class_ids") and class_names[:len(old_classes)] != old_classes:
                raise ValueError("当前会话已停用类别的编号和名称必须保持稳定")
            context = normalize_class_context(class_names, request.get("class_context"), previous_context)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        inactive = context["inactive_class_ids"]
        if inactive:
            for row in db.query(AnnotationDraftAsset).filter(AnnotationDraftAsset.draft_id == draft.id).all():
                row.annotations_json = without_inactive_annotations(row.annotations_json or [], inactive, class_names)
            for item in payload["assets"]:
                item["annotations"] = without_inactive_annotations(item.get("annotations") or [], inactive, class_names)
        try:
            for item in payload['assets']:
                validate_session_annotation_ids(item.get('annotations') or [], class_names)
        except ValueError as exc:
            raise HTTPException(status_code=409, detail=str(exc)) from exc
        draft.class_context_json = context
    last_page = (
        payload["last_page"]
        if any(key in request for key in ("last_page", "image_page"))
        else int(draft.last_page or 1) if draft else 1
    )
    last_image_path = (
        payload["last_image_path"]
        if any(key in request for key in ("last_image_path", "selected_image_path"))
        else (draft.last_image_path if draft else "")
    )
    draft = upsert_draft(
        db,
        draft=draft,
        owner_id=owner_id,
        owner_username=str(current_user.get("username") or get_request_current_username(http_request)),
        dataset_id=dataset_id,
        source_version_id=version.id,
        annotation_kind=kind,
        model_path=model_path,
        model_name=model_name,
        annotation_geometry=geometry,
        class_names=class_names,
        class_bindings=class_bindings,
        last_page=last_page,
        last_image_path=last_image_path,
        assets=payload["assets"],
    )
    batch_result = None
    if request.get("annotation_task_id"):
        if request.get("clear_all_annotations"):
            raise HTTPException(status_code=400, detail="清空标注与同步 AI 结果不能同时提交")
        try:
            batch_result = import_draft_batch(db, draft, version, working_version,
                request["annotation_task_id"], inactive_ids=inactive_ids,
                override_paths={item["relative_path"] for item in payload["assets"]})
        except ValueError as exc:
            db.rollback()
            raise HTTPException(status_code=409, detail=str(exc)) from exc
    db.commit()
    db.refresh(draft)
    return {**serialize_draft(db, draft), "batch_sync": batch_result}


@app.delete("/datasets/{dataset_id}/annotation-drafts/{kind}")
@app.delete("/api/datasets/{dataset_id}/annotation-drafts/{kind}")
async def delete_annotation_draft(
    dataset_id: str,
    kind: str,
    request: Request,
    db=Depends(get_db),
):
    try:
        kind = normalize_annotation_kind(kind)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    current_user = get_request_current_user(request)
    require_dataset_manage(db, current_user, dataset_id, "original")
    draft = get_draft(db, _draft_http_user_id(current_user), dataset_id, kind)
    _delete_draft_temporary_dataset(db, draft)
    deleted = delete_draft(db, draft)
    db.commit()
    return {"dataset_id": dataset_id, "annotation_kind": kind, "deleted": deleted}


@app.get("/datasets/{dataset_id}/images")
@app.get("/api/datasets/{dataset_id}/images")
async def list_dataset_images(
    dataset_id: str,
    request: Request,
    page: int = 1,
    page_size: int = 100,
    include_dimensions: bool = False,
    search: str = "",
    dataset_role: str = "original",
    draft_kind: str = "",
    draft_id: str = "",
    db=Depends(get_db),
):
    """从 dataset_assets 索引分页列出图片，不扫描文件系统。"""
    try:
        dataset_role = str(dataset_role or "original").strip().lower()
        if dataset_role not in {"original", "training"}:
            raise HTTPException(status_code=400, detail="dataset_role 只能是 original 或 training")
        current_user = get_request_current_user(request)
        require_dataset_read(db, current_user, dataset_id, dataset_role)
        version = (
            db.query(DatasetVersion)
            .filter(DatasetVersion.dataset_id == dataset_id, DatasetVersion.dataset_role == dataset_role, DatasetVersion.status == "ready")
            .order_by(DatasetVersion.created_at.desc())
            .first()
        )
        if not version:
            raise HTTPException(status_code=409, detail="数据集尚未建立可用资产索引")
        draft = None
        if str(draft_kind or "").strip():
            if dataset_role != "original":
                raise HTTPException(status_code=400, detail="标注草稿只能叠加到原始数据集")
            try:
                normalized_draft_kind = normalize_annotation_kind(draft_kind)
            except ValueError as exc:
                raise HTTPException(status_code=400, detail=str(exc)) from exc
            draft = get_draft(db, _draft_http_user_id(current_user), dataset_id, normalized_draft_kind)
            if draft is None:
                raise HTTPException(status_code=404, detail="标注草稿不存在")
            if draft_id and str(draft.id) != draft_id:
                raise HTTPException(status_code=409, detail="当前标注草稿已被替换，请重新进入")
            if str(draft.source_version_id or "") != str(version.id):
                raise HTTPException(status_code=409, detail="标注草稿对应的数据集版本已变化")
            _validate_annotation_model_selection(db, current_user, draft.model_path, normalized_draft_kind)
        source_version = version
        source_dataset_path = Path(source_version.root_path)
        dataset_info = build_dataset_info_from_metadata(source_dataset_path)
        dataset_name = dataset_info.get("name") or dataset_id
        if draft is not None:
            version = _draft_working_version(db, dataset_id, draft, source_version)
        dataset_path = Path(version.root_path)
        page = max(1, int(page or 1))
        page_size = max(1, min(int(page_size or 100), 500))
        query = db.query(DatasetAsset).filter(DatasetAsset.dataset_version_id == version.id)
        search_text = str(search or "").strip()
        if search_text:
            query = query.filter(DatasetAsset.relative_path.ilike(f"%{search_text}%"))
        total = query.count()
        assets = query.order_by(DatasetAsset.relative_path.asc()).offset((page - 1) * page_size).limit(page_size).all()
        draft_assets = {}
        if draft is not None and assets:
            draft_assets = {
                row.relative_path: row
                for row in db.query(AnnotationDraftAsset)
                .filter(
                    AnnotationDraftAsset.draft_id == draft.id,
                    AnnotationDraftAsset.relative_path.in_([asset.relative_path for asset in assets]),
                )
                .all()
            }
        class_version = version if isolated_draft(draft) else source_version
        images = []
        for asset in assets:
            source_image = (dataset_path / asset.relative_path).resolve()
            width, height = int(asset.width or 0), int(asset.height or 0)
            try:
                source_image.relative_to(dataset_path.resolve())
                detected_width, detected_height = get_uploaded_image_size(source_image)
                if detected_width > 0 and detected_height > 0:
                    width, height = int(detected_width), int(detected_height)
            except Exception:
                pass
            draft_asset = draft_assets.get(asset.relative_path)
            images.append({
                "id": asset.id,
                "asset_id": asset.id,
                "filename": asset.basename,
                "name": asset.basename,
                "dataset_id": version.dataset_id,
                "dataset_name": dataset_name,
                "dataset_version_id": version.id,
                "relative_path": asset.relative_path,
                "url": f"/api/datasets/{version.dataset_id}/assets/{asset.id}/content",
                "fetch_url": f"/api/datasets/{version.dataset_id}/assets/{asset.id}/content",
                "thumbnail_url": f"/api/datasets/{version.dataset_id}/assets/{asset.id}/thumbnail",
                "path": str(dataset_path / asset.relative_path),
                "saved_path": asset.relative_path,
                "source": dataset_name,
                "sourcePath": str(dataset_path),
                "size": asset.size,
                "width": width,
                "height": height,
                "coordinate_space": "raw",
                "type": mimetypes.guess_type(asset.basename)[0] or "application/octet-stream",
                "created_at": version.created_at.isoformat() if version.created_at else None,
                "updated_at": version.created_at.isoformat() if version.created_at else None,
                "content_hash": asset.content_hash,
                "annotations": (
                    without_inactive_annotations(draft_asset.annotations_json or [],
                                                 get_inactive_annotation_class_ids(class_version),
                                                 get_dataset_annotation_classes(class_version)[0])
                    if draft_asset is not None
                    else without_inactive_annotations(
                        manifest_frontend_annotations(session_base_annotations(version, asset.annotations_json))
                        if isolated_draft(draft) else asset.annotations_json or [],
                                                      get_inactive_annotation_class_ids(class_version),
                                                      get_dataset_annotation_classes(class_version)[0])
                ),
            })

        return {
            "dataset_id": dataset_id,
            "dataset_name": dataset_name,
            "dataset_path": str(dataset_path),
            "annotation_classes": get_dataset_annotation_classes(version)[0],
            "total": total,
            "count": len(images),
            "page": page,
            "page_size": page_size,
            "has_more": page * page_size < total,
            "include_dimensions": include_dimensions,
            "draft_kind": normalized_draft_kind if draft is not None else None,
            "draft_id": draft.id if draft is not None else None,
            "draft_revision": int(draft.revision or 0) if draft is not None else None,
            "draft_source_version_id": draft.source_version_id if draft is not None else None,
            "draft_temporary_dataset_id": draft.temporary_dataset_id if draft is not None else None,
            "draft_last_page": int(draft.last_page or 1) if draft is not None else None,
            "draft_last_image_path": draft.last_image_path if draft is not None else None,
            "draft_model_path": draft.model_path if draft is not None else None,
            "draft_model_name": draft.model_name if draft is not None else None,
            "draft_annotation_geometry": draft.annotation_geometry if draft is not None else None,
            "draft_class_names": draft.class_names_json if draft is not None else None,
            "draft_class_bindings": draft.class_bindings_json if draft is not None else None,
            "draft_class_scope": (draft.class_context_json or {}).get("scope", "dataset") if draft else None,
            "draft_class_context": draft.class_context_json or {} if draft else {},
            "images": images
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"获取数据集图片列表失败: dataset={dataset_id}, error={e}")
        raise HTTPException(status_code=500, detail=f"获取数据集图片列表失败: {str(e)}")


def _get_latest_original_dataset_version(db, dataset_id: str) -> DatasetVersion:
    version = (
        db.query(DatasetVersion)
        .filter(
            DatasetVersion.dataset_id == dataset_id,
            DatasetVersion.dataset_role == "original",
            DatasetVersion.status == "ready",
        )
        .order_by(DatasetVersion.created_at.desc(), DatasetVersion.id.desc())
        .first()
    )
    if not version:
        raise HTTPException(status_code=409, detail="数据集尚未建立可用资产索引")
    return version


@app.get("/datasets/{dataset_id}/annotation-classes")
@app.get("/api/datasets/{dataset_id}/annotation-classes")
async def get_dataset_annotation_classes_api(
    dataset_id: str,
    request: Request,
    db=Depends(get_db),
):
    require_dataset_read(db, get_request_current_user(request), dataset_id, "original")
    version = _get_latest_original_dataset_version(db, dataset_id)
    context = get_dataset_annotation_class_context(version)
    return {
        "dataset_id": dataset_id,
        "version_id": version.id,
        "classes": context["classes"],
        "annotation_classes": context["classes"],
        "labels_zh": context["labels_zh"],
        "inactive_class_ids": context["inactive_class_ids"],
        "bindings": context["bindings"],
        "annotation_class_bindings": context["bindings"],
        "source": context["source"],
    }


@app.put("/datasets/{dataset_id}/annotation-classes")
@app.put("/api/datasets/{dataset_id}/annotation-classes")
async def update_dataset_annotation_classes(
    dataset_id: str,
    request: Dict[str, Any],
    http_request: Request,
    db=Depends(get_db),
):
    current_user = get_request_current_user(http_request)
    require_dataset_manage(db, current_user, dataset_id, "original")
    version = _get_latest_original_dataset_version(db, dataset_id)
    version = db.query(DatasetVersion).filter(DatasetVersion.id == version.id).with_for_update().populate_existing().one()
    raw_classes = request.get("classes")
    if raw_classes is None:
        raw_classes = request.get("class_names")
    if not isinstance(raw_classes, (list, tuple)):
        raise HTTPException(status_code=400, detail="classes 必须是类别名称数组")
    try:
        classes = normalize_class_names(raw_classes)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    previous_classes, _ = get_dataset_annotation_classes(version)
    if len(classes) < len(previous_classes):
        raise HTTPException(status_code=409, detail="训练类别编号必须保持稳定；请使用停用类别并删除其标注")
    prior_inactive = get_inactive_annotation_class_ids(version)
    if prior_inactive and classes[:len(previous_classes)] != previous_classes:
        raise HTTPException(status_code=409, detail="已停用类别的数据集必须保持原训练类别编号和名称")
    raw_retired = request.get("retire_class_ids") or []
    if not isinstance(raw_retired, list) or any(
        not isinstance(item, int) or isinstance(item, bool) or item < 0 or item >= len(previous_classes)
        for item in raw_retired
    ):
        raise HTTPException(status_code=400, detail="停用类别编号无效")
    retired_ids = sorted(set(raw_retired) - set(prior_inactive))
    inactive_ids = sorted(set(prior_inactive) | set(retired_ids))
    if retired_ids:
        if request.get("expected_classes") != previous_classes or request.get("expected_inactive_class_ids") != prior_inactive:
            raise HTTPException(status_code=409, detail="类别列表已变化，请刷新后重试删除")
        if classes[:len(previous_classes)] != previous_classes:
            raise HTTPException(status_code=400, detail="停用类别时不能同时修改训练英文名称或编号")
        if len(inactive_ids) >= len(previous_classes):
            raise HTTPException(status_code=400, detail="至少保留一个可用类别")
        active_task = db.execute(text("""
            SELECT EXISTS (
              SELECT 1 FROM job_tasks
              WHERE type IN ('annotation', 'dataset_export', 'dataset_augmentation')
                AND (input_json::jsonb->>'dataset_id' = :dataset_id
                     OR input_json::jsonb->>'original_dataset_id' = :dataset_id)
                AND status IN ('pending', 'queued', 'waiting', 'running', 'cancel_requested', 'cancelling')
            )
        """), {"dataset_id": dataset_id}).scalar()
        if active_task:
            raise HTTPException(status_code=409, detail="该数据集仍有进行中的标注、导出或增强任务，请等待任务结束后删除类别")
        drafts = db.query(AnnotationDraft).filter(AnnotationDraft.dataset_id == dataset_id).with_for_update().all()
        drafts = [draft for draft in drafts if not isolated_draft(draft)]
        for draft in drafts:
            draft_classes = ordered_class_names(draft.class_names_json)
            if draft_classes and draft_classes[:len(previous_classes)] != previous_classes:
                raise HTTPException(status_code=409, detail="已有草稿使用不同的类别编号，请先处理该草稿")
        version_ids = [version.id]
        for draft in drafts:
            if draft.temporary_dataset_id:
                version_ids.extend(row[0] for row in db.query(DatasetVersion.id).filter(
                    DatasetVersion.dataset_id == draft.temporary_dataset_id,
                    DatasetVersion.dataset_role == "original",
                ).all())
        if len(version_ids) > 1:
            for temporary_version in db.query(DatasetVersion).filter(
                DatasetVersion.id.in_(version_ids[1:]),
            ).all():
                temporary_metadata = dict(temporary_version.metadata_json or {})
                temporary_metadata[ANNOTATION_CLASSES_METADATA_KEY] = classes
                temporary_metadata[ANNOTATION_INACTIVE_CLASS_IDS_METADATA_KEY] = inactive_ids
                temporary_version.metadata_json = temporary_metadata
        deleted_count = 0
        changed_assets_by_version = {}
        for asset in db.query(DatasetAsset).filter(DatasetAsset.dataset_version_id.in_(version_ids)).yield_per(200):
            original = asset.annotations_json or []
            filtered = without_inactive_annotations(original, retired_ids, previous_classes)
            if len(filtered) != len(original):
                asset.annotations_json = filtered
                deleted_count += len(original) - len(filtered)
                changed_assets_by_version.setdefault(asset.dataset_version_id, {})[asset.relative_path] = filtered
        draft_ids = [draft.id for draft in drafts]
        if draft_ids:
            for asset in db.query(AnnotationDraftAsset).filter(AnnotationDraftAsset.draft_id.in_(draft_ids)).yield_per(200):
                original = asset.annotations_json or []
                filtered = without_inactive_annotations(original, retired_ids, previous_classes)
                if len(filtered) != len(original):
                    asset.annotations_json = filtered
                    deleted_count += len(original) - len(filtered)
            for draft in drafts:
                draft.revision = int(draft.revision or 0) + 1
        db.flush()
    source = str(request.get("source") or "user").strip().lower()
    if source not in {"user", "trained_model", "default_new"}:
        source = "user"
    metadata = dict(version.metadata_json or {})
    metadata[ANNOTATION_CLASSES_METADATA_KEY] = classes
    metadata[ANNOTATION_INACTIVE_CLASS_IDS_METADATA_KEY] = inactive_ids
    if retired_ids:
        metadata["annotation_retirement_last"] = {
            "class_ids": retired_ids, "deleted_annotation_count": deleted_count,
            "at": datetime.now().isoformat(),
        }
    if "labels_zh" in request:
        raw_labels = request["labels_zh"]
        if not isinstance(raw_labels, dict):
            raise HTTPException(status_code=400, detail="labels_zh 必须是英文类别到中文名称的映射")
        if any(not isinstance(key, str) or not isinstance(value, str) or len(value.strip()) > 80
               for key, value in raw_labels.items()):
            raise HTTPException(status_code=400, detail="中文名称格式无效或超过 80 字")
        metadata[ANNOTATION_CLASS_LABELS_METADATA_KEY] = {
            name: raw_labels[name].strip() for name in classes
            if name in raw_labels and raw_labels[name].strip()
        }
    else:
        stored_labels = metadata.get(ANNOTATION_CLASS_LABELS_METADATA_KEY)
        if not isinstance(stored_labels, dict):
            stored_labels = {}
        metadata[ANNOTATION_CLASS_LABELS_METADATA_KEY] = {
            name: label for name, label in stored_labels.items()
            if name in classes and isinstance(label, str) and label.strip()
        }
    if ANNOTATION_CLASS_BINDINGS_METADATA_KEY in request:
        raw_bindings = request.get(ANNOTATION_CLASS_BINDINGS_METADATA_KEY)
    else:
        raw_bindings = metadata.get(ANNOTATION_CLASS_BINDINGS_METADATA_KEY)
    metadata[ANNOTATION_CLASS_BINDINGS_METADATA_KEY] = normalize_annotation_class_bindings(raw_bindings, classes)
    metadata["annotation_classes_source"] = source
    metadata["annotation_classes_updated_at"] = datetime.now().isoformat()
    version.metadata_json = metadata
    manifest_replacements = []
    try:
        if retired_ids:
            from dataset_versioning import manifest_content_hash
            for changed_version_id, changed_assets in changed_assets_by_version.items():
                changed_version = db.query(DatasetVersion).filter(
                    DatasetVersion.id == changed_version_id,
                ).with_for_update().one()
                manifest_path = Path(changed_version.manifest_path)
                previous_manifest = manifest_path.read_bytes()
                manifest = json.loads(previous_manifest)
                found_paths = set()
                for manifest_asset in manifest.get("assets") or []:
                    relative_path = manifest_asset.get("relative_path")
                    if relative_path in changed_assets:
                        manifest_asset["annotations"] = changed_assets[relative_path]
                        found_paths.add(relative_path)
                if found_paths != set(changed_assets):
                    raise HTTPException(status_code=409, detail="数据集清单与资产不一致，请先修复清单再删除类别")
                changed_version.content_hash = manifest_content_hash(manifest)
                manifest["content_hash"] = changed_version.content_hash
                db.flush()
                temporary_manifest = manifest_path.with_name(f"{manifest_path.name}.{uuid.uuid4().hex}.tmp")
                try:
                    temporary_manifest.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
                    os.replace(temporary_manifest, manifest_path)
                finally:
                    temporary_manifest.unlink(missing_ok=True)
                manifest_replacements.append((manifest_path, previous_manifest))
        db.commit()
    except Exception:
        db.rollback()
        for manifest_path, previous_manifest in reversed(manifest_replacements):
            manifest_path.write_bytes(previous_manifest)
        raise
    db.refresh(version)
    return {
        "dataset_id": dataset_id,
        "version_id": version.id,
        "classes": classes,
        "annotation_classes": classes,
        "inactive_class_ids": inactive_ids,
        "deleted_annotation_count": deleted_count if retired_ids else 0,
        "labels_zh": metadata[ANNOTATION_CLASS_LABELS_METADATA_KEY],
        "bindings": metadata[ANNOTATION_CLASS_BINDINGS_METADATA_KEY],
        "annotation_class_bindings": metadata[ANNOTATION_CLASS_BINDINGS_METADATA_KEY],
        "source": source,
    }


def stream_file_range(path: Path, start: int, end: int, chunk_size: int = 1024 * 1024):
    with open(path, "rb") as file_handle:
        file_handle.seek(start)
        remaining = end - start + 1
        while remaining > 0:
            chunk = file_handle.read(min(chunk_size, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            yield chunk


@app.get("/datasets/{dataset_id}/assets/{asset_id}/content")
@app.get("/api/datasets/{dataset_id}/assets/{asset_id}/content")
async def get_dataset_asset_content(dataset_id: str, asset_id: int, request: Request, db=Depends(get_db)):
    asset = db.query(DatasetAsset).filter(DatasetAsset.id == asset_id).first()
    version = db.query(DatasetVersion).filter(DatasetVersion.id == asset.dataset_version_id).first() if asset else None
    if not asset or not version or version.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail="图片资产不存在")
    require_dataset_read(db, get_request_current_user(request), dataset_id, version.dataset_role)
    image_file = resolve_relative_file_under_root(Path(version.root_path), asset.relative_path)
    etag = f'"{asset.content_hash}"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag, "Cache-Control": "private, max-age=86400, immutable"})
    size = image_file.stat().st_size
    start, end, status_code = 0, max(0, size - 1), 200
    range_header = request.headers.get("range", "")
    if range_header.startswith("bytes="):
        try:
            start_text, end_text = range_header[6:].split("-", 1)
            start = int(start_text) if start_text else 0
            end = int(end_text) if end_text else end
            if start < 0 or end < start or start >= size:
                raise ValueError
            end = min(end, size - 1)
            status_code = 206
        except ValueError:
            raise HTTPException(status_code=416, detail="无效 Range")
    headers = {
        "Accept-Ranges": "bytes",
        "Content-Length": str(end - start + 1),
        "Cache-Control": "private, max-age=86400, immutable",
        "ETag": etag,
    }
    if status_code == 206:
        headers["Content-Range"] = f"bytes {start}-{end}/{size}"
    return StreamingResponse(
        stream_file_range(image_file, start, end),
        status_code=status_code,
        media_type=mimetypes.guess_type(image_file.name)[0] or "application/octet-stream",
        headers=headers,
    )


@app.delete("/datasets/{dataset_id}/assets/{asset_id}")
@app.delete("/api/datasets/{dataset_id}/assets/{asset_id}")
async def delete_dataset_asset(dataset_id: str, asset_id: int, request: Request, db=Depends(get_db)):
    """删除原始数据集中的单个图片资产及其关联文件。"""
    asset = db.query(DatasetAsset).filter(DatasetAsset.id == asset_id).first()
    version = db.query(DatasetVersion).filter(DatasetVersion.id == asset.dataset_version_id).first() if asset else None
    if not asset or not version or version.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail="图片资产不存在")
    if version.dataset_role != "original":
        raise HTTPException(status_code=409, detail="仅支持删除原始数据集图片")
    latest_version = (
        db.query(DatasetVersion)
        .filter(
            DatasetVersion.dataset_id == dataset_id,
            DatasetVersion.dataset_role == "original",
            DatasetVersion.status == "ready",
        )
        .order_by(DatasetVersion.created_at.desc(), DatasetVersion.id.desc())
        .first()
    )
    if not latest_version or latest_version.id != version.id:
        raise HTTPException(status_code=409, detail="图片资产不是当前原始数据集版本")

    require_dataset_manage(db, get_request_current_user(request), dataset_id, "original")
    dataset_root = Path(version.root_path).resolve()
    image_file = resolve_relative_file_under_root(dataset_root, asset.relative_path)
    related_files = [image_file]
    if asset.label_path:
        label_file = (dataset_root / str(asset.label_path).replace("\\", "/")).resolve()
        if _safe_relative_to(label_file, dataset_root) and label_file.is_file():
            related_files.append(label_file)

    staged_files = []
    image_name = asset.basename
    image_size = int(asset.size or 0)
    try:
        for target in related_files:
            staged = stage_directory_for_deletion(target)
            if staged is not None:
                staged_files.append((staged, target))

        db.delete(asset)
        db.flush()
        remaining_image_count, remaining_label_count, remaining_size = db.query(
            func.count(DatasetAsset.id),
            func.count(DatasetAsset.label_path),
            func.coalesce(func.sum(DatasetAsset.size), 0),
        ).filter(DatasetAsset.dataset_version_id == version.id).one()
        db.commit()
    except Exception:
        db.rollback()
        for staged, target in reversed(staged_files):
            try:
                restore_staged_directory(staged, target)
            except OSError:
                logger.exception("图片删除失败后的文件恢复失败: %s", target)
        raise

    for staged, _target in staged_files:
        try:
            purge_staged_directory(staged)
        except OSError:
            logger.exception("图片删除后的隔离文件清理失败: %s", staged)

    thumbnail_dir = DATA_DIR / "thumbnails" / version.id
    for thumbnail in thumbnail_dir.glob(f"{asset_id}-*.jpg"):
        try:
            thumbnail.unlink()
        except OSError:
            logger.warning("图片缩略图缓存清理失败: %s", thumbnail)

    metadata = read_dataset_metadata(dataset_root)
    if metadata is not None:
        metadata.update({
            "image_count": int(remaining_image_count or 0),
            "label_count": int(remaining_label_count or 0),
            "size": int(remaining_size or 0),
        })
        write_dataset_metadata(dataset_root, metadata)

    return {
        "message": "图片已删除",
        "dataset_id": dataset_id,
        "asset_id": asset_id,
        "filename": image_name,
        "deleted_size": image_size,
        "remaining_image_count": int(remaining_image_count or 0),
    }


def copy_raw_image_without_exif(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with Image.open(source) as image:
        output_format = str(image.format or "").upper()
        if output_format not in {"JPEG", "PNG", "WEBP", "BMP"}:
            output_format = "JPEG"
        if output_format == "JPEG" and image.mode not in {"RGB", "L"}:
            image = image.convert("RGB")
        save_kwargs = {"quality": 95} if output_format == "JPEG" else {}
        image.save(destination, format=output_format, **save_kwargs)


def create_cached_thumbnail(source: Path, destination: Path, max_size: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.{uuid.uuid4().hex}.tmp")
    with Image.open(source) as image:
        image = image.convert("RGB")
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        image.save(temporary, format="JPEG", quality=82, optimize=True)
    os.replace(temporary, destination)


@app.get("/datasets/{dataset_id}/assets/{asset_id}/thumbnail")
@app.get("/api/datasets/{dataset_id}/assets/{asset_id}/thumbnail")
async def get_dataset_asset_thumbnail(
    dataset_id: str,
    asset_id: int,
    request: Request,
    max_size: int = 320,
    db=Depends(get_db),
):
    max_size = max(64, min(int(max_size), 1024))
    asset = db.query(DatasetAsset).filter(DatasetAsset.id == asset_id).first()
    version = db.query(DatasetVersion).filter(DatasetVersion.id == asset.dataset_version_id).first() if asset else None
    if not asset or not version or version.dataset_id != dataset_id:
        raise HTTPException(status_code=404, detail="图片资产不存在")
    require_dataset_read(db, get_request_current_user(request), dataset_id, version.dataset_role)
    etag = f'"{asset.content_hash}-{max_size}"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers={"ETag": etag, "Cache-Control": "private, max-age=86400, immutable"})
    source = resolve_relative_file_under_root(Path(version.root_path), asset.relative_path)
    thumbnail = DATA_DIR / "thumbnails" / version.id / f"{asset.id}-{max_size}.jpg"
    if not thumbnail.is_file():
        await asyncio.to_thread(create_cached_thumbnail, source, thumbnail, max_size)
    return FileResponse(
        thumbnail,
        media_type="image/jpeg",
        headers={
            "Cache-Control": "private, max-age=86400, immutable",
            "ETag": etag,
        },
    )


@app.get("/datasets/{dataset_id}/images/data")
@app.get("/api/datasets/{dataset_id}/images/data")
async def get_dataset_image_data(dataset_id: str, image_path: str, request: Request, db=Depends(get_db)):
    """读取指定数据集图片并返回 base64 data URL。"""
    try:
        require_dataset_read(db, get_request_current_user(request), dataset_id, "original")
        dataset_path = resolve_managed_dataset_root(dataset_id, constants.DATASETS_DIR)
        image_file = resolve_relative_file_under_root(dataset_path, image_path)

        encoded_data = image_processor.encode_image_file_to_base64(str(image_file))
        if not encoded_data:
            raise HTTPException(status_code=422, detail="图片 raw 编码失败")
        data_url, mime_type = encoded_data

        dataset_info = build_dataset_info_from_metadata(dataset_path)
        image_record = build_dataset_image_record(
            dataset_id=dataset_id,
            dataset_name=dataset_info.get("name") or dataset_id,
            dataset_path=dataset_path,
            image_path=image_file,
            include_dimensions=True
        )
        image_record["data"] = data_url
        image_record["mime_type"] = mime_type
        return image_record

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"读取数据集图片失败: dataset={dataset_id}, image={image_path}, error={e}")
        raise HTTPException(status_code=500, detail=f"读取数据集图片失败: {str(e)}")


@app.get("/training-datasets")
@app.get("/api/training-datasets")
def list_training_datasets(
    request: Request,
    page: int = 1,
    page_size: int = 20,
    name: str = "",
    creator: str = "",
    db=Depends(get_db),
):
    """获取训练数据集列表。训练数据集与原始数据集分开存放。"""
    try:
        page = max(1, int(page or 1))
        page_size = max(1, min(int(page_size or 20), 100))
        current_user = get_request_current_user(request)
        total, datasets_list = list_persisted_dataset_infos(
            db, "training", page, page_size, current_user, name=name, creator=creator
        )

        return {
            "total_datasets": total,
            "total": total,
            "page": page,
            "page_size": page_size,
            "datasets": datasets_list,
            "creator_options": list_dataset_creators(db, "training", current_user),
        }

    except Exception as e:
        logger.error(f"获取训练数据集列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取训练数据集列表失败: {str(e)}")


@app.post("/upload/training-dataset")
@app.post("/api/upload/training-dataset")
async def upload_training_dataset(
    request: Request,
    file: UploadFile = File(...),
    owner_role_id: Optional[int] = Form(None),
    db=Depends(get_db),
):
    """上传已经符合 YOLO 格式的训练数据集 ZIP。"""
    original_filename = file.filename or "training_dataset.zip"
    if not original_filename.lower().endswith(".zip"):
        raise HTTPException(status_code=400, detail="只支持ZIP格式的数据集")

    dataset_path = build_unique_upload_dir(
        TRAINING_DATASETS_DIR,
        sanitize_original_stem(original_filename, fallback="training_dataset"),
        fallback="training_dataset"
    )
    dataset_id = dataset_path.name
    uploader_name = get_request_current_username(request)
    current_user = get_request_current_user(request)
    owner_role = select_upload_role(db, current_user, owner_role_id)
    archive_limits = ArchiveLimits.from_env()

    try:
        dataset_path.mkdir(parents=True, exist_ok=True)
        zip_path = save_upload_file_keep_name(
            file,
            dataset_path,
            fallback=sanitize_original_filename(original_filename, "training_dataset.zip"),
            max_bytes=archive_limits.max_upload_bytes,
        )

        safe_extract_zip(zip_path, dataset_path, archive_limits)
        file_manager.cleanup_file(str(zip_path))

        flatten_single_yolo_root(dataset_path)

        if not training_dataset_manager.validate_structure(dataset_path):
            raise HTTPException(status_code=400, detail="训练数据集必须满足 YOLO 格式：train/images、train/labels、val/images、val/labels")

        ensure_training_dataset_yaml(dataset_path)
        training_dataset_manager.fix_yaml_config(dataset_path)
        stats = training_dataset_manager.analyze_dataset(dataset_path)
        now = datetime.now().isoformat()
        dataset_info = {
            "id": dataset_id,
            "name": Path(sanitize_original_filename(original_filename, "training_dataset.zip")).stem,
            "original_filename": original_filename,
            "path": str(dataset_path),
            "created_at": now,
            "updated_at": now,
            "status": constants.DATASET_STATUS_READY if stats.get("image_count", 0) > 0 else constants.DATASET_STATUS_EMPTY,
            "dataset_role": "training",
            "uploader": uploader_name,
            "upload_user": uploader_name,
            "created_by": uploader_name,
            **stats
        }
        write_training_dataset_metadata(dataset_path, dataset_info)
        sync_dataset_metadata_to_database(
            db,
            dataset_path,
            dataset_info,
            dataset_role="training",
            source_type="training_upload",
        )
        ensure_dataset_resource(db, dataset_id, "training", current_user, owner_role)

        return {
            "dataset_id": dataset_id,
            "dataset_path": str(dataset_path),
            "message": "训练数据集上传成功",
            "dataset_info": dataset_info
        }

    except ArchiveSafetyError as exc:
        file_manager.cleanup_dir(str(dataset_path))
        raise HTTPException(status_code=413, detail=str(exc))
    except (zipfile.BadZipFile, RuntimeError) as exc:
        file_manager.cleanup_dir(str(dataset_path))
        raise HTTPException(status_code=400, detail=f"无效的 ZIP 文件: {exc}")
    except HTTPException:
        file_manager.cleanup_dir(str(dataset_path))
        raise
    except Exception as e:
        file_manager.cleanup_dir(str(dataset_path))
        logger.error(f"训练数据集上传失败: {e}")
        raise HTTPException(status_code=500, detail=f"训练数据集上传失败: {str(e)}")


@app.delete("/training-datasets/{dataset_id}")
@app.delete("/api/training-datasets/{dataset_id}")
async def delete_training_dataset(dataset_id: str, request: Request, db=Depends(get_db)):
    """删除单个训练数据集。"""
    dataset_path = resolve_managed_dataset_root(dataset_id, TRAINING_DATASETS_DIR)
    try:
        require_dataset_manage(db, get_request_current_user(request), dataset_id, "training")
        deleted = delete_persisted_dataset(db, dataset_id, "training", dataset_path)
        return {
            "message": f"训练数据集 {dataset_id} 已删除",
            "deleted_dataset_id": dataset_id,
            **deleted,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除训练数据集失败: {str(e)}")


@app.get("/datasets/{dataset_id}/versions")
@app.get("/api/datasets/{dataset_id}/versions")
async def list_dataset_versions(dataset_id: str, request: Request, db=Depends(get_db)):
    current_user = get_request_current_user(request)
    versions = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.dataset_id == dataset_id)
        .order_by(DatasetVersion.created_at.desc())
        .all()
    )
    readable_versions = []
    for item in versions:
        resource_role = "original" if item.dataset_role == "annotation" else item.dataset_role
        try:
            require_dataset_read(db, current_user, item.dataset_id, resource_role)
            readable_versions.append(item)
        except HTTPException:
            continue
    return {"dataset_id": dataset_id, "versions": [item.to_dict() for item in readable_versions]}


@app.post("/datasets/{dataset_id}/versions/index")
@app.post("/api/datasets/{dataset_id}/versions/index")
async def index_dataset_version(
    dataset_id: str,
    dataset_role: str = "original",
    admin_user: Dict[str, Any] = Depends(require_admin),
    db=Depends(get_db),
):
    if dataset_role not in {"original", "training"}:
        raise HTTPException(status_code=400, detail="dataset_role 只能是 original 或 training")
    root = (
        resolve_managed_dataset_root(dataset_id, constants.DATASETS_DIR)
        if dataset_role == "original"
        else resolve_managed_dataset_root(dataset_id, TRAINING_DATASETS_DIR)
    )
    version, report, created = create_initial_version(
        db,
        dataset_id=dataset_id,
        root_path=root,
        dataset_role=dataset_role,
        manifest_root=DATA_DIR / "manifests",
        created_by=str(admin_user.get("username") or "admin"),
    )
    dataset_info = (
        build_dataset_info_from_metadata(root)
        if dataset_role == "original"
        else build_training_dataset_info_from_metadata(root)
    )
    persist_dataset_business_metadata(
        db,
        version,
        root,
        dataset_info,
        source_type="manual_index",
    )
    owner_role = select_upload_role(db, admin_user)
    ensure_dataset_resource(db, dataset_id, dataset_role, admin_user, owner_role)
    return {
        "created": created,
        "version": version.to_dict(),
        "validation": {
            "valid": report["valid"],
            "errors": report["errors"],
            "warnings": report["warnings"],
            "statistics": report["statistics"],
        },
    }


@app.get("/dataset-versions/{version_id}/validation")
@app.get("/api/dataset-versions/{version_id}/validation")
async def get_dataset_version_validation(version_id: str, request: Request, db=Depends(get_db)):
    version = db.query(DatasetVersion).filter(DatasetVersion.id == version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="数据集版本不存在")
    resource_role = "original" if version.dataset_role == "annotation" else version.dataset_role
    require_dataset_read(db, get_request_current_user(request), version.dataset_id, resource_role)
    manifest_path = Path(version.manifest_path).resolve()
    manifest_root = (DATA_DIR / "manifests").resolve()
    if not _safe_relative_to(manifest_path, manifest_root) or not manifest_path.is_file():
        raise HTTPException(status_code=500, detail="数据集版本 manifest 缺失或越界")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {
        "version_id": version.id,
        "dataset_id": version.dataset_id,
        "content_hash": version.content_hash,
        "validation": manifest.get("validation", {}),
        "statistics": manifest.get("statistics", {}),
    }


@app.get("/dataset-versions/{version_id}/assets")
@app.get("/api/dataset-versions/{version_id}/assets")
async def list_dataset_version_assets(
    version_id: str,
    request: Request,
    page: int = 1,
    page_size: int = 100,
    db=Depends(get_db),
):
    version = db.query(DatasetVersion).filter(DatasetVersion.id == version_id).first()
    if not version:
        raise HTTPException(status_code=404, detail="数据集版本不存在")
    resource_role = "original" if version.dataset_role == "annotation" else version.dataset_role
    require_dataset_read(db, get_request_current_user(request), version.dataset_id, resource_role)
    page = max(1, int(page))
    page_size = max(1, min(int(page_size), 500))
    query = db.query(DatasetAsset).filter(DatasetAsset.dataset_version_id == version_id)
    total = query.count()
    has_annotations = query.filter(func.json_array_length(DatasetAsset.annotations_json) > 0).first() is not None
    assets = (
        query.order_by(DatasetAsset.relative_path.asc())
        .offset((page - 1) * page_size)
        .limit(page_size)
        .all()
    )
    return {
        "version_id": version_id,
        "page": page,
        "page_size": page_size,
        "total": total,
        "has_annotations": has_annotations,
        "assets": [item.to_dict() for item in assets],
    }


@app.post("/datasets/{dataset_id}/annotation-versions")
@app.post("/api/datasets/{dataset_id}/annotation-versions")
async def save_dataset_annotation_version(
    dataset_id: str,
    request: Dict[str, Any],
    http_request: Request,
    db=Depends(get_db),
):
    current_user = get_request_current_user(http_request)
    require_dataset_manage(db, current_user, dataset_id, "original")
    class_names = request.get("classes") or request.get("class_names") or []
    images = request.get("images") or []
    if not isinstance(images, list) or not images:
        raise HTTPException(status_code=400, detail="没有可保存的标注资产")
    if len(images) > 100_000:
        raise HTTPException(status_code=413, detail="单次标注版本资产数超过 100000")

    source_version_id = str(request.get("source_version_id") or "").strip()
    source_version = None
    if source_version_id:
        source_version = db.query(DatasetVersion).filter(
            DatasetVersion.id == source_version_id,
            DatasetVersion.dataset_id == dataset_id,
        ).first()
    else:
        source_version = (
            db.query(DatasetVersion)
            .filter(DatasetVersion.dataset_id == dataset_id, DatasetVersion.dataset_role == "original")
            .order_by(DatasetVersion.created_at.desc())
            .first()
        )
    if not source_version:
        source_root = resolve_managed_dataset_root(dataset_id, constants.DATASETS_DIR)
        source_version, _, _ = create_initial_version(
            db,
            dataset_id=dataset_id,
            root_path=source_root,
            dataset_role="original",
            manifest_root=DATA_DIR / "manifests",
            created_by=get_request_current_username(http_request),
        )

    source_root = Path(source_version.root_path).resolve()
    updates: Dict[str, Dict[str, Any]] = {}
    try:
        for item in images:
            if not isinstance(item, dict):
                raise ValueError("images 中包含非对象条目")
            raw_relative = str(item.get("relative_path") or "").strip()
            if not raw_relative:
                raise ValueError("每个标注资产必须包含 relative_path")
            source_image = resolve_relative_file_under_root(source_root, raw_relative)
            relative_path = str(source_image.relative_to(source_root)).replace("\\", "/")
            width = int(item.get("width") or 0)
            height = int(item.get("height") or 0)
            if not width or not height:
                width, height = get_uploaded_image_size(source_image)
            annotations = []
            for annotation in item.get("annotations") or []:
                line = normalize_yolo_annotation_line(annotation, width, height)
                if not line:
                    raise ValueError(f"{relative_path} 包含无法转换的标注")
                annotations.append({"yolo_line": line})
            source_group = str(item.get("source_group") or "").strip() or None
            if not source_group:
                match = re.match(r"(.+)_frame_\d+$", Path(relative_path).stem)
                source_group = match.group(1) if match else None
            updates[relative_path] = {
                "width": width,
                "height": height,
                "source_group": source_group,
                "annotations": annotations,
            }
        version, created = create_annotation_version(
            db,
            source_version=source_version,
            updates=updates,
            class_names=class_names,
            manifest_root=DATA_DIR / "manifests",
            created_by=get_request_current_username(http_request),
        )
        return {"created": created, "version": version.to_dict()}
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/dataset-versions/{version_id}/derive-classes")
@app.post("/api/dataset-versions/{version_id}/derive-classes")
async def derive_dataset_version_classes(
    version_id: str,
    request: Dict[str, Any],
    http_request: Request,
    db=Depends(get_db),
):
    source_version = db.query(DatasetVersion).filter(DatasetVersion.id == version_id).first()
    if not source_version:
        raise HTTPException(status_code=404, detail="数据集版本不存在")
    selected = request.get("class_names") or request.get("classes") or []
    current_user = get_request_current_user(http_request)
    resource_role = "original" if source_version.dataset_role == "annotation" else source_version.dataset_role
    try:
        if bool(request.get("dry_run", False)):
            require_dataset_read(db, current_user, source_version.dataset_id, resource_role)
        else:
            require_dataset_manage(db, current_user, source_version.dataset_id, resource_role)
        plan = class_subset_plan(db, source_version, selected)
        if bool(request.get("dry_run", False)):
            return {"dry_run": True, "plan": plan}
        version, plan, created = derive_class_subset(
            db,
            source_version=source_version,
            selected_names=selected,
            derived_root=DATA_DIR / "derived_datasets",
            staging_root=DATA_DIR / "staging",
            manifest_root=DATA_DIR / "manifests",
            created_by=get_request_current_username(http_request),
        )
        return {"created": created, "version": version.to_dict(), "plan": plan}
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc))


def annotation_task_matches_source_version(annotation_task: JobTask, source_version: DatasetVersion) -> bool:
    task_dataset_id = str((annotation_task.input_json or {}).get("dataset_id") or "").strip()
    if task_dataset_id == source_version.dataset_id:
        return True
    metadata = dict(source_version.metadata_json or {})
    return bool(
        metadata.get("source_type") == "annotation_temporary"
        and str(metadata.get("source_dataset_id") or "").strip() == task_dataset_id
    )


def annotation_export_source_view(db, source_version, input_data):
    draft_id = str(input_data.get("annotation_draft_id") or "").strip()
    if not draft_id:
        return source_version
    try:
        owner_id = int(input_data.get("owner_id"))
    except (TypeError, ValueError) as exc:
        raise ValueError("标注草稿缺少有效归属用户") from exc
    draft = get_draft_by_id(db, owner_id, draft_id)
    if draft is None:
        raise ValueError("标注草稿不存在或不属于当前用户")
    original = _draft_current_version_or_conflict(db, draft.dataset_id, draft)
    working = _draft_working_version(db, draft.dataset_id, draft, original)
    if str(source_version.id) != str(working.id):
        raise ValueError("标注草稿与导出数据集不匹配")
    if isolated_draft(draft):
        requested = input_data.get("classes") or input_data.get("class_names")
        if requested and list(requested) != list(draft.class_names_json or []):
            raise ValueError("标注会话类别已变化，请先保存草稿再导出")
        return working
    return source_version


def build_annotation_export_updates(
    db,
    source_version: DatasetVersion,
    input_data: Dict[str, Any],
) -> Tuple[Dict[str, Dict[str, Any]], List[str]]:
    """Build complete server-side annotation updates from a persisted AI job plus UI overrides."""
    source_version = annotation_export_source_view(db, source_version, input_data)
    source_assets = (
        db.query(DatasetAsset)
        .filter(DatasetAsset.dataset_version_id == source_version.id)
        .order_by(DatasetAsset.relative_path.asc())
        .all()
    )
    assets_by_path = {asset.relative_path: asset for asset in source_assets}
    known_paths = list(assets_by_path)
    source_root = Path(source_version.root_path).resolve()
    inactive_ids = get_inactive_annotation_class_ids(source_version)
    stable_classes, _ = get_dataset_annotation_classes(source_version)
    absolute_paths, basenames = build_asset_path_indexes(source_root, known_paths)
    class_names = normalize_class_names(input_data.get("classes") or input_data.get("class_names") or [])
    if inactive_ids:
        if class_names and class_names != stable_classes:
            raise ValueError("训练类别编号和名称必须与停用前保持一致")
        class_names = list(stable_classes)
    class_mode = str(input_data.get("class_mode") or "auto").strip().lower()
    annotation_mode = str(input_data.get("annotation_mode") or "replace").strip().lower()
    target_class = input_data.get("target_class")
    clear_all_annotations = bool(input_data.get("clear_all_annotations", False))
    updates: Dict[str, Dict[str, Any]] = {}
    if isolated_version(source_version):
        updates = {asset.relative_path: {
            "width": int(asset.width or 0), "height": int(asset.height or 0),
            "source_group": asset.source_group, "annotations": without_inactive_annotations(
                session_base_annotations(source_version, asset.annotations_json), inactive_ids, stable_classes)
        } for asset in source_assets}
    annotation_draft = None

    def raw_asset_dimensions(asset: DatasetAsset) -> Tuple[int, int]:
        source_image = (source_root / asset.relative_path).resolve()
        try:
            source_image.relative_to(source_root)
            width, height = get_uploaded_image_size(source_image)
            if width > 0 and height > 0:
                return int(width), int(height)
        except Exception:
            pass
        return int(asset.width or 0), int(asset.height or 0)

    if clear_all_annotations:
        for asset in source_assets:
            width, height = raw_asset_dimensions(asset)
            updates[asset.relative_path] = {
                "width": width,
                "height": height,
                "source_group": asset.source_group,
                "annotations": [],
            }

    annotation_draft_id = str(input_data.get("annotation_draft_id") or "").strip()
    if annotation_draft_id:
        try:
            owner_id = int(input_data.get("owner_id"))
        except (TypeError, ValueError):
            raise ValueError("标注草稿缺少有效归属用户")
        annotation_draft = get_draft_by_id(db, owner_id, annotation_draft_id)
        if annotation_draft is None:
            raise ValueError("标注草稿不存在或不属于当前用户")
        source_metadata = dict(source_version.metadata_json or {})
        source_is_draft_temporary = bool(
            str(annotation_draft.temporary_dataset_id or "") == str(source_version.dataset_id)
            and source_metadata.get("source_type") == "annotation_temporary"
            and str(source_metadata.get("source_dataset_id") or "") == str(annotation_draft.dataset_id)
        )
        if str(annotation_draft.dataset_id) != str(source_version.dataset_id) and not source_is_draft_temporary:
            raise ValueError("标注草稿与导出数据集不匹配")
        current_original_version = _get_latest_original_dataset_version(db, annotation_draft.dataset_id)
        if str(annotation_draft.source_version_id or "") != str(current_original_version.id):
            raise ValueError("标注草稿对应的数据集版本已变化")
        _validate_annotation_model_selection(
            db,
            {
                "id": owner_id,
                "username": input_data.get("owner_username"),
                "role": input_data.get("owner_role"),
                "roleKeys": input_data.get("owner_roleKeys") or [],
            },
            annotation_draft.model_path,
            annotation_draft.annotation_kind,
        )
        if not class_names:
            class_names = normalize_class_names(annotation_draft.class_names_json or [])
        draft_assets = (
            db.query(AnnotationDraftAsset)
            .filter(AnnotationDraftAsset.draft_id == annotation_draft.id)
            .all()
        )
        for draft_asset in draft_assets:
            asset = assets_by_path.get(draft_asset.relative_path)
            if asset is None:
                raise ValueError(f"标注草稿包含未知资产: {draft_asset.relative_path}")
            width = int(draft_asset.width or asset.width or 0)
            height = int(draft_asset.height or asset.height or 0)
            updates[draft_asset.relative_path] = {
                "width": width,
                "height": height,
                "source_group": asset.source_group,
                "annotations": [
                    frontend_annotation_to_manifest(
                        annotation,
                        image_width=width,
                        image_height=height,
                    )
                    for annotation in without_inactive_annotations(
                        draft_asset.annotations_json or [], inactive_ids, stable_classes,
                    )
                ],
            }

    annotation_task_id = str(input_data.get("annotation_task_id") or "").strip()
    if annotation_task_id:
        annotation_task = db.query(JobTask).filter(
            JobTask.id == annotation_task_id,
            JobTask.type == "annotation",
        ).first()
        if not annotation_task:
            raise ValueError("一键 AI 标注任务不存在")
        if annotation_task.status != "completed":
            raise ValueError("一键 AI 标注任务尚未完成")
        if isolated_version(source_version) and str((annotation_task.input_json or {}).get("annotation_draft_id") or "") != annotation_draft_id:
            raise ValueError("AI 标注任务不属于当前标注会话")
        if not annotation_task_matches_source_version(annotation_task, source_version):
            raise ValueError("一键 AI 标注任务与当前数据集不匹配")

        events = (
            db.query(JobEvent)
            .filter(JobEvent.task_id == annotation_task_id, JobEvent.event_type == "annotation_result")
            .order_by(JobEvent.sequence.asc())
            .all()
        )
        for event in events:
            item = ((event.payload_json or {}).get("item") or {})
            relative_path = resolve_result_relative_path(
                item,
                source_root,
                known_paths,
                absolute_paths,
                basenames,
            )
            asset = assets_by_path[relative_path]
            width, height = raw_asset_dimensions(asset)
            generated = []
            for annotation in without_inactive_annotations(
                item.get("annotations"), inactive_ids, stable_classes,
            ):
                class_id = resolve_ai_class_id(
                    annotation,
                    class_names,
                    class_mode=class_mode,
                    target_class=target_class,
                    inactive_ids=inactive_ids,
                )
                generated.append(frontend_annotation_to_manifest(
                    annotation,
                    class_id=class_id,
                    image_width=width,
                    image_height=height,
                ))
            annotations = generated
            if annotation_mode == "append" and not clear_all_annotations:
                annotations = [*without_inactive_annotations(
                    session_base_annotations(source_version, asset.annotations_json), inactive_ids, stable_classes,
                ), *generated]
            updates[relative_path] = {
                "width": width,
                "height": height,
                "source_group": asset.source_group,
                "annotations": annotations,
            }

    # Loaded images are small, explicit overrides. They preserve any manual edits
    # made after the AI task without forcing every asset into the browser DOM.
    for item in input_data.get("images") or []:
        if not isinstance(item, dict):
            raise ValueError("images 中包含非对象条目")
        relative_path = str(item.get("relative_path") or "").replace("\\", "/").strip().lstrip("./")
        asset = assets_by_path.get(relative_path)
        if not asset:
            raise ValueError(f"标注包含未知资产: {relative_path or 'unknown'}")
        width, height = raw_asset_dimensions(asset)
        annotations = [
            frontend_annotation_to_manifest(annotation, image_width=width, image_height=height)
            for annotation in without_inactive_annotations(
                item.get("annotations"), inactive_ids, stable_classes,
            )
        ]
        updates[relative_path] = {
            "width": width,
            "height": height,
            "source_group": asset.source_group,
            "annotations": annotations,
        }

    return updates, class_names


def run_annotation_export_task(task_id: str) -> None:
    """Create the annotation version and training dataset outside the ASGI event loop."""
    db = SessionLocal()
    try:
        task = db.query(JobTask).filter(JobTask.id == task_id, JobTask.type == "dataset_export").first()
        if not task:
            return
        input_data = dict(task.input_json or {})
        update_job(
            db,
            task_id,
            status="running",
            stage="collecting_annotations",
            progress=5,
            message="正在汇总标注结果",
        )
        source_version = db.query(DatasetVersion).filter(
            DatasetVersion.id == str(input_data.get("source_version_id") or ""),
            DatasetVersion.dataset_role == "original",
        ).first()
        if not source_version:
            raise ValueError("原始数据集版本不存在")

        source_version = annotation_export_source_view(db, source_version, input_data)
        updates, class_names = build_annotation_export_updates(db, source_version, input_data)
        try:
            export_profile = resolve_tiling_profile(
                str(input_data.get("profile") or input_data.get("tiling_profile") or TILING_POLICY_VERSION)
            )
        except ValueError as exc:
            raise ValueError(str(exc))
        source_assets = (
            db.query(DatasetAsset)
            .filter(DatasetAsset.dataset_version_id == source_version.id)
            .order_by(DatasetAsset.relative_path.asc())
            .all()
        )
        required_geometry = str(input_data.get("required_geometry") or "bbox").strip().lower()
        annotation_summary = summarize_annotation_assets(
            source_assets,
            updates,
            required_geometry=str(input_data.get("required_geometry") or "bbox"),
        )
        quality_updates = {
            path: {
                **update,
                "annotations": filter_annotations_by_geometry(
                    update.get("annotations") or [],
                    required_geometry=required_geometry,
                ),
            }
            for path, update in updates.items()
        }
        quality = summarize_mobilephone_export_quality(
            source_assets,
            class_names,
            annotations_by_path=quality_updates,
            profile=export_profile["version"],
            train_ratio=float(input_data.get("train_ratio", 0.8)),
            test_ratio=float(input_data.get("test_ratio", 0.0)),
            seed=int(input_data.get("seed", 42)),
            required_geometry=required_geometry,
        )
        _add_quality_sample_preview_urls(quality, source_version.dataset_id)
        if quality["errors"]:
            raise ValueError(f"训练数据质量门阻止导出: {quality['errors'][:3]}")
        warning_codes = sorted({str(item.get("code")) for item in quality["warnings"] if item.get("code")})
        supplied_confirmation = input_data.get("quality_confirmation") or {}
        supplied_codes = sorted({
            str(code)
            for code in (supplied_confirmation.get("warning_codes") or [])
            if str(code).strip()
        }) if isinstance(supplied_confirmation, dict) else []
        confirmed = bool(supplied_confirmation.get("confirmed")) if isinstance(supplied_confirmation, dict) else False
        if warning_codes and (not confirmed or supplied_codes != warning_codes):
            raise ValueError("导出质量告警尚未确认或预检结果已过期，请重新预检")
        quality_confirmation = {
            "confirmed": confirmed,
            "warning_codes": warning_codes,
            "thresholds": quality.get("thresholds") or {},
        }
        quality["confirmation"] = quality_confirmation
        annotation_summary["quality"] = quality
        exclude_unlabeled = bool(input_data.get("exclude_unlabeled_images", False))
        update_job(
            db,
            task_id,
            stage="saving_annotation_version",
            progress=25,
            message=(
                f"\u6b63\u5728\u4fdd\u7559 {annotation_summary['annotated_count']} \u5f20\u6709\u6807\u6ce8\u56fe\u7247，"
                f"\u5254\u9664 {annotation_summary['unlabeled_count']} \u5f20\u672a\u6807\u6ce8\u56fe\u7247"
                if exclude_unlabeled
                else f"正在保存 {len(updates)} 张图片的标注变更"
            ),
            result={"annotation_export": annotation_summary},
        )
        annotation_version, annotation_created = create_annotation_version(
            db,
            source_version=source_version,
            updates=updates,
            class_names=class_names,
            manifest_root=DATA_DIR / "manifests",
            created_by=str(input_data.get("owner_username") or "system"),
            required_geometry=str(input_data.get("required_geometry") or "bbox"),
            drop_unannotated=exclude_unlabeled,
        )
        inactive_ids = get_inactive_annotation_class_ids(source_version)
        if inactive_ids:
            annotation_metadata = dict(annotation_version.metadata_json or {})
            annotation_metadata[ANNOTATION_CLASSES_METADATA_KEY] = list(class_names)
            annotation_metadata[ANNOTATION_INACTIVE_CLASS_IDS_METADATA_KEY] = inactive_ids
            annotation_version.metadata_json = annotation_metadata
            db.commit()
            db.refresh(annotation_version)
        session_annotation_model = build_annotation_model_binding(
            input_data.get("annotation_model_path") or input_data.get("model_path"),
            input_data.get("annotation_model_name") or input_data.get("model_name"),
            input_data.get("annotation_geometry") or input_data.get("required_geometry"),
            selected_by=str(input_data.get("owner_username") or "system"),
            selected_at=datetime.utcnow().isoformat() + "Z",
        )
        if session_annotation_model:
            annotation_metadata = dict(annotation_version.metadata_json or {})
            annotation_metadata.update({
                "annotation_model": session_annotation_model,
                "annotation_model_path": session_annotation_model["path"],
                "annotation_model_name": session_annotation_model["name"],
                "annotation_geometry": session_annotation_model["geometry"],
                "annotation_model_selection_mode": "session",
            })
            annotation_version.metadata_json = annotation_metadata
            db.commit()
            db.refresh(annotation_version)

        update_job(
            db,
            task_id,
            stage="exporting_files",
            progress=45,
            message="正在生成训练数据集文件",
            result={"annotation_version_id": annotation_version.id},
        )

        def report_export_progress(processed: int, total: int) -> None:
            progress_db = SessionLocal()
            try:
                percentage = 45 + (max(0, processed) / max(1, total)) * 35
                update_job(
                    progress_db,
                    task_id,
                    stage="exporting_files",
                    progress=round(min(80, percentage), 2),
                    message=f"正在生成训练数据集文件 {processed}/{total}",
                )
            except Exception as progress_exc:
                progress_db.rollback()
                logger.warning("更新数据集导出进度失败: task=%s, error=%s", task_id, progress_exc)
            finally:
                progress_db.close()

        fixed_export_classes = None
        if inactive_ids:
            fixed_export_classes = get_training_model_class_names(
                db, str(input_data.get("annotation_model_path") or input_data.get("model_path") or ""),
            )
            if not fixed_export_classes:
                raise ValueError("旧模型未记录训练类别顺序，不能导出可安全续训的数据集")
        version, summary, created = export_training_version(
            db,
            source_version=annotation_version,
            output_name=str(input_data["name"]),
            training_root=TRAINING_DATASETS_DIR,
            staging_root=DATA_DIR / "staging",
            manifest_root=DATA_DIR / "manifests",
            created_by=str(input_data.get("owner_username") or "system"),
            seed=int(input_data.get("seed", 42)),
            train_ratio=float(input_data.get("train_ratio", 0.8)),
            test_ratio=float(input_data.get("test_ratio", 0.0)),
            required_geometry=str(input_data.get("required_geometry") or "bbox"),
            progress_callback=report_export_progress,
            profile=export_profile["version"],
            quality_confirmation=quality_confirmation,
            fixed_class_names=fixed_export_classes,
        )
        if inactive_ids:
            training_metadata = dict(version.metadata_json or {})
            training_metadata[ANNOTATION_CLASSES_METADATA_KEY] = list(fixed_export_classes)
            training_metadata[ANNOTATION_INACTIVE_CLASS_IDS_METADATA_KEY] = [
                fixed_export_classes.index(class_names[index]) for index in inactive_ids
                if class_names[index] in fixed_export_classes
            ]
            version.metadata_json = training_metadata
            db.commit()
            db.refresh(version)

        update_job(
            db,
            task_id,
            stage="indexing_dataset",
            progress=85,
            message="正在写入训练数据集索引",
            result={"annotation_version_id": annotation_version.id, "dataset_id": version.dataset_id},
        )
        output_path = Path(version.root_path)
        stats = training_dataset_manager.analyze_dataset(output_path)
        now = datetime.now().isoformat()
        source_annotation_model = (
            session_annotation_model
            or get_dataset_annotation_model(db, annotation_version, allow_source_fallback=False)
            or get_dataset_annotation_model(db, source_version)
        )
        dataset_info = {
            "id": version.dataset_id,
            "name": str(input_data["name"]),
            "path": str(output_path),
            "version_id": version.id,
            "parent_version_id": annotation_version.id,
            "created_at": now,
            "updated_at": now,
            "status": constants.DATASET_STATUS_READY,
            "dataset_role": "training",
            "source_dataset_id": str(input_data.get("source_dataset_id") or source_version.dataset_id),
            "source_working_dataset_id": source_version.dataset_id,
            "required_geometry": str(input_data.get("required_geometry") or "bbox"),
            "annotation_export": annotation_summary,
            "image_size": summary.get("image_size"),
            "model_binding_mode": "fixed" if source_annotation_model else "none",
            "annotation_model_selection_mode": "fixed" if source_annotation_model else "none",
            "annotation_model_locked": bool(source_annotation_model),
            "image_size": summary.get("image_size"),
            "optimal_image_size": (summary.get("image_size") or {}).get("optimal_size"),
            "image_size_policy": (summary.get("image_size") or {}).get("policy"),
            "uploader": str(input_data.get("owner_username") or "system"),
            "created_by": str(input_data.get("owner_username") or "system"),
            **({
                "annotation_model": source_annotation_model,
                "annotation_model_path": source_annotation_model.get("path"),
                "annotation_model_name": source_annotation_model.get("name"),
                "annotation_geometry": source_annotation_model.get("geometry"),
            } if source_annotation_model else {}),
            **stats,
        }
        write_training_dataset_metadata(output_path, dataset_info)
        persist_dataset_business_metadata(
            db,
            version,
            output_path,
            dataset_info,
            source_type="annotation_export",
        )
        owner_role = db.query(Role).filter(
            Role.id == int(input_data["owner_role_id"]),
            Role.status == "active",
        ).first()
        if not owner_role:
            raise ValueError("训练数据集归属角色不存在或已停用")
        current_user = {
            "id": input_data.get("owner_id"),
            "username": input_data.get("owner_username"),
        }
        ensure_dataset_resource(db, version.dataset_id, "training", current_user, owner_role)
        db.commit()
        draft_cleanup = {"requested": bool(input_data.get("annotation_draft_id")), "deleted": False}
        annotation_draft_id = str(input_data.get("annotation_draft_id") or "").strip()
        temporary_cleanup = {
            "requested": bool(input_data.get("temporary_dataset_id")),
            "deleted": False,
        }
        temporary_dataset_id = str(input_data.get("temporary_dataset_id") or "").strip()
        if temporary_dataset_id:
            temporary_version = (
                db.query(DatasetVersion)
                .filter(
                    DatasetVersion.dataset_id == temporary_dataset_id,
                    DatasetVersion.dataset_role == "original",
                    DatasetVersion.status == "ready",
                )
                .order_by(DatasetVersion.created_at.desc(), DatasetVersion.id.desc())
                .first()
            )
            temporary_metadata = dict(temporary_version.metadata_json or {}) if temporary_version else {}
            if not temporary_version or temporary_metadata.get("source_type") != "annotation_temporary":
                temporary_cleanup["error"] = "导出源不是受支持的临时增强数据集"
            elif str(temporary_metadata.get("source_dataset_id") or "") != str(input_data.get("source_dataset_id") or ""):
                temporary_cleanup["error"] = "临时增强数据集不属于当前原始数据集"
            else:
                try:
                    temporary_cleanup["deleted"] = _delete_annotation_temporary_dataset(
                        db,
                        temporary_dataset_id,
                        str(input_data.get("source_dataset_id") or ""),
                    )
                except Exception as cleanup_exc:
                    db.rollback()
                    temporary_cleanup["error"] = str(cleanup_exc)
                    logger.exception("清理临时增强数据集失败: dataset=%s", temporary_dataset_id)
        if annotation_draft_id and not temporary_cleanup.get("error"):
            draft = get_draft_by_id(db, int(input_data["owner_id"]), annotation_draft_id)
            draft_cleanup["deleted"] = delete_draft(db, draft)
            db.commit()
        elif annotation_draft_id:
            draft_cleanup["error"] = "临时增强数据集清理失败，草稿已保留以便重试"
        update_job(
            db,
            task_id,
            status="completed",
            stage="completed",
            progress=100,
            message="训练数据集导出完成",
            result={
                "created": created,
                "annotation_created": annotation_created,
                "annotation_version_id": annotation_version.id,
                "dataset_id": version.dataset_id,
                "dataset_info": dataset_info,
                "annotation_export": annotation_summary,
                "summary": summary,
                "temporary_cleanup": temporary_cleanup,
                "annotation_draft_cleanup": draft_cleanup,
            },
        )
    except Exception as exc:
        db.rollback()
        logger.exception("后台导出标注训练数据集失败: task=%s", task_id)
        try:
            task = db.query(JobTask).filter(JobTask.id == task_id).first()
            if task and task.status in ACTIVE_STATUSES:
                update_job(
                    db,
                    task_id,
                    status="failed",
                    stage="failed",
                    message=f"训练数据集导出失败: {str(exc)}",
                    error_code="dataset_export_failed",
                    error_detail=str(exc),
                )
        except Exception:
            db.rollback()
            logger.exception("写入导出失败状态失败: task=%s", task_id)
    finally:
        db.close()


AUGMENTATION_OPERATION_NAMES = {
    "rotate": "旋转",
    "flip_horizontal": "水平镜像",
    "flip_vertical": "垂直镜像",
    "brightness": "亮度",
    "contrast": "对比度",
}


def normalize_augmentation_operations(value: Any) -> List[str]:
    if not isinstance(value, list):
        return []
    result = []
    for item in value:
        operation = str(item or "").strip().lower()
        if operation in AUGMENTATION_OPERATION_NAMES and operation not in result:
            result.append(operation)
    return result


def apply_dataset_augmentation(image: Image.Image, operation: str, variant_index: int) -> Tuple[Image.Image, str]:
    """Apply one bounded deterministic augmentation and return its filename token."""
    if operation == "rotate":
        angle = (90, 180, 270)[variant_index % 3]
        return image.rotate(angle, expand=True), f"rotate{angle}"
    if operation == "flip_horizontal":
        return ImageOps.mirror(image), "flip-h"
    if operation == "flip_vertical":
        return ImageOps.flip(image), "flip-v"
    if operation == "brightness":
        factor = 1.15 if variant_index % 2 == 0 else 0.85
        return ImageEnhance.Brightness(image).enhance(factor), f"brightness-{variant_index + 1}"
    if operation == "contrast":
        factor = 1.2 if variant_index % 2 == 0 else 0.8
        return ImageEnhance.Contrast(image).enhance(factor), f"contrast-{variant_index + 1}"
    raise ValueError(f"不支持的数据增强操作: {operation}")


def save_augmented_image(image: Image.Image, target_path: Path) -> None:
    target_path.parent.mkdir(parents=True, exist_ok=True)
    suffix = target_path.suffix.lower()
    save_image = image
    if suffix in {".jpg", ".jpeg"} and save_image.mode not in {"RGB", "L"}:
        save_image = save_image.convert("RGB")
    save_image.save(target_path)


def run_dataset_augmentation_task(task_id: str) -> None:
    """Generate a new original dataset from a source dataset outside the ASGI loop."""
    db = SessionLocal()
    output_path: Optional[Path] = None
    completed = False
    try:
        task = db.query(JobTask).filter(
            JobTask.id == task_id,
            JobTask.type == "dataset_augmentation",
        ).first()
        if not task:
            return

        input_data = dict(task.input_json or {})
        operations = normalize_augmentation_operations(input_data.get("operations"))
        include_original = bool(input_data.get("include_original", True))
        copies_per_image = max(1, min(int(input_data.get("copies_per_image") or 1), 3))
        source_dataset_id = str(input_data.get("source_dataset_id") or "").strip()
        source_version_id = str(input_data.get("source_version_id") or "").strip()
        output_name = str(input_data.get("name") or "").strip()
        if not source_dataset_id or not source_version_id or not output_name or not operations:
            raise ValueError("数据增强任务参数不完整")

        if is_cancel_requested(task_id):
            update_job(db, task_id, status="cancelled", stage="cancelled", message="数据增强任务已取消")
            return

        update_job(
            db,
            task_id,
            status="running",
            stage="collecting_assets",
            progress=2,
            message="正在读取原始数据集资产",
        )
        source_version = db.query(DatasetVersion).filter(
            DatasetVersion.id == source_version_id,
            DatasetVersion.dataset_id == source_dataset_id,
            DatasetVersion.dataset_role == "original",
            DatasetVersion.status == "ready",
        ).first()
        if not source_version:
            raise ValueError("原始数据集版本不存在或不可用")

        source_root = Path(source_version.root_path).resolve()
        assets = db.query(DatasetAsset).filter(
            DatasetAsset.dataset_version_id == source_version.id,
        ).order_by(DatasetAsset.relative_path.asc()).all()
        assets = [asset for asset in assets if Path(asset.relative_path).suffix.lower() in IMAGE_FILE_EXTENSIONS]
        if not assets:
            raise ValueError("原始数据集中没有可增强的图片")
        if len(assets) > 100_000:
            raise ValueError("数据集图片数量超过增强上限 100000")

        output_count = (1 if include_original else 0) + len(operations) * copies_per_image
        estimated_bytes = sum(max(0, int(asset.size or 0)) for asset in assets) * output_count
        if estimated_bytes > shutil.disk_usage(constants.DATASETS_DIR).free:
            raise OSError("磁盘可用空间不足，无法生成增强数据集")

        output_path = build_unique_upload_dir(constants.DATASETS_DIR, output_name, fallback="augmented-dataset")
        output_path.mkdir(parents=True, exist_ok=True)
        total_outputs = len(assets) * output_count
        processed = 0
        update_job(
            db,
            task_id,
            stage="generating_images",
            progress=5,
            message=f"正在生成增强图片 0/{total_outputs}",
            event_payload={"total": total_outputs},
        )

        for asset in assets:
            if is_cancel_requested(task_id):
                update_job(db, task_id, status="cancelled", stage="cancelled", progress=processed / max(1, total_outputs) * 100, message="数据增强任务已取消")
                return
            source_file = resolve_relative_file_under_root(source_root, asset.relative_path)
            relative_path = Path(asset.relative_path)
            if include_original:
                target_original = output_path / relative_path
                target_original.parent.mkdir(parents=True, exist_ok=True)
                copy_raw_image_without_exif(source_file, target_original)
                processed += 1

            with Image.open(source_file) as opened:
                base_image = opened.copy()
            try:
                for operation in operations:
                    for variant_index in range(copies_per_image):
                        if is_cancel_requested(task_id):
                            update_job(db, task_id, status="cancelled", stage="cancelled", progress=processed / max(1, total_outputs) * 100, message="数据增强任务已取消")
                            return
                        augmented, token = apply_dataset_augmentation(base_image, operation, variant_index)
                        target_name = f"{relative_path.stem}__aug_{token}{relative_path.suffix.lower()}"
                        target_path = output_path / relative_path.parent / target_name
                        if target_path.exists():
                            target_name = f"{relative_path.stem}__aug_{token}_{variant_index + 1}{relative_path.suffix.lower()}"
                            target_path = output_path / relative_path.parent / target_name
                        save_augmented_image(augmented, target_path)
                        processed += 1
                        if processed == total_outputs or processed % 20 == 0:
                            progress_db = SessionLocal()
                            try:
                                update_job(
                                    progress_db,
                                    task_id,
                                    stage="generating_images",
                                    progress=round(5 + processed / max(1, total_outputs) * 80, 2),
                                    message=f"正在生成增强图片 {processed}/{total_outputs}",
                                    event_payload={"processed": processed, "total": total_outputs},
                                )
                            finally:
                                progress_db.close()
            finally:
                base_image.close()

        stats = analyze_original_dataset(output_path)
        now = datetime.now().isoformat()
        source_annotation_model = get_dataset_annotation_model(db, source_version, allow_source_fallback=False)
        owner_username = str(input_data.get("owner_username") or "system")
        dataset_info = {
            "id": output_path.name,
            "name": output_name,
            "path": str(output_path),
            "created_at": now,
            "updated_at": now,
            "status": constants.DATASET_STATUS_READY,
            "dataset_role": "original",
            "source_type": "augmentation",
            "source_dataset_id": source_dataset_id,
            "augmentation_operations": operations,
            "include_original": include_original,
            "copies_per_image": copies_per_image,
            "uploader": owner_username,
            "upload_user": owner_username,
            "created_by": owner_username,
            **({
                "annotation_model": source_annotation_model,
                "annotation_model_path": source_annotation_model.get("path"),
                "annotation_model_name": source_annotation_model.get("name"),
                "annotation_geometry": source_annotation_model.get("geometry"),
            } if source_annotation_model else {}),
            **stats,
        }
        uploaded_datasets[output_path.name] = dataset_info
        write_dataset_metadata(output_path, dataset_info)
        version = sync_dataset_metadata_to_database(
            db,
            output_path,
            dataset_info,
            dataset_role="original",
            source_type="augmentation",
        )
        owner_role = db.query(Role).filter(
            Role.id == int(input_data["owner_role_id"]),
            Role.status == "active",
        ).first()
        if not owner_role:
            raise ValueError("数据集归属角色不存在或已停用")
        ensure_dataset_resource(
            db,
            output_path.name,
            "original",
            {"id": input_data.get("owner_id"), "username": owner_username},
            owner_role,
        )
        completed = True
        update_job(
            db,
            task_id,
            status="completed",
            stage="completed",
            progress=100,
            message="数据增强完成",
            result={"dataset_id": output_path.name, "dataset_info": dataset_info, "version_id": version.id},
        )
    except Exception as exc:
        db.rollback()
        logger.exception("后台数据增强失败: task=%s", task_id)
        try:
            task = db.query(JobTask).filter(JobTask.id == task_id).first()
            if task and task.status in ACTIVE_STATUSES:
                update_job(
                    db,
                    task_id,
                    status="failed",
                    stage="failed",
                    message=f"数据增强失败: {str(exc)}",
                    error_code="dataset_augmentation_failed",
                    error_detail=str(exc),
                )
        except Exception:
            db.rollback()
            logger.exception("写入数据增强失败状态失败: task=%s", task_id)
    finally:
        db.close()
        if output_path is not None and not completed:
            shutil.rmtree(output_path, ignore_errors=True)
            uploaded_datasets.pop(output_path.name, None)
            cleanup_db = SessionLocal()
            try:
                delete_dataset_version_records(cleanup_db, output_path.name, "original")
                cleanup_db.query(DatasetResource).filter(
                    DatasetResource.dataset_id == output_path.name,
                    DatasetResource.dataset_role == "original",
                ).delete(synchronize_session=False)
                cleanup_db.commit()
            except Exception:
                cleanup_db.rollback()
            finally:
                cleanup_db.close()
            shutil.rmtree(DATA_DIR / "manifests" / output_path.name, ignore_errors=True)



def _resolve_annotation_export_profile(request: Dict[str, Any]) -> Dict[str, Any]:
    requested = str(request.get("profile") or request.get("tiling_profile") or TILING_POLICY_VERSION).strip()
    try:
        return resolve_tiling_profile(requested)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

def _resolve_export_split_ratios(request: Mapping[str, Any]) -> Tuple[float, float]:
    """Parse train/test ratios once so preflight and export use identical splits."""
    try:
        train_ratio = float(request.get("train_ratio", 0.8))
        test_ratio = float(request.get("test_ratio", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("训练集划分参数无效") from exc
    if not 0 < train_ratio < 1:
        raise ValueError("train_ratio 必须在 0 和 1 之间")
    if not 0 <= test_ratio < 1:
        raise ValueError("test_ratio 必须在 0 和 1 之间")
    if train_ratio + test_ratio >= 1:
        raise ValueError("train_ratio + test_ratio 必须小于 1")
    return train_ratio, test_ratio

def _add_quality_sample_preview_urls(quality: Dict[str, Any], dataset_id: str) -> None:
    previews = []
    for item in quality.get("mobilephone_sample_previews") or []:
        asset_id = item.get("asset_id")
        if asset_id is None:
            continue
        previews.append({
            **item,
            "thumbnail_url": f"/api/datasets/{dataset_id}/assets/{int(asset_id)}/thumbnail",
        })
    quality["mobilephone_sample_previews"] = previews[:100]

def _prepare_saved_version_quality_gate(
    db,
    source_version: DatasetVersion,
    request: Dict[str, Any],
    export_profile: Dict[str, Any],
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply the same quality gate to the legacy synchronous export route."""
    required_geometry = str(request.get("required_geometry") or "bbox").strip().lower()
    if required_geometry not in {"bbox", "polygon"}:
        raise ValueError("标注框版本仅支持 bbox 或 polygon")
    train_ratio, test_ratio = _resolve_export_split_ratios(request)

    requested_classes = request.get("classes") or request.get("class_names")
    class_names = (
        normalize_class_names(requested_classes)
        if requested_classes
        else ordered_class_names(source_version.class_map_json)
    )
    if not class_names:
        raise ValueError("标注版本缺少有效类别列表")
    source_assets = (
        db.query(DatasetAsset)
        .filter(DatasetAsset.dataset_version_id == source_version.id)
        .order_by(DatasetAsset.relative_path.asc())
        .all()
    )
    quality_updates = {
        asset.relative_path: {
            "width": int(asset.width or 0),
            "height": int(asset.height or 0),
            "annotations": filter_annotations_by_geometry(
                asset.annotations_json or [],
                required_geometry=required_geometry,
            ),
        }
        for asset in source_assets
    }
    quality = summarize_mobilephone_export_quality(
        source_assets,
        class_names,
        annotations_by_path=quality_updates,
        profile=export_profile["version"],
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        seed=int(request.get("seed", 42)),
        required_geometry=required_geometry,
    )
    if quality["errors"]:
        raise ValueError(f"训练数据质量门阻止导出: {quality['errors'][:3]}")
    warning_codes = sorted({
        str(item.get("code"))
        for item in quality["warnings"]
        if item.get("code")
    })
    supplied_confirmation = request.get("quality_confirmation") or {}
    supplied_codes = sorted({
        str(code)
        for code in (supplied_confirmation.get("warning_codes") or [])
        if str(code).strip()
    }) if isinstance(supplied_confirmation, dict) else []
    confirmed = bool(supplied_confirmation.get("confirmed")) if isinstance(supplied_confirmation, dict) else False
    if warning_codes and (not confirmed or supplied_codes != warning_codes):
        raise ValueError("导出质量告警尚未确认或预检结果已过期，请重新预检")
    confirmation = {
        "confirmed": confirmed,
        "warning_codes": warning_codes,
        "thresholds": quality.get("thresholds") or {},
    }
    quality["confirmation"] = confirmation
    return quality, confirmation


@app.post("/annotation/export-training-dataset/preflight")
@app.post("/api/annotation/export-training-dataset/preflight")
async def preflight_annotation_training_dataset_export(
    request: Dict[str, Any],
    http_request: Request,
    db=Depends(get_db),
):
    """Count effective unlabeled assets before the user confirms export."""
    dataset_id = str(request.get("dataset_id") or "").strip()
    if not dataset_id:
        raise HTTPException(status_code=400, detail="\\u7f3a\\u5c11 dataset_id")
    export_profile = _resolve_annotation_export_profile(request)
    required_geometry = str(request.get("required_geometry") or "bbox").strip().lower()
    if required_geometry not in {"bbox", "polygon"}:
        raise HTTPException(status_code=400, detail="\\u6807\\u6ce8\\u6846\\u7248\\u672c\\u4ec5\\u652f\\u6301 bbox \\u6216 polygon")
    images = request.get("images") or []
    if not isinstance(images, list) or len(images) > 100_000:
        raise HTTPException(status_code=413, detail="\\u6807\\u6ce8\\u8986\\u76d6\\u8d44\\u4ea7\\u6570\\u8d85\\u8fc7 100000")

    current_user = get_request_current_user(http_request)
    require_dataset_manage(db, current_user, dataset_id, "original")
    source_version = (
        db.query(DatasetVersion)
        .filter(
            DatasetVersion.dataset_id == dataset_id,
            DatasetVersion.dataset_role == "original",
            DatasetVersion.status == "ready",
        )
        .order_by(DatasetVersion.created_at.desc())
        .first()
    )
    if not source_version:
        raise HTTPException(status_code=409, detail="\\u6570\\u636e\\u96c6\\u5c1a\\u672a\\u5efa\\u7acb\\u53ef\\u7528\\u8d44\\u4ea7\\u7d22\\u5f15")

    draft_id = str(request.get("annotation_draft_id") or "").strip()
    if draft_id:
        draft = get_draft_by_id(db, _draft_http_user_id(current_user), draft_id)
        if draft is None or str(draft.dataset_id) != dataset_id:
            raise HTTPException(status_code=404, detail="\\u6807\\u6ce8\\u8349\\u7a3f\\u4e0d\\u5b58\\u5728\\u6216\\u4e0d\\u5c5e\\u4e8e\\u5f53\\u524d\\u6570\\u636e\\u96c6")
        if str(draft.source_version_id or "") != str(source_version.id):
            raise HTTPException(status_code=409, detail="\\u6807\\u6ce8\\u8349\\u7a3f\\u5bf9\\u5e94\\u7684\\u6570\\u636e\\u96c6\\u7248\\u672c\\u5df2\\u53d8\\u5316")
        _validate_annotation_model_selection(db, current_user, draft.model_path, draft.annotation_kind)
        source_version = _draft_working_version(db, dataset_id, draft, source_version)
        request = {
            **request,
            "owner_id": current_user.get("id"),
            "owner_username": current_user.get("username"),
            "owner_role": current_user.get("role"),
            "owner_roleKeys": current_user.get("roleKeys") or [],
        }

    annotation_task_id = str(request.get("annotation_task_id") or "").strip()
    if annotation_task_id:
        annotation_task = db.query(JobTask).filter(
            JobTask.id == annotation_task_id,
            JobTask.type == "annotation",
        ).first()
        if not annotation_task:
            raise HTTPException(status_code=404, detail="\\u4e00\\u952e AI \\u6807\\u6ce8\\u4efb\\u52a1\\u4e0d\\u5b58\\u5728")
        require_job_visibility(annotation_task, http_request)
        if annotation_task.status != "completed":
            raise HTTPException(status_code=409, detail="\\u4e00\\u952e AI \\u6807\\u6ce8\\u4efb\\u52a1\\u5c1a\\u672a\\u5b8c\\u6210")
        if not annotation_task_matches_source_version(annotation_task, source_version):
            raise HTTPException(status_code=409, detail="\\u4e00\\u952e AI \\u6807\\u6ce8\\u4efb\\u52a1\\u4e0e\\u5f53\\u524d\\u6570\\u636e\\u96c6\\u4e0d\\u5339\\u914d")
        task_source_version_id = str((annotation_task.input_json or {}).get("source_version_id") or "").strip()
        if task_source_version_id and task_source_version_id != source_version.id and not annotation_task_matches_source_version(annotation_task, source_version):
            task_source_version = db.query(DatasetVersion).filter(
                DatasetVersion.id == task_source_version_id,
                DatasetVersion.dataset_id == dataset_id,
                DatasetVersion.dataset_role == "original",
            ).first()
            if not task_source_version:
                raise HTTPException(status_code=409, detail="\\u4e00\\u952e AI \\u6807\\u6ce8\\u4efb\\u52a1\\u5bf9\\u5e94\\u7684\\u539f\\u59cb\\u6570\\u636e\\u96c6\\u7248\\u672c\\u5df2\\u4e0d\\u5b58\\u5728")
            source_version = task_source_version

    updates, _ = build_annotation_export_updates(db, source_version, request)
    source_assets = (
        db.query(DatasetAsset)
        .filter(DatasetAsset.dataset_version_id == source_version.id)
        .order_by(DatasetAsset.relative_path.asc())
        .all()
    )
    summary = summarize_annotation_assets(source_assets, updates, required_geometry=required_geometry)
    quality_updates = {
        path: {
            **update,
            "annotations": filter_annotations_by_geometry(
                update.get("annotations") or [],
                required_geometry=required_geometry,
            ),
        }
        for path, update in updates.items()
    }
    try:
        train_ratio, test_ratio = _resolve_export_split_ratios(request)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    quality = summarize_mobilephone_export_quality(
        source_assets,
        request.get("classes") or request.get("class_names") or [],
        annotations_by_path=quality_updates,
        profile=export_profile["version"],
        train_ratio=train_ratio,
        test_ratio=test_ratio,
        seed=int(request.get("seed", 42)),
        required_geometry=required_geometry,
    )
    _add_quality_sample_preview_urls(quality, dataset_id)
    return {
        "dataset_id": dataset_id,
        "source_version_id": source_version.id,
        "required_geometry": required_geometry,
        **summary,
        "profile": export_profile["version"],
        "train_ratio": train_ratio,
        "test_ratio": test_ratio,
        "quality": quality,
        "can_export": bool(quality["can_export"]),
        "warnings": quality["warnings"],
        "errors": quality["errors"],
        "message": f"\\u5f53\\u524d\\u5b58\\u5728 {summary['unlabeled_count']} \\u5f20\\u56fe\\u7247\\u672a\\u6807\\u6ce8\\uff0c\\u5bfc\\u51fa\\u540e\\u5c06\\u5264\\u9664\\u8fd9\\u4e9b\\u56fe\\u7247",
    }


@app.post("/annotation/export-training-dataset/start")
@app.post("/api/annotation/export-training-dataset/start")
async def start_annotation_training_dataset_export(
    request: Dict[str, Any],
    http_request: Request,
    db=Depends(get_db),
):
    dataset_id = str(request.get("dataset_id") or "").strip()
    if not dataset_id:
        raise HTTPException(status_code=400, detail="缺少 dataset_id")
    export_profile = _resolve_annotation_export_profile(request)
    output_name = str(request.get("name") or f"{dataset_id}_标注训练集").strip()
    if not output_name or len(output_name) > 100 or any(ord(char) < 32 for char in output_name):
        raise HTTPException(status_code=400, detail="训练数据集名称格式不正确")
    try:
        train_ratio, test_ratio = _resolve_export_split_ratios(request)
        seed = int(request.get("seed", 42))
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="训练集划分参数无效")
    if not 0 < train_ratio < 1:
        raise HTTPException(status_code=400, detail="train_ratio 必须在 0 和 1 之间")
    annotation_model_path = str(
        request.get("annotation_model_path")
        or request.get("model_path")
        or request.get("model")
        or ""
    ).strip()
    annotation_model_name = str(
        request.get("annotation_model_name")
        or request.get("model_name")
        or request.get("modelName")
        or annotation_model_path
    ).strip()
    annotation_geometry = str(
        request.get("annotation_geometry")
        or request.get("geometry")
        or ""
    ).strip().lower()
    if annotation_geometry not in {"bbox", "rect", "polygon"}:
        annotation_geometry = model_annotation_geometry(annotation_model_path)
    requested_geometry = str(request.get("required_geometry") or "").strip().lower()
    required_geometry = requested_geometry or ("polygon" if annotation_geometry == "polygon" else "bbox")
    if required_geometry not in {"bbox", "polygon"}:
        raise HTTPException(status_code=400, detail="标注框版本仅支持 bbox 或 polygon")

    images = request.get("images") or []
    if not isinstance(images, list) or len(images) > 100_000:
        raise HTTPException(status_code=413, detail="标注覆盖资产数超过 100000")
    current_user = get_request_current_user(http_request)
    source_resource = require_dataset_manage(db, current_user, dataset_id, "original")
    source_version = (
        db.query(DatasetVersion)
        .filter(
            DatasetVersion.dataset_id == dataset_id,
            DatasetVersion.dataset_role == "original",
            DatasetVersion.status == "ready",
        )
        .order_by(DatasetVersion.created_at.desc())
        .first()
    )
    if not source_version:
        raise HTTPException(status_code=409, detail="数据集尚未建立可用资产索引")

    annotation_draft_id = str(request.get("annotation_draft_id") or "").strip()
    annotation_draft = None
    if annotation_draft_id:
        annotation_draft = get_draft_by_id(db, _draft_http_user_id(current_user), annotation_draft_id)
        if annotation_draft is None or str(annotation_draft.dataset_id) != dataset_id:
            raise HTTPException(status_code=404, detail="标注草稿不存在或不属于当前数据集")
        if str(annotation_draft.source_version_id or "") != str(source_version.id):
            raise HTTPException(status_code=409, detail="标注草稿对应的数据集版本已变化")
        _validate_annotation_model_selection(
            db,
            current_user,
            annotation_draft.model_path,
            annotation_draft.annotation_kind,
        )
        if not annotation_model_path:
            annotation_model_path = annotation_draft.model_path
            annotation_model_name = annotation_draft.model_name
        if not request.get("annotation_geometry") and not request.get("geometry"):
            annotation_geometry = annotation_draft.annotation_geometry
            if not requested_geometry:
                required_geometry = "polygon" if annotation_geometry == "polygon" else "bbox"
        source_version = _draft_working_version(db, dataset_id, annotation_draft, source_version)

    annotation_task_id = str(request.get("annotation_task_id") or "").strip()
    if annotation_task_id:
        annotation_task = db.query(JobTask).filter(
            JobTask.id == annotation_task_id,
            JobTask.type == "annotation",
        ).first()
        if not annotation_task:
            raise HTTPException(status_code=404, detail="一键 AI 标注任务不存在")
        require_job_visibility(annotation_task, http_request)
        if annotation_task.status != "completed":
            raise HTTPException(status_code=409, detail="一键 AI 标注任务尚未完成")
        if not annotation_task_matches_source_version(annotation_task, source_version):
            raise HTTPException(status_code=409, detail="一键 AI 标注任务与当前数据集不匹配")
        task_source_version_id = str((annotation_task.input_json or {}).get("source_version_id") or "").strip()
        if task_source_version_id and task_source_version_id != source_version.id and not annotation_task_matches_source_version(annotation_task, source_version):
            task_source_version = db.query(DatasetVersion).filter(
                DatasetVersion.id == task_source_version_id,
                DatasetVersion.dataset_id == dataset_id,
                DatasetVersion.dataset_role == "original",
            ).first()
            if not task_source_version:
                raise HTTPException(status_code=409, detail="一键 AI 标注任务对应的原始数据集版本已不存在")
            source_version = task_source_version
        task_input = dict(annotation_task.input_json or {})
        task_model_binding = build_annotation_model_binding(
            task_input.get("model_path"),
            task_input.get("model_name"),
            task_input.get("annotation_geometry"),
        )
        if (
            annotation_model_path
            and task_model_binding
            and not annotation_model_matches(annotation_model_path, task_model_binding)
        ):
            raise HTTPException(
                status_code=409,
                detail="导出模型与一键 AI 标注任务使用的模型不一致",
            )
        if not annotation_model_path:
            annotation_model_path = str(task_input.get("model_path") or "").strip()
            annotation_model_name = annotation_model_name or annotation_model_path
        if not request.get("annotation_geometry") and not request.get("geometry"):
            annotation_geometry = model_annotation_geometry(annotation_model_path)
            if not requested_geometry:
                required_geometry = "polygon" if annotation_geometry == "polygon" else "bbox"

    requested_owner_role_id = request.get("owner_role_id")
    owner_role = None
    if requested_owner_role_id is None and source_resource.owner_role_id is not None:
        owner_role = db.query(Role).filter(
            Role.id == source_resource.owner_role_id,
            Role.status == "active",
        ).first()
    if owner_role is None:
        owner_role = select_upload_role(db, current_user, requested_owner_role_id)

    input_data = {
        "source_dataset_id": dataset_id,
        "source_version_id": source_version.id,
        "annotation_task_id": annotation_task_id,
        "annotation_draft_id": annotation_draft_id,
        "classes": request.get("classes") or request.get("class_names") or [],
        "images": images,
        "clear_all_annotations": bool(request.get("clear_all_annotations", False)),
        "class_mode": request.get("class_mode") or "auto",
        "target_class": request.get("target_class"),
        "annotation_mode": request.get("annotation_mode") or "replace",
        "required_geometry": required_geometry,
        "exclude_unlabeled_images": bool(request.get("exclude_unlabeled_images", True)),
        "annotation_model_path": annotation_model_path,
        "annotation_model_name": annotation_model_name or annotation_model_path,
        "annotation_geometry": annotation_geometry,
        "name": output_name,
        "seed": seed,
        "train_ratio": train_ratio,
        "test_ratio": test_ratio,
        "profile": export_profile["version"],
        "quality_confirmation": request.get("quality_confirmation") or {},
        "owner_role_id": owner_role.id,
        "owner_id": current_user.get("id"),
        "owner_username": str(current_user.get("username") or get_request_current_username(http_request)),
        "owner_role": current_user.get("role"),
        "owner_roleKeys": current_user.get("roleKeys") or [],
        "temporary_dataset_id": (
            str(annotation_draft.temporary_dataset_id or "").strip()
            if annotation_draft is not None
            else str(request.get("temporary_dataset_id") or "").strip()
        ),
    }
    job = create_job(
        db,
        job_type="dataset_export",
        owner_username=input_data["owner_username"],
        owner_id=current_user.get("id"),
        input_data=input_data,
        message="训练数据集导出任务已创建",
    )
    handle = asyncio.create_task(asyncio.to_thread(run_annotation_export_task, job.id))
    annotation_export_task_handles[job.id] = handle
    handle.add_done_callback(lambda _task, export_id=job.id: annotation_export_task_handles.pop(export_id, None))
    return {
        "task_id": job.id,
        "status": job.status,
        "stage": job.stage,
        "progress": job.progress,
        "message": job.message,
    }


@app.post("/annotation/export-training-dataset")
@app.post("/api/annotation/export-training-dataset")
async def export_annotation_training_dataset(
    request: Dict[str, Any],
    http_request: Request,
    db=Depends(get_db),
):
    annotation_version_id = str(request.get("annotation_version_id") or request.get("version_id") or "").strip()
    if not annotation_version_id:
        raise HTTPException(status_code=400, detail="缺少 annotation_version_id；请先保存标注版本")
    source_version = db.query(DatasetVersion).filter(DatasetVersion.id == annotation_version_id).first()
    if not source_version:
        raise HTTPException(status_code=404, detail="标注版本不存在")
    if source_version.dataset_role not in {"annotation", "training"}:
        raise HTTPException(status_code=400, detail="指定版本不是可导出的标注/训练版本")
    output_name = str(request.get("name") or f"{source_version.dataset_id}_标注训练集").strip()
    if not output_name:
        raise HTTPException(status_code=400, detail="训练数据集名称不能为空")
    if len(output_name) > 100 or any(ord(char) < 32 for char in output_name):
        raise HTTPException(status_code=400, detail="训练数据集名称格式不正确")
    export_profile = _resolve_annotation_export_profile(request)
    current_user = get_request_current_user(http_request)
    source_resource_role = "original" if source_version.dataset_role == "annotation" else "training"
    source_resource = require_dataset_manage(db, current_user, source_version.dataset_id, source_resource_role)
    requested_owner_role_id = request.get("owner_role_id")
    owner_role = None
    if requested_owner_role_id is None and source_resource.owner_role_id is not None:
        owner_role = db.query(Role).filter(
            Role.id == source_resource.owner_role_id,
            Role.status == "active",
        ).first()
    if owner_role is None:
        owner_role = select_upload_role(db, current_user, requested_owner_role_id)
    try:
        requested_annotation_model = build_annotation_model_binding(
            request.get("annotation_model_path") or request.get("model_path") or request.get("model"),
            request.get("annotation_model_name") or request.get("model_name") or request.get("modelName"),
            request.get("annotation_geometry") or request.get("geometry") or request.get("required_geometry"),
            selected_by=get_request_current_username(http_request),
            selected_at=datetime.utcnow().isoformat() + "Z",
        )
        source_annotation_model = get_dataset_annotation_model(
            db,
            source_version,
            allow_source_fallback=False,
        )
        if source_version.dataset_role != "training":
            source_annotation_model = requested_annotation_model or source_annotation_model
        train_ratio, test_ratio = _resolve_export_split_ratios(request)
        quality_request = dict(request)
        if request.get("annotation_geometry") in {"polygon", "segmentation"}:
            quality_request.setdefault("required_geometry", "polygon")
        quality, quality_confirmation = _prepare_saved_version_quality_gate(db, source_version, quality_request, export_profile)
        legacy_inactive_ids = get_inactive_annotation_class_ids(source_version)
        legacy_fixed_classes = None
        if legacy_inactive_ids:
            legacy_fixed_classes = get_training_model_class_names(
                db, str((source_annotation_model or {}).get("path") or ""),
            )
            if not legacy_fixed_classes:
                raise ValueError("旧模型未记录训练类别顺序，不能导出可安全续训的数据集")
        version, summary, created = export_training_version(
            db,
            source_version=source_version,
            output_name=output_name,
            training_root=TRAINING_DATASETS_DIR,
            staging_root=DATA_DIR / "staging",
            manifest_root=DATA_DIR / "manifests",
            created_by=get_request_current_username(http_request),
            seed=int(request.get("seed", 42)),
            train_ratio=train_ratio,
            test_ratio=test_ratio,
            profile=export_profile["version"],
            quality_confirmation=quality_confirmation,
            fixed_class_names=legacy_fixed_classes,
            required_geometry=(
                "polygon"
                if (request.get("required_geometry") or request.get("annotation_geometry")) in {"polygon", "segmentation"}
                else ("bbox" if request.get("required_geometry") or request.get("annotation_geometry") else None)
            ),
        )
        summary["quality"] = quality
        output_path = Path(version.root_path)
        stats = training_dataset_manager.analyze_dataset(output_path)
        now = datetime.now().isoformat()
        dataset_info = {
            "id": version.dataset_id,
            "name": output_name,
            "path": str(output_path),
            "version_id": version.id,
            "parent_version_id": source_version.id,
            "created_at": now,
            "updated_at": now,
            "status": constants.DATASET_STATUS_READY,
            "dataset_role": "training",
            "source_dataset_id": source_version.dataset_id,
            "annotation_export": {"quality": quality},
            "model_binding_mode": "fixed" if source_annotation_model else "none",
            "annotation_model_selection_mode": "fixed" if source_annotation_model else "none",
            "annotation_model_locked": bool(source_annotation_model),
            "image_size": summary.get("image_size"),
            "optimal_image_size": (summary.get("image_size") or {}).get("optimal_size"),
            "image_size_policy": (summary.get("image_size") or {}).get("policy"),
            "uploader": get_request_current_username(http_request),
            "created_by": get_request_current_username(http_request),
            **({
                "annotation_model": source_annotation_model,
                "annotation_model_path": source_annotation_model.get("path"),
                "annotation_model_name": source_annotation_model.get("name"),
                "annotation_geometry": source_annotation_model.get("geometry"),
            } if source_annotation_model else {}),
            **stats,
        }
        write_training_dataset_metadata(output_path, dataset_info)
        persist_dataset_business_metadata(
            db,
            version,
            output_path,
            dataset_info,
            source_type="annotation_export",
        )
        ensure_dataset_resource(db, version.dataset_id, "training", current_user, owner_role)
        if legacy_inactive_ids:
            metadata = dict(version.metadata_json or {})
            metadata[ANNOTATION_CLASSES_METADATA_KEY] = list(legacy_fixed_classes)
            source_names = ordered_class_names(source_version.class_map_json)
            metadata[ANNOTATION_INACTIVE_CLASS_IDS_METADATA_KEY] = [
                legacy_fixed_classes.index(source_names[index]) for index in legacy_inactive_ids
                if source_names[index] in legacy_fixed_classes
            ]
            version.metadata_json = metadata
        return {
            "created": created,
            "message": "标注版本已导出为可复现训练数据集",
            "dataset_id": version.dataset_id,
            "dataset_path": str(output_path),
            "version": version.to_dict(),
            "dataset_info": dataset_info,
            "summary": summary,
        }
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=400, detail=str(exc))


async def export_annotation_training_dataset_legacy(request: Dict[str, Any], http_request: Request):
    """将标注页当前数据集保存为训练数据集，而不是返回浏览器下载包。"""
    source_dataset_id = str(request.get("dataset_id") or request.get("datasetId") or "").strip()
    if not source_dataset_id:
        raise HTTPException(status_code=400, detail="缺少原始数据集ID")

    images = request.get("images") or []
    if not isinstance(images, list) or not images:
        raise HTTPException(status_code=400, detail="没有可导出的图片")

    class_names = request.get("classes") or request.get("class_names") or []
    if not isinstance(class_names, list) or not class_names:
        class_names = ["object"]
    class_names = [str(name or f"class_{index}").strip() or f"class_{index}" for index, name in enumerate(class_names)]

    source_dataset_path = resolve_managed_dataset_root(source_dataset_id, constants.DATASETS_DIR)
    source_info = build_dataset_info_from_metadata(source_dataset_path)
    source_name = str(request.get("dataset_name") or source_info.get("name") or source_dataset_id)
    output_name = str(request.get("name") or f"{source_name}_标注训练集")
    output_path = build_unique_upload_dir(TRAINING_DATASETS_DIR, output_name, fallback="training_dataset")
    output_id = output_path.name
    uploader_name = get_request_current_username(http_request)

    try:
        for split in ("train", "val", "test"):
            (output_path / split / "images").mkdir(parents=True, exist_ok=True)
            (output_path / split / "labels").mkdir(parents=True, exist_ok=True)

        total_images = len(images)
        train_count = int(total_images * 0.8)
        if total_images <= 1:
            train_count = total_images
        else:
            train_count = max(1, min(train_count, total_images - 1))

        used_names = set()
        exported_count = 0

        for index, item in enumerate(images):
            if not isinstance(item, dict):
                continue

            relative_path = item.get("relative_path") or item.get("saved_path") or item.get("path") or item.get("filename") or item.get("name")
            source_image = resolve_relative_file_under_root(source_dataset_path, relative_path)
            split = "train" if index < train_count else "val"
            target_name = build_unique_training_image_name(item.get("name") or source_image.name, used_names)

            target_image = output_path / split / "images" / target_name
            target_label = output_path / split / "labels" / f"{Path(target_name).stem}.txt"
            copy_raw_image_without_exif(source_image, target_image)

            # The persisted asset dimensions may belong to a previous EXIF-
            # canonicalized pipeline. Always derive label dimensions from the
            # exact raw image written to the training workspace.
            width, height = get_uploaded_image_size(source_image)

            label_lines = []
            annotations = item.get("annotations") or []
            if isinstance(annotations, list):
                for annotation in annotations:
                    if not isinstance(annotation, dict):
                        continue
                    line = normalize_yolo_annotation_line(annotation, width, height)
                    if line:
                        label_lines.append(line)

            with open(target_label, "w", encoding="utf-8") as f:
                f.write("\n".join(label_lines))
                if label_lines:
                    f.write("\n")

            exported_count += 1

        if exported_count == 0:
            raise HTTPException(status_code=400, detail="没有成功导出的图片")

        dataset_yaml = (
            "names:\n"
            + "\n".join([f"  {index}: {name}" for index, name in enumerate(class_names)])
            + f"\nnc: {len(class_names)}\n"
            + f"path: {output_path}\n"
            + "train: train/images\n"
            + "val: val/images\n"
            + "test: test/images\n"
        )
        with open(output_path / "dataset.yaml", "w", encoding="utf-8") as f:
            f.write(dataset_yaml)

        with open(output_path / "classes.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(class_names) + "\n")

        with open(output_path / "README.md", "w", encoding="utf-8") as f:
            f.write(
                "# YOLO Training Dataset\n\n"
                f"- Source dataset: {source_name}\n"
                f"- Exported images: {exported_count}\n"
                f"- Classes: {', '.join(class_names)}\n"
            )

        stats = training_dataset_manager.analyze_dataset(output_path)
        now = datetime.now().isoformat()
        dataset_info = {
            "id": output_id,
            "name": output_path.name,
            "path": str(output_path),
            "created_at": now,
            "updated_at": now,
            "status": constants.DATASET_STATUS_READY if stats.get("image_count", 0) > 0 else constants.DATASET_STATUS_EMPTY,
            "dataset_role": "training",
            "source_dataset_id": source_dataset_id,
            "source_dataset_name": source_name,
            "uploader": uploader_name,
            "upload_user": uploader_name,
            "created_by": uploader_name,
            **stats
        }
        write_training_dataset_metadata(output_path, dataset_info)

        return {
            "message": "标注数据集已保存到训练数据集",
            "dataset_id": output_id,
            "dataset_path": str(output_path),
            "dataset_info": dataset_info
        }

    except HTTPException:
        file_manager.cleanup_dir(str(output_path))
        raise
    except Exception as e:
        file_manager.cleanup_dir(str(output_path))
        logger.error(f"保存训练数据集失败: {e}")
        raise HTTPException(status_code=500, detail=f"保存训练数据集失败: {str(e)}")


@app.post("/datasets/{dataset_id}/augmentation")
@app.post("/api/datasets/{dataset_id}/augmentation")
async def start_dataset_augmentation(
    dataset_id: str,
    payload: Dict[str, Any],
    request: Request,
    db=Depends(get_db),
):
    """Queue whole-dataset augmentation; generated images are appended and originals remain intact."""
    current_user = get_request_current_user(request)
    source_resource = require_dataset_manage(db, current_user, dataset_id, "original")
    config = {
        "rotations": payload.get("rotations") or [],
        "mirror": payload.get("mirror") or "none",
        "brightnessLevels": payload.get("brightnessLevels") or payload.get("brightness_levels") or [],
        "grayscaleLevels": payload.get("grayscaleLevels") or payload.get("grayscale_levels") or [],
        "contrastLevels": payload.get("contrastLevels") or payload.get("contrast_levels") or [],
    }
    try:
        operations = normalize_fine_augmentation_operations(config)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))

    source_version = (
        db.query(DatasetVersion)
        .filter(
            DatasetVersion.dataset_id == dataset_id,
            DatasetVersion.dataset_role == "original",
            DatasetVersion.status == "ready",
        )
        .order_by(DatasetVersion.created_at.desc(), DatasetVersion.id.desc())
        .first()
    )
    if not source_version:
        raise HTTPException(status_code=409, detail="数据集尚未建立可用资产索引")

    annotation_draft = None
    annotation_draft_id = str(payload.get("annotation_draft_id") or "").strip()
    temporary_dataset = False
    if annotation_draft_id:
        try:
            draft_kind = normalize_annotation_kind(payload.get("annotation_kind"))
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        annotation_draft = get_draft(db, _draft_http_user_id(current_user), dataset_id, draft_kind)
        if annotation_draft is None or str(annotation_draft.id) != annotation_draft_id:
            raise HTTPException(status_code=404, detail="标注草稿不存在或不属于当前用户")
        _draft_current_version_or_conflict(db, dataset_id, annotation_draft)
        try:
            expected_revision = int(payload.get("draft_revision"))
        except (TypeError, ValueError):
            raise HTTPException(status_code=409, detail="标注草稿修订号无效")
        if expected_revision != int(annotation_draft.revision or 0):
            raise HTTPException(status_code=409, detail="标注草稿已在其他标签页更新，请刷新后重试")
        source_version = _draft_working_version(db, dataset_id, annotation_draft, source_version)
        temporary_dataset = True

    raw_source_paths = payload.get("source_relative_paths")
    source_relative_paths = None
    annotation_overrides = []
    create_new_dataset = not bool(annotation_draft.temporary_dataset_id) if annotation_draft else bool(payload.get("create_new_dataset"))
    output_name = (
        f"annotation-draft-{annotation_draft.id}"
        if annotation_draft and create_new_dataset
        else str(payload.get("output_name") or f"{dataset_id}_增强").strip()
    )
    if create_new_dataset and (not output_name or len(output_name) > 128 or any(ord(char) < 32 for char in output_name)):
        raise HTTPException(status_code=400, detail="增强数据集名称格式不正确")
    if raw_source_paths is not None:
        if not isinstance(raw_source_paths, list) or not raw_source_paths or len(raw_source_paths) > 100:
            raise HTTPException(status_code=400, detail="指定增强图片列表无效")
        source_relative_paths = list(dict.fromkeys(
            str(value or "").replace("\\", "/").strip().lstrip("./")
            for value in raw_source_paths
        ))
        if any(not value for value in source_relative_paths):
            raise HTTPException(status_code=400, detail="指定增强图片缺少资产路径")
        source_assets = db.query(DatasetAsset).filter(
            DatasetAsset.dataset_version_id == source_version.id,
            DatasetAsset.relative_path.in_(source_relative_paths),
        ).all()
        assets_by_path = {str(asset.relative_path).replace("\\", "/"): asset for asset in source_assets}
        missing_path = next((path for path in source_relative_paths if path not in assets_by_path), None)
        if missing_path:
            raise HTTPException(status_code=400, detail=f"指定增强图片不是数据集资产: {missing_path}")

        raw_overrides = payload.get("annotation_overrides") or []
        if not isinstance(raw_overrides, list) or len(raw_overrides) > len(source_relative_paths):
            raise HTTPException(status_code=400, detail="增强标注覆盖格式不正确")
        for item in raw_overrides:
            if not isinstance(item, dict):
                raise HTTPException(status_code=400, detail="增强标注覆盖包含非对象条目")
            relative_path = str(item.get("relative_path") or "").replace("\\", "/").strip().lstrip("./")
            asset = assets_by_path.get(relative_path)
            if not asset:
                raise HTTPException(status_code=400, detail=f"增强标注覆盖不是选中资产: {relative_path or 'unknown'}")
            source_image = resolve_relative_file_under_root(Path(source_version.root_path), relative_path)
            width, height = get_uploaded_image_size(source_image)
            try:
                annotations = [
                    frontend_annotation_to_manifest(annotation, image_width=width, image_height=height)
                    for annotation in (item.get("annotations") or [])
                ]
            except (TypeError, ValueError) as exc:
                raise HTTPException(status_code=400, detail=f"增强标注格式不正确: {exc}")
            annotation_overrides.append({
                "relative_path": relative_path,
                "annotations": annotations,
            })


    active_tasks = db.query(JobTask).filter(
        JobTask.type == "dataset_augmentation",
        JobTask.status.in_(ACTIVE_STATUSES),
    ).all()
    if any(
        str((task.input_json or {}).get("original_dataset_id") or (task.input_json or {}).get("dataset_id") or "") == dataset_id
        for task in active_tasks
    ):
        raise HTTPException(status_code=409, detail="该数据集已有增强任务正在执行")

    task = create_job(
        db,
        job_type="dataset_augmentation",
        owner_username=get_request_current_username(request),
        owner_id=get_request_current_user_id(current_user),
        input_data={
            "dataset_id": source_version.dataset_id,
            "original_dataset_id": dataset_id,
            "source_version_id": source_version.id,
            "config": config,
            "operations": operations,
            "source_relative_paths": source_relative_paths,
            "annotation_overrides": annotation_overrides,
            "create_new_dataset": create_new_dataset,
            "temporary_dataset": temporary_dataset,
            "output_name": output_name if create_new_dataset else "",
            "annotation_draft_id": annotation_draft.id if annotation_draft else "",
            "annotation_draft_owner_id": annotation_draft.owner_user_id if annotation_draft else None,
            "draft_source_version_id": annotation_draft.source_version_id if annotation_draft else "",
            "annotation_draft_revision": int(annotation_draft.revision or 0) if annotation_draft else None,
            "owner_role_id": source_resource.owner_role_id,
            "owner_id": source_resource.owner_user_id,
            "owner_username": source_resource.owner_username,
        },
        message=(
            f"已创建增强数据集任务，将生成 {len(operations)} 张图片"
            if create_new_dataset
            else f"已创建单图增强任务，将生成 {len(operations)} 张图片"
            if source_relative_paths and not create_new_dataset
            else f"已创建数据集增强任务，将处理 {len(operations)} 种操作"
        ),
    )
    return task.to_dict()


@app.post("/datasets/{dataset_id}/refresh")
async def refresh_dataset_stats(dataset_id: str, request: Request, db=Depends(get_db)):
    """
    手动刷新单个数据集统计信息。
    这个接口会扫描数据集目录，适合用户点击“刷新统计”时调用。
    """
    try:
        require_dataset_manage(db, get_request_current_user(request), dataset_id, "original")
        dataset_info = refresh_dataset_metadata(dataset_id)
        sync_dataset_metadata_to_database(
            db,
            Path(constants.DATASETS_DIR) / dataset_id,
            dataset_info,
            dataset_role="original",
            source_type="refresh",
        )

        return {
            "message": "数据集统计信息刷新成功",
            "dataset_info": dataset_info
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"刷新数据集统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"刷新数据集统计失败: {str(e)}")

@app.post("/datasets/refresh-all")
async def refresh_all_dataset_stats(request: Request, db=Depends(get_db)):
    """
    手动刷新所有数据集统计信息。
    注意：大数据集很多时，这个接口可能耗时较长，不建议前端页面自动调用。
    """
    try:
        datasets_dir = Path(constants.DATASETS_DIR)

        if not datasets_dir.exists():
            return {
                "message": "数据集目录不存在",
                "updated_count": 0,
                "failed": []
            }

        updated_count = 0
        failed = []
        current_user = get_request_current_user(request)

        for dataset_id_dir in datasets_dir.iterdir():
            if not dataset_id_dir.is_dir():
                continue

            dataset_id = dataset_id_dir.name

            try:
                require_dataset_manage(db, current_user, dataset_id, "original")
                dataset_info = refresh_dataset_metadata(dataset_id)
                sync_dataset_metadata_to_database(
                    db,
                    dataset_id_dir,
                    dataset_info,
                    dataset_role="original",
                    source_type="refresh",
                )
                updated_count += 1
            except Exception as e:
                failed.append({
                    "id": dataset_id,
                    "error": str(e)
                })

        return {
            "message": f"刷新完成，成功刷新 {updated_count} 个数据集",
            "updated_count": updated_count,
            "failed": failed
        }

    except Exception as e:
        logger.error(f"批量刷新数据集统计失败: {e}")
        raise HTTPException(status_code=500, detail=f"批量刷新数据集统计失败: {str(e)}")

# --- 训练管理接口 ---

@app.get("/jobs")
@app.get("/api/jobs")
async def list_jobs(
    request: Request,
    type: Optional[str] = None,
    status: Optional[str] = None,
    page: int = 1,
    page_size: int = 50,
    db=Depends(get_db),
):
    page = max(1, int(page))
    page_size = max(1, min(int(page_size), 200))
    query = db.query(JobTask)
    current_user = get_request_current_user(request)
    if current_user.get("role") != "admin":
        query = query.filter(JobTask.owner_username == str(current_user.get("username") or ""))
    if type:
        query = query.filter(JobTask.type == type)
    if status:
        query = query.filter(JobTask.status == status)
    total = query.count()
    tasks = query.order_by(JobTask.created_at.desc()).offset((page - 1) * page_size).limit(page_size).all()
    return {"total": total, "page": page, "page_size": page_size, "tasks": [task.to_dict() for task in tasks]}


@app.get("/jobs/{task_id}")
@app.get("/api/jobs/{task_id}")
async def get_job(task_id: str, request: Request, db=Depends(get_db)):
    task = db.query(JobTask).filter(JobTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    require_job_visibility(task, request)
    return task.to_dict()


@app.get("/jobs/{task_id}/events")
@app.get("/api/jobs/{task_id}/events")
async def get_job_events(
    task_id: str,
    request: Request,
    after_sequence: int = 0,
    limit: int = 200,
    db=Depends(get_db),
):
    task = db.query(JobTask).filter(JobTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    require_job_visibility(task, request)
    limit = max(1, min(int(limit), 1000))
    events = (
        db.query(JobEvent)
        .filter(JobEvent.task_id == task_id, JobEvent.sequence > max(0, int(after_sequence)))
        .order_by(JobEvent.sequence.asc())
        .limit(limit)
        .all()
    )
    next_sequence = events[-1].sequence if events else max(0, int(after_sequence))
    return {"task_id": task_id, "events": [event.to_dict() for event in events], "next_sequence": next_sequence}


@app.post("/jobs/{task_id}/cancel")
@app.post("/api/jobs/{task_id}/cancel")
async def cancel_job(task_id: str, request: Request, db=Depends(get_db)):
    task = db.query(JobTask).filter(JobTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    require_job_visibility(task, request)
    task = request_job_cancel(db, task_id)
    return task.to_dict()

def _resolve_training_dataset_version(db, dataset_path: str) -> Tuple[Path, DatasetVersion]:
    requested_path = Path(dataset_path).resolve()
    dataset_root = (
        requested_path.parent
        if requested_path.is_file() and requested_path.suffix.lower() in {".yaml", ".yml"}
        else requested_path
    )
    if not dataset_root.exists():
        raise HTTPException(status_code=404, detail="数据集路径不存在")
    if not _safe_relative_to(dataset_root, TRAINING_DATASETS_DIR.resolve()):
        raise HTTPException(status_code=400, detail="训练仅允许使用受管训练数据集目录")
    version = (
        db.query(DatasetVersion)
        .filter(DatasetVersion.root_path == str(dataset_root), DatasetVersion.dataset_role == "training")
        .order_by(DatasetVersion.created_at.desc())
        .first()
    )
    if not version or version.status != "ready":
        raise HTTPException(status_code=409, detail="训练数据集尚未建立可用版本或校验未通过")
    return requested_path, version


def _latest_annotation_source(db, training_version: DatasetVersion) -> DatasetVersion:
    parent = None
    if training_version.parent_version_id:
        parent = db.query(DatasetVersion).filter(
            DatasetVersion.id == training_version.parent_version_id,
            DatasetVersion.dataset_role == "annotation",
            DatasetVersion.status == "ready",
        ).first()
    if parent is None:
        return training_version
    return (
        db.query(DatasetVersion)
        .filter(
            DatasetVersion.dataset_id == parent.dataset_id,
            DatasetVersion.dataset_role == "annotation",
            DatasetVersion.status == "ready",
        )
        .order_by(DatasetVersion.created_at.desc())
        .first()
        or parent
    )


def _dataset_optimal_image_size(version: DatasetVersion) -> Optional[Dict[str, Any]]:
    """Read the persisted crop-size decision for one training dataset."""
    metadata = dict(version.metadata_json or {})
    summary = metadata.get("image_size")
    summary = dict(summary) if isinstance(summary, dict) else {}
    split_config = dict(version.split_config_json or {})
    nested_split_config = metadata.get("split_config")
    if isinstance(nested_split_config, dict):
        split_config = {**nested_split_config, **split_config}
    raw_candidates = (
        metadata.get("optimal_image_size"),
        metadata.get("image_size_optimal"),
        summary.get("optimal_size"),
        summary.get("recommended_size"),
        split_config.get("optimal_image_size"),
    )
    optimal_size = None
    for raw_value in raw_candidates:
        try:
            candidate = int(raw_value)
        except (TypeError, ValueError):
            continue
        if candidate in SUPPORTED_IMAGE_SIZES:
            optimal_size = candidate
            break
    if optimal_size is None:
        target_counts = (
            summary.get("target_size_counts")
            or split_config.get("image_size_target_counts")
            or split_config.get("target_size_counts")
            or {}
        )
        optimal_size = optimal_image_size_from_counts(target_counts)
    if optimal_size is None:
        return None
    return {
        "optimal_size": optimal_size,
        "summary": summary,
        "source": "persisted_annotation_safe_export",
    }


def _build_training_preflight(
    db,
    config: TrainingConfig,
    dataset_path: str,
    current_user: Dict[str, Any],
) -> Tuple[Dict[str, Any], Path, DatasetVersion, DatasetVersion]:
    requested_path, selected_version = _resolve_training_dataset_version(db, dataset_path)
    require_dataset_read(db, current_user, selected_version.dataset_id, "training")
    annotation_model_binding = get_dataset_annotation_model(db, selected_version)
    selected_metadata = dict(selected_version.metadata_json or {})
    has_explicit_training_binding = bool(
        selected_version.dataset_role == "training"
        and (
            isinstance(selected_metadata.get("annotation_model"), dict)
            or str(selected_metadata.get("annotation_model_path") or "").strip()
        )
    )
    binding_mode = str(
        selected_metadata.get("model_binding_mode")
        or selected_metadata.get("annotation_model_selection_mode")
        or ""
    ).strip().lower()
    model_locked = bool(
        has_explicit_training_binding
        and annotation_model_binding
        and binding_mode != "reselectable"
    )
    if model_locked and not annotation_model_matches(config.model_name, annotation_model_binding):
        raise HTTPException(
            status_code=409,
            detail={
                "code": "training_model_mismatch",
                "message": "训练模型必须使用该训练数据集导出时绑定的模型",
                "annotation_model": annotation_model_binding,
            },
        )
    model_record = db.query(ModelRegistry).filter(ModelRegistry.path == config.model_name).first()
    if model_record is None and Path(config.model_name).name == config.model_name:
        matches = (
            db.query(ModelRegistry)
            .filter(ModelRegistry.name == config.model_name)
            .order_by(ModelRegistry.updated_at.desc())
            .limit(2)
            .all()
        )
        model_record = matches[0] if len(matches) == 1 else None
    model_metadata = dict(model_record.metadata_json or {}) if model_record else {}
    if model_record is not None and model_record.source == "trained":
        model_metadata = merge_preserved_model_metadata(
            trained_model_metadata(Path(model_record.path)),
            model_metadata,
            preserve_all=True,
        )
    try:
        profile = resolve_training_profile(
            config.training_profile,
            config.model_name,
            model_metadata.get("base_model"),
            model_metadata,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    source_version = (
        _latest_annotation_source(db, selected_version)
        if profile == "yoloe_pe_segment" and config.prepare_latest_annotations
        else selected_version
    )
    lineage_version = source_version
    for _ in range(3):
        if lineage_version.dataset_role == "original":
            retirement = (lineage_version.metadata_json or {}).get("annotation_retirement_last") or {}
            if isinstance(retirement, dict) and retirement.get("at"):
                try:
                    retired_at = datetime.fromisoformat(str(retirement["at"]))
                except ValueError:
                    raise HTTPException(status_code=409, detail="类别停用时间记录无效，请检查数据集版本")
                if source_version.created_at is None or source_version.created_at <= retired_at:
                    raise HTTPException(status_code=409, detail="当前训练数据仍是类别删除前的版本，请重新导出标注训练集")
            break
        if not lineage_version.parent_version_id:
            break
        parent = db.query(DatasetVersion).filter(DatasetVersion.id == lineage_version.parent_version_id).first()
        if parent is None:
            break
        lineage_version = parent
    if source_version.dataset_role != "training":
        require_dataset_read(db, current_user, source_version.dataset_id, source_version.dataset_role)
    analysis = analyze_training_version(
        db,
        source_version,
        required_geometry="polygon" if profile == "yoloe_pe_segment" else None,
        major_class_threshold=config.major_class_threshold,
    )
    fixed_training_classes = None
    if get_inactive_annotation_class_ids(source_version):
        fixed_training_classes = get_training_model_class_names(db, config.model_name)
        if not fixed_training_classes:
            raise HTTPException(status_code=409, detail="旧模型未记录训练类别顺序，无法保证安全续训")
        source_classes = ordered_class_names(source_version.class_map_json)
        checkpoint_keys = {annotation_class_key(name) for name in fixed_training_classes}
        unsupported = [item["name"] for item in analysis["class_stats"]
                       if item["instances"] > 0 and annotation_class_key(item["name"]) not in checkpoint_keys]
        try:
            resolve_fixed_training_class_map(source_classes, fixed_training_classes, analysis["class_stats"],
                                             get_inactive_annotation_class_ids(source_version))
        except ValueError as exc:
            raise HTTPException(status_code=409, detail={
                "code": "checkpoint_class_mismatch",
                "message": "旧模型类别与当前有效标注无法一一对应",
                "unmapped_annotations": unsupported,
                "reason": str(exc),
            })
        if profile != "yoloe_pe_segment" or not config.prepare_latest_annotations:
            if source_classes != fixed_training_classes:
                raise HTTPException(status_code=409, detail="当前训练数据集类别顺序与旧模型不同，请使用可按旧模型类别表重新生成训练数据的流程")
    if int(config.major_class_threshold) != 20:
        analysis["errors"].append(
            {
                "code": "major_class_threshold_fixed",
                "message": "大类阈值固定为实例数大于 20",
            }
        )
    effective = normalize_training_config(config.dict(), profile)
    dataset_image_size = _dataset_optimal_image_size(selected_version)
    if dataset_image_size:
        effective["img_size"] = dataset_image_size["optimal_size"]
    model_lineage = build_model_lineage(
        config.model_name,
        model_metadata,
        profile,
        parent_registry_id=model_record.id if model_record else None,
    )
    preflight = {
        **analysis,
        "model_family": "yoloe" if profile == "yoloe_pe_segment" else "yolo",
        "task": "segment" if profile == "yoloe_pe_segment" else "auto",
        "training_profile": profile,
        "model_lineage": model_lineage,
        "selected_dataset_version_id": selected_version.id,
        "source_dataset_version_id": source_version.id,
        "source_content_hash": source_version.content_hash,
        "using_latest_annotations": source_version.id != selected_version.id,
        "fixed_training_classes": fixed_training_classes,
        "annotation_model": annotation_model_binding,
        "model_locked": model_locked,
        "model_binding_mode": "fixed" if model_locked else "reselectable",
        "valid_classes": [item for item in analysis["class_stats"] if not item["is_zero"]],
        "zero_sample_classes": [item for item in analysis["class_stats"] if item["is_zero"]],
        "sparse_classes": [item for item in analysis["class_stats"] if item["is_sparse"]],
        "major_classes": [item for item in analysis["class_stats"] if item["is_major"]],
        "planned_split": {"train": 0.8, "val": 0.1, "test": 0.1, "seed": 42},
        "recommended_config": recommended_training_config(profile),
        "image_size": dataset_image_size.get("summary") if dataset_image_size else {},
        "optimal_image_size": dataset_image_size.get("optimal_size") if dataset_image_size else None,
        "image_size_source": dataset_image_size.get("source") if dataset_image_size else None,
        "effective_config": {
            key: effective[key]
            for key in (
                "epochs",
                "batch_size",
                "img_size",
                "learning_rate",
                "optimizer",
                "patience",
                "major_class_threshold",
            )
        },
    }
    return preflight, requested_path, selected_version, source_version


@app.post("/training/preflight")
@app.post("/api/training/preflight")
def training_preflight(
    config: TrainingConfig,
    dataset_path: str,
    request: Request,
    db=Depends(get_db),
):
    preflight, _, _, _ = _build_training_preflight(
        db,
        config,
        dataset_path,
        get_request_current_user(request),
    )
    return preflight


@app.post("/train")
@app.post("/api/training-jobs")
@serialize_training_submission
def start_training(
    config: TrainingConfig,
    dataset_path: str,
    request: Request,
    db=Depends(get_db),
):
    """
    开始模型训练（使用独立进程）

    参数:
    - config: 训练配置参数 (TrainingConfig)
    - dataset_path: 数据集所在的根目录路径
    
    返回:
    - task_id: 生成的任务唯一标识符
    - status: 初始状态 (pending)
    - message: 启动结果消息
    """
    
    requested_device = str(config.device or "auto").strip().lower()
    gpu_waiting = training_device_requires_gpu(requested_device) and not gpu_runtime_is_ready(probe_gpu())

    if not training_worker_is_available():
        raise HTTPException(
            status_code=503,
            detail={
                "code": "training_worker_unavailable",
                "message": "训练服务异常，请等待 Training Worker 恢复后重试",
            },
        )

    active_training = db.query(JobTask).filter(JobTask.type == "training", JobTask.status.in_(ACTIVE_STATUSES)).count()
    if active_training >= 1:
        raise HTTPException(status_code=409, detail="GPU 训练并发上限为 1，请等待当前任务结束")

    current_user = get_request_current_user(request)
    preflight, requested_path, selected_version, source_version = _build_training_preflight(
        db,
        config,
        dataset_path,
        current_user,
    )
    if preflight["errors"]:
        raise HTTPException(
            status_code=400,
            detail={"code": "training_preflight_failed", "preflight": preflight},
        )

    task_id = str(uuid.uuid4())
    version = selected_version
    training_workspace_root = None
    reuse_prepared_dataset = can_reuse_prepared_segmentation_dataset(
        selected_version, source_version, preflight["training_profile"]
    )
    if reuse_prepared_dataset:
        preflight = {
            **preflight,
            "reused_prepared_dataset": True,
            "ephemeral_workspace": False,
            "split_config": dict(selected_version.split_config_json or {}),
        }
    if preflight["training_profile"] == "yoloe_pe_segment" and not reuse_prepared_dataset:
        try:
            version, export_summary, _ = export_training_version(
                db,
                source_version=source_version,
                output_name=f"task_{task_id}",
                training_root=TRAINING_WORKSPACES_DIR,
                staging_root=TRAINING_WORKSPACES_DIR / ".staging",
                manifest_root=DATA_DIR / "manifests",
                created_by=str(current_user.get("username") or get_request_current_username(request)),
                seed=42,
                train_ratio=0.8,
                drop_zero_classes=not bool(preflight.get("fixed_training_classes")),
                fixed_class_names=preflight.get("fixed_training_classes"),
                required_geometry="polygon",
                major_class_threshold=config.major_class_threshold,
                persist_version=False,
            )
        except ValueError as exc:
            db.rollback()
            raise HTTPException(status_code=400, detail=str(exc))
        training_workspace_root = str(Path(version.root_path).resolve())
        requested_path = Path(version.root_path) / "dataset.yaml"
        preflight = {
            **preflight,
            "prepared_dataset_version_id": version.id,
            "prepared_dataset_id": version.dataset_id,
            "content_hash": version.content_hash,
            "split_config": export_summary["split_config"],
            "ephemeral_workspace": True,
        }
        prepared_image_size = (export_summary.get("image_size") or {}).get("optimal_size")
        if prepared_image_size in SUPPORTED_IMAGE_SIZES:
            preflight["image_size"] = export_summary.get("image_size") or {}
            preflight["optimal_image_size"] = prepared_image_size
            preflight["image_size_source"] = "prepared_training_workspace"
            preflight["effective_config"] = {
                **dict(preflight.get("effective_config") or {}),
                "img_size": prepared_image_size,
            }

    config_payload = config.dict()
    if preflight.get("optimal_image_size") in SUPPORTED_IMAGE_SIZES:
        config_payload["img_size"] = preflight["optimal_image_size"]
    config_data = normalize_training_config(config_payload, preflight["training_profile"])
    config_data["_dataset_version_id"] = selected_version.id
    config_data["_training_preflight"] = preflight
    config_data["_model_lineage"] = preflight["model_lineage"]
    if training_workspace_root:
        config_data["_training_workspace_root"] = training_workspace_root
    try:
        job = create_job(
            db,
            job_type="training",
            owner_username=str(current_user.get("username") or get_request_current_username(request)),
            owner_id=current_user.get("id"),
            input_data={
                "dataset_path": str(requested_path),
                "dataset_version_id": selected_version.id,
                "config": config_data,
            },
            task_id=task_id,
            message=WAIT_MESSAGE if gpu_waiting else "训练任务已创建",
        )
    except Exception:
        if training_workspace_root:
            cleanup_training_workspace(
                training_workspace_root,
                workspace_root=TRAINING_WORKSPACES_DIR,
            )
        raise

    return {
        "task_id": task_id,
        "status": "pending",
        "message": WAIT_MESSAGE if gpu_waiting else "训练任务已进入 Worker 队列",
        "training_profile": preflight["training_profile"],
        "dataset_version_id": selected_version.id,
        "preflight": preflight,
    }


# --- 推理与检测接口 ---

@app.post("/predict")
@gpu_http_guard
async def predict_image(
    request: Request,
    db=Depends(get_db),
    file: UploadFile = File(...),
    model_path: str = "yolov8n.pt",
    training_task_id: Optional[str] = None,
    confidence: float = 0.25,
    iou_threshold: float = 0.7,
    max_detections: int = 100,
    max_det: Optional[int] = None,
    save_results: bool = True
):
    """
    单张图片推理检测
    
    参数:
    - file: 上传的图片文件
    - model_path: 使用的模型路径或名称
    - config: 推理配置参数 (InferenceConfig)
    
    返回:
    - prediction_id: 推理任务ID
    - status: 完成状态
    - results: 包含检测详情、结果图片路径等
    """
    
    if training_task_id:
        task = db.query(JobTask).filter(
            JobTask.id == str(training_task_id), JobTask.type == "training"
        ).first()
        if task is None:
            raise HTTPException(status_code=404, detail="训练任务不存在")
        require_job_visibility(task, request)
        model_path, _ = resolve_training_test_model(task, db, resolve_managed_model_path)

    # 检查文件类型
    if not (file.content_type or "").lower().startswith('image/'):
        raise HTTPException(status_code=400, detail="只支持图片文件")

    config = normalize_inference_config(
        confidence=confidence,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
        max_det=max_det,
        save_results=save_results
    )

    logger.info(
        f"[PREDICT CONFIG] model={model_path}, "
        f"confidence={config.confidence}, "
        f"iou_threshold={config.iou_threshold}, "
        f"max_detections={config.max_detections}, "
        f"save_results={config.save_results}"
    )
    predict_temp_dir = None
    try:
        # 单图推理的临时文件也保留上传原始文件名，避免 YOLO 输出和页面显示变成随机临时名。
        prediction_id = generate_task_id()
        predict_temp_dir = Path(constants.TEMP_DIR) / f"predict_{prediction_id}"
        tmp_path = await asyncio.to_thread(
            save_upload_file_keep_name,
            file,
            predict_temp_dir,
            fallback=sanitize_original_filename(file.filename, "image.jpg"),
            max_bytes=IMAGE_UPLOAD_MAX_BYTES,
        )

        # 加载模型(使用InferenceManager的缓存)
        trainer = inference_manager.get_or_load_model(model_path)

        # 执行推理
        results = trainer.predict(
            source=str(tmp_path),
            save_dir="/app/runs/predict",
            name=f"prediction_{prediction_id}",
            conf=config.confidence,
            iou=config.iou_threshold,
            save_results=config.save_results
        )
        
        # 清理临时目录
        file_manager.cleanup_dir(str(predict_temp_dir))
        
        # 提取检测详情
        detections = []
        if results and len(results) > 0:
            result = results[0]
            if hasattr(result, 'boxes') and result.boxes is not None:
                detections = inference_manager._extract_detections(result.boxes, result)
        
        # 使用ResultManager复制结果到静态目录；同时加入本接口自己的兜底复制逻辑。
        # 兜底逻辑可以避免前端拿到 2010_006324.jpg_annotated 这种不可访问字符串。
        try:
            result_images, annotated_images = result_manager.copy_prediction_results(
                prediction_id,
                result_type="predict"  # YOLO使用predict目录,不是prediction
            )
        except Exception as e:
            logger.warning(f"ResultManager复制单图推理结果失败，改用兜底逻辑: {e}")
            result_images, annotated_images = [], []

        fallback_annotated_url = _build_single_annotated_image_url(
            prediction_id,
            results,
            file.filename
        )

        valid_annotated_images = []
        if fallback_annotated_url:
            valid_annotated_images.append(fallback_annotated_url)
        valid_annotated_images.extend(_extract_valid_url_list(annotated_images))
        valid_annotated_images.extend(_extract_valid_url_list(result_images))

        # 去重并保持顺序
        valid_annotated_images = list(dict.fromkeys(valid_annotated_images))
        annotated_image_url = valid_annotated_images[0] if valid_annotated_images else None

        return {
            "prediction_id": prediction_id,
            "status": constants.TASK_STATUS_COMPLETED,
            "results": {
                "file_name": file.filename,
                "original_name": file.filename,
                "result_images": valid_annotated_images,
                "annotated_images": valid_annotated_images,
                "annotated_image": annotated_image_url,
                "annotated_image_url": annotated_image_url,
                "detections": detections,
                "detections_count": len(detections),
                "confidence_threshold": config.confidence,
                "iou_threshold": config.iou_threshold,
                "max_detections": config.max_detections,
                "save_results": config.save_results,
                "model_used": model_path
            }
        }
        
    except HTTPException:
        if predict_temp_dir:
            file_manager.cleanup_dir(str(predict_temp_dir))
        raise
    except Exception as e:
        if predict_temp_dir:
            file_manager.cleanup_dir(str(predict_temp_dir))
        raise HTTPException(status_code=500, detail=f"推理失败: {str(e)}")


@app.post("/predict/batch")
@gpu_http_guard
async def predict_batch(
    request: Request,
    db=Depends(get_db),
    files: List[UploadFile] = File(...),
    model_path: str = "yolov8n.pt",
    training_task_id: Optional[str] = None,
    confidence: float = 0.25,
    iou_threshold: float = 0.7,
    max_detections: int = 100,
    max_det: Optional[int] = None,
    save_results: bool = True
):
    """
    批量图片推理检测

    参数:
    - files: 上传的多个图片文件列表
    - model_path: 使用的模型路径
    - config: 推理配置参数

    返回:
    - batch_id: 批次ID
    - status: 完成状态
    - results: 包含处理数量、逐图结果、总检测数等
    """

    if training_task_id:
        task = db.query(JobTask).filter(
            JobTask.id == str(training_task_id), JobTask.type == "training"
        ).first()
        if task is None:
            raise HTTPException(status_code=404, detail="训练任务不存在")
        require_job_visibility(task, request)
        model_path, _ = resolve_training_test_model(task, db, resolve_managed_model_path)

    config = normalize_inference_config(
        confidence=confidence,
        iou_threshold=iou_threshold,
        max_detections=max_detections,
        max_det=max_det,
        save_results=save_results
    )

    logger.info(
        f"[BATCH PREDICT CONFIG] model={model_path}, "
        f"confidence={config.confidence}, "
        f"iou_threshold={config.iou_threshold}, "
        f"max_detections={config.max_detections}, "
        f"save_results={config.save_results}"
    )

    if len(files) > constants.MAX_BATCH_IMAGES:
        raise HTTPException(
            status_code=400,
            detail=f"批量推理最多支持{constants.MAX_BATCH_IMAGES}张图片"
        )

    temp_dir = None
    try:
        # 创建临时目录
        batch_id = generate_task_id()
        temp_dir = file_manager.create_temp_dir(prefix=f"batch_{batch_id}_")

        # 保存所有上传文件，并记录“保存文件名 -> 原始文件名”映射。
        # 这里不再强制改成 image_0.jpg，而是保留上传文件名。
        saved_files = []
        file_name_map = {}  # {"用户原始文件名.jpg": "用户原始文件名.jpg"}
        total_upload_bytes = 0

        for i, file in enumerate(files):
            if not (file.content_type or "").lower().startswith('image/'):
                continue

            saved_path = await asyncio.to_thread(
                save_upload_file_keep_name,
                file,
                temp_dir,
                fallback=f"image_{i}.jpg",
                max_bytes=IMAGE_UPLOAD_MAX_BYTES,
            )
            total_upload_bytes += saved_path.stat().st_size
            if total_upload_bytes > BATCH_INFERENCE_MAX_TOTAL_BYTES:
                raise HTTPException(status_code=413, detail="批量推理上传总大小超过限制")
            temp_file_name = Path(saved_path).name

            saved_files.append(str(saved_path))
            file_name_map[temp_file_name] = file.filename

        if not saved_files:
            raise HTTPException(status_code=400, detail="没有有效的图片文件")

        # 加载模型
        trainer = inference_manager.get_or_load_model(model_path)

        # 批量推理：显式指定保存路径
        results = trainer.predict(
            source=temp_dir,
            save_dir="/app/runs/batch",
            name=f"batch_{batch_id}",
            conf=config.confidence,
            iou=config.iou_threshold,
            save_results=config.save_results
        )

        # 复制结果图到静态目录，并拿到“文件名 -> URL”的映射。
        # ResultManager 在部分版本中可能返回 xxx.jpg_annotated 这样的非 URL 字符串，
        # 因此这里额外用 YOLO 实际输出目录构造一份可靠 URL 映射，并优先使用它。
        try:
            manager_annotated_image_map = result_manager.copy_batch_result_images(batch_id) or {}
        except Exception as e:
            logger.warning(f"ResultManager复制批量推理结果失败，改用兜底逻辑: {e}")
            manager_annotated_image_map = {}

        fallback_annotated_image_map = _build_batch_annotated_image_map(
            batch_id,
            results,
            file_name_map
        )

        annotated_image_map = {
            **manager_annotated_image_map,
            **fallback_annotated_image_map
        }

        # 构造逐图返回结果
        items = []
        total_detections = 0

        if results:
            for idx, result in enumerate(results):
                # YOLO 的 result.path 是输入图片路径
                input_path = getattr(result, "path", "") or ""
                file_name = os.path.basename(input_path) if input_path else f"image_{idx}.jpg"
                original_name = file_name_map.get(file_name, file_name)

                detections = []
                if hasattr(result, 'boxes') and result.boxes is not None:
                    detections = inference_manager._extract_detections(result.boxes, result)

                detections_count = len(detections)
                total_detections += detections_count

                annotated_image_url = (
                    annotated_image_map.get(file_name)
                    or annotated_image_map.get(original_name)
                    or annotated_image_map.get(Path(file_name).stem)
                    or annotated_image_map.get(Path(original_name).stem)
                )

                if not _is_http_or_static_url(annotated_image_url):
                    annotated_image_url = None

                items.append({
                    "index": idx,
                    "file_name": original_name,         # 前端展示原始文件名
                    "temp_file_name": file_name,        # 临时保存名，例如 image_0.jpg
                    "original_name": original_name,     # 用户上传时的原始文件名
                    "annotated_image": annotated_image_url,
                    "annotated_image_url": annotated_image_url,
                    "detections": detections,
                    "detections_count": detections_count
                })

        # 可选：保留旧字段，兼容旧前端。只返回真正可访问的 URL。
        result_images = list(dict.fromkeys(_extract_valid_url_list(annotated_image_map)))

        # 清理临时目录
        file_manager.cleanup_dir(temp_dir)

        return {
            "batch_id": batch_id,
            "status": constants.TASK_STATUS_COMPLETED,
            "results": {
                "processed_images": len(saved_files),
                "total_detections": total_detections,
                "confidence_threshold": config.confidence,
                "iou_threshold": config.iou_threshold,
                "max_detections": config.max_detections,
                "save_results": config.save_results,
                "model_used": model_path,
                "result_images": result_images,   # 兼容旧前端
                "items": items                    # 新增：逐图详情
            }
        }

    except HTTPException:
        if temp_dir:
            file_manager.cleanup_dir(temp_dir)
        raise
    except Exception as e:
        if temp_dir:
            file_manager.cleanup_dir(temp_dir)
        raise HTTPException(status_code=500, detail=f"批量推理失败: {str(e)}")


# --- 模型导出接口 ---

@app.post("/export")
async def export_model(
    model_path: str,
    config: ExportConfig = ExportConfig()
):
    """
    将训练好的模型导出为其他格式（如 ONNX, TFLite 等）
    
    参数:
    - model_path: 源模型路径 (.pt)
    - config: 导出配置 (ExportConfig)
    """
    
    if not os.path.exists(model_path):
        raise HTTPException(status_code=404, detail="模型文件不存在")
    
    try:
        # 加载模型
        trainer = YOLOTrainer(model_name=model_path)
        trainer.load_model()
        
        # 导出模型
        export_id = generate_task_id()
        export_dir = Path(constants.EXPORTS_DIR) / f"export_{export_id}"
        
        trainer.export_model(
            format=config.format,
            save_dir=str(export_dir)
        )
        
        # 查找导出的文件
        exported_files = []
        if export_dir.exists():
            for file_path in export_dir.glob("*"):
                if file_path.suffix in ['.onnx', '.pt', '.tflite', '.mlmodel']:
                    exported_files.append(str(file_path))
        
        return {
            "export_id": export_id,
            "status": constants.TASK_STATUS_COMPLETED,
            "format": config.format,
            "exported_files": exported_files,
            "message": f"模型已成功导出为 {config.format} 格式"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"模型导出失败: {str(e)}")


# --- 训练状态与日志接口 ---

@app.get("/status/{task_id}", response_model=TaskStatus)
async def get_training_status(task_id: str, request: Request, db=Depends(get_db)):
    """
    查询指定训练任务的实时状态
    """
    
    task = db.query(JobTask).filter(JobTask.id == task_id, JobTask.type == "training").first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    require_job_visibility(task, request)
    payload = task.to_dict()
    payload["test_model"] = describe_training_test_model(task, db, resolve_managed_model_path)
    return TaskStatus(**payload)


@app.post("/stop/{task_id}")
async def stop_training(task_id: str, request: Request, db=Depends(get_db)):
    """
    强制停止正在进行的训练任务
    """
    
    try:
        task = db.query(JobTask).filter(JobTask.id == task_id, JobTask.type == "training").first()
        if not task:
            raise HTTPException(status_code=404, detail="任务不存在")
        require_job_visibility(task, request)
        task = request_job_cancel(db, task_id)
        return {
            "task_id": task_id,
            "status": task.status,
            "message": task.message,
        }
        
    except Exception as e:
        logger.error(f"停止训练任务失败: {e}")
        raise HTTPException(status_code=500, detail=f"停止训练任务失败: {str(e)}")


@app.get("/logs/{task_id}")
async def get_training_logs(task_id: str, request: Request, lines: int = 50, after_sequence: int = 0, db=Depends(get_db)):
    """
    获取训练任务的最后 N 行日志内容
    """
    
    task = db.query(JobTask).filter(JobTask.id == task_id, JobTask.type == "training").first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    require_job_visibility(task, request)
    lines = max(1, min(int(lines), 1000))
    events = (
        db.query(JobEvent)
        .filter(JobEvent.task_id == task_id, JobEvent.sequence > max(0, int(after_sequence)))
        .order_by(JobEvent.sequence.asc())
        .limit(lines)
        .all()
    )
    return {
        "task_id": task_id,
        "returned_lines": len(events),
        "next_sequence": events[-1].sequence if events else max(0, int(after_sequence)),
        "logs": [event.message for event in events],
        "events": [event.to_dict() for event in events],
    }


@app.websocket("/ws/logs/{task_id}")
async def training_logs_websocket(websocket: WebSocket, task_id: str):
    """
    通过 WebSocket 实时推送训练日志
    """
    behavior_owner = await authenticate_websocket(websocket)
    if behavior_owner is None:
        return

    db = SessionLocal()
    try:
        task = db.query(JobTask).filter(JobTask.id == task_id, JobTask.type == "training").first()
    finally:
        db.close()
    if not task:
        await websocket.close(code=4404, reason="task not found")
        return
    if behavior_owner.get("role") != "admin" and task.owner_username != str(behavior_owner.get("username") or ""):
        await websocket.close(code=4404, reason="task not found")
        return

    await websocket.accept()

    log_file = TRAINING_LOG_ROOT / f"{task.id}.log"
    last_position = 0
    
    try:
        while True:
            if log_file.exists():
                try:
                    # 检查文件是否有新内容
                    with open(log_file, 'r', encoding='utf-8') as f:
                        f.seek(last_position)
                        new_lines = f.readlines()
                        last_position = f.tell()
                    
                    # 发送新的日志行
                    if new_lines:
                        for line in new_lines:
                            await websocket.send_text(line.strip())
                    
                except Exception as e:
                    logger.error(f"读取日志文件失败: {e}")
            
            # 等待一秒后继续检查
            await asyncio.sleep(1)
            
    except WebSocketDisconnect:
        logger.info(f"训练日志WebSocket断开连接: {task_id}")
    except Exception as e:
        logger.error(f"训练日志WebSocket错误: {e}")
        await websocket.close()


@app.get("/tasks")
async def list_tasks(request: Request, db=Depends(get_db)):
    """
    列出所有历史和当前的训练任务
    """
    
    query = db.query(JobTask).filter(JobTask.type == "training")
    current_user = get_request_current_user(request)
    if current_user.get("role") != "admin":
        query = query.filter(JobTask.owner_username == str(current_user.get("username") or ""))
    tasks = query.order_by(JobTask.created_at.desc()).limit(200).all()
    task_ids = [task.id for task in tasks]
    registry_ids = [
        str((task.result_json or {}).get("model_registry_id") or "").strip()
        for task in tasks
        if str((task.result_json or {}).get("model_registry_id") or "").strip()
    ]
    model_paths = [
        str((task.result_json or {}).get("model_path") or "").strip()
        for task in tasks
        if str((task.result_json or {}).get("model_path") or "").strip()
    ]
    model_records = []
    if task_ids or registry_ids or model_paths:
        model_records = db.query(ModelRegistry).filter(
            or_(
                ModelRegistry.task_id.in_(task_ids),
                ModelRegistry.id.in_(registry_ids),
                ModelRegistry.path.in_(model_paths),
            )
        ).all()
    payloads = []
    for task in tasks:
        payload = task.to_dict()
        payload["test_model"] = describe_training_test_model(
            task,
            db,
            resolve_managed_model_path,
            model_records,
        )
        payloads.append(payload)
    return {
        "total_tasks": len(tasks),
        "tasks": payloads
    }


# --- 模型库管理接口 ---

MODEL_UPLOAD_ENABLED = os.getenv("ENABLE_MODEL_UPLOAD", "false").strip().lower() in {"1", "true", "yes"}
MODEL_UPLOAD_MAX_BYTES = int(os.getenv("MODEL_UPLOAD_MAX_BYTES", str(1024**3)))


def find_active_model_record(
    db,
    model_ref: str,
    *,
    allowed_statuses: Tuple[str, ...] = ("active",),
) -> Optional[ModelRegistry]:
    """Find one permitted registry entry without allowing ambiguous filename deletes."""
    raw = str(model_ref or "").strip().replace("\\", "/")
    if not raw:
        return None

    record = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.status.in_(allowed_statuses))
        .filter((ModelRegistry.id == raw) | (ModelRegistry.path == raw))
        .first()
    )
    if record or "/" in raw:
        return record

    matches = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.status.in_(allowed_statuses), ModelRegistry.name == Path(raw).name)
        .limit(2)
        .all()
    )
    return matches[0] if len(matches) == 1 else None


def resolve_managed_model_path(
    model_path: str,
    registry_record: Optional[ModelRegistry] = None,
    *,
    require_file: bool = True,
) -> Path:
    """Resolve a model only inside managed model/run roots; never fall back to cwd."""
    raw = str(model_path or "").strip().replace("\\", "/")
    if not raw or ".." in Path(raw).parts:
        raise HTTPException(status_code=400, detail="模型路径不合法")

    roots = [
        MODELS_DIR.resolve(), RUNS_DIR.resolve(),
        Path("/app/data/models"), Path("/app/data/runs"),
        Path("/app/models"), Path("/app/runs"),
    ]
    requested = Path(raw)
    candidates = []
    if registry_record is not None:
        candidates.append(Path(registry_record.path))
    candidates.extend([requested] if requested.is_absolute() else [
        MODELS_DIR / requested,
        RUNS_DIR / requested,
        RUNS_DIR / "train" / requested,
    ])
    if not requested.is_absolute() and len(requested.parts) == 2:
        candidates.append(RUNS_DIR / "train" / requested.parts[0] / "weights" / requested.parts[1])

    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved.suffix.lower() not in {".pt", ".onnx"}:
            continue
        if not any(_safe_relative_to(resolved, root) for root in roots):
            continue
        if not require_file or resolved.is_file():
            return resolved
    raise HTTPException(status_code=404, detail="模型文件不存在或不在受管目录")


def resolve_registered_model_path(record: ModelRegistry) -> Path:
    return resolve_managed_model_path(record.path, record, require_file=False)


def _update_model_export_job(task_id: str, **kwargs) -> None:
    db = SessionLocal()
    try:
        update_job(db, task_id, **kwargs)
    finally:
        db.close()


def run_model_export_task(task_id: str) -> None:
    """Run one persisted PT-to-ONNX conversion without mutating the source PT."""
    db = SessionLocal()
    try:
        task = db.query(JobTask).filter(JobTask.id == task_id, JobTask.type == "model_export").first()
        if task is None:
            return
        input_data = dict(task.input_json or {})
        if task.cancel_requested or task.status == "cancelling":
            update_job(db, task_id, status="cancelled", stage="cancelled", message="模型转换任务已取消")
            return

        source_model_id = str(input_data.get("source_model_id") or "").strip()
        source_record = (
            db.query(ModelRegistry)
            .filter(
                ModelRegistry.id == source_model_id,
                ModelRegistry.status.in_(("active", "candidate")),
            )
            .first()
        )
        if source_record is None or source_record.source != "trained":
            raise ValueError("仅支持转换已注册的训练模型")

        source_path = resolve_managed_model_path(source_record.path, source_record)
        if source_path.suffix.lower() != ".pt":
            raise ValueError("源模型必须是 .pt 文件")
        onnx_path = source_path.with_suffix(".onnx")
        existing_onnx = db.query(ModelRegistry).filter(ModelRegistry.path == str(onnx_path).replace("\\", "/")).first()
        if existing_onnx is not None and existing_onnx.status == "deleted":
            raise ValueError("目标 ONNX 文件已被删除，请先清理同名文件后重试")

        source_record_id = source_record.id
        source_task_id = source_record.task_id
        source_dataset_version_id = source_record.dataset_version_id
        owner_username = str(task.owner_username or "admin")
        source_metadata = dict(source_record.metadata_json or {})
        imgsz = max(32, int(input_data.get("imgsz") or 640))

        if existing_onnx is not None and existing_onnx.status in {"active", "candidate"} and onnx_path.is_file():
            result = {
                "source_model_id": source_record_id,
                "source_pt_path": str(source_path),
                "onnx_model_id": existing_onnx.id,
                "onnx_path": str(onnx_path),
                "format": "onnx",
                "already_exists": True,
            }
            db.close()
            db = None
            _update_model_export_job(
                task_id,
                status="completed",
                stage="completed",
                progress=100,
                message="ONNX 模型已存在，无需重复转换",
                result=result,
            )
            return

        db.close()
        db = None

        _update_model_export_job(
            task_id,
            status="running",
            stage="loading_model",
            progress=5,
            message="正在加载 PT 模型并准备 ONNX 转换",
        )
        _update_model_export_job(
            task_id,
            stage="exporting_onnx",
            progress=15,
            message="正在转换为 ONNX，源 PT 文件保持不变",
        )
        converted_path = convert_pt_to_onnx(source_path, task_id=task_id, imgsz=imgsz)

        register_db = SessionLocal()
        try:
            metadata = {
                **source_metadata,
                "display_name": source_metadata.get("display_name") or f"{source_path.parent.parent.name} - best",
                "weight_type": source_path.stem,
                "source_model_registry_id": source_record_id,
                "source_pt_path": str(source_path),
                "export_format": "onnx",
                "export_imgsz": imgsz,
            }
            onnx_record = register_model_file(
                register_db,
                converted_path,
                source="trained",
                status="active",
                task_id=source_task_id,
                dataset_version_id=source_dataset_version_id,
                created_by=owner_username,
                metadata=metadata,
            )
            result = {
                "source_model_id": source_record_id,
                "source_pt_path": str(source_path),
                "onnx_model_id": onnx_record.id,
                "onnx_path": str(converted_path),
                "format": "onnx",
                "imgsz": imgsz,
            }
        finally:
            register_db.close()

        _update_model_export_job(
            task_id,
            status="completed",
            stage="completed",
            progress=100,
            message="ONNX 模型转换完成",
            result=result,
        )
    except Exception as exc:
        logger.exception("模型 ONNX 转换失败: task=%s", task_id)
        try:
            _update_model_export_job(
                task_id,
                status="failed",
                stage="failed",
                progress=100,
                message=f"ONNX 转换失败: {exc}",
                error_code="model_export_failed",
                error_detail=str(exc),
            )
        except Exception:
            logger.exception("写入模型 ONNX 转换失败状态失败: task=%s", task_id)
    finally:
        if db is not None:
            db.close()


def is_protected_registered_model(db, record: ModelRegistry, path: Path) -> bool:
    return is_protected_pretrained_model(record.path, path) or is_model_in_active_use(db, record, path)


@app.post("/models/reconcile")
@app.post("/api/models/reconcile")
async def reconcile_model_registry(
    admin_user: Dict[str, Any] = Depends(require_admin),
    db=Depends(get_db),
):
    return reconcile_models(
        db,
        model_root=MODELS_DIR,
        runs_root=RUNS_DIR,
        pretrained_names=PRETRAINED_MODEL_NAMES,
        created_by=str(admin_user.get("username") or "admin"),
    )

@app.post("/models/{model_id}/convert-onnx")
@app.post("/api/models/{model_id}/convert-onnx")
async def start_model_onnx_conversion(
    model_id: str,
    admin_user: Dict[str, Any] = Depends(require_admin),
    db=Depends(get_db),
):
    source_record = (
        db.query(ModelRegistry)
        .filter(
            ModelRegistry.id == model_id,
            ModelRegistry.status.in_(("active", "candidate")),
        )
        .first()
    )
    if source_record is None:
        raise HTTPException(status_code=404, detail="训练模型不存在")
    if source_record.source != "trained" or str(source_record.format or "").lower() != "pt":
        raise HTTPException(status_code=409, detail="仅支持将训练模型的 .pt 文件转换为 ONNX")

    source_path = resolve_managed_model_path(source_record.path, source_record)
    onnx_path = source_path.with_suffix(".onnx")
    existing_onnx = db.query(ModelRegistry).filter(ModelRegistry.path == str(onnx_path).replace("\\", "/")).first()
    if existing_onnx is not None and existing_onnx.status == "deleted":
        raise HTTPException(status_code=409, detail="同名 ONNX 文件存在删除记录，请先清理后重试")
    if existing_onnx is not None and existing_onnx.status in {"active", "candidate"} and onnx_path.is_file():
        return {
            "task_id": None,
            "status": "completed",
            "already_exists": True,
            "model": existing_onnx.to_dict(),
        }

    request_key = {"source_model_id": source_record.id, "format": "onnx"}
    active_tasks = (
        db.query(JobTask)
        .filter(
            JobTask.type == "model_export",
            JobTask.status.in_(("pending", "running", "cancelling")),
        )
        .order_by(JobTask.created_at.desc())
        .all()
    )
    for task in active_tasks:
        if dict(task.input_json or {}) == request_key:
            return {"task_id": task.id, "status": task.status, "task": task.to_dict(), "deduplicated": True}

    task = create_job(
        db,
        job_type="model_export",
        owner_username=str(admin_user.get("username") or "admin"),
        owner_id=str(admin_user.get("id") or "") or None,
        input_data=request_key,
        message="ONNX 模型转换任务已创建",
    )
    handle = asyncio.create_task(asyncio.to_thread(run_model_export_task, task.id))
    model_export_task_handles[task.id] = handle
    handle.add_done_callback(lambda _task, export_id=task.id: model_export_task_handles.pop(export_id, None))
    return {"task_id": task.id, "status": task.status, "task": task.to_dict()}

@app.post("/models/{model_id}/activation-evaluation")
@app.post("/api/models/{model_id}/activation-evaluation")
async def evaluate_candidate_activation(
    model_id: str,
    payload: CandidateActivationRequest,
    admin_user: Dict[str, Any] = Depends(require_admin),
    db=Depends(get_db),
):
    model = db.query(ModelRegistry).filter(ModelRegistry.id == model_id).first()
    if not model:
        raise HTTPException(status_code=404, detail="候选模型不存在")
    if model.status not in {"candidate", "active"}:
        raise HTTPException(status_code=409, detail="模型当前状态不能执行候选验收")
    try:
        evaluation = evaluate_yoloe_candidate(
            payload.current_best_metrics,
            payload.candidate_metrics,
            payload.smoke_tests,
        )
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    metadata = dict(model.metadata_json or {})
    metadata["activation_evaluation"] = {
        **evaluation,
        "pretrained_metrics": payload.pretrained_metrics,
        "current_best_metrics": payload.current_best_metrics,
        "candidate_metrics": payload.candidate_metrics,
        "smoke_tests": payload.smoke_tests,
        "evaluated_by": str(admin_user.get("username") or "admin"),
        "evaluated_at": datetime.now().isoformat(),
    }
    metadata["activation_status"] = "active" if evaluation["passed"] else "candidate"
    model.metadata_json = metadata
    model.status = "active" if evaluation["passed"] else "candidate"
    model.updated_at = datetime.now()
    db.commit()
    db.refresh(model)
    return {"model": model.to_dict(), "evaluation": evaluation}


@app.get("/models/candidates")
@app.get("/api/models/candidates")
async def list_candidate_models(
    admin_user: Dict[str, Any] = Depends(require_admin),
    db=Depends(get_db),
):
    records = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.status == "candidate")
        .order_by(ModelRegistry.updated_at.desc())
        .all()
    )
    return {"models": [item.to_dict() for item in filter_available_model_records(db, records)]}


@app.get("/models")
@app.get("/api/models")
async def list_models(request: Request, db=Depends(get_db)):
    """
    列出所有可用模型，并按“逻辑模型”去重。

    背景：当前 docker-compose 为了兼容新旧代码，同时可能存在：
    - ./data/models:/app/data/models
    - ./data/models:/app/models
    - ./data/runs:/app/data/runs
    - ./data/runs:/app/runs

    这会导致同一个宿主机 pt 文件在容器内出现两条路径。
    因此不能只按 resolve() 后的容器路径去重，还要按模型类型、文件名、实验名、大小等逻辑信息去重。
    """
    registry_items = (
        db.query(ModelRegistry)
        .filter(ModelRegistry.status.in_(("active", "candidate")))
        .order_by(ModelRegistry.source.asc(), ModelRegistry.updated_at.desc())
        .all()
    )
    registry_items = filter_available_model_records(db, registry_items)
    current_user = get_request_current_user(request)
    registry_items = [
        item for item in registry_items
        if _model_registry_record_visible_to_user(db, item, current_user)
    ]
    registry_by_path = {
        str(Path(record.path).resolve()).replace("\\", "/"): record
        for record in registry_items
    }
    linked_version_ids = {
        str(record.dataset_version_id)
        for record in registry_items
        if record.source == "trained" and record.dataset_version_id
    }
    linked_versions = {
        version.id: version
        for version in (
            db.query(DatasetVersion)
            .filter(DatasetVersion.id.in_(linked_version_ids))
            .all()
            if linked_version_ids
            else []
        )
    }
    payloads = []
    for item in registry_items:
        payload = normalize_visible_model_payload(item.to_dict())
        model_format = str(payload.get("format") or Path(payload.get("name") or "").suffix.lstrip(".") or "pt").lower()
        payload["format"] = model_format
        payload["supports_training"] = model_format == "pt"
        if item.source == "trained":
            discovered = trained_model_metadata(Path(item.path))
            payload = hydrate_trained_model_payload(payload, discovered)
            trained_class_names = ordered_class_names(
                payload.get("trained_class_names")
                or payload.get("class_names")
                or payload.get("names")
            )
            if not trained_class_names:
                linked_version = linked_versions.get(str(item.dataset_version_id or ""))
                trained_class_names = ordered_class_names(
                    linked_version.class_map_json if linked_version else None
                )
            payload["trained_class_names"] = trained_class_names
            payload["class_names"] = trained_class_names
            if model_format == "pt":
                onnx_path = Path(item.path).with_suffix(".onnx").resolve()
                onnx_record = registry_by_path.get(str(onnx_path).replace("\\", "/"))
                payload["onnx_available"] = bool(onnx_path.is_file() or onnx_record is not None)
                payload["onnx_model_id"] = onnx_record.id if onnx_record is not None else None
        else:
            payload["trained_class_names"] = []
            payload["supports_text_prompts"] = (
                bool(payload.get("supports_text_prompts"))
                or model_supports_text_prompts(
                    payload.get("name"),
                    payload.get("path"),
                    payload.get("base_model"),
                )
            )
        payloads.append(payload)
    latest_trained_set = False
    for payload in payloads:
        payload["is_latest"] = (
            payload.get("source") == "trained"
            and payload.get("format") == "pt"
            and not latest_trained_set
        )
        if payload["is_latest"]:
            latest_trained_set = True
    return {
        "total_models": len(payloads),
        "models": payloads,
        "registry_source": "postgresql",
    }

    # 以下旧扫描实现保留一个版本用于回滚参考，正常请求不会执行。
    models: List[Dict[str, Any]] = []
    seen_paths = set()
    seen_logical_keys = set()

    def normalize_model_record_path(path_value: Path) -> str:
        """返回前端可直接提交给后端的模型路径。"""
        return str(Path(path_value)).replace("\\", "/")

    def _path_priority(path_value: Path) -> int:
        """
        路径优先级：优先返回 /app/data 或 data 目录下的统一路径，避免前端拿到旧兼容路径。
        数值越小优先级越高。
        """
        text = normalize_model_record_path(path_value)
        if text.startswith("/app/data/"):
            return 0
        if "/data/" in text or text.startswith("data/"):
            return 1
        if text.startswith("/app/models") or text.startswith("/app/runs"):
            return 2
        return 3

    def _replace_existing_if_better(payload: Dict[str, Any], logical_key: tuple) -> bool:
        """如果同一个逻辑模型已经存在，但新路径更优，则替换已有记录。"""
        for index, item in enumerate(models):
            if item.get("_logical_key") == logical_key:
                old_priority = _path_priority(Path(item.get("path", "")))
                new_priority = _path_priority(Path(payload.get("path", "")))
                old_mtime = float(item.get("modified_time") or 0)
                new_mtime = float(payload.get("modified_time") or 0)
                if (new_priority, -new_mtime) < (old_priority, -old_mtime):
                    models[index] = payload
                return True
        return False

    def add_model_record(
        model_file: Path,
        model_type: str,
        display_name: str = None,
        extra: Dict[str, Any] = None,
        logical_key: tuple = None,
    ) -> None:
        try:
            model_file = Path(model_file)
            if not model_file.exists() or not model_file.is_file() or model_file.suffix.lower() != ".pt":
                return

            stat = model_file.stat()
            normalized_path = normalize_model_record_path(model_file)
            resolved_key = str(model_file.resolve()).replace("\\", "/")

            if logical_key is None:
                logical_key = (model_type, model_file.name.lower(), int(stat.st_size))

            payload = {
                "name": model_file.name,
                "type": model_type,
                "path": normalized_path,
                "size": stat.st_size,
                "created_time": stat.st_ctime,
                "modified_time": stat.st_mtime,
                "display_name": display_name or model_file.stem.replace("_", " "),
                "is_latest": False,
                "_logical_key": logical_key,
            }
            if extra:
                payload.update(extra)

            # 先按逻辑模型去重，解决 /app/models 与 /app/data/models 同时挂载导致的重复。
            if logical_key in seen_logical_keys:
                _replace_existing_if_better(payload, logical_key)
                return

            # 再按容器路径去重，解决同一路径被多个扫描规则扫到。
            if resolved_key in seen_paths:
                return

            seen_logical_keys.add(logical_key)
            seen_paths.add(resolved_key)
            models.append(payload)
        except Exception as e:
            logger.warning("添加模型记录失败: %s, error=%s", model_file, e)

    pretrained_models = list(PRETRAINED_MODEL_NAMES)
    pretrained_name_set = PRETRAINED_MODEL_NAME_SET

    # 先按固定预训练模型名扫描；候选顺序优先统一数据目录。
    for model_name in pretrained_models:
        candidates = [
            PRETRAINED_MODELS_DIR / model_name,
            MODELS_DIR / model_name,
            Path("/app/data/models/pretrained") / model_name,
            Path("/app/data/models") / model_name,
            Path("/app/models/pretrained") / model_name,
            Path("/app/models") / model_name,
            PROJECT_ROOT / model_name,
            BACKEND_DIR / model_name,
            Path(model_name),
        ]
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                add_model_record(
                    candidate,
                    "pretrained",
                    model_name.replace(".pt", "").upper(),
                    logical_key=("pretrained", model_name.lower()),
                )
                break

    # 递归扫描模型目录。保留 /app/models 兼容旧路径，但通过 logical_key 去重。
    # 显式加入 /app/data/models/pretrained，保证行为分析页“物品检测模型路径”能读取该目录下所有 .pt。
    model_roots = [
        PRETRAINED_MODELS_DIR,
        MODELS_DIR,
        Path("/app/data/models/pretrained"),
        Path("/app/data/models"),
        Path("/app/models/pretrained"),
        Path("/app/models"),
    ]
    scanned_model_roots = set()
    for root in model_roots:
        try:
            root = Path(root)
            if not root.exists():
                continue
            root_key = normalize_model_record_path(root.resolve())
            if root_key in scanned_model_roots:
                continue
            scanned_model_roots.add(root_key)

            logger.info("扫描模型目录: %s", root)
            for model_file in root.rglob("*.pt"):
                if not model_file.is_file():
                    continue

                lower_parts = [part.lower() for part in model_file.parts]
                lower_name = model_file.name.lower()

                is_pretrained = (
                    _safe_relative_to(model_file, PRETRAINED_MODELS_DIR)
                    or "pretrained" in lower_parts
                    or lower_name in pretrained_name_set
                )
                is_behavior = (
                    _safe_relative_to(model_file, BEHAVIOR_MODELS_DIR)
                    or "behavior" in lower_parts
                )

                if is_pretrained:
                    model_type = "pretrained"
                    display_name = model_file.stem.upper()
                    logical_key = ("pretrained", lower_name)
                elif is_behavior:
                    model_type = "behavior"
                    display_name = model_file.stem.replace("_", " ")
                    logical_key = ("behavior", lower_name)
                else:
                    model_type = "uploaded"
                    display_name = model_file.stem.replace("_", " ")
                    try:
                        stat_size = int(model_file.stat().st_size)
                    except Exception:
                        stat_size = 0
                    logical_key = ("uploaded", lower_name, stat_size)

                add_model_record(model_file, model_type, display_name, logical_key=logical_key)
        except Exception as e:
            logger.warning("扫描模型目录失败: %s, error=%s", root, e)

    # 训练得到的模型：递归扫描 runs/**/weights/best.pt。
    trained_models: List[Dict[str, Any]] = []
    run_roots = [RUNS_DIR, Path("/app/data/runs"), Path("/app/runs"), PROJECT_ROOT / "runs"]
    scanned_run_roots = set()

    for runs_root in run_roots:
        try:
            runs_root = Path(runs_root)
            if not runs_root.exists():
                continue
            root_key = normalize_model_record_path(runs_root.resolve())
            if root_key in scanned_run_roots:
                continue
            scanned_run_roots.add(root_key)

            for weight_file in runs_root.rglob("weights/*.pt"):
                if not weight_file.is_file() or weight_file.name != "best.pt":
                    continue

                stat = weight_file.stat()
                exp_dir = weight_file.parent.parent
                logical_key = ("trained", exp_dir.name.lower(), weight_file.name.lower(), int(stat.st_size))

                if logical_key in seen_logical_keys:
                    # 同一个训练实验在 /app/runs 与 /app/data/runs 下重复出现时，只保留一条。
                    continue

                resolved_key = str(weight_file.resolve()).replace("\\", "/")
                if resolved_key in seen_paths:
                    continue

                dataset_info = "未知数据集"
                epochs_info = ""
                args_file = exp_dir / "args.yaml"
                if args_file.exists():
                    try:
                        import yaml
                        with open(args_file, "r", encoding="utf-8") as f:
                            args_data = yaml.safe_load(f) or {}
                        dataset_path = args_data.get("data")
                        if isinstance(dataset_path, str):
                            dataset_info = Path(dataset_path).stem
                        if "epochs" in args_data:
                            epochs_info = f" ({args_data['epochs']}轮)"
                    except Exception as e:
                        logger.warning("读取训练参数失败: %s, error=%s", args_file, e)

                seen_logical_keys.add(logical_key)
                seen_paths.add(resolved_key)
                trained_models.append({
                    "name": f"{exp_dir.name}/{weight_file.name}",
                    "type": "trained",
                    "path": normalize_model_record_path(weight_file),
                    "size": stat.st_size,
                    "created_time": stat.st_ctime,
                    "modified_time": stat.st_mtime,
                    "display_name": f"{exp_dir.name} - {weight_file.stem}",
                    "experiment_name": exp_dir.name,
                    "weight_type": weight_file.stem,
                    "dataset": dataset_info,
                    "epochs": epochs_info,
                    "is_latest": False,
                    "_logical_key": logical_key,
                })
        except Exception as e:
            logger.warning("扫描训练模型目录失败: %s, error=%s", runs_root, e)

    trained_models.sort(key=lambda item: item.get("modified_time", 0), reverse=True)
    if trained_models:
        trained_models[0]["is_latest"] = True

    models.extend(trained_models)

    type_order = {"pretrained": 0, "uploaded": 1, "trained": 2, "behavior": 3}
    models.sort(key=lambda item: (type_order.get(item.get("type"), 99), -float(item.get("modified_time") or 0)))

    # 内部去重 key 不返回给前端。
    for item in models:
        item.pop("_logical_key", None)

    return {
        "total_models": len(models),
        "models": models,
        "model_roots": {
            "models": str(MODELS_DIR),
            "pretrained": str(PRETRAINED_MODELS_DIR),
            "data_pretrained": "/app/data/models/pretrained",
            "legacy_pretrained": "/app/models/pretrained",
            "behavior": str(BEHAVIOR_MODELS_DIR),
            "runs": str(RUNS_DIR),
            "legacy_models": "/app/models",
            "legacy_runs": "/app/runs",
            "dedupe": "logical_key",
        }
    }


@app.post("/models/upload")
@app.post("/api/models/upload")
async def upload_model(
    file: UploadFile = File(...),
    name: str = Form(...),
    description: str = Form(None),
    _admin_user: Dict[str, Any] = Depends(require_admin),
):
    """
    上传自定义模型文件 (.pt 格式)
    """
    
    if not MODEL_UPLOAD_ENABLED:
        raise HTTPException(status_code=503, detail="模型上传功能尚未完成隔离校验，当前已关闭")

    # 检查文件类型
    if not file.filename.endswith('.pt'):
        raise HTTPException(status_code=400, detail="只支持.pt格式的模型文件")
    
    try:
        # M4 完成隔离审批前功能默认关闭；启用时也禁止覆盖现有模型。
        custom_models_dir = MODELS_DIR
        custom_models_dir.mkdir(parents=True, exist_ok=True)
        
        # 使用上传时的原始文件名，仅做必要安全清理，避免路径穿越。
        model_filename = sanitize_original_filename(file.filename, "model.pt")
        model_path = custom_models_dir / model_filename
        
        if model_path.exists():
            raise HTTPException(status_code=409, detail="同名模型已存在，禁止覆盖")
        
        # 保存模型文件
        size = save_upload_file_limited(file, model_path, MODEL_UPLOAD_MAX_BYTES)
        
        logger.info(f"模型上传成功: {model_path}, 大小: {size} bytes")
        
        return {
            "message": "模型上传成功",
            "filename": model_filename,
            "path": str(model_path),
            "size": size,
            "overwritten": False
        }
    except HTTPException:
        raise
    except ArchiveSafetyError as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    except Exception as e:
        logger.error(f"模型上传失败: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"模型上传失败: {str(e)}"
        )


@app.get("/models/{model_id}/download")
@app.get("/api/models/{model_id}/download")
async def download_registered_model(model_id: str, request: Request, db=Depends(get_db)):
    current_user = getattr(request.state, "current_user", None)
    if not current_user:
        raise HTTPException(status_code=401, detail="Authentication required")
    record = db.query(ModelRegistry).filter(
        ModelRegistry.id == model_id,
        ModelRegistry.status.in_(("active", "candidate")),
    ).first()
    if record is None or not _model_registry_record_visible_to_user(db, record, current_user):
        raise HTTPException(status_code=404, detail="Model not found")
    path = resolve_managed_model_path(record.path, record)
    return FileResponse(path=path, filename=path.name, media_type="application/octet-stream")


@app.get("/download/model/{model_name}")
async def download_model(model_name: str):
    """下载模型文件"""
    model_path = resolve_managed_model_path(model_name)
    
    return FileResponse(
        path=model_path,
        filename=os.path.basename(model_path),
        media_type='application/octet-stream'
    )


@app.delete("/task/{task_id}")
async def delete_task(task_id: str, request: Request, db=Depends(get_db)):
    """Delete one completed training task, its events, and its linked model files."""
    task = db.query(JobTask).filter(JobTask.id == task_id).first()
    if not task:
        raise HTTPException(status_code=404, detail="任务不存在")
    require_job_visibility(task, request)
    if task.type != "training":
        raise HTTPException(status_code=404, detail="训练任务不存在")
    if task.status not in {"completed", "failed", "cancelled"}:
        worker = read_worker_heartbeat()
        if worker["status"] == "available" and worker.get("active_task_id") == task.id:
            raise HTTPException(status_code=409, detail="训练任务仍在运行，请先停止后再删除")

        # The task can outlive a stopped or restarted worker.  Do not keep a
        # non-terminal database row from blocking deletion when no worker owns it.
        task = update_job(
            db,
            task.id,
            status="failed",
            stage="failed",
            message="训练 Worker 不可用，任务已标记为失败",
            error_code="worker_lost",
            error_detail="删除时未发现持有该任务的可用训练 Worker",
            event_type="worker_lost",
            event_payload={"worker": worker},
            level="warning",
        )

    try:
        deleted = delete_training_task_and_models(
            db,
            task_id,
            resolve_path=resolve_registered_model_path,
            is_protected=lambda record, path: is_protected_registered_model(db, record, path),
        )
    except HTTPException:
        raise
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc))
    except ModelDeletionRecoveryError:
        logger.exception("删除训练关联模型回滚不完整: %s", task_id)
        raise HTTPException(status_code=500, detail="删除未完成，部分模型仍在隔离区，请使用恢复工具处理")
    except OSError as exc:
        logger.exception("删除训练关联模型失败: %s", task_id)
        raise HTTPException(status_code=500, detail=f"删除训练关联模型失败: {exc}")

    return {
        "message": "训练记录及关联模型已删除",
        "deleted_task_id": deleted["task_id"],
        "deleted_events": deleted["deleted_events"],
        "deleted_model_count": deleted["deleted_model_count"],
        "operation_id": deleted.get("operation_id"),
        "cleanup_pending": deleted.get("cleanup_pending", False),
    }


@app.delete("/models/{model_path:path}")
@app.delete("/api/models/{model_path:path}")
async def delete_model(
    model_path: str,
    _admin_user: Dict[str, Any] = Depends(require_admin),
    db=Depends(get_db),
):
    """
    删除训练模型
    注意：基础预训练模型（如yolov8n.pt等）不允许删除
    """

    # 先按请求路径和常见预训练权重名拦截，防止用户直接调用 DELETE API。
    model_filename = os.path.basename(model_path)
    if is_protected_pretrained_model(model_path):
        raise HTTPException(
            status_code=403, 
            detail=f"基础预训练模型 '{model_filename}' 受保护，不允许删除"
        )
    
    registry_item = find_active_model_record(
        db,
        model_path,
        allowed_statuses=("active", "candidate"),
    )
    full_path = resolve_managed_model_path(model_path, registry_item)

    if is_protected_pretrained_model(model_path, full_path):
        raise HTTPException(
            status_code=403,
            detail=f"基础预训练模型 '{model_filename}' 受保护，不允许删除"
        )
    
    try:
        if registry_item is None:
            registry_item = db.query(ModelRegistry).filter(
                ModelRegistry.path == str(full_path.resolve()).replace("\\", "/")
            ).first()

        if registry_item is None:
            raise HTTPException(
                status_code=409,
                detail="模型尚未纳入模型登记，请先执行模型对账后再删除",
            )

        deleted = delete_model_and_linked_training_task(
            db,
            registry_item,
            resolve_path=resolve_registered_model_path,
            is_protected=lambda record, path: is_protected_registered_model(db, record, path),
        )
        deleted_task_id = deleted["deleted_training_task_id"]
        deleted_model_count = deleted["deleted_model_count"]
        cleanup_pending = deleted.get("cleanup_pending", False)
        operation_id = deleted.get("operation_id")

        logger.info("已删除模型文件: %s", full_path)
        
        # 只对训练模型检查并清理空的实验目录
        # 自定义上传的模型不需要这个逻辑
        if "runs/train" in str(full_path):
            # 检查weights目录是否为空，如果为空则删除整个实验目录
            weights_dir = full_path.parent
            quarantine_dir = weights_dir / ".delete-quarantine"
            if quarantine_dir.is_dir() and not any(quarantine_dir.iterdir()):
                quarantine_dir.rmdir()
            if weights_dir.exists() and weights_dir.name == "weights" and not any(weights_dir.iterdir()):
                experiment_dir = weights_dir.parent
                logger.info(f"weights目录为空，删除整个实验目录: {experiment_dir}")
                file_manager.cleanup_dir(str(experiment_dir))
        
        return {
            "message": f"模型 {model_path} 已成功删除",
            "deleted_model": str(full_path),
            "deleted_model_count": deleted_model_count,
            "deleted_training_task_id": deleted_task_id,
            "operation_id": operation_id,
            "cleanup_pending": cleanup_pending,
        }
        
    except PermissionError:
        db.rollback()
        raise HTTPException(
            status_code=403, 
            detail="没有权限删除该模型文件"
        )
    except ValueError as exc:
        db.rollback()
        raise HTTPException(status_code=409, detail=str(exc))
    except ModelDeletionRecoveryError:
        db.rollback()
        logger.exception("删除模型回滚不完整: %s", model_path)
        raise HTTPException(status_code=500, detail="删除未完成，部分模型仍在隔离区，请使用恢复工具处理")
    except Exception as e:
        db.rollback()
        logger.error(f"删除模型失败: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"删除模型失败: {str(e)}"
        )


@app.delete("/datasets/{dataset_id}")
async def delete_dataset(dataset_id: str, request: Request, db=Depends(get_db)):
    """删除单个数据集"""
    
    dataset_path = resolve_managed_dataset_root(dataset_id, constants.DATASETS_DIR)
    
    try:
        require_dataset_manage(db, get_request_current_user(request), dataset_id, "original")
        deleted = delete_persisted_dataset(db, dataset_id, "original", dataset_path)
        
        # 从缓存中移除
        if dataset_id in uploaded_datasets:
            del uploaded_datasets[dataset_id]
        
        return {
            "message": f"数据集 {dataset_id} 已删除",
            "deleted_dataset_id": dataset_id,
            **deleted,
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除数据集失败: {str(e)}")


@app.post("/datasets/batch/delete")
async def batch_delete_datasets(payload: dict, http_request: Request, db=Depends(get_db)):
    """批量删除数据集"""
    
    dataset_ids = payload.get("dataset_ids", [])
    current_user = get_request_current_user(http_request)
    
    if not dataset_ids:
        raise HTTPException(status_code=400, detail="请提供要删除的数据集ID列表")
    
    deleted_count = 0
    deleted_dataset_ids = []
    failed_deletions = []
    
    for dataset_id in dataset_ids:
        try:
            require_dataset_manage(db, current_user, dataset_id, "original")
            dataset_path = resolve_managed_dataset_root(dataset_id, constants.DATASETS_DIR)
            delete_persisted_dataset(db, dataset_id, "original", dataset_path)
            
            # 从缓存中移除
            if dataset_id in uploaded_datasets:
                del uploaded_datasets[dataset_id]
            
            deleted_count += 1
            deleted_dataset_ids.append(dataset_id)
        except HTTPException as e:
            failed_deletions.append({"id": dataset_id, "error": str(e.detail)})
        except Exception as e:
            failed_deletions.append({"id": dataset_id, "error": str(e)})
    
    return {
        "message": f"批量删除完成，成功删除 {deleted_count} 个数据集",
        "deleted_count": deleted_count,
        "deleted_dataset_ids": deleted_dataset_ids,
        "total_requested": len(dataset_ids),
        "failed_deletions": failed_deletions
    }


@app.websocket("/ws/realtime/{session_id}")
async def websocket_realtime_inference(websocket: WebSocket, session_id: str):
    """实时推理WebSocket端点。运行中配置会被保存并持续应用。"""
    if str(session_id).startswith("monitor_"):
        await websocket.close(code=4403, reason="reserved monitoring session")
        return
    behavior_owner = await authenticate_websocket(websocket)
    if behavior_owner is None:
        return

    existing_context = behavior_session_contexts.get(session_id) or {}
    if existing_context and not _same_websocket_owner(existing_context, behavior_owner):
        await websocket.close(code=4403, reason="session belongs to another user")
        return
    await websocket.accept()

    previous_websocket = realtime_connections.get(session_id)
    previous_task = realtime_stream_tasks.get(session_id)
    if previous_websocket is not None or (previous_task is not None and not previous_task.done()):
        cleanup_complete = await request_realtime_cleanup(
            session_id,
            websocket=previous_websocket,
            timeout_seconds=15.0,
        )
        if not cleanup_complete:
            await websocket.close(code=1013, reason="previous session is still stopping")
            return

    behavior_session_contexts[session_id] = {
        "owner_id": behavior_owner.get("id"),
        "owner_username": behavior_owner.get("username") or "",
        "mode": "realtime",
        "source_id": session_id,
        "session_id": session_id,
    }
    realtime_connections[session_id] = websocket
    realtime_send_locks[session_id] = asyncio.Lock()
    realtime_frame_delivery[session_id] = {}
    realtime_stop_events[session_id] = asyncio.Event()
    realtime_session_states[session_id] = "active"
    configure_realtime_frame_delivery(session_id, supports_ack=False)

    # 初始化会话配置和统计
    default_config = get_default_realtime_config_dict()
    realtime_session_configs[session_id] = default_config
    realtime_frame_counters[session_id] = 0
    realtime_last_process_time[session_id] = 0

    realtime_stats[session_id] = {
        "session_id": session_id,
        "frames_processed": 0,
        "total_detections": 0,
        "inference_times": [],
        "start_time": datetime.now().isoformat(),
        "last_update": datetime.now().isoformat(),
        "is_active": True,
        "config": default_config
    }

    counted_ids = set()
    object_counts = {}

    try:
        await send_realtime_message_safe(session_id, websocket, {
            "type": "config_updated",
            "message": "实时监控会话已建立，默认配置已应用",
            "config": default_config
        })

        while True:
            message = await websocket.receive_json()
            if realtime_session_should_stop(session_id):
                break
            message_type = message.get("type")

            if message_type == "ping":
                await send_realtime_message_safe(session_id, websocket, {
                    "type": "pong",
                    "client_timestamp": message.get("timestamp"),
                    "server_timestamp": time.time(),
                })
                continue

            if message_type == "frame_ack":
                acknowledge_realtime_image(session_id, message.get("frame_sequence"))
                continue

            if message_type == "frame":
                # 前端每帧携带的 config 会与会话配置合并，确保滑条/模型/开关变化能即时生效。
                config_data = message.get("config", {})
                config, source_changed = update_realtime_session_config(session_id, config_data)
                if behavior_monitoring_requires_login(config, behavior_owner):
                    await send_realtime_message_safe(session_id, websocket, {
                        "type": "error",
                        "code": "behavior_auth_required",
                        "message": "启用行为告警时必须使用已登录会话",
                    })
                    continue
                if source_changed:
                    logger.info(
                        "实时监控视频源配置已更新: session=%s, source=%s, url=%s",
                        session_id,
                        config.source_type,
                        redact_stream_url(config.source_url),
                    )

                if not should_process_realtime_frame(session_id, config):
                    # 静默丢弃超过 max_fps 或 frame_skip 的帧，防止前端高频发帧压垮后端。
                    continue

                if config.source_type == "webcam" and "data" in message:
                    image_data = message["data"]
                    image = await asyncio.to_thread(decode_base64_image, image_data)
                else:
                    image = await asyncio.to_thread(get_frame_from_source, session_id, config)
                if realtime_session_should_stop(session_id):
                    break

                if image is None:
                    await send_realtime_message_safe(session_id, websocket, {
                        "type": "error",
                        "message": "无法获取图像数据"
                    })
                    continue

                result = await asyncio.to_thread(process_realtime_frame, image, config.model_path, config)
                if result.get("gpu_waiting"):
                    await send_realtime_message_safe(session_id, websocket, {
                        "type": "gpu_waiting", "message": WAIT_MESSAGE,
                    })
                    await asyncio.sleep(2)
                    continue
                if realtime_session_should_stop(session_id):
                    break
                behavior_context = behavior_session_contexts.get(session_id, {})
                behavior_context.update({
                "source_id": redact_stream_url(config.source_url) or session_id,
                    "session_id": session_id,
                })
                behavior_result = submit_behavior_frame(
                    session_id,
                    image,
                    result,
                    config.behavior_config,
                    context=behavior_context,
                    timestamp=time.time(),
                )

                # 更新唯一对象计数
                if "detections" in result:
                    for detection in result["detections"]:
                        track_id = detection.get("track_id")
                        if track_id is not None and track_id != -1 and track_id not in counted_ids:
                            counted_ids.add(track_id)
                            class_name = detection.get("class") or detection.get("class_name") or "object"
                            object_counts[class_name] = object_counts.get(class_name, 0) + 1
                            logger.info(f"发现新跟踪对象: track_id={track_id}, class_name='{class_name}'")

                annotated_image = get_realtime_annotated_image(image, result, config)
                annotated_image_base64 = None
                if annotated_image is not None:
                    annotated_image_base64 = await asyncio.to_thread(encode_image_to_base64, annotated_image)
                    if realtime_session_should_stop(session_id):
                        break
                saved_frame_url = await asyncio.to_thread(
                    save_realtime_frame_if_needed,
                    session_id,
                    annotated_image if annotated_image is not None else image,
                    config,
                )
                if realtime_session_should_stop(session_id):
                    break

                response = update_realtime_stats_and_build_response(
                    session_id,
                    result,
                    annotated_image_base64,
                    config,
                    object_counts=object_counts,
                    saved_frame_url=saved_frame_url,
                    behavior_result=behavior_result,
                )

                for alarm in response.get("alarm_events") or []:
                    if not await send_realtime_message_safe(session_id, websocket, {
                        "type": "alarm_event",
                        "session_id": session_id,
                        "event": alarm,
                        "timestamp": alarm.get("alerted_at") or response.get("timestamp"),
                    }):
                        break
                if not await send_realtime_message_safe(session_id, websocket, response):
                    break

            elif message_type == "start_server_stream":
                config_data = message.get("config", {})
                config, source_changed = update_realtime_session_config(session_id, config_data)
                if behavior_monitoring_requires_login(config, behavior_owner):
                    await send_realtime_message_safe(session_id, websocket, {
                        "type": "error",
                        "code": "behavior_auth_required",
                        "message": "启用行为告警时必须使用已登录会话",
                    })
                    continue
                configure_realtime_frame_delivery(
                    session_id,
                    supports_ack=bool(message.get("supports_frame_ack", False)),
                )

                existing_task = realtime_stream_tasks.get(session_id)

                if config.source_type in ["rtsp", "http", "file"]:
                    if existing_task is None or existing_task.done():
                        realtime_stream_tasks[session_id] = asyncio.create_task(
                            process_server_stream(websocket, session_id, config)
                        )

                    await send_realtime_message_safe(session_id, websocket, {
                        "type": "server_stream_started",
                        "message": "服务器端视频流已启动，当前配置已应用",
                        "config": normalize_realtime_config_data(config.dict())
                    })
                else:
                    await send_realtime_message_safe(session_id, websocket, {
                        "type": "server_stream_skipped",
                        "message": "当前为前端摄像头模式，将使用 frame 消息中的图像数据",
                        "config": normalize_realtime_config_data(config.dict())
                    })
                continue

            elif message_type == "config_update":
                config_data = message.get("config", {})
                config, source_changed = update_realtime_session_config(session_id, config_data)

                if session_id in realtime_stats:
                    realtime_stats[session_id]["config"] = normalize_realtime_config_data(config.dict())
                    realtime_stats[session_id]["last_update"] = datetime.now().isoformat()

                await send_realtime_message_safe(session_id, websocket, {
                    "type": "config_updated",
                    "message": "配置已更新并已应用",
                    "source_changed": source_changed,
                    "config": normalize_realtime_config_data(config.dict())
                })

            elif message_type == "get_stats":
                stats = realtime_stats.get(session_id, {})
                stats["config"] = realtime_session_configs.get(session_id, get_default_realtime_config_dict())
                await send_realtime_message_safe(session_id, websocket, {
                    "type": "stats",
                    "data": stats
                })

            else:
                await send_realtime_message_safe(session_id, websocket, {
                    "type": "error",
                    "message": f"未知消息类型: {message_type}"
                })

    except WebSocketDisconnect:
        logger.info(f"WebSocket连接断开: {session_id}")
    except Exception as e:
        logger.error(f"WebSocket错误 {session_id}: {e}")
        try:
            await send_realtime_message_safe(session_id, websocket, {
                "type": "error",
                "message": f"连接错误: {str(e)}"
            })
        except Exception:
            pass
    finally:
        # A resumed connection may already own this session id. Only the
        # current owner may tear down shared state.
        if realtime_connections.get(session_id) is websocket:
            await request_realtime_cleanup(
                session_id,
                websocket=websocket,
                timeout_seconds=15.0,
            )


@app.get("/realtime/stats/{session_id}")
async def get_realtime_stats(session_id: str):
    """获取实时推理统计"""
    if session_id not in realtime_stats:
        raise HTTPException(status_code=404, detail="会话不存在")
    
    return realtime_stats[session_id]


def redact_stream_url(value: Any) -> str:
    """Remove URL userinfo before a stream address reaches logs or APIs."""
    return re.sub(r"(?i)([a-z][a-z0-9+.-]*://)([^/@\s]+)@", r"\1[REDACTED]@", str(value or ""))


def public_realtime_config(value: Any) -> Dict[str, Any]:
    if hasattr(value, "dict"):
        value = value.dict()
    payload = dict(value or {})
    for key in ("source_url", "sourceUrl", "url", "path"):
        if key in payload:
            payload[key] = redact_stream_url(payload[key])
    return payload


@app.get("/realtime/sessions")
async def list_realtime_sessions():
    """列出所有实时推理会话"""
    active_sessions = []
    for session_id, stats in realtime_stats.items():
        if str(session_id).startswith("monitor_"):
            continue
        if stats.get("is_active", False):
            active_sessions.append({
                "session_id": session_id,
                "frames_processed": stats["frames_processed"],
                "total_detections": stats["total_detections"],
                "start_time": stats["start_time"],
                "last_update": stats["last_update"],
                "config": public_realtime_config(
                    stats.get("config", realtime_session_configs.get(session_id, get_default_realtime_config_dict()))
                )
            })
    
    return {
        "active_sessions": len(active_sessions),
        "sessions": active_sessions
    }


@app.delete("/realtime/session/{session_id}")
async def stop_realtime_session(session_id: str):
    """停止实时推理会话"""
    if str(session_id).startswith("monitor_"):
        raise HTTPException(status_code=409, detail="托管持续监控请使用持续监控专用停止接口")
    cleanup_complete = await request_realtime_cleanup(session_id, timeout_seconds=15.0)
    payload = {
        "session_id": session_id,
        "status": "stopped" if cleanup_complete else "stopping",
        "cleanup_complete": cleanup_complete,
        "message": "实时推理会话已停止" if cleanup_complete else "会话正在安全回收，请稍后重试",
    }
    if cleanup_complete:
        return payload
    return JSONResponse(status_code=202, content=payload)


class VideoChunkUploadStartRequest(BaseModel):
    filename: str
    size: int
    content_type: str = ""
    total_chunks: int


@app.post("/upload/video/chunks/start")
async def start_video_chunk_upload(req: VideoChunkUploadStartRequest):
    await asyncio.to_thread(cleanup_stale_video_upload_sessions)
    validated = validate_video_upload_metadata(req.filename, req.content_type, req.size)
    total_chunks = max(1, int(req.total_chunks))
    expected_chunks = max(1, int((int(req.size) + VIDEO_UPLOAD_CHUNK_BYTES - 1) // VIDEO_UPLOAD_CHUNK_BYTES))
    if total_chunks != expected_chunks or total_chunks > VIDEO_UPLOAD_MAX_CHUNKS:
        raise HTTPException(status_code=400, detail="分片数量与文件大小不一致")
    upload_id = uuid.uuid4().hex
    session_dir = _video_upload_session_dir(upload_id)
    session_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "upload_id": upload_id,
        "filename": validated["filename"],
        "original_filename": req.filename,
        "content_type": validated["content_type"],
        "size": int(req.size),
        "total_chunks": total_chunks,
        "chunk_bytes": VIDEO_UPLOAD_CHUNK_BYTES,
        "created_at": datetime.now().isoformat(),
    }
    _write_video_upload_session(session_dir, metadata)
    return {**metadata, "uploaded_chunks": []}


@app.get("/upload/video/chunks/{upload_id}")
async def get_video_chunk_upload(upload_id: str):
    session_dir, metadata = _read_video_upload_session(upload_id)
    uploaded_chunks = sorted(
        int(path.stem)
        for path in session_dir.glob("*.part")
        if path.stem.isdigit()
    )
    return {**metadata, "uploaded_chunks": uploaded_chunks}


@app.post("/upload/video/chunks/{upload_id}/parts/{chunk_index}")
async def upload_video_chunk(upload_id: str, chunk_index: int, chunk: UploadFile = File(...)):
    session_dir, metadata = _read_video_upload_session(upload_id)
    total_chunks = int(metadata["total_chunks"])
    if chunk_index < 0 or chunk_index >= total_chunks:
        raise HTTPException(status_code=400, detail="分片序号超出范围")
    expected_size = VIDEO_UPLOAD_CHUNK_BYTES
    if chunk_index == total_chunks - 1:
        expected_size = int(metadata["size"]) - VIDEO_UPLOAD_CHUNK_BYTES * (total_chunks - 1)
    chunk_path = session_dir / f"{chunk_index}.part"
    try:
        written = await asyncio.to_thread(save_upload_file_limited, chunk, chunk_path, VIDEO_UPLOAD_CHUNK_BYTES)
    except ArchiveSafetyError as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    if written != expected_size:
        chunk_path.unlink(missing_ok=True)
        raise HTTPException(status_code=400, detail=f"分片大小错误: expected={expected_size}, actual={written}")
    return {"upload_id": upload_id, "chunk_index": chunk_index, "size": written}


@app.post("/upload/video/chunks/{upload_id}/complete")
async def complete_video_chunk_upload(upload_id: str):
    video_info = await asyncio.to_thread(_complete_video_chunk_upload_sync, upload_id)
    return {"video_id": video_info["id"], "message": "视频分片上传成功", "video_info": video_info}


@app.post("/upload/video")
async def upload_video_file(file: UploadFile = File(...)):
    """上传较小视频；大文件由分片接口处理。"""
    original_filename = file.filename or "video.mp4"
    validate_video_upload_metadata(original_filename, file.content_type)
    try:
        video_path = await asyncio.to_thread(
            save_upload_file_keep_name,
            file,
            get_video_upload_dir(),
            sanitize_original_filename(original_filename, "video.mp4"),
            VIDEO_UPLOAD_MAX_BYTES,
        )
        video_info = await asyncio.to_thread(register_uploaded_video, video_path, original_filename)
        return {"video_id": video_info["id"], "message": "视频上传成功", "video_info": video_info}
    except HTTPException:
        raise
    except ArchiveSafetyError as exc:
        raise HTTPException(status_code=413, detail=str(exc))
    except Exception as exc:
        logger.exception("视频上传失败: %s", exc)
        raise HTTPException(status_code=400, detail=f"视频上传失败: {exc}")

@app.post("/annotation/video/upload")
async def upload_video_for_annotation(file: UploadFile = File(...)):
    """上传视频文件用于数据标注，复用受限的视频源存储与注册流程。"""
    original_filename = file.filename or "annotation_video.mp4"
    validate_video_upload_metadata(original_filename, file.content_type)
    try:
        video_path = await asyncio.to_thread(
            save_upload_file_keep_name,
            file,
            get_video_upload_dir(),
            fallback=sanitize_original_filename(original_filename, "annotation_video.mp4"),
            max_bytes=VIDEO_UPLOAD_MAX_BYTES,
        )
        video_info = await asyncio.to_thread(register_uploaded_video, video_path, original_filename)

        return {
            "video_id": video_info["id"],
            "video_path": video_info["path"],
            "filename": video_path.name,
            "original_filename": original_filename,
            "size": video_info["size"],
            "message": "视频上传成功",
            **{key: value for key, value in video_info.items() if key not in {"id", "path", "name", "original_filename", "filename", "size"}},
        }

    except HTTPException:
        raise
    except ArchiveSafetyError as e:
        raise HTTPException(status_code=413, detail=str(e))
    except Exception as e:
        logger.error(f"视频上传失败: {str(e)}")
        raise HTTPException(status_code=400, detail=f"视频上传失败: {str(e)}")

def _resolve_managed_video_path(video_id: Any) -> Path:
    text_id = str(video_id or "").strip()
    if not text_id or Path(text_id).name != text_id or text_id in {".", ".."}:
        raise HTTPException(status_code=400, detail="无效的视频 ID")
    load_video_sources_from_disk()
    source_info = video_sources.get(text_id)
    if not isinstance(source_info, dict) or str(source_info.get("type") or "file").lower() != "file":
        raise HTTPException(status_code=404, detail="视频源不存在")
    path_text = str(source_info.get("path") or "").strip()
    if not path_text:
        raise HTTPException(status_code=404, detail="视频文件不存在")
    candidate = Path(path_text)
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    try:
        resolved = candidate.resolve(strict=False)
    except OSError:
        raise HTTPException(status_code=400, detail="无效的视频路径")
    allowed_roots = (get_video_upload_dir().resolve(),)
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise HTTPException(status_code=403, detail="视频路径不在受管目录内")
    if resolved.suffix.lower() not in VIDEO_FILE_EXTENSIONS:
        raise HTTPException(status_code=400, detail="不支持的视频格式")
    if not resolved.is_file():
        raise HTTPException(status_code=404, detail="视频文件不存在")
    return resolved


def _extract_annotation_frames_sync(video_id: str, resolved_path: Path, frame_interval: float, max_frames: int) -> Dict[str, Any]:
    cap = video_processor.create_capture("file", str(resolved_path))
    if not cap:
        raise HTTPException(status_code=400, detail="无法打开视频文件")
    try:
        video_info = video_processor.get_video_info(cap)
        fps = float(video_info.get("fps") or 0.0)
        if fps <= 0:
            fps = 25.0
        interval_seconds = min(max(float(frame_interval), 0.05), ANNOTATION_VIDEO_MAX_INTERVAL_SECONDS)
        interval = max(1, int(round(fps * interval_seconds)))
        bounded_max_frames = max(1, min(int(max_frames), ANNOTATION_VIDEO_MAX_FRAMES))
        frame_count = max(0, int(video_info.get("frame_count") or 0))
        total_extract_frames = min(bounded_max_frames, (frame_count + interval - 1) // interval if frame_count else bounded_max_frames)

        extracted_frames = []
        frame_index = 0
        while len(extracted_frames) < total_extract_frames:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret:
                break
            if ANNOTATION_FRAME_MAX_WIDTH > 0 and frame.shape[1] > ANNOTATION_FRAME_MAX_WIDTH:
                scale = ANNOTATION_FRAME_MAX_WIDTH / float(frame.shape[1])
                frame = cv2.resize(frame, (ANNOTATION_FRAME_MAX_WIDTH, max(1, int(frame.shape[0] * scale))))
            frame_base64 = image_processor.encode_image_to_base64(frame)
            if sum(len(item.get("data") or "") for item in extracted_frames) + len(frame_base64) > ANNOTATION_EXTRACT_MAX_RESPONSE_BYTES:
                break
            extracted_frames.append({
                "frame_index": frame_index,
                "timestamp": round(frame_index / fps, 2),
                "data": frame_base64,
                "width": int(frame.shape[1]),
                "height": int(frame.shape[0]),
            })
            frame_index += interval
        return {
            "video_id": video_id,
            "total_frames": frame_count,
            "extracted_frames": len(extracted_frames),
            "fps": fps,
            "frames": extracted_frames,
            "message": f"成功提取 {len(extracted_frames)} 帧",
        }
    finally:
        video_processor.release_capture(cap)


@app.post("/annotation/video/extract-frames")
async def extract_frames_from_video(
    video_id: str,
    frame_interval: float = 1.0,
    max_frames: int = 100,
    video_path: Optional[str] = None,
):
    """按已注册 video_id 从受管视频中有界提取帧；旧 video_path 参数仅兼容且不被信任。"""
    try:
        resolved_path = _resolve_managed_video_path(video_id)
        return await asyncio.to_thread(
            _extract_annotation_frames_sync,
            str(video_id).strip(),
            resolved_path,
            frame_interval,
            max_frames,
        )
    except HTTPException:
        raise
    except (TypeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=f"提取视频帧参数无效: {exc}")
    except Exception as exc:
        logger.exception("提取视频帧失败: %s", exc)
        raise HTTPException(status_code=400, detail=f"提取视频帧失败: {exc}")


@app.post("/annotation/ai-assist")
@app.post("/api/annotation/ai-assist")
@gpu_http_guard
async def ai_assist_annotation(request: dict, http_request: Request, db=Depends(get_db)):
    """AI辅助预标注 - 使用YOLO模型自动检测目标并生成标注框"""
    try:
        # 从请求体中获取参数
        image_data = request.get("image_data")
        model_path = str(request.get("model_path") or request.get("model") or "yolo11n.pt").strip() or "yolo11n.pt"
        confidence = _safe_ratio(request.get("confidence", constants.DEFAULT_CONFIDENCE), constants.DEFAULT_CONFIDENCE)
        prompt_classes = get_annotation_prompt_classes_from_request(request)
        raw_page_classes = request.get("annotation_classes", request.get("page_classes"))
        try:
            page_classes = normalize_class_names(raw_page_classes) if raw_page_classes is not None else None
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc))
        page_bindings = request.get("annotation_class_bindings", request.get("class_bindings"))
        prompt_page_indices = request.get("prompt_page_indices")
        target_page_index = request.get("target_page_index")
        model_metadata = request.get("model_metadata") if isinstance(request.get("model_metadata"), dict) else {}
        
        if not image_data:
            raise HTTPException(status_code=400, detail="缺少图片数据")
        
        # 使用统一函数加载图片：支持 base64，也支持 /image/xxx、/api/image/xxx、./data/image/xxx
        img = load_annotation_image_from_request_value(image_data)
        if img is None:
            raise HTTPException(status_code=400, detail="无效的图片数据")
        
        # 转换为BGR格式供OpenCV使用
        img_bgr = image_processor.convert_rgb_to_bgr(img)
        
        # 获取图片尺寸
        img_height, img_width = image_processor.get_image_size(img)
        
        # 根据服务器注册元数据选择普通YOLO或可验证的YOLOE加载路径。
        requested_model_id = str(
            request.get("model_id")
            or model_metadata.get("id")
            or model_metadata.get("model_id")
            or ""
        ).strip()
        model_path, registry_model, model_metadata = _resolve_annotation_request_model(
            db,
            model_path,
            requested_model_id,
            model_metadata,
        )
        annotation_draft_id = str(request.get("annotation_draft_id") or "").strip()
        if annotation_draft_id:
            draft = _annotation_request_draft(db, str(request.get("dataset_id") or ""),
                annotation_draft_id, get_request_current_user(http_request), model_path)
            if isolated_draft(draft):
                if page_classes is not None and page_classes != list(draft.class_names_json or []):
                    raise HTTPException(status_code=409, detail="标注会话类别已变化，请先保存草稿")
                page_classes = list(draft.class_names_json or [])
                page_bindings = list(draft.class_bindings_json or [])
        supports_text_prompts = annotation_model_supports_text_prompts(model_path, model_metadata)
        supports_dynamic_text_prompts = annotation_model_supports_dynamic_text_prompts(
            model_path,
            model_metadata,
        )
        effective_prompt_classes = prompt_classes or (page_classes if supports_text_prompts else []) or []
        model = load_annotation_inference_model(
            model_path,
            effective_prompt_classes,
            model_metadata,
        )
        supports_text_prompts = annotation_model_supports_text_prompts(model_path, model_metadata, model)
        applied_prompt_classes = apply_annotation_prompt_classes(
            model,
            effective_prompt_classes,
            supports_text_prompts=supports_text_prompts,
            supports_dynamic_text_prompts=supports_dynamic_text_prompts,
        )
        routing = build_annotation_class_routing(
            model,
            page_classes,
            page_bindings,
            prompt_classes=applied_prompt_classes,
            prompt_page_indices=prompt_page_indices,
            target_page_index=target_page_index,
            text_prompt_model=supports_text_prompts,
            dynamic_text_prompt_model=supports_dynamic_text_prompts,
        )

        # 执行推理
        if routing.get("allowed_model_ids") is not None and not routing["allowed_model_ids"]:
            results = []
        else:
            results = _run_annotation_model(
                model,
                img_bgr,
                confidence,
                routing.get("allowed_model_ids"),
            )
        
        # 提取检测结果
        annotations = []
        if len(results) > 0:
            result = results[0]
            boxes = getattr(result, "boxes", None)
            
            if boxes is not None:
                for index, box in enumerate(boxes):
                    annotation = build_ai_annotation_from_box(
                        model,
                        result,
                        box,
                        index,
                        img_width,
                        img_height,
                        routing,
                    )
                    if annotation:
                        annotations.append(annotation)
        annotations = enforce_annotation_prompt_results(annotations, applied_prompt_classes)
        
        return {
            "annotations": annotations,
            "count": len(annotations),
            "coordinate_space": "raw",
            "image_width": int(img_width),
            "image_height": int(img_height),
            "image_orientation": 1,
            "model": model_path,
            "model_id": str(registry_model.id) if registry_model is not None else "",
            "prompt_classes": applied_prompt_classes,
            "annotation_classes": page_classes,
            "annotation_class_bindings": routing["bindings"],
            "unmapped_page_classes": routing["unmapped_page_classes"],
            "unmapped_page_class_names": routing["unmapped_page_class_names"],
            "filtered_model_class_ids": sorted(routing["allowed_model_ids"] or []),
            "supports_text_prompts": supports_text_prompts,
            "supports_dynamic_text_prompts": supports_dynamic_text_prompts,
            "message": f"检测到 {len(annotations)} 个目标"
        }
    
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"AI辅助预标注失败: {str(e)}")
        raise HTTPException(status_code=500, detail=f"AI辅助预标注失败: {str(e)}")

def build_video_source_dedupe_key(source_info: Dict[str, Any]) -> str:
    """
    视频源去重键。
    file 类型优先按真实文件路径去重；
    rtsp/http 类型按 URL 去重。
    """
    source_type = str(source_info.get("type") or source_info.get("source_type") or "").lower()

    if source_type == "file":
        path_value = (
            source_info.get("path")
            or source_info.get("source_url")
            or source_info.get("url")
            or source_info.get("filename")
            or source_info.get("name")
            or source_info.get("id")
            or ""
        )

        text = str(path_value).replace("\\", "/").strip()

        if text.startswith("/static/videos/"):
            text = "static/videos/" + text.split("/static/videos/", 1)[1]
        elif text.startswith("/api/static/videos/"):
            text = "static/videos/" + text.split("/api/static/videos/", 1)[1]
        elif text.startswith("/app/static/videos/"):
            text = "static/videos/" + text.split("/app/static/videos/", 1)[1]

        filename = Path(text).name or str(source_info.get("filename") or source_info.get("name") or "")
        size = str(source_info.get("size") or 0)
        duration = str(round(float(source_info.get("duration") or 0), 2))

        return f"file:{filename.lower()}:{size}:{duration}"

    url_value = source_info.get("url") or source_info.get("source_url") or source_info.get("path") or source_info.get("id") or ""
    return f"{source_type}:{str(url_value).strip().lower()}"

@app.get("/video/sources")
async def list_video_sources():
    """获取视频源列表。会从 static/videos 自动恢复已上传的视频文件。"""
    try:
        load_video_sources_from_disk()
        sources = []
        
        # 添加默认摄像头源
        sources.append({
            "id": "webcam",
            "name": "摄像头",
            "type": "webcam",
            "url": "",
            "status": "available"
        })
        
        seen_source_keys = set()
        # 添加所有视频源（文件和RTSP/HTTP）
        for video_id, video_info in video_sources.items():
            video_type = video_info.get("type", "unknown")
            
            # 检查视频源是否可用
            is_available = True
            if video_type == "file":
                # 文件类型检查路径是否存在
                is_available = "path" in video_info and os.path.exists(video_info["path"])
                source_url = video_info.get("path", "")
            else:
                # RTSP/HTTP类型使用URL
                source_url = video_info.get("url", "")
            
            if is_available:
                source_data = {
                    "id": video_id,
                    "name": video_info["name"],
                    "type": video_type,
                    "url": source_url,
                    "fps": video_info.get("fps", 30),
                    "resolution": f"{video_info.get('width', 0)}x{video_info.get('height', 0)}",
                    "created_at": video_info.get("created_at", ""),
                    "status": "available"
                }
                
                # 为文件类型添加额外信息
                if video_type == "file":
                    source_data.update({
                        "duration": video_info.get("duration", 0),
                        "size": video_info.get("size", 0)
                    })
                
                dedupe_key = build_video_source_dedupe_key(source_data)
                if dedupe_key in seen_source_keys:
                    continue
                seen_source_keys.add(dedupe_key)
                
                sources.append(source_data)
        
        return {
            "total_sources": len(sources),
            "sources": sources
        }
        
    except Exception as e:
        logger.error(f"获取视频源列表失败: {e}")
        raise HTTPException(status_code=500, detail=f"获取视频源列表失败: {str(e)}")


@app.post("/video/source/add")
async def add_video_source(config: VideoSourceConfig):
    """添加视频源（RTSP/HTTP）"""
    try:
        source_id = generate_task_id()
        
        # 验证视频源连接
        cap = create_video_capture(config.source_type, config.source_url)
        if cap is None:
            raise HTTPException(status_code=400, detail="无法连接到视频源")
        
        # 获取视频信息
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or config.fps
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        cap.release()
        
        # 保存视频源信息
        source_info = {
            "id": source_id,
            "name": config.name or f"{config.source_type.upper()}源",
            "type": config.source_type,
            "url": config.source_url,
            "fps": fps,
            "width": width,
            "height": height,
            "resolution": config.resolution,
            "created_at": datetime.now().isoformat(),
            "status": "available"
        }
        
        video_sources[source_id] = source_info
        save_video_sources_metadata()
        
        return {
            "source_id": source_id,
            "message": "视频源添加成功",
            "source_info": source_info
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"添加视频源失败: {str(e)}")


@app.delete("/video/source/{source_id}")
async def delete_video_source(source_id: str):
    """删除视频源，同时同步更新 static/videos/video_sources.json。"""
    load_video_sources_from_disk()

    if source_id not in video_sources:
        raise HTTPException(status_code=404, detail="视频源不存在")
    
    try:
        source_info = video_sources[source_id]
        
        # 如果是上传的视频文件，删除文件
        if source_info["type"] == "file" and "path" in source_info:
            if os.path.exists(source_info["path"]):
                os.remove(source_info["path"])
        
        # 删除视频源记录，并同步 metadata，避免刷新后又恢复出来
        del video_sources[source_id]
        save_video_sources_metadata()
        
        return {
            "message": f"视频源 {source_id} 已删除",
            "deleted_source": source_info["name"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"删除视频源失败: {str(e)}")


@app.post("/video/source/test")
async def test_video_source(config: VideoSourceConfig):
    """测试视频源连接"""
    try:
        cap = create_video_capture(config.source_type, config.source_url)
        if cap is None:
            return {
                "status": "failed",
                "message": "无法连接到视频源"
            }
        
        # 尝试读取一帧
        ret, frame = cap.read()
        cap.release()
        
        if not ret:
            return {
                "status": "failed",
                "message": "无法从视频源读取帧"
            }
        
        return {
            "status": "success",
            "message": "视频源连接成功",
            "frame_size": frame.shape[:2] if frame is not None else None
        }
        
    except Exception as e:
        return {
            "status": "failed",
            "message": f"测试失败: {str(e)}"
        }


@app.websocket("/ws/grid/{grid_id}")
async def websocket_grid_inference(websocket: WebSocket, grid_id: str):
    """
    九宫格推理 WebSocket 端点。

    关键修复：
    1. 每个格子独立后台任务，任务句柄可停止、可清理；
    2. 多个格子共用同一个 WebSocket 时，通过 grid_send_locks 串行发送，避免并发 send_json 抢连接；
    3. 视频读取与 YOLO 推理放入线程池，避免某一路同步阻塞导致其它路无法刷新；
    4. 兼容前端传入的 type/url、source_type/source_url、path 等多种字段名。
    """
    if str(grid_id).startswith("monitor_"):
        await websocket.close(code=4403, reason="reserved monitoring session")
        return
    behavior_owner = await authenticate_websocket(websocket)
    if behavior_owner is None:
        return

    existing_grid = grid_sessions.get(grid_id) or {}
    if existing_grid and existing_grid.get("is_active") and not _same_websocket_owner(existing_grid, behavior_owner):
        await websocket.close(code=4403, reason="grid belongs to another user")
        return
    await websocket.accept()

    grid_connections[grid_id] = websocket
    grid_send_locks[grid_id] = asyncio.Lock()
    grid_stop_events[grid_id] = asyncio.Event()

    grid_sessions[grid_id] = {
        "grid_id": grid_id,
        "owner_id": behavior_owner.get("id"),
        "owner_username": behavior_owner.get("username") or "",
        "grid_name": "九宫格监控",
        "sessions": {},
        "created_at": datetime.now().isoformat(),
        "is_active": True,
        "total_detections": 0,
        "active_sources": 0
    }

    try:
        while True:
            message = await websocket.receive_json()
            message_type = message.get("type")

            if message_type == "start_grid":
                config_data = message.get("config", {}) or {}
                grid_config = GridConfig(**config_data)
                if behavior_monitoring_requires_login(
                    {"shared_behavior_config": grid_config.shared_behavior_config},
                    behavior_owner,
                ):
                    await send_grid_message_safe(grid_id, websocket, {
                        "type": "error",
                        "code": "behavior_auth_required",
                        "message": "启用行为告警时必须使用已登录会话",
                    })
                    continue
                supports_frame_ack = bool(message.get("supports_frame_ack", False))
                client_side_overlay = bool(grid_config.client_side_overlay)

                # 如果同一个 grid_id 之前已经启动过，先停掉旧任务，避免多批任务同时推送。
                await stop_grid_tasks(grid_id)
                grid_stop_events[grid_id] = asyncio.Event()
                grid_send_locks.setdefault(grid_id, asyncio.Lock())

                sessions = {}
                tasks = {}
                skipped_positions = []

                for i, raw_source_config in enumerate(grid_config.video_sources or []):
                    normalized_source = normalize_grid_source_config(raw_source_config)

                    if normalized_source is None:
                        if raw_source_config:
                            skipped_positions.append(i)
                            await send_grid_message_safe(grid_id, websocket, {
                                "type": "grid_cell_result",
                                "grid_position": i,
                                "position": i,
                                "session_id": f"{grid_id}_cell_{i}",
                                "detections": [],
                                "detections_count": 0,
                                "inference_time": 0,
                                "annotated_image": None,
                                "stats": {
                                    "frames_processed": 0,
                                    "total_detections": 0,
                                    "avg_inference_time": 0,
                                    "current_fps": 0
                                },
                                "timestamp": datetime.now().isoformat(),
                                "error": "该宫格视频源无效或不支持。九宫格后端仅支持 rtsp/http/file。"
                            })
                        continue

                    session_id = f"{grid_id}_cell_{i}"
                    realtime_config = RealtimeConfig(
                        model_path=grid_config.shared_model,
                        confidence=grid_config.shared_confidence,
                        iou_threshold=grid_config.shared_iou_threshold,
                        max_fps=max(1, int(grid_config.max_fps or 1)),
                        frame_skip=1,
                        source_type=normalized_source["type"],
                        source_url=normalized_source["url"],
                        loop_video=True,
                        show_labels=True,
                        show_confidence=True,
                        client_side_overlay=client_side_overlay,
                        behavior_config=normalize_behavior_config(
                            grid_config.shared_behavior_config,
                            normalized_source.get("behavior_config") or {},
                        ),
                    )

                    realtime_stats[session_id] = {
                        "session_id": session_id,
                        "frames_processed": 0,
                        "total_detections": 0,
                        "inference_times": [],
                        "avg_inference_time": 0,
                        "current_fps": 0,
                        "start_time": datetime.now().isoformat(),
                        "last_update": datetime.now().isoformat(),
                        "is_active": True,
                        "grid_position": i,
                        "source_type": realtime_config.source_type,
                        "source_url": realtime_config.source_url,
                        "config": normalize_realtime_config_data(realtime_config.dict())
                    }
                    configure_realtime_frame_delivery(
                        session_id,
                        supports_ack=supports_frame_ack,
                    )

                    sessions[i] = {
                        "session_id": session_id,
                        "config": realtime_config,
                        "source_info": normalized_source
                    }
                    behavior_session_contexts[session_id] = {
                        "owner_id": behavior_owner.get("id"),
                        "owner_username": behavior_owner.get("username") or "",
                        "mode": "grid",
                        "source_id": normalized_source["url"] or session_id,
                        "session_id": session_id,
                        "grid_id": grid_id,
                        "grid_position": str(i),
                    }

                grid_sessions[grid_id] = {
                    "grid_id": grid_id,
                    "grid_name": grid_config.grid_name,
                    "grid_size": grid_config.grid_size,
                    "sessions": sessions,
                    "created_at": grid_sessions.get(grid_id, {}).get("created_at", datetime.now().isoformat()),
                    "is_active": True,
                    "total_detections": 0,
                    "active_sources": len(sessions),
                    "skipped_positions": skipped_positions,
                    "supports_frame_ack": supports_frame_ack,
                    "client_side_overlay": client_side_overlay,
                }

                def process_grid_batch(
                    images: List[np.ndarray],
                    model_path: str,
                    confidence: float,
                    iou_threshold: float,
                ) -> List[Dict[str, Any]]:
                    return process_realtime_batch(
                        images,
                        model_path,
                        confidence,
                        iou_threshold,
                        render_annotations=not client_side_overlay,
                    )

                grid_inference_schedulers[grid_id] = GridBatchInferenceScheduler(
                    process_grid_batch,
                    max_batch_size=_bounded_env_int("GRID_BATCH_MAX_SIZE", 9, 1, 64),
                    max_wait_ms=_bounded_env_int("GRID_BATCH_WAIT_MS", 4, 0, 100),
                )

                await send_grid_message_safe(grid_id, websocket, {
                    "type": "grid_started",
                    "grid_id": grid_id,
                    "active_sources": len(sessions),
                    "skipped_positions": skipped_positions,
                    "message": f"九宫格监控已启动，{len(sessions)}个视频源"
                })

                # grid_started 先返回给前端，再启动各格子后台任务，避免第一帧结果早于启动确认。
                for position, session_info in sessions.items():
                    session_id = session_info["session_id"]
                    task = asyncio.create_task(
                        process_grid_stream(
                            grid_id=grid_id,
                            websocket=websocket,
                            session_id=session_id,
                            config=session_info["config"],
                            grid_position=int(position)
                        )
                    )
                    tasks[session_id] = task

                grid_stream_tasks[grid_id] = tasks

            elif message_type == "grid_frame_ack":
                session_id = str(message.get("session_id") or "")
                sessions = grid_sessions.get(grid_id, {}).get("sessions", {})
                valid_session_ids = {
                    str(session.get("session_id") or "")
                    for session in sessions.values()
                }
                if session_id in valid_session_ids:
                    acknowledge_realtime_image(
                        session_id,
                        message.get("frame_sequence"),
                    )

            elif message_type == "stop_grid":
                await stop_grid_tasks(grid_id)

                if grid_id in grid_sessions:
                    grid_sessions[grid_id]["is_active"] = False
                    grid_sessions[grid_id]["active_sources"] = 0

                await send_grid_message_safe(grid_id, websocket, {
                    "type": "grid_stopped",
                    "message": "九宫格监控已停止"
                })

            elif message_type in {"update_grid_config", "grid_config_update", "config_update"}:
                # 运行中动态更新九宫格推理参数。
                # 前端配置弹窗中的置信度阈值 / IoU 阈值 / 最大帧率调整后，会通过该消息即时作用到
                # 已经启动的每一路 RealtimeConfig；无需停止并重新启动九宫格。
                config_data = message.get("config", {}) or {}
                updated_count, applied_config = update_running_grid_config(grid_id, config_data)

                await send_grid_message_safe(grid_id, websocket, {
                    "type": "grid_config_updated",
                    "grid_id": grid_id,
                    "updated_sessions": updated_count,
                    "config": applied_config,
                    "silent": bool(message.get("silent", False)),
                    "message": f"九宫格运行参数已更新，已应用到 {updated_count} 路视频源"
                })

            elif message_type == "get_grid_stats":
                grid_data = grid_sessions.get(grid_id, {})
                sessions_stats = {}

                for position, session_info in grid_data.get("sessions", {}).items():
                    session_id = session_info["session_id"]
                    if session_id in realtime_stats:
                        sessions_stats[position] = realtime_stats[session_id]

                await send_grid_message_safe(grid_id, websocket, {
                    "type": "grid_stats",
                    "grid_id": grid_id,
                    "sessions": sessions_stats,
                    "total_detections": grid_data.get("total_detections", 0),
                    "active_sources": grid_data.get("active_sources", 0)
                })

            else:
                await send_grid_message_safe(grid_id, websocket, {
                    "type": "error",
                    "message": f"未知九宫格消息类型: {message_type}"
                })

    except WebSocketDisconnect:
        logger.info(f"九宫格WebSocket连接断开: {grid_id}")
    except Exception as e:
        logger.error(f"九宫格WebSocket错误 {grid_id}: {e}")
        try:
            await send_grid_message_safe(grid_id, websocket, {
                "type": "error",
                "message": f"九宫格连接错误: {str(e)}"
            })
        except Exception:
            pass
    finally:
        await stop_grid_tasks(grid_id)

        if grid_connections.get(grid_id) is websocket:
            grid_connections.pop(grid_id, None)

        grid_send_locks.pop(grid_id, None)
        cleanup_task = grid_cleanup_tasks.get(grid_id)
        if cleanup_task is None or cleanup_task.done():
            grid_stop_events.pop(grid_id, None)

        if grid_id in grid_sessions:
            grid_sessions[grid_id]["is_active"] = False
            grid_sessions[grid_id]["active_sources"] = 0


def normalize_grid_source_config(source_config: Any) -> Optional[Dict[str, Any]]:
    """统一清洗九宫格视频源配置，兼容不同前端字段名。"""
    if not source_config or not isinstance(source_config, dict):
        return None

    raw_type = (
        source_config.get("type")
        or source_config.get("source_type")
        or source_config.get("sourceType")
        or ""
    )
    raw_url = (
        source_config.get("url")
        or source_config.get("source_url")
        or source_config.get("sourceUrl")
        or source_config.get("path")
        or ""
    )

    source_type = str(raw_type or "").strip().lower()
    source_url = str(raw_url or "").strip()

    # 如果类型缺失，则尽量从 URL/路径推断。
    lowered_url = source_url.lower()
    if not source_type:
        if lowered_url.startswith("rtsp://") or lowered_url.startswith("rtsps://"):
            source_type = "rtsp"
        elif lowered_url.startswith("http://") or lowered_url.startswith("https://"):
            source_type = "http"
        elif source_url:
            source_type = "file"

    if source_type in {"rtsps"}:
        source_type = "rtsp"
    elif source_type in {"https", "stream", "mjpeg", "mjpg"}:
        source_type = "http"
    elif source_type in {"video", "uploaded", "upload"}:
        source_type = "file"

    # 浏览器 webcam 不能通过后端九宫格接口直接并发取 9 路帧，这里明确跳过。
    if source_type not in {"rtsp", "http", "file"}:
        return None

    if not source_url:
        return None

    # 当前后端运行在容器内，OpenCV 打开本地视频文件时需要文件路径。
    # 如果前端只传了 /static/... URL，则转换为容器内相对路径 static/...
    if source_type == "file" and source_url.startswith("/static/"):
        source_url = source_url.lstrip("/")

    if source_type in {"rtsp", "http"}:
        parsed = urlparse(source_url)
        expected_schemes = {"rtsp", "rtsps"} if source_type == "rtsp" else {"http", "https"}
        if parsed.scheme.lower() not in expected_schemes or not parsed.hostname:
            return None
    else:
        source_url = _grid_allowed_file_path(source_url)
        if not source_url:
            return None

    normalized = dict(source_config)
    normalized.update({
        "type": source_type,
        "source_type": source_type,
        "url": source_url,
        "source_url": source_url,
        "name": source_config.get("name") or f"{source_type.upper()}源"
    })
    return normalized


async def send_grid_message_safe(grid_id: str, websocket: WebSocket, payload: Dict[str, Any]) -> bool:
    """九宫格多个后台任务共用一个 WebSocket 时，必须串行发送。"""
    if grid_connections.get(grid_id) is not websocket:
        return False

    lock = grid_send_locks.setdefault(grid_id, asyncio.Lock())

    try:
        async with lock:
            if grid_connections.get(grid_id) is not websocket:
                return False
            await websocket.send_json(payload)
        return True
    except Exception as e:
        logger.warning(f"九宫格WebSocket发送失败: grid={grid_id}, error={e}")
        return False


async def _reap_grid_tasks(grid_id: str, tasks: Dict[str, asyncio.Task]):
    try:
        await asyncio.gather(*tasks.values(), return_exceptions=True)
        grid_stream_tasks.pop(grid_id, None)
        grid_stop_events.pop(grid_id, None)
    finally:
        if grid_cleanup_tasks.get(grid_id) is asyncio.current_task():
            grid_cleanup_tasks.pop(grid_id, None)


async def stop_grid_tasks(grid_id: str):
    """停止某个九宫格会话的所有后台流任务并释放视频资源。"""
    stop_event = grid_stop_events.get(grid_id)
    if stop_event is not None:
        stop_event.set()

    scheduler = grid_inference_schedulers.pop(grid_id, None)
    if scheduler is not None:
        await scheduler.close()

    tasks = grid_stream_tasks.get(grid_id, {}) or {}
    if tasks:
        _done, pending = await asyncio.wait(tasks.values(), timeout=15.0)
        if pending:
            reaper = grid_cleanup_tasks.get(grid_id)
            if reaper is None or reaper.done():
                grid_cleanup_tasks[grid_id] = asyncio.create_task(
                    _reap_grid_tasks(grid_id, tasks)
                )
            return False
        grid_stream_tasks.pop(grid_id, None)

    sessions = grid_sessions.get(grid_id, {}).get("sessions", {})
    for session_info in sessions.values():
        session_id = session_info.get("session_id")
        if not session_id:
            continue

        await asyncio.to_thread(cleanup_video_capture, session_id, 2.0)
        behavior_stopped = await stop_behavior_frame_scheduler(session_id, 15.0)
        if not behavior_stopped:
            return False
        await asyncio.to_thread(close_behavior_session_sync, session_id, time.time())
        realtime_frame_delivery.pop(session_id, None)
        if session_id in realtime_stats:
            realtime_stats[session_id]["is_active"] = False
            realtime_stats[session_id]["last_update"] = datetime.now().isoformat()
    return True


def update_running_grid_config(grid_id: str, config_data: Dict[str, Any]) -> tuple:
    """
    动态更新已经启动的九宫格运行参数。

    注意：每个 process_grid_stream 持有的是 session_info["config"] 这个 RealtimeConfig 对象的引用，
    因此这里直接修改对象属性，下一帧推理就会读取到新的 confidence / iou_threshold / max_fps。
    """
    grid_data = grid_sessions.get(grid_id) or {}
    sessions = grid_data.get("sessions", {}) or {}

    # 如果前端只传了部分字段，未传字段沿用每一路当前配置。
    updated_count = 0
    sample_config = None

    for session_info in sessions.values():
        realtime_config = session_info.get("config")
        if realtime_config is None:
            continue

        current_model = getattr(realtime_config, "model_path", "yolov8n.pt")
        current_confidence = getattr(realtime_config, "confidence", 0.25)
        current_iou = getattr(realtime_config, "iou_threshold", 0.7)
        current_max_fps = getattr(realtime_config, "max_fps", 10)
        current_behavior = getattr(realtime_config, "behavior_config", {})

        model_path = (
            config_data.get("shared_model")
            or config_data.get("model_path")
            or config_data.get("modelPath")
            or current_model
        )
        confidence = config_data.get(
            "shared_confidence",
            config_data.get("confidence", config_data.get("conf", current_confidence))
        )
        iou_threshold = config_data.get(
            "shared_iou_threshold",
            config_data.get("iou_threshold", config_data.get("iouThreshold", config_data.get("iou", current_iou)))
        )
        max_fps = config_data.get("max_fps", config_data.get("maxFps", current_max_fps))
        behavior_config = config_data.get(
            "shared_behavior_config",
            config_data.get("behavior_config", current_behavior),
        )

        realtime_config.model_path = str(model_path).strip() or current_model
        realtime_config.confidence = _safe_ratio(confidence, current_confidence)
        realtime_config.iou_threshold = _safe_ratio(iou_threshold, current_iou)
        realtime_config.max_fps = _safe_int(max_fps, default=current_max_fps, min_value=1, max_value=60)
        realtime_config.behavior_config = normalize_behavior_config(current_behavior, behavior_config)

        session_id = session_info.get("session_id")
        if session_id in realtime_stats:
            realtime_stats[session_id]["config"] = normalize_realtime_config_data(realtime_config.dict())
            realtime_stats[session_id]["last_update"] = datetime.now().isoformat()

        sample_config = realtime_config
        updated_count += 1

    if sample_config is not None:
        applied_config = {
            "shared_model": sample_config.model_path,
            "shared_confidence": sample_config.confidence,
            "shared_iou_threshold": sample_config.iou_threshold,
            "max_fps": sample_config.max_fps,
            "shared_behavior_config": sample_config.behavior_config,
        }
        grid_data["shared_model"] = sample_config.model_path
        grid_data["shared_confidence"] = sample_config.confidence
        grid_data["shared_iou_threshold"] = sample_config.iou_threshold
        grid_data["max_fps"] = sample_config.max_fps
    else:
        applied_config = {
            "shared_model": config_data.get("shared_model", ""),
            "shared_confidence": _safe_ratio(config_data.get("shared_confidence", 0.25), 0.25),
            "shared_iou_threshold": _safe_ratio(config_data.get("shared_iou_threshold", 0.7), 0.7),
            "max_fps": _safe_int(config_data.get("max_fps", 60), default=60, min_value=1, max_value=60),
            "shared_behavior_config": normalize_behavior_config(config_data.get("shared_behavior_config")),
        }

    grid_sessions[grid_id] = grid_data
    return updated_count, applied_config


async def process_grid_stream(
    grid_id: str,
    websocket: WebSocket,
    session_id: str,
    config: RealtimeConfig,
    grid_position: int
):
    """处理九宫格中单个视频流。"""
    consecutive_read_failures = 0
    last_stream_status = None
    frame_sequence = 0
    monitor_source_sequence = 0
    monitor_mode = 'monitoring_algorithms' in behavior_session_contexts.get(session_id, {})
    monitor_sample_interval_seconds = 30.0 if monitor_mode else 0.0
    next_monitor_sample_at = time.monotonic()

    try:
        logger.info(
            f"启动九宫格视频流: {session_id}, 位置: {grid_position}, "
            f"源类型: {config.source_type}, 地址: {redact_stream_url(config.source_url)}"
        )
        # Recorder alarm sessions take one snapshot every 30 seconds. Live
        # display uses a separate, owner-scoped raw capture in monitoring_workspace.
        while True:
            stop_event = grid_stop_events.get(grid_id)
            if stop_event is not None and stop_event.is_set():
                break
            if grid_connections.get(grid_id) is not websocket:
                break
            if not grid_sessions.get(grid_id, {}).get("is_active", False):
                break
            if grid_sessions.get(grid_id, {}).get('sessions', {}).get(grid_position, {}).get('stop_requested'):
                break

            if monitor_mode:
                wait_seconds = next_monitor_sample_at - time.monotonic()
                if wait_seconds > 0:
                    try:
                        await asyncio.wait_for(grid_stop_events[grid_id].wait(), timeout=wait_seconds)
                    except asyncio.TimeoutError:
                        pass
                    continue
                next_monitor_sample_at = time.monotonic() + monitor_sample_interval_seconds

            loop_start_time = time.time()
            # max_fps 支持运行中动态调整，因此每一轮都重新计算帧间隔。
            frame_interval = 1.0 / max(1, int(getattr(config, "max_fps", 1) or 1))

            # RTSP/HTTP use the supervised reader so each cell exposes reconnect state.
            if config.source_type in {"rtsp", "http"}:
                image, stream_health = await asyncio.to_thread(
                    video_processor.read_supervised_frame,
                    session_id,
                    config.source_type,
                    config.source_url,
                    1.0,
                )
            else:
                image = await asyncio.to_thread(get_frame_from_source, session_id, config)
                stream_health = {
                    "session_id": session_id,
                    "source_type": config.source_type,
                    "status": "live" if image is not None else "waiting",
                    "last_error": None,
                    "reconnect_count": 0,
                }
            stop_event = grid_stop_events.get(grid_id)
            if stop_event is not None and stop_event.is_set():
                break

            stream_status = (stream_health or {}).get("status", "waiting")
            if grid_sessions.get(grid_id, {}).get('sessions', {}).get(grid_position, {}).get('stop_requested'):
                break
            if session_id in realtime_stats:
                realtime_stats[session_id]["stream_health"] = stream_health

            transition_event = build_grid_stream_transition_event(
                grid_id,
                grid_position,
                session_id,
                last_stream_status,
                stream_status,
                stream_health,
            )
            if stream_status != last_stream_status:
                if not await send_grid_message_safe(grid_id, websocket, {
                    "type": "grid_cell_status",
                    "grid_position": grid_position,
                    "position": grid_position,
                    "session_id": session_id,
                    "status": stream_status,
                    "health": stream_health or {},
                    "timestamp": time.time(),
                }):
                    break
                last_stream_status = stream_status

            if transition_event:
                write_realtime_stream_event_log(transition_event)
                if transition_event["level"] == "error":
                    logger.error(
                        "Grid stream disconnected: grid=%s, position=%s, session=%s, source_type=%s, reason=%s, reconnect_count=%s",
                        grid_id,
                        grid_position,
                        session_id,
                        transition_event.get("source_type"),
                        transition_event.get("reason"),
                        transition_event.get("reconnect_count"),
                    )
                if not await send_grid_message_safe(grid_id, websocket, {
                    "type": "grid_stream_event",
                    **transition_event,
                }):
                    break

            if image is None:
                consecutive_read_failures += 1
                if monitor_mode:
                    # Retry an unavailable RTSP snapshot promptly. The 30-second
                    # cadence applies to frames actually submitted for inference.
                    next_monitor_sample_at = time.monotonic() + 1.0

                # 不要每一轮都刷 error，避免前端和日志被刷爆。
                # Network captures start asynchronously. Treat the first empty
                # read as warm-up instead of a fatal cell error; sustained
                # failures are still surfaced and retained by the soak.
                failure_thresholds = {1, 3} if monitor_mode else {30, 100}
                if consecutive_read_failures in failure_thresholds:
                    await send_grid_message_safe(grid_id, websocket, {
                        "type": "grid_cell_result",
                        "grid_position": grid_position,
                        "position": grid_position,
                        "session_id": session_id,
                        "detections": [],
                        "detections_count": 0,
                        "inference_time": 0,
                        "annotated_image": None,
                        "stats": build_grid_stats_payload(session_id, config),
                        "stream_status": stream_status,
                        "stream_health": stream_health or {},
                        "timestamp": datetime.now().isoformat(),
                        "error": f"无法从视频源读取帧，已连续失败 {consecutive_read_failures} 次"
                    })

                await asyncio.sleep(min(1.0, frame_interval))
                continue

            consecutive_read_failures = 0

            scheduler = grid_inference_schedulers.get(grid_id)
            if scheduler is None:
                raise RuntimeError("九宫格批量推理调度器未初始化或已经停止")
            from monitoring_inference import infer_selected_models
            monitoring_models = behavior_session_contexts.get(session_id, {}).get('monitoring_models')
            if monitoring_models:
                enabled_events = behavior_session_contexts.get(session_id, {}).get('monitoring_algorithms', [])
                result = await infer_selected_models(scheduler, image, monitoring_models, config.confidence, config.iou_threshold,
                    lambda: bool(grid_stop_events.get(grid_id) and grid_stop_events[grid_id].is_set()) or bool(
                        grid_sessions.get(grid_id, {}).get('sessions', {}).get(grid_position, {}).get('stop_requested')),
                    enabled_events=enabled_events,
                    revision=behavior_session_contexts.get(session_id, {}).get('inference_revision'))
                if result is None:
                    break
            else:
                result = await scheduler.infer(image, config.model_path, config.confidence, config.iou_threshold)
            if result.get("gpu_waiting"):
                await send_grid_message_safe(grid_id, websocket, {
                    "type": "gpu_waiting", "position": grid_position, "message": WAIT_MESSAGE,
                })
                await asyncio.sleep(2)
                continue
            stop_event = grid_stop_events.get(grid_id)
            if stop_event is not None and stop_event.is_set():
                break
            if grid_sessions.get(grid_id, {}).get('sessions', {}).get(grid_position, {}).get('stop_requested'):
                break
            behavior_context = behavior_session_contexts.get(session_id, {})
            monitor_source_sequence += 1
            behavior_context = dict(
                behavior_context, _monitor_source_sequence=monitor_source_sequence,
                capture_timestamp=float((stream_health or {}).get('last_frame_at') or loop_start_time),
                _monitor_capture_sequence=(stream_health or {}).get('sequence'),
            )
            behavior_started_at = time.perf_counter()
            if 'monitoring_algorithms' in behavior_context:
                # Recorder snapshots and their alarm decisions stay paired.
                # The preview scheduler may replace queued frames under load.
                behavior_result = await asyncio.to_thread(
                    process_behavior_frame_sync, session_id, image, result,
                    config.behavior_config, context=behavior_context,
                    timestamp=time.time(),
                )
            else:
                behavior_result = submit_behavior_frame(
                    session_id, image, result, config.behavior_config,
                    context=behavior_context, timestamp=time.time(),
                )
            behavior_time_ms = (time.perf_counter() - behavior_started_at) * 1000.0

            if 'monitoring_algorithms' in behavior_context:
                allowed = behavior_context['monitoring_algorithms']
                behavior_result = filter_monitor_events(behavior_result, allowed)
                result = dict(result)
                result['detections'] = filter_monitor_detections(result.get('detections') or [], allowed, config.behavior_config)
                result['detections_count'] = len(result['detections'])
            stats_payload = update_grid_realtime_stats(session_id, result, config)
            stats_payload["batch_size"] = int(result.get("batch_size", 1) or 1)
            stats_payload["batch_queue_wait_ms"] = round(float(result.get("batch_queue_wait_ms", 0) or 0), 2)
            confirmed_count = sum(
                1
                for item in (behavior_result.get("action_events") or [])
                if item.get("state") == "confirmed"
            )
            session_stats = realtime_stats.setdefault(session_id, {})
            session_stats["total_action_events"] = int(session_stats.get("total_action_events") or 0) + confirmed_count
            stats_payload["total_action_events"] = session_stats["total_action_events"]
            alarm_events = [
                item for item in (behavior_result.get("action_events") or [])
                if item.get("state") == "confirmed" and item.get("should_alert")
            ]
            session_stats["total_alarm_events"] = int(session_stats.get("total_alarm_events") or 0) + len(alarm_events)
            stats_payload["total_alarm_events"] = session_stats["total_alarm_events"]

            force_result_delivery = bool(
                result.get("error")
                or behavior_result.get("_monitor_preview")
                or behavior_result.get("action_events")
                or behavior_result.get("temporal_shadow_events")
                or behavior_result.get("temporal_score_snapshots")
                or behavior_result.get("error")
                or behavior_result.get("persistence_warning")
            )
            result_fps = min(
                max(1, int(config.max_fps or 1)),
                _bounded_env_int("GRID_RESULT_MAX_FPS", 60, 1, 60),
            )
            if "monitoring_algorithms" not in behavior_context and not force_result_delivery and not should_send_grid_result(
                session_id,
                max_fps=result_fps,
            ):
                update_grid_stage_telemetry(
                    session_id,
                    result,
                    behavior_time_ms=behavior_time_ms,
                )
                elapsed = time.time() - loop_start_time
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                continue

            monitor_mode = "monitoring_algorithms" in behavior_context
            behavior_result.pop('_monitor_preview', None)
            preview_image = image
            display_detections = result.get("detections", [])
            preview_timestamp = result.get("timestamp", datetime.now().isoformat())
            if monitor_mode:
                # The independent raw-frame pump owns monitor images; inference stays background-only.
                preview_image = None
                display_detections = []
            annotated_image_base64 = None
            annotated_image = preview_image if config.client_side_overlay else get_realtime_annotated_image(preview_image, result, config)
            preview_meta = {}
            preview_encode_time_ms = 0.0
            frame_sequence += 1
            if not monitor_mode:
                preview_fps = min(
                    max(1, int(config.max_fps or 1)),
                    _bounded_env_int("GRID_PREVIEW_MAX_FPS", 60, 1, 60),
                    getattr(websocket, "preview_max_fps", 60),
                )
                grid_preview_max_width = _bounded_env_int("GRID_OUTPUT_MAX_WIDTH", 640, 320, 1920)
                grid_preview_quality = _bounded_env_int("GRID_JPEG_QUALITY", 60, 40, 90)
                if not getattr(websocket, "no_preview", False) and annotated_image is not None and should_send_realtime_image(session_id, max_fps=preview_fps):
                    preview_encode_started_at = time.perf_counter()
                    annotated_image_base64, preview_meta = await asyncio.to_thread(
                        encode_realtime_preview,
                        annotated_image,
                        max_width=grid_preview_max_width,
                        quality=grid_preview_quality,
                    )
                    preview_encode_time_ms = (time.perf_counter() - preview_encode_started_at) * 1000.0
                    preview_meta["source_width"] = int(preview_image.shape[1])
                    preview_meta["source_height"] = int(preview_image.shape[0])
                    if annotated_image_base64:
                        mark_realtime_image_sent(session_id, frame_sequence, preview_meta)

            response = {
                "type": "grid_cell_result",
                "grid_position": grid_position,
                "position": grid_position,
                "session_id": session_id,
                "detections": display_detections,
                "detections_count": len(display_detections),
                "action_events": behavior_result.get("action_events") or [],
                "alarm_events": alarm_events,
                "active_events": behavior_result.get("active_events") or [],
                "temporal_shadow_events": behavior_result.get("temporal_shadow_events") or [],
                "temporal_score_snapshots": behavior_result.get("temporal_score_snapshots") or [],
                "temporal_model": behavior_result.get("temporal_model") or {},
                "behavior_stats": behavior_result.get("behavior_stats") or {"enabled": False},
                "behavior_persistence_status": behavior_result.get("persistence_status", "persisted"),
                "inference_time": result.get("inference_time", 0),
                "annotated_image": annotated_image_base64,
                "client_side_overlay": bool(config.client_side_overlay) and not monitor_mode,
                "frame_sequence": frame_sequence if annotated_image_base64 else None,
                "preview": preview_meta,
                "stats": stats_payload,
                "stream_status": stream_status,
                "stream_health": stream_health or {},
                "timestamp": preview_timestamp
            }

            if "error" in result:
                response["error"] = result["error"]
            if behavior_result.get("error"):
                response["behavior_error"] = behavior_result["error"]
            if behavior_result.get("persistence_warning"):
                response["behavior_persistence_warning"] = behavior_result["persistence_warning"]

            websocket_send_started_at = time.perf_counter()
            for alarm in alarm_events:
                if not await send_grid_message_safe(grid_id, websocket, {
                    "type": "alarm_event",
                    "grid_id": grid_id,
                    "grid_position": grid_position,
                    "session_id": session_id,
                    "event": {**alarm, "grid_position": grid_position},
                    "timestamp": alarm.get("alerted_at") or response.get("timestamp"),
                }):
                    break
            send_ok = await send_grid_message_safe(grid_id, websocket, response)
            websocket_send_time_ms = (time.perf_counter() - websocket_send_started_at) * 1000.0
            update_grid_stage_telemetry(
                session_id,
                result,
                behavior_time_ms=behavior_time_ms,
                preview_encode_time_ms=preview_encode_time_ms,
                websocket_send_time_ms=websocket_send_time_ms,
                result_sent=send_ok,
                image_sent=bool(annotated_image_base64 and send_ok),
            )
            if not send_ok:
                break
            mark_grid_result_sent(session_id)

            if not monitor_mode:
                elapsed = time.time() - loop_start_time
                sleep_time = max(0, frame_interval - elapsed)
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)

    except asyncio.CancelledError:
        logger.info(f"九宫格视频流任务已取消: {session_id}")
    except Exception as e:
        logger.error(f"九宫格视频流处理错误 {session_id}: {e}")
        await send_grid_message_safe(grid_id, websocket, {
            "type": "grid_cell_result",
            "grid_position": grid_position,
            "position": grid_position,
            "session_id": session_id,
            "detections": [],
            "detections_count": 0,
            "inference_time": 0,
            "annotated_image": None,
            "stats": build_grid_stats_payload(session_id, config),
            "timestamp": datetime.now().isoformat(),
            "error": str(e)
        })
    finally:
        logger.info(f"九宫格视频流处理结束: {session_id}")
        await asyncio.to_thread(cleanup_video_capture, session_id, 2.0)
        if session_id in realtime_stats:
            realtime_stats[session_id]["is_active"] = False
            realtime_stats[session_id]["last_update"] = datetime.now().isoformat()


def update_grid_realtime_stats(session_id: str, result: Dict[str, Any], config: RealtimeConfig) -> Dict[str, Any]:
    """更新九宫格单格统计并返回前端需要的 stats。"""
    stats = realtime_stats.setdefault(session_id, {
        "session_id": session_id,
        "frames_processed": 0,
        "total_detections": 0,
        "inference_times": [],
        "result_times": [],
        "start_time": datetime.now().isoformat(),
        "last_update": datetime.now().isoformat(),
        "is_active": True
    })

    detections_count = int(result.get("detections_count", 0) or 0)
    inference_time = float(result.get("inference_time", 0) or 0)

    stats["frames_processed"] = int(stats.get("frames_processed", 0)) + 1
    stats["total_detections"] = int(stats.get("total_detections", 0)) + detections_count
    stats.setdefault("inference_times", []).append(inference_time)
    completed_at = time.monotonic()
    result_times = stats.setdefault("result_times", [])
    result_times.append(completed_at)
    fps_window_seconds = _bounded_env_int("GRID_FPS_WINDOW_SECONDS", 2, 1, 10)
    cutoff = completed_at - float(fps_window_seconds)
    stats["result_times"] = [timestamp for timestamp in result_times if timestamp >= cutoff]

    if len(stats["inference_times"]) > 30:
        stats["inference_times"] = stats["inference_times"][-30:]

    avg_inference_time = (
        sum(stats["inference_times"]) / len(stats["inference_times"])
        if stats["inference_times"] else 0
    )
    recent_times = stats["result_times"]
    elapsed = recent_times[-1] - recent_times[0] if len(recent_times) > 1 else 0
    current_fps = (len(recent_times) - 1) / elapsed if elapsed > 0 else 0

    stats["avg_inference_time"] = avg_inference_time
    stats["current_fps"] = current_fps
    stats["last_update"] = datetime.now().isoformat()
    stats["is_active"] = True

    # 回填九宫格整体统计。
    grid_id = session_id.rsplit("_cell_", 1)[0] if "_cell_" in session_id else None
    if grid_id and grid_id in grid_sessions:
        sessions = grid_sessions[grid_id].get("sessions", {})
        total_detections = 0
        active_sources = 0
        for session_info in sessions.values():
            sid = session_info.get("session_id")
            sid_stats = realtime_stats.get(sid, {})
            total_detections += int(sid_stats.get("total_detections", 0) or 0)
            if sid_stats.get("is_active", False):
                active_sources += 1
        grid_sessions[grid_id]["total_detections"] = total_detections
        grid_sessions[grid_id]["active_sources"] = active_sources

    return build_grid_stats_payload(session_id, config)


def update_grid_stage_telemetry(
    session_id: str,
    result: Dict[str, Any],
    *,
    behavior_time_ms: float = 0.0,
    preview_encode_time_ms: float = 0.0,
    websocket_send_time_ms: float = 0.0,
    result_sent: bool = False,
    image_sent: bool = False,
) -> Dict[str, Any]:
    """Persist a bounded per-cell sample for grid throughput comparisons."""
    stats = realtime_stats.setdefault(session_id, {})
    sample = {
        "batch_size": int(result.get("batch_size", 1) or 1),
        "batch_queue_wait_ms": round(float(result.get("batch_queue_wait_ms", 0) or 0), 2),
        "inference_time_ms": round(float(result.get("inference_time", 0) or 0), 2),
        "annotation_time_ms": round(float(result.get("annotation_time_ms", 0) or 0), 2),
        "behavior_time_ms": round(float(behavior_time_ms or 0), 2),
        "preview_encode_time_ms": round(float(preview_encode_time_ms or 0), 2),
        "websocket_send_time_ms": round(float(websocket_send_time_ms or 0), 2),
        "delivery": {
            **get_realtime_delivery_snapshot(session_id),
            "result_sent": bool(result_sent),
            "image_sent": bool(image_sent),
        },
    }
    samples = stats.setdefault("stage_samples", [])
    samples.append(sample)
    if len(samples) > 30:
        del samples[:-30]

    timing_fields = (
        "batch_queue_wait_ms",
        "inference_time_ms",
        "annotation_time_ms",
        "behavior_time_ms",
        "preview_encode_time_ms",
        "websocket_send_time_ms",
    )
    stats["last_stage_timing"] = sample
    stats["avg_stage_timing_ms"] = {
        field: round(sum(float(item.get(field, 0) or 0) for item in samples) / len(samples), 2)
        for field in timing_fields
    }
    return sample


def build_grid_stats_payload(session_id: str, config: RealtimeConfig) -> Dict[str, Any]:
    """构造九宫格单格统计返回值。"""
    stats = realtime_stats.get(session_id, {})
    last_stage_timing = stats.get("last_stage_timing") or {}
    return {
        "frames_processed": int(stats.get("frames_processed", 0) or 0),
        "total_detections": int(stats.get("total_detections", 0) or 0),
        "avg_inference_time": round(float(stats.get("avg_inference_time", 0) or 0), 2),
        "current_fps": round(float(stats.get("current_fps", 0) or 0), 1),
        "max_fps": config.max_fps,
        "source_type": config.source_type,
        "batch_size": int(last_stage_timing.get("batch_size", 1) or 1),
        "batch_queue_wait_ms": float(last_stage_timing.get("batch_queue_wait_ms", 0) or 0),
        "stage_timing": last_stage_timing,
        "avg_stage_timing_ms": stats.get("avg_stage_timing_ms") or {},
        "delivery": last_stage_timing.get("delivery") or get_realtime_delivery_snapshot(session_id),
    }


# --- 系统资源监控接口 ---

def _safe_float(value, default=0.0):
    """安全转换为 float。"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _round_number(value, digits=2):
    """统一数值保留小数。"""
    return round(_safe_float(value), digits)


def _read_proc_cpu_snapshot():
    """
    从 /proc/stat 读取 CPU 快照。
    返回: (idle_time, total_time)。
    """
    try:
        with open('/proc/stat', 'r', encoding='utf-8') as f:
            first_line = f.readline().strip()

        parts = first_line.split()
        if not parts or parts[0] != 'cpu':
            return None

        values = [_safe_float(v) for v in parts[1:]]
        if len(values) < 4:
            return None

        idle = values[3] + (values[4] if len(values) > 4 else 0)
        total = sum(values)
        return idle, total
    except Exception:
        return None


def _get_cpu_percent():
    """
    获取 CPU 使用率。
    优先使用 psutil；没有 psutil 时用 /proc/stat 兜底。
    """
    try:
        if psutil is not None:
            return _round_number(psutil.cpu_percent(interval=0.2))
    except Exception:
        pass

    first = _read_proc_cpu_snapshot()
    if not first:
        return 0.0

    time.sleep(0.2)

    second = _read_proc_cpu_snapshot()
    if not second:
        return 0.0

    idle_delta = second[0] - first[0]
    total_delta = second[1] - first[1]

    if total_delta <= 0:
        return 0.0

    cpu_percent = (1.0 - idle_delta / total_delta) * 100
    return _round_number(max(0.0, min(100.0, cpu_percent)))


def _get_memory_info():
    """
    获取内存信息。
    返回单位：GB。
    优先使用 psutil；没有 psutil 时用 /proc/meminfo 兜底。
    """
    try:
        if psutil is not None:
            memory = psutil.virtual_memory()
            total_gb = memory.total / 1024 / 1024 / 1024
            used_gb = memory.used / 1024 / 1024 / 1024
            return {
                'percent': _round_number(memory.percent),
                'used': _round_number(used_gb),
                'total': _round_number(total_gb)
            }
    except Exception:
        pass

    try:
        meminfo = {}
        with open('/proc/meminfo', 'r', encoding='utf-8') as f:
            for line in f:
                key, value = line.split(':', 1)
                meminfo[key] = _safe_float(value.strip().split()[0])  # kB

        total_kb = meminfo.get('MemTotal', 0.0)
        available_kb = meminfo.get('MemAvailable', 0.0)
        used_kb = max(0.0, total_kb - available_kb)
        percent = (used_kb / total_kb * 100) if total_kb > 0 else 0.0

        return {
            'percent': _round_number(percent),
            'used': _round_number(used_kb / 1024 / 1024),
            'total': _round_number(total_kb / 1024 / 1024)
        }
    except Exception:
        return {
            'percent': 0.0,
            'used': 0.0,
            'total': 0.0
        }


def _get_gpu_probe() -> Dict[str, Any]:
    return probe_gpu()


def _get_gpu_info():
    """Backward-compatible GPU list backed by the shared runtime probe."""
    return _get_gpu_probe().get("gpus", [])



# --- 一阶段行为动作检测：YOLO-Pose + 跟踪 + 规则引擎 ---
# 说明：
# 1. 告警类别为使用手机、未佩戴工牌、人员摔倒；
# 2. 支持同步分析和后台任务分析；
# 3. 结果保存到 static/behavior_results，可通过 /static/behavior_results/... 访问。
behavior_analysis_tasks: Dict[str, Dict[str, Any]] = {}
behavior_analysis_task_handles: Dict[str, asyncio.Task] = {}

# 动作复核分类器统一使用 ActionTraining.vue 训练输出的路径。
# 兼容旧前端曾经传入的 /app/data/models/behavior/action_event_classifier.pkl，
# 避免行为分析页仍拿旧路径导致找不到模型。
ACTION_CLASSIFIER_CANONICAL_MODEL_PATH = "/app/models/action_event_classifier.pkl"
ACTION_CLASSIFIER_LEGACY_MODEL_PATHS = {
    "/app/data/models/behavior/action_event_classifier.pkl",
    "data/models/behavior/action_event_classifier.pkl",
    str(BEHAVIOR_MODELS_DIR / "action_event_classifier.pkl"),
}


def resolve_action_classifier_model_path(model_path: Any = None, for_training: bool = False) -> str:
    """统一动作复核分类器路径。

    训练页 ActionTraining.vue 默认保存到 /app/models/action_event_classifier.pkl。
    如果行为分析页或旧缓存传入 /app/data/models/behavior/action_event_classifier.pkl，
    这里会自动切回 /app/models/action_event_classifier.pkl。
    """
    text = str(model_path or "").strip()
    if not text or text in ACTION_CLASSIFIER_LEGACY_MODEL_PATHS:
        text = ACTION_CLASSIFIER_CANONICAL_MODEL_PATH

    # 兼容只传文件名的情况。
    if text == "action_event_classifier.pkl":
        text = ACTION_CLASSIFIER_CANONICAL_MODEL_PATH

    # 训练时确保目标目录存在。
    if for_training:
        try:
            Path(text).parent.mkdir(parents=True, exist_ok=True)
        except Exception:
            pass
        return text

    # 如果传入路径不存在，但标准路径存在，则优先使用训练页输出的标准模型。
    if not os.path.exists(text) and os.path.exists(ACTION_CLASSIFIER_CANONICAL_MODEL_PATH):
        return ACTION_CLASSIFIER_CANONICAL_MODEL_PATH

    return text


class BehaviorAnalyzeRequest(BaseModel):
    """行为分析请求。video_path 和 source_id 至少提供一个。"""
    video_path: Optional[str] = None
    source_id: Optional[str] = None
    model_path: str = "yolov8n-pose.pt"
    object_model_path: str = "/app/models/yolov8n.pt"
    conf: float = 0.35
    iou: float = 0.50
    object_conf: float = 0.25
    imgsz: int = 640
    object_imgsz: Optional[int] = None
    tracker: str = "bytetrack.yaml"
    sample_stride: int = 1
    max_frames: Optional[int] = None
    generate_annotated_video: bool = False
    annotated_video_fps: float = 6.0
    enabled_events: Optional[List[str]] = [
        "phone_use",
        "missing_badge",
        "fall_down",
    ]
    phone_classes: Optional[List[str]] = None
    transfer_classes: Optional[List[str]] = None
    object_owner_distance_ratio: float = 0.35
    transfer_max_seconds: float = 3.0
    transfer_min_owner_frames: int = 3
    transfer_person_distance_ratio: float = 0.85
    draw_objects: bool = False
    extract_event_clips: bool = True
    clip_pre_seconds: float = 1.0
    clip_post_seconds: float = 1.0
    clip_min_seconds: float = 2.0
    clip_max_seconds: float = 8.0
    behavior_config: Optional[Dict[str, Any]] = None
    temporal_mode: str = "off"
    temporal_model_id: Optional[str] = None
    temporal_model_path: Optional[str] = None
    temporal_thresholds: Optional[Dict[str, float]] = None
    temporal_clip_length: int = TEMPORAL_DEFAULT_CLIP_LENGTH
    temporal_sample_fps: float = 6.0

    # Legacy classifier fields are accepted for request compatibility only.
    enable_model_review: bool = False
    require_model_review: bool = False
    classifier_model_path: str = ACTION_CLASSIFIER_CANONICAL_MODEL_PATH
    classifier_threshold: float = 0.65
    classifier_max_frames: int = 96
    keep_candidate_events: bool = True
    keep_rejected_events: bool = True
    fallback_to_rule_events: bool = False
    verifier_mode: str = "off"
    verifier_model_id: Optional[str] = None
    verifier_model_path: Optional[str] = None
    verifier_threshold: float = 0.65


def _safe_behavior_float(value: Any, default: float, min_value: float, max_value: float) -> float:
    """行为分析参数清洗。"""
    try:
        result = float(value)
    except (TypeError, ValueError):
        result = default
    return max(min_value, min(max_value, result))


def _safe_behavior_int(value: Any, default: int, min_value: int, max_value: int) -> int:
    """行为分析整数参数清洗。"""
    try:
        result = int(value)
    except (TypeError, ValueError):
        result = default
    return max(min_value, min(max_value, result))


def resolve_behavior_video_path(req: BehaviorAnalyzeRequest) -> str:
    """
    解析行为分析视频路径。

    支持：
    - video_path: /app/static/videos/a.mp4
    - video_path: static/videos/a.mp4
    - video_path: /static/videos/a.mp4
    - video_path: http/https/rtsp/rtmp 地址
    - source_id: 已上传视频源 ID
    """
    if req.source_id:
        source = video_sources.get(str(req.source_id))
        if not source:
            raise HTTPException(status_code=404, detail=f"视频源不存在: {req.source_id}")

        source_type = source.get("type") or source.get("source_type") or "file"
        if source_type == "file":
            candidate = source.get("path") or source.get("source_url") or source.get("url")
        else:
            candidate = source.get("url") or source.get("source_url")

        if not candidate:
            raise HTTPException(status_code=400, detail=f"视频源没有可用地址: {req.source_id}")
    else:
        candidate = req.video_path

    if not candidate:
        raise HTTPException(status_code=400, detail="video_path 和 source_id 至少提供一个")

    text = str(candidate).strip()
    lower = text.lower()
    if lower.startswith(("rtsp://", "rtmp://", "http://", "https://")):
        return text

    normalized = text.replace("\\", "/")
    if normalized.startswith("/static/videos/"):
        path = Path(constants.STATIC_DIR) / "videos" / normalized.split("/static/videos/", 1)[1]
    elif normalized.startswith("static/videos/"):
        path = Path(normalized)
    elif normalized.startswith("/app/static/videos/"):
        path = Path(normalized)
    elif normalized.startswith("./static/videos/"):
        path = Path(normalized[2:])
    else:
        path = Path(text)

    if not path.exists():
        raise HTTPException(status_code=404, detail=f"视频文件不存在: {path}")

    return str(path)


def resolve_behavior_verifier_request(req: BehaviorAnalyzeRequest) -> Dict[str, Any]:
    mode = str(req.verifier_mode or "off").strip().lower()
    if req.enable_model_review and mode == "off":
        mode = "shadow"
    if mode not in {"off", "shadow", "filter"}:
        raise HTTPException(status_code=400, detail=f"不支持的人工反馈验证模式: {mode}")
    if mode == "off":
        return {
            "mode": "off",
            "model_path": req.verifier_model_path or req.classifier_model_path,
            "model_id": None,
            "model_status": None,
        }

    db = SessionLocal()
    try:
        record = None
        if req.verifier_model_id:
            record = db.query(ModelRegistry).filter(
                ModelRegistry.id == str(req.verifier_model_id),
                ModelRegistry.source == "behavior_verifier",
            ).first()
            if record is None:
                raise HTTPException(status_code=404, detail="人工反馈验证器版本不存在")
        elif not req.verifier_model_path:
            record = db.query(ModelRegistry).filter(
                ModelRegistry.source == "behavior_verifier",
                ModelRegistry.status == "active",
            ).order_by(ModelRegistry.updated_at.desc()).first()
        model_path = str(
            (record.path if record else None)
            or req.verifier_model_path
            or req.classifier_model_path
            or ACTION_CLASSIFIER_CANONICAL_MODEL_PATH
        )
        model_status = record.status if record else None
        if mode == "filter" and (record is None or model_status != "active"):
            raise HTTPException(status_code=409, detail="过滤模式只能使用已激活的人工反馈验证器")
        return {
            "mode": mode,
            "model_path": model_path,
            "model_id": record.id if record else None,
            "model_status": model_status,
        }
    finally:
        db.close()


def build_behavior_analyzer_from_request(req: BehaviorAnalyzeRequest):
    """根据请求创建行为分析器。"""
    from behavior.analyzer import BehaviorAnalyzer

    verifier = resolve_behavior_verifier_request(req)
    return BehaviorAnalyzer(
        model_path=req.model_path,
        output_root=str(Path(constants.STATIC_DIR) / "behavior_results"),
        conf=_safe_behavior_float(req.conf, 0.35, 0.0, 1.0),
        iou=_safe_behavior_float(req.iou, 0.50, 0.0, 1.0),
        tracker=req.tracker or "bytetrack.yaml",
        imgsz=_safe_behavior_int(req.imgsz, 640, 160, 1920),
        object_model_path=req.object_model_path,
        object_conf=_safe_behavior_float(req.object_conf, 0.25, 0.0, 1.0),
        object_imgsz=req.object_imgsz,
        transfer_classes=req.transfer_classes,
        object_owner_distance_ratio=req.object_owner_distance_ratio,
        transfer_max_seconds=req.transfer_max_seconds,
        transfer_min_owner_frames=req.transfer_min_owner_frames,
        transfer_person_distance_ratio=req.transfer_person_distance_ratio,
        draw_objects=req.draw_objects,
        extract_event_clips=req.extract_event_clips,
        clip_pre_seconds=req.clip_pre_seconds,
        clip_post_seconds=req.clip_post_seconds,
        clip_min_seconds=req.clip_min_seconds,
        clip_max_seconds=req.clip_max_seconds,
        keep_candidate_events=req.keep_candidate_events,
        keep_rejected_events=req.keep_rejected_events,
        enable_model_review=verifier["mode"] != "off",
        require_model_review=req.require_model_review,
        classifier_model_path=verifier["model_path"],
        classifier_threshold=_safe_behavior_float(req.verifier_threshold, req.classifier_threshold, 0.0, 1.0),
        classifier_max_frames=_safe_behavior_int(req.classifier_max_frames, 96, 8, 512),
        verifier_mode=verifier["mode"],
        fallback_to_rule_events=True if verifier["mode"] == "shadow" else req.fallback_to_rule_events,
        temporal_mode=str(req.temporal_mode or "off").strip().lower(),
        temporal_model_id=str(req.temporal_model_id or ""),
        temporal_model_path=str(req.temporal_model_path or ""),
        temporal_thresholds=req.temporal_thresholds,
        temporal_clip_length=_safe_behavior_int(req.temporal_clip_length, TEMPORAL_DEFAULT_CLIP_LENGTH, 4, 48),
        temporal_sample_fps=_safe_behavior_float(req.temporal_sample_fps, 6.0, 1.0, 15.0),
    )


def make_behavior_time_task_id() -> str:
    """生成行为分析任务 ID；同时作为 static/behavior_results 下的输出目录名。"""
    output_root = Path(constants.STATIC_DIR) / "behavior_results"
    output_root.mkdir(parents=True, exist_ok=True)

    time_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    for index in range(1000):
        task_id = time_name if index == 0 else f"{time_name}_{index:02d}"
        if task_id in behavior_analysis_tasks:
            continue
        if (output_root / task_id).exists():
            continue
        return task_id

    return f"{time_name}_{int(time.time() * 1000)}"


def run_behavior_analysis_sync(req: BehaviorAnalyzeRequest, progress_callback=None, output_task_id: Optional[str] = None) -> Dict[str, Any]:
    """同步执行行为分析。供同步接口和后台任务共用。"""
    payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    payload["video_path"] = resolve_behavior_video_path(req)
    payload["source_id"] = req.source_id
    verifier = resolve_behavior_verifier_request(req)
    payload["verifier_mode"] = verifier["mode"]
    payload["verifier_model_path"] = verifier["model_path"]
    return run_behavior_analysis_payload(
        payload,
        output_task_id=output_task_id,
        progress_callback=progress_callback,
    )


def build_behavior_task_status_payload(task: JobTask, db, include_result: bool = False) -> Dict[str, Any]:
    """构造 PostgreSQL 持久化行为分析任务状态。"""
    payload = task.to_dict()
    payload["error"] = task.error_detail
    payload["completed_at"] = task.finished_at.isoformat() if task.finished_at else None
    latest_progress = (
        db.query(JobEvent)
        .filter(JobEvent.task_id == task.id, JobEvent.event_type == "progress")
        .order_by(JobEvent.sequence.desc())
        .first()
    )
    progress_payload = dict(latest_progress.payload_json or {}) if latest_progress else {}
    for key in ("frame_idx", "processed_frames", "total_frames", "elapsed_seconds"):
        payload[key] = progress_payload.get(key, 0)
    if not include_result:
        payload.pop("result", None)
        payload.pop("results", None)
    payload["worker"] = read_behavior_worker_heartbeat()
    return payload


def _behavior_result_media_path(value: Any) -> Optional[str]:
    text_value = str(value or "").strip()
    if not text_value:
        return None
    if text_value.startswith("/api/static/"):
        return "/app/static/" + text_value[len("/api/static/"):]
    if text_value.startswith("/static/"):
        return "/app/static/" + text_value[len("/static/"):]
    return text_value


def persist_batch_behavior_result(
    task_info: Dict[str, Any],
    req: BehaviorAnalyzeRequest,
    result: Dict[str, Any],
) -> None:
    owner_id = task_info.get("owner_id")
    if owner_id is None:
        return
    created_at = datetime.fromisoformat(task_info["created_at"])
    base_timestamp = created_at.timestamp()
    confirmed = []
    resolved = []
    for event in result.get("events") or []:
        event_id = str(event.get("event_id") or uuid.uuid4())
        start_time = base_timestamp + float(event.get("start_time") or 0.0)
        confirmed_time = base_timestamp + float(
            event.get("confirmed_time", event.get("start_time")) or 0.0
        )
        end_time = base_timestamp + float(event.get("end_time") or event.get("start_time") or 0.0)
        item = {
            "event_id": event_id,
            "event_type": event.get("event_type"),
            "state": "confirmed",
            "severity": event.get("severity") or {
                "phone_use": "medium",
                "fall_down": "high",
                "missing_badge": "medium",
            }.get(event.get("event_type"), "low"),
            "rule_score": float(event.get("rule_score", event.get("confidence", 0.0)) or 0.0),
            "should_alert": False,
            "track_ids": list(event.get("persons") or []),
            "object_id": event.get("object_id"),
            "object_class": event.get("object_class"),
            "reason": dict(event.get("reason") or {}),
            "decision": {
                "rule": event.get("rule_decision") or {},
                "temporal": event.get("temporal_decision") or {},
                "final": event.get("final_decision") or {},
                "input_quality": event.get("input_quality") or {},
                "temporal_span": event.get("temporal_span") or {},
                "fallback": bool(event.get("fallback", True)),
                "training_feature_vector": event.get("training_feature_vector") or [],
            },
            "decision_model_id": event.get("model_id") or (event.get("reason") or {}).get("model_id"),
            "decision_model_version": event.get("model_version") or (event.get("reason") or {}).get("model_version"),
            "started_at": start_time,
            "confirmed_at": confirmed_time,
            "snapshot_path": _behavior_result_media_path(event.get("snapshot")),
            "clip_path": _behavior_result_media_path(event.get("clip")),
            "evidence_status": "ready" if event.get("snapshot") or event.get("clip") else "failed",
            "evidence_error": event.get("clip_error"),
        }
        confirmed.append(item)
        resolved.append({**item, "state": "resolved", "ended_at": end_time})
    context = {
        "owner_id": int(owner_id),
        "owner_username": str(task_info.get("owner_username") or ""),
        "mode": "batch",
        "source_id": str(req.source_id or req.video_path or result.get("video_path") or task_info["task_id"]),
        "session_id": task_info["task_id"],
    }
    persist_behavior_transitions(confirmed, **context)
    persist_behavior_transitions(resolved, **context)


async def run_behavior_analysis_task(task_id: str, req: BehaviorAnalyzeRequest):
    """后台执行行为分析，避免长视频导致 HTTP 请求超时。"""
    task_info = behavior_analysis_tasks.get(task_id)
    if not task_info:
        return

    def update_progress(payload: Dict[str, Any]):
        task_info["progress"] = payload.get("progress", task_info.get("progress", 0.0))
        task_info["message"] = payload.get("message", task_info.get("message", "正在分析视频"))
        task_info["frame_idx"] = payload.get("frame_idx", task_info.get("frame_idx", 0))
        task_info["processed_frames"] = payload.get("processed_frames", task_info.get("processed_frames", 0))
        task_info["updated_at"] = datetime.now().isoformat()

    try:
        task_info.update({
            "status": "running",
            "progress": 1.0,
            "message": "正在加载行为分析模型...",
            "updated_at": datetime.now().isoformat(),
        })

        result = await asyncio.to_thread(run_behavior_analysis_sync, req, update_progress, output_task_id=task_id)
        await asyncio.to_thread(persist_batch_behavior_result, task_info, req, result)

        task_info.update({
            "status": "completed",
            "progress": 100.0,
            "message": "行为分析完成",
            "result": result,
            "completed_at": datetime.now().isoformat(),
            "updated_at": datetime.now().isoformat(),
        })
    except Exception as e:
        logger.exception(f"行为分析任务失败: task_id={task_id}, error={e}")
        task_info.update({
            "status": "failed",
            "message": f"行为分析失败: {str(e)}",
            "error": str(e),
            "updated_at": datetime.now().isoformat(),
        })
    finally:
        behavior_analysis_task_handles.pop(task_id, None)


@app.post("/behavior/analyze_video")
@app.post("/api/behavior/analyze_video")
@gpu_http_guard
async def analyze_behavior_video(req: BehaviorAnalyzeRequest):
    """
    同步行为分析接口。

    适合短视频或先调试算法链路；长视频建议使用 /behavior/tasks/start。
    """
    try:
        result = await asyncio.to_thread(run_behavior_analysis_sync, req)
        return {
            "success": True,
            "message": "行为分析完成",
            "data": result,
        }
    except HTTPException:
        raise
    except Exception as e:
        logger.exception(f"行为分析失败: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/behavior/tasks/start")
@app.post("/api/behavior/tasks/start")
async def start_behavior_analysis_task(req: BehaviorAnalyzeRequest, http_request: Request, db=Depends(get_db)):
    """启动后台行为分析任务。"""
    payload = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    payload["video_path"] = resolve_behavior_video_path(req)
    verifier = resolve_behavior_verifier_request(req)
    payload["verifier_mode"] = verifier["mode"]
    payload["verifier_model_path"] = verifier["model_path"]
    current_user = get_request_current_user(http_request)
    task = create_job(
        db,
        job_type="behavior_analysis",
        owner_username=str(current_user.get("username") or get_request_current_username(http_request)),
        owner_id=current_user.get("id"),
        input_data=payload,
        message="行为分析任务已进入队列",
    )
    return build_behavior_task_status_payload(task, db, include_result=False)


@app.get("/behavior/tasks/status/{task_id}")
@app.get("/api/behavior/tasks/status/{task_id}")
async def get_behavior_analysis_task_status(task_id: str, request: Request, db=Depends(get_db)):
    """查询行为分析任务进度。"""
    task = db.query(JobTask).filter(JobTask.id == task_id, JobTask.type == "behavior_analysis").first()
    if not task:
        raise HTTPException(status_code=404, detail="行为分析任务不存在")
    require_job_visibility(task, request)
    return build_behavior_task_status_payload(task, db, include_result=False)


@app.get("/behavior/tasks/result/{task_id}")
@app.get("/api/behavior/tasks/result/{task_id}")
async def get_behavior_analysis_task_result(task_id: str, request: Request, db=Depends(get_db)):
    """查询行为分析任务结果。"""
    task = db.query(JobTask).filter(JobTask.id == task_id, JobTask.type == "behavior_analysis").first()
    if not task:
        raise HTTPException(status_code=404, detail="行为分析任务不存在")
    require_job_visibility(task, request)
    return build_behavior_task_status_payload(task, db, include_result=True)


@app.delete("/behavior/tasks/{task_id}")
@app.delete("/api/behavior/tasks/{task_id}")
async def delete_behavior_analysis_task(task_id: str, request: Request, db=Depends(get_db)):
    """活动任务转为协作取消；终态任务删除持久化记录。"""
    task = db.query(JobTask).filter(JobTask.id == task_id, JobTask.type == "behavior_analysis").first()
    if not task:
        raise HTTPException(status_code=404, detail="行为分析任务不存在")
    require_job_visibility(task, request)
    if task.status in ACTIVE_STATUSES:
        task = request_job_cancel(db, task_id)
        return {"success": True, "cancel_requested": True, "task": task.to_dict()}
    result = delete_terminal_job(db, task_id, job_type="behavior_analysis")
    return {"success": True, "message": "行为分析任务记录已删除", **result}


# --- 第三阶段第 2 小步：事件片段人工复核接口 ---
try:
    from behavior.sample_manager import BehaviorSampleManager
except Exception as sample_manager_import_error:
    BehaviorSampleManager = None
    logger.warning(f"行为样本复核模块导入失败: {sample_manager_import_error}")

ALLOWED_REVIEW_LABELS = {
    "phone_use",
    "missing_badge",
    "fall_down",
    "normal",
    "uncertain",
    "delete",
}



BEHAVIOR_MEDIA_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v",
    ".jpg", ".jpeg", ".png", ".bmp", ".webp",
}


def build_static_url_from_any_path(value: Any) -> Optional[str]:
    """
    将 sample_manager / analyzer 返回的各种文件路径统一转换成浏览器可访问的 /static/... URL。

    支持：
    - /app/static/behavior_results/xxx/a.mp4
    - static/behavior_results/xxx/a.mp4
    - /static/behavior_results/xxx/a.mp4
    - /api/static/behavior_results/xxx/a.mp4
    - constants.STATIC_DIR 下的绝对路径
    """
    if value is None:
        return None

    text = str(value).strip()
    if not text:
        return None

    lower = text.lower()
    if lower.startswith(("http://", "https://", "data:", "blob:")):
        return text

    try:
        from urllib.parse import urlparse, unquote, quote
        parsed = urlparse(text)
        path_text = unquote(parsed.path if parsed.scheme else text)
    except Exception:
        from urllib.parse import quote
        path_text = text

    normalized = path_text.replace("\\", "/").strip()

    if normalized.startswith("/api/static/"):
        relative = normalized.split("/api/static/", 1)[1]
    elif normalized.startswith("/static/"):
        relative = normalized.split("/static/", 1)[1]
    elif normalized.startswith("static/"):
        relative = normalized.split("static/", 1)[1]
    elif normalized.startswith("./static/"):
        relative = normalized.split("./static/", 1)[1]
    elif normalized.startswith("/app/static/"):
        relative = normalized.split("/app/static/", 1)[1]
    else:
        try:
            relative_path = Path(normalized).resolve().relative_to(STATIC_ROOT)
            relative = "/".join(relative_path.parts)
        except Exception:
            try:
                # 兜底兼容 constants.STATIC_DIR 不是 STATIC_ROOT 字符串形式时的情况。
                relative_path = Path(normalized).resolve().relative_to(Path(constants.STATIC_DIR).resolve())
                relative = "/".join(relative_path.parts)
            except Exception:
                return None

    relative = relative.lstrip("/")
    if not relative:
        return None

    # 分段 quote，保留 /，兼容中文文件名、空格。
    return "/static/" + "/".join(quote(part) for part in relative.split("/"))


def normalize_behavior_review_media_payload(payload: Any) -> Any:
    """
    递归规范化行为复核接口返回值。

    不破坏 sample_manager 的原字段，只额外补齐浏览器可访问 URL：
    - xxx_path -> xxx_url
    - clip_path/video_path -> clip_url/video_url
    - snapshot_path/image_path/preview_path -> snapshot_url/image_url/preview_url
    同时给每个媒体项兜底填充 url 字段，方便旧前端直接使用。
    """
    media_path_keys = {
        "path", "file_path", "saved_path", "clip_path", "video_path", "image_path",
        "snapshot_path", "preview_path", "thumbnail_path", "frame_path",
    }
    media_url_keys = {
        "url", "clip_url", "video_url", "image_url", "snapshot_url", "preview_url",
        "thumbnail_url", "frame_url",
    }

    if isinstance(payload, list):
        return [normalize_behavior_review_media_payload(item) for item in payload]

    if not isinstance(payload, dict):
        return payload

    item = {key: normalize_behavior_review_media_payload(value) for key, value in payload.items()}

    discovered_urls: List[str] = []
    for key, value in list(item.items()):
        if not isinstance(value, str):
            continue

        candidate_url = build_static_url_from_any_path(value)
        if not candidate_url:
            continue

        suffix = Path(str(value).split("?", 1)[0]).suffix.lower()
        if suffix and suffix not in BEHAVIOR_MEDIA_EXTENSIONS:
            continue

        discovered_urls.append(candidate_url)

        if key in media_url_keys:
            item[key] = candidate_url
            continue

        if key in media_path_keys or key.endswith("_path"):
            if key.endswith("_path"):
                item[key[:-5] + "_url"] = candidate_url
            elif key == "path":
                item.setdefault("file_url", candidate_url)

    if discovered_urls:
        item.setdefault("url", discovered_urls[0])

        # 根据文件类型补齐最常用字段。
        for url in discovered_urls:
            suffix = Path(url.split("?", 1)[0]).suffix.lower()
            if suffix in {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}:
                item.setdefault("clip_url", url)
                item.setdefault("video_url", url)
            elif suffix in {".jpg", ".jpeg", ".png", ".bmp", ".webp"}:
                item.setdefault("image_url", url)
                item.setdefault("snapshot_url", url)
                item.setdefault("preview_url", url)

    return item

class BehaviorReviewRequest(BaseModel):
    task_id: str
    event_id: int
    label: str
    note: str = ""
    review_reason: str = ""
    reason_code: str = ""
    input_quality: str = ""
    evaluability: str = ""
    occlusion: str = ""
    media_quality: str = ""


def _get_behavior_sample_manager():
    if BehaviorSampleManager is None:
        raise RuntimeError("样本复核模块不可用，请检查 backend/behavior/sample_manager.py")
    return BehaviorSampleManager(
        behavior_results_dir=Path(constants.STATIC_DIR) / "behavior_results",
        samples_root=Path("data") / "action_samples",
    )


def _behavior_review_owner(request: Request) -> Tuple[Optional[int], str]:
    current_user = get_request_current_user(request)
    return get_request_current_user_id(current_user), str(current_user.get("username") or "unknown")


def migrate_legacy_behavior_review_samples(db) -> int:
    """Import legacy JSON-manifest reviews once without duplicating re-reviewed samples."""
    try:
        samples = (_get_behavior_sample_manager().dataset() or {}).get("samples", []) or []
    except Exception as exc:
        logger.warning("读取历史行为复核样本失败，跳过迁移: %s", exc)
        return 0

    migrated = 0
    seen_sources = set()
    for sample in samples:
        source_task_id = str(sample.get("source_task_id") or "").strip()
        source_event_id = str(sample.get("source_event_id") or "").strip()
        label = str(sample.get("label") or "").strip()
        if not source_task_id or not source_event_id or label in {"", "delete"}:
            continue
        source_key = (source_task_id, source_event_id)
        if source_key in seen_sources:
            continue
        seen_sources.add(source_key)
        exists = db.query(BehaviorReviewSample.id).filter(
            BehaviorReviewSample.source_task_id == source_task_id,
            BehaviorReviewSample.source_event_id == source_event_id,
        ).first()
        if exists:
            continue
        db.add(BehaviorReviewSample(
            owner_id=None,
            owner_username="legacy",
            source_task_id=source_task_id,
            source_event_id=source_event_id,
            source_event_type=sample.get("source_event_type"),
            review_label=label,
            review_status="reviewed",
            sample_type=str(sample.get("sample_type") or "clip"),
            clip_path=sample.get("clip_path"),
            image_path=sample.get("image_path"),
            evidence_json={"legacy_manifest": True},
            note=str(sample.get("note") or ""),
            reviewed_by="legacy_import",
        ))
        migrated += 1
    if migrated:
        db.commit()
    return migrated


def _upsert_behavior_review_sample(
    db,
    request: Request,
    req: BehaviorReviewRequest,
    payload: Dict[str, Any],
) -> Optional[BehaviorReviewSample]:
    owner_id, owner_username = _behavior_review_owner(request)
    source_event_id = str(req.event_id)
    row = db.query(BehaviorReviewSample).filter(
        BehaviorReviewSample.owner_username == owner_username,
        BehaviorReviewSample.source_task_id == str(req.task_id),
        BehaviorReviewSample.source_event_id == source_event_id,
    ).first()
    if row is None:
        row = db.query(BehaviorReviewSample).filter(
            BehaviorReviewSample.owner_username == "legacy",
            BehaviorReviewSample.source_task_id == str(req.task_id),
            BehaviorReviewSample.source_event_id == source_event_id,
        ).first()
    if req.label == "delete":
        if row is not None:
            db.delete(row)
            db.commit()
        return None

    review = dict(payload.get("review") or {})
    evaluability = str(req.evaluability or req.input_quality or "").strip().lower()
    is_evaluable = evaluability not in {"not_evaluable", "not-evaluable"}
    requested_label = str(req.label)
    source_event_type = str(review.get("source_event_type") or "").strip()
    feature_vector = review.get("training_feature_vector") or []
    model_family = {
        "phone_use": "pose_object",
        "missing_badge": "badge_image",
        "fall_down": "pose",
    }.get(source_event_type or requested_label, "pose")
    decision_evidence = {
        "model_family": model_family,
        "training_feature_vector": feature_vector,
        "feature_schema_version": review.get("feature_schema_version"),
    }
    values = {
        "owner_id": owner_id,
        "owner_username": owner_username,
        "source_task_id": str(req.task_id),
        "source_event_id": source_event_id,
        "source_event_type": review.get("source_event_type"),
        "review_label": requested_label if is_evaluable else "uncertain",
        "review_status": "reviewed",
        "sample_type": str(review.get("sample_type") or "clip"),
        "clip_path": review.get("reviewed_clip_path"),
        "image_path": review.get("reviewed_image_path"),
        "evidence_json": {
            "clip": review.get("clip"),
            "snapshot": review.get("snapshot"),
            "source_image": review.get("source_image"),
            "reason": review.get("reason") or {},
            "training_feature_vector": feature_vector,
            "decision": decision_evidence,
            "negative_for": [source_event_type] if requested_label == "normal" and source_event_type else [],
            "persons": review.get("persons") or [],
            "object_id": review.get("object_id"),
            "object_class": review.get("object_class"),
            "review_reason": str(req.reason_code or req.review_reason or ""),
            "input_quality": str(req.input_quality or ""),
            "evaluability": str(req.evaluability or ""),
            "evaluable": is_evaluable,
            "review_status": "reviewed" if is_evaluable else "not_evaluable",
            "requested_label": requested_label,
            "occlusion": str(req.occlusion or ""),
            "media_quality": str(req.media_quality or ""),
        },
        "note": str(req.note or ""),
        "reviewed_by": owner_username,
        "reviewed_at": datetime.now(),
    }
    if row is None:
        row = BehaviorReviewSample(**values)
        db.add(row)
    else:
        for key, value in values.items():
            setattr(row, key, value)
    db.commit()
    db.refresh(row)
    return row


def _behavior_review_rows(db, request: Request) -> List[BehaviorReviewSample]:
    current_user = get_request_current_user(request)
    query = db.query(BehaviorReviewSample)
    if current_user.get("role") != "admin":
        query = query.filter(
            BehaviorReviewSample.owner_username == str(current_user.get("username") or "")
        )
    return query.order_by(BehaviorReviewSample.reviewed_at.desc()).all()


def _build_behavior_review_task_option(task_dir: Path, manager: Any = None) -> Dict[str, Any]:
    """把 static/behavior_results 下真实存在的一个任务目录转换为前端下拉框选项。"""
    task_dir = Path(task_dir)

    # 关键修复：这里必须实时判断目录是否仍然存在。
    # 用户在宿主机删除 data/static/behavior_results/<task_id> 后，
    # 刷新路径时不能再把历史内存任务或已删除目录返回给前端。
    if not task_dir.exists() or not task_dir.is_dir():
        raise FileNotFoundError(f"行为分析结果目录不存在: {task_dir}")

    task_id = task_dir.name
    stat = task_dir.stat()

    clip_count = 0
    reviewed_count = 0
    pending_count = 0
    list_error = None

    if manager is not None:
        try:
            payload = manager.list_task_clips(task_id)
            clips = payload.get("clips", []) if isinstance(payload, dict) else []
            clip_count = len(clips)
            reviewed_count = len([item for item in clips if item.get("review_status") == "reviewed"])
            pending_count = max(0, clip_count - reviewed_count)
        except FileNotFoundError:
            # 目录在扫描和读取之间被删除时，直接向上抛出，让列表接口跳过该项。
            raise
        except Exception as e:
            # 目录存在但暂时不是可复核任务时，也返回给前端，方便定位路径。
            list_error = str(e)

    if clip_count == 0:
        try:
            media_suffixes = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".jpg", ".jpeg", ".png", ".bmp", ".webp"}
            clip_count = len([
                p for p in task_dir.rglob("*")
                if p.is_file() and p.suffix.lower() in media_suffixes
            ])
            pending_count = max(0, clip_count - reviewed_count)
        except FileNotFoundError:
            raise
        except Exception:
            pass

    created_at = datetime.fromtimestamp(stat.st_ctime).isoformat()
    updated_at = datetime.fromtimestamp(stat.st_mtime).isoformat()

    return {
        "task_id": task_id,
        "id": task_id,
        "name": task_id,
        "path": str(task_dir),
        "result_path": str(task_dir),
        "url": f"/static/behavior_results/{task_id}",
        "clip_count": clip_count,
        "reviewed_count": reviewed_count,
        "pending_count": pending_count,
        "created_at": created_at,
        "updated_at": updated_at,
        "modified_time": stat.st_mtime,
        "list_error": list_error,
    }


@app.get("/behavior/review/tasks")
@app.get("/api/behavior/review/tasks")
async def list_behavior_review_tasks():
    """
    从 static/behavior_results 实时读取可复核的行为分析任务目录。

    关键规则：接口只返回真实存在的目录。
    用户从宿主机删除 data/static/behavior_results/<task_id> 后，刷新路径时不会再返回该任务。
    """
    behavior_results_dir = Path(constants.STATIC_DIR) / "behavior_results"
    behavior_results_dir.mkdir(parents=True, exist_ok=True)

    try:
        manager = _get_behavior_sample_manager()
    except Exception as e:
        logger.warning(f"行为样本管理器不可用，仅返回目录列表: {e}")
        manager = None

    tasks: List[Dict[str, Any]] = []
    skipped_missing: List[str] = []
    removed_stale_memory_tasks: List[str] = []

    try:
        # 只扫描当前磁盘真实存在的目录，避免返回已被手动删除的历史路径。
        for task_dir in behavior_results_dir.iterdir():
            if not task_dir.exists() or not task_dir.is_dir():
                continue
            try:
                tasks.append(_build_behavior_review_task_option(task_dir, manager))
            except FileNotFoundError:
                skipped_missing.append(task_dir.name)
                continue
    except Exception as e:
        logger.exception("读取行为分析结果目录失败")
        raise HTTPException(status_code=500, detail=f"读取行为分析结果目录失败: {str(e)}")

    # 清理已经没有磁盘目录的历史内存任务，避免后端热运行期间继续把删除过的路径返回给前端。
    # 仅保留 pending/running/analyzing 这类正在生成目录的任务。
    active_statuses = {"pending", "running", "analyzing"}
    existing_task_ids = {item["task_id"] for item in tasks}

    for task_id, task_info in list(behavior_analysis_tasks.items()):
        task_dir = behavior_results_dir / task_id
        status = str(task_info.get("status", "")).lower()

        if task_id in existing_task_ids:
            continue

        if task_dir.exists() and task_dir.is_dir():
            try:
                tasks.append(_build_behavior_review_task_option(task_dir, manager))
                existing_task_ids.add(task_id)
            except FileNotFoundError:
                skipped_missing.append(task_id)
            continue

        if status in active_statuses:
            # 新任务刚创建时，目录可能还没生成；保留运行中任务，但明确标记。
            tasks.append({
                "task_id": task_id,
                "id": task_id,
                "name": task_id,
                "path": str(task_dir),
                "result_path": str(task_dir),
                "url": f"/static/behavior_results/{task_id}",
                "clip_count": 0,
                "reviewed_count": 0,
                "pending_count": 0,
                "status": task_info.get("status", "unknown"),
                "created_at": task_info.get("created_at"),
                "updated_at": task_info.get("updated_at"),
                "modified_time": time.time(),
                "list_error": "任务正在生成，结果目录尚未写入完成",
            })
        else:
            behavior_analysis_tasks.pop(task_id, None)
            removed_stale_memory_tasks.append(task_id)

    tasks.sort(key=lambda item: float(item.get("modified_time") or 0), reverse=True)

    return {
        "success": True,
        "root_path": str(behavior_results_dir),
        "root_url": "/static/behavior_results",
        "total": len(tasks),
        "tasks": normalize_behavior_review_media_payload(tasks),
        "items": normalize_behavior_review_media_payload(tasks),
        "data": normalize_behavior_review_media_payload(tasks),
        "skipped_missing": skipped_missing,
        "removed_stale_memory_tasks": removed_stale_memory_tasks,
    }


@app.get("/behavior/review/tasks/{task_id}/clips")
@app.get("/api/behavior/review/tasks/{task_id}/clips")
async def list_behavior_review_clips(task_id: str):
    """列出某个行为分析任务生成的事件片段。"""
    try:
        manager = _get_behavior_sample_manager()
        payload = manager.list_task_clips(task_id)
        return normalize_behavior_review_media_payload(payload)
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("获取行为复核片段失败")
        raise HTTPException(status_code=500, detail=f"获取行为复核片段失败: {str(e)}")


@app.post("/behavior/review/clips/review")
@app.post("/api/behavior/review/clips/review")
async def review_behavior_clip(req: BehaviorReviewRequest, http_request: Request, db=Depends(get_db)):
    """保存单个事件片段的人工复核结果，并归档为训练样本。"""
    try:
        if req.label not in ALLOWED_REVIEW_LABELS:
            raise HTTPException(status_code=400, detail=f"不支持的复核标签: {req.label}")
        manager = _get_behavior_sample_manager()
        payload = manager.review_clip(
            task_id=req.task_id,
            event_id=req.event_id,
            label=req.label,
            note=req.note,
        )
        persisted = _upsert_behavior_review_sample(db, http_request, req, payload)
        payload["persisted_sample"] = persisted.to_dict() if persisted else None
        return normalize_behavior_review_media_payload(payload)
    except HTTPException:
        raise
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except KeyError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("保存行为复核结果失败")
        raise HTTPException(status_code=500, detail=f"保存行为复核结果失败: {str(e)}")


@app.get("/behavior/review/stats")
@app.get("/api/behavior/review/stats")
async def get_behavior_review_stats(http_request: Request, db=Depends(get_db)):
    """获取已复核样本统计。"""
    try:
        rows = _behavior_review_rows(db, http_request)
        labels: Dict[str, int] = {}
        for row in rows:
            labels[row.review_label] = labels.get(row.review_label, 0) + 1
        return {
            "total_reviewed": len(rows),
            "labels": labels,
            "trainable": sum(1 for row in rows if row.review_label not in {"uncertain", "delete"}),
            "storage": "postgresql",
        }
    except Exception as e:
        logger.exception("获取行为复核统计失败")
        raise HTTPException(status_code=500, detail=f"获取行为复核统计失败: {str(e)}")


@app.get("/behavior/review/dataset")
@app.get("/api/behavior/review/dataset")
async def get_behavior_review_dataset(http_request: Request, db=Depends(get_db)):
    """获取可用于训练的复核样本清单。"""
    try:
        rows = _behavior_review_rows(db, http_request)
        payload = {
            "total": len(rows),
            "samples": [row.to_training_sample() for row in rows],
            "labels": sorted({row.review_label for row in rows}),
            "storage": "postgresql",
        }
        return normalize_behavior_review_media_payload(payload)
    except Exception as e:
        logger.exception("获取行为训练样本清单失败")
        raise HTTPException(status_code=500, detail=f"获取行为训练样本清单失败: {str(e)}")


# --- 第三阶段第 3/4 小步：动作复核分类器训练与预测接口 ---
try:
    from behavior.action_classifier import (
        DEFAULT_MODEL_PATH as ACTION_CLASSIFIER_DEFAULT_MODEL_PATH,
        get_model_info as get_action_classifier_model_info,
        load_action_verifier,
        train_classifier_from_embedded_features,
        train_classifier_from_samples,
    )
except Exception as action_classifier_import_error:
    ACTION_CLASSIFIER_DEFAULT_MODEL_PATH = ACTION_CLASSIFIER_CANONICAL_MODEL_PATH
    get_action_classifier_model_info = None
    load_action_verifier = None
    train_classifier_from_embedded_features = None
    train_classifier_from_samples = None
    logger.warning(f"动作复核分类器模块导入失败: {action_classifier_import_error}")


class ActionClassifierTrainRequest(BaseModel):
    model_path: str = ACTION_CLASSIFIER_CANONICAL_MODEL_PATH
    behavior_type: str = ""
    labels: Optional[List[str]] = None
    include_uncertain: bool = False
    min_samples_per_class: int = 20
    max_frames_per_clip: int = 96


class TemporalFusionTrainRequest(BaseModel):
    """Persistent training request for the shared three-head temporal model."""

    event_types: List[str] = list(TEMPORAL_EVENT_TYPES)
    include_all_reviewers: bool = True
    clip_length: int = TEMPORAL_DEFAULT_CLIP_LENGTH
    image_size: int = TEMPORAL_DEFAULT_IMAGE_SIZE
    epochs: int = 3
    batch_size: int = 2
    learning_rate: float = 0.0005
    device: str = "auto"
    thresholds: Dict[str, float] = dict(TEMPORAL_DEFAULT_THRESHOLDS)


class ActionClassifierPredictRequest(BaseModel):
    clip_path: str
    model_path: str = ACTION_CLASSIFIER_CANONICAL_MODEL_PATH
    max_frames_per_clip: int = 96


class BehaviorModelActivationPayload(BaseModel):
    reason: str = ""


action_classifier_task_handles: Dict[str, asyncio.Task] = {}
require_algorithm_training_menu = require_any_menu("/algorithm-training", "/action-training")


def _build_temporal_training_task_payload(
    db,
    task_id: str,
    http_request: Request,
    include_result: bool = False,
) -> Dict[str, Any]:
    task = db.query(JobTask).filter(
        JobTask.id == task_id,
        JobTask.type == "behavior_temporal_training",
    ).first()
    if task is None:
        raise HTTPException(status_code=404, detail="端到端行为训练任务不存在")
    require_job_visibility(task, http_request)
    payload = task.to_dict()
    payload["error"] = task.error_detail
    if not include_result:
        payload.pop("result", None)
        payload.pop("results", None)
    return payload


@app.post("/behavior/temporal-models/train")
@app.post("/api/behavior/temporal-models/train")
async def start_temporal_fusion_training(
    req: TemporalFusionTrainRequest,
    http_request: Request,
    current_user: Dict[str, Any] = Depends(require_algorithm_training_menu),
    db=Depends(get_db),
):
    requested_events = {str(item).strip() for item in (req.event_types or []) if str(item).strip()}
    if requested_events != set(TEMPORAL_EVENT_TYPES):
        raise HTTPException(
            status_code=400,
            detail={"message": "端到端模型必须同时覆盖三类行为", "event_types": list(TEMPORAL_EVENT_TYPES)},
        )
    owner_username = str(current_user.get("username") or get_request_current_username(http_request))
    config = req.model_dump() if hasattr(req, "model_dump") else req.dict()
    config["event_types"] = list(TEMPORAL_EVENT_TYPES)
    config["owner_username"] = owner_username
    config["owner_id"] = current_user.get("id")
    config["model_family"] = TEMPORAL_MODEL_FAMILY
    config["include_all_reviewers"] = bool(req.include_all_reviewers)
    active_or_pending = db.query(JobTask).filter(
        JobTask.type == "behavior_temporal_training",
        JobTask.owner_username == owner_username,
        JobTask.status.in_(ACTIVE_STATUSES),
    ).order_by(JobTask.created_at.desc()).all()
    request_key = {
        "event_types": list(TEMPORAL_EVENT_TYPES),
        "clip_length": int(req.clip_length),
        "image_size": int(req.image_size),
        "epochs": int(req.epochs),
        "batch_size": int(req.batch_size),
        "learning_rate": float(req.learning_rate),
        "thresholds": dict(req.thresholds or {}),
    }
    for active in active_or_pending:
        existing = dict(active.input_json or {})
        existing_key = {key: existing.get(key) for key in request_key}
        if existing_key == request_key:
            return {"task_id": active.id, "status": active.status, "task": active.to_dict(), "deduplicated": True}
    task_id = generate_task_id()
    model_root = Path(constants.STATIC_DIR) / "behavior_models" / "temporal"
    model_root.mkdir(parents=True, exist_ok=True)
    config["model_path"] = str(model_root / f"{task_id}.pt")
    task = create_job(
        db,
        job_type="behavior_temporal_training",
        owner_username=owner_username,
        owner_id=current_user.get("id"),
        input_data=config,
        task_id=task_id,
        message="端到端行为模型训练任务已进入队列",
    )
    return _build_temporal_training_task_payload(db, task.id, http_request, include_result=False)


@app.get("/behavior/temporal-models/training/{task_id}")
@app.get("/api/behavior/temporal-models/training/{task_id}")
async def get_temporal_fusion_training_status(task_id: str, http_request: Request, db=Depends(get_db)):
    return _build_temporal_training_task_payload(db, task_id, http_request, include_result=False)


@app.get("/behavior/temporal-models/training/{task_id}/result")
@app.get("/api/behavior/temporal-models/training/{task_id}/result")
async def get_temporal_fusion_training_result(task_id: str, http_request: Request, db=Depends(get_db)):
    return _build_temporal_training_task_payload(db, task_id, http_request, include_result=True)


@app.get("/behavior/temporal-models")
@app.get("/api/behavior/temporal-models")
async def list_temporal_fusion_models(
    http_request: Request,
    _current_user: Dict[str, Any] = Depends(require_algorithm_training_menu),
    db=Depends(get_db),
):
    current_user = get_request_current_user(http_request)
    query = db.query(ModelRegistry).filter(ModelRegistry.source == "behavior_temporal")
    if current_user.get("role") != "admin":
        query = query.filter(ModelRegistry.created_by == str(current_user.get("username") or ""))
    records = query.order_by(ModelRegistry.created_at.desc()).all()
    return {"models": [record.to_dict() for record in records], "supported_event_types": list(TEMPORAL_EVENT_TYPES)}


@app.get("/behavior/temporal-models/{model_id}/evaluation")
@app.get("/api/behavior/temporal-models/{model_id}/evaluation")
async def get_temporal_fusion_model_evaluation(
    model_id: str,
    _current_user: Dict[str, Any] = Depends(require_algorithm_training_menu),
    db=Depends(get_db),
):
    record = db.query(ModelRegistry).filter(
        ModelRegistry.id == model_id,
        ModelRegistry.source == "behavior_temporal",
    ).first()
    if record is None:
        raise HTTPException(status_code=404, detail="端到端行为模型不存在")
    metadata = dict(record.metadata_json or {})
    return {
        "model_id": model_id,
        "model": record.to_dict(),
        "metrics": metadata.get("grouped_validation") or {},
        "grouped_validation": metadata.get("grouped_validation") or {},
        "failed_samples": metadata.get("failed_samples") or [],
        "thresholds": metadata.get("thresholds") or {},
    }


@app.get("/behavior/temporal-models/{model_id}/shadow-comparison")
@app.get("/api/behavior/temporal-models/{model_id}/shadow-comparison")
async def get_temporal_fusion_shadow_comparison(
    model_id: str,
    limit: int = 1000,
    _current_user: Dict[str, Any] = Depends(require_algorithm_training_menu),
    db=Depends(get_db),
):
    record = db.query(ModelRegistry).filter(
        ModelRegistry.id == model_id,
        ModelRegistry.source == "behavior_temporal",
    ).first()
    if record is None:
        raise HTTPException(status_code=404, detail="端到端行为模型不存在")
    rows = db.query(BehaviorEvent).order_by(BehaviorEvent.confirmed_at.desc()).limit(
        _safe_behavior_int(limit, default=1000, min_value=1, max_value=10000)
    ).all()
    comparisons = []
    for row in rows:
        decision = row.decision_json if isinstance(row.decision_json, dict) else {}
        shadow = decision.get("temporal_shadow") if isinstance(decision.get("temporal_shadow"), dict) else {}
        if str(shadow.get("model_id") or "") == model_id:
            comparisons.append(shadow)
    return {
        "model_id": model_id,
        "total": len(comparisons),
        "would_add_alerts": sum(1 for item in comparisons if item.get("would_add_alert")),
        "would_suppress_alerts": sum(1 for item in comparisons if item.get("would_suppress_alert")),
        "average_confidence": round(
            sum(float(item.get("confidence") or 0.0) for item in comparisons) / len(comparisons), 4
        ) if comparisons else None,
    }


@app.post("/behavior/temporal-models/{model_id}/activate")
@app.post("/api/behavior/temporal-models/{model_id}/activate")
async def activate_temporal_fusion_model(
    model_id: str,
    payload: Optional[BehaviorModelActivationPayload] = None,
    current_user: Dict[str, Any] = Depends(require_algorithm_training_menu),
    db=Depends(get_db),
):
    record = db.query(ModelRegistry).filter(
        ModelRegistry.id == model_id,
        ModelRegistry.source == "behavior_temporal",
        ModelRegistry.status.in_(("candidate", "active")),
    ).first()
    if record is None:
        raise HTTPException(status_code=404, detail="端到端行为模型候选版本不存在")
    return _activate_behavior_model_record(db, record, current_user, payload.reason if payload else "", rollback=False)


@app.post("/behavior/temporal-models/{model_id}/rollback")
@app.post("/api/behavior/temporal-models/{model_id}/rollback")
async def rollback_temporal_fusion_model(
    model_id: str,
    payload: BehaviorModelActivationPayload,
    current_user: Dict[str, Any] = Depends(require_algorithm_training_menu),
    db=Depends(get_db),
):
    record = db.query(ModelRegistry).filter(
        ModelRegistry.id == model_id,
        ModelRegistry.source == "behavior_temporal",
        ModelRegistry.status == "archived",
    ).first()
    if record is None:
        raise HTTPException(status_code=404, detail="可回滚的端到端行为模型不存在")
    return _activate_behavior_model_record(db, record, current_user, payload.reason, rollback=True)


def _get_review_samples_for_training(
    owner_username: str,
    *,
    include_all_reviewers: bool = False,
) -> List[Dict[str, Any]]:
    db = SessionLocal()
    try:
        query = db.query(BehaviorReviewSample).filter(
            BehaviorReviewSample.review_status == "reviewed",
            ~BehaviorReviewSample.review_label.in_(("uncertain", "delete")),
        )
        if not include_all_reviewers:
            query = query.filter(BehaviorReviewSample.owner_username == str(owner_username))
        rows = query.order_by(BehaviorReviewSample.reviewed_at.asc()).all()
        return [row.to_training_sample() for row in rows]
    finally:
        db.close()


def _build_action_classifier_task_payload(
    db,
    task_id: str,
    http_request: Request,
    include_result: bool = False,
) -> Dict[str, Any]:
    task = db.query(JobTask).filter(
        JobTask.id == task_id,
        JobTask.type == "behavior_verifier_training",
    ).first()
    if task is None:
        raise HTTPException(status_code=404, detail="动作复核分类器训练任务不存在")
    require_job_visibility(task, http_request)
    payload = task.to_dict()
    payload["error"] = task.error_detail
    if not include_result:
        payload.pop("result", None)
        payload.pop("results", None)
    return payload


def _run_action_classifier_training_sync(task_id: str, config: Dict[str, Any]) -> Dict[str, Any]:
    if train_classifier_from_samples is None:
        raise RuntimeError("动作复核分类器模块不可用，请检查 backend/behavior/action_classifier.py")
    db = SessionLocal()
    try:
        update_job(
            db,
            task_id,
            status="running",
            stage="loading_review_samples",
            progress=10.0,
            message="正在读取 PostgreSQL 复核样本...",
            event_type="started",
        )
    finally:
        db.close()
    samples = _get_review_samples_for_training(
        str(config.get("owner_username") or ""),
        include_all_reviewers=bool(config.get("include_all_reviewers", False)),
    )
    behavior_type = str(config.get("behavior_type") or "").strip()
    if behavior_type:
        samples = select_binary_behavior_samples(samples, behavior_type)
        source_groups = {
            str(item.get("source_task_id") or item.get("source_video") or item.get("source_id") or item.get("id"))
            for item in samples
        }
        if len(source_groups) < 3:
            raise ValueError(f"至少需要 3 个独立设备/视频来源组；当前仅 {len(source_groups)} 个")
    db = SessionLocal()
    try:
        update_job(
            db,
            task_id,
            stage="extracting_pose_features",
            progress=30.0,
            message=f"已读取 {len(samples)} 个复核样本，正在提取 Pose 特征...",
        )
    finally:
        db.close()
    target_model_path = resolve_action_classifier_model_path(
        config.get("model_path") or ACTION_CLASSIFIER_DEFAULT_MODEL_PATH,
        for_training=True,
    )
    minimum_samples = _safe_int(
        config.get("min_samples_per_class"),
        default=20 if behavior_type else 1,
        min_value=20 if behavior_type else 1,
        max_value=1000,
    )
    specialized_family = {
        "phone_use": "pose_object",
        "missing_badge": "badge_image",
    }.get(behavior_type)
    if specialized_family:
        if train_classifier_from_embedded_features is None:
            raise RuntimeError("专用行为验证器训练模块不可用")
        result = train_classifier_from_embedded_features(
            samples=samples,
            model_path=target_model_path,
            labels=[behavior_type, "normal"],
            feature_family=specialized_family,
            min_samples_per_class=minimum_samples,
        )
    else:
        result = train_classifier_from_samples(
            samples=samples,
            model_path=target_model_path,
            labels=[behavior_type, "normal"] if behavior_type else config.get("labels"),
            include_uncertain=bool(config.get("include_uncertain", False)),
            min_samples_per_class=minimum_samples,
            max_frames_per_clip=_safe_int(config.get("max_frames_per_clip"), default=96, min_value=8, max_value=512),
        )
    grouped_validation = (result.get("metadata") or {}).get("grouped_validation") or {}
    if behavior_type and not grouped_validation.get("available"):
        generated_path = Path(str(result.get("model_path") or ""))
        if generated_path.is_file():
            generated_path.unlink()
        raise ValueError(
            f"独立来源分组验证未完成: {grouped_validation.get('reason') or 'unknown reason'}"
        )
    model_path = Path(str(result["model_path"])).resolve()
    model_id = str(uuid.uuid4())
    model_record = ModelRegistry(
        id=model_id,
        name=model_path.name,
        path=str(model_path),
        sha256=sha256_file(model_path),
        size=model_path.stat().st_size,
        source="behavior_verifier",
        status="candidate",
        format="pkl",
        task_id=task_id,
        created_by=str(config.get("owner_username") or "unknown"),
        metadata_json={
            **dict(result.get("metadata") or {}),
            "display_name": f"行为验证器 {task_id[:8]}",
            "verifier_mode": "shadow",
            "review_scope": "all_reviewers" if config.get("include_all_reviewers") else "owner",
            "behavior_type": behavior_type,
            "model_family": specialized_family or "pose",
            "supported_event_types": [behavior_type] if behavior_type else list(result.get("labels") or []),
        },
    )
    db = SessionLocal()
    try:
        db.add(model_record)
        db.flush()
        result.update({
            "model_id": model_id,
            "model_status": "candidate",
            "model_sha256": model_record.sha256,
        })
        update_job(
            db,
            task_id,
            status="completed",
            stage="candidate_ready",
            progress=100.0,
            message="人工反馈验证器训练完成，已登记为候选版本",
            result=result,
            event_type="completed",
            event_payload={"model_id": model_id, "model_status": "candidate"},
        )
    finally:
        db.close()
    return result


async def _run_action_classifier_training_task(task_id: str, config: Dict[str, Any]) -> None:
    try:
        while True:
            blocker = _behavior_training_resource_blocker() if training_device_requires_gpu(config.get("device")) else None
            if not blocker:
                break
            db = SessionLocal()
            try:
                task = db.query(JobTask).filter(JobTask.id == task_id).first()
                if task is None or task.status in {"cancelled", "cancelling"}:
                    return
                update_job(
                    db,
                    task_id,
                    stage="waiting_realtime_resources",
                    progress=0.0,
                    message=blocker,
                )
            finally:
                db.close()
            await asyncio.sleep(5)
        await _gpu_job_call(task_id, _run_action_classifier_training_sync, task_id, config,
                            requires_gpu=training_device_requires_gpu(config.get("device")))
    except Exception as e:
        logger.exception(f"动作复核分类器训练失败: task_id={task_id}")
        db = SessionLocal()
        try:
            task = db.query(JobTask).filter(JobTask.id == task_id).first()
            if task is not None and task.status not in {"completed", "failed", "cancelled"}:
                update_job(
                    db,
                    task_id,
                    status="failed",
                    stage="failed",
                    progress=100.0,
                    message=f"训练失败: {str(e)}",
                    error_code="behavior_verifier_training_failed",
                    error_detail=str(e),
                    event_type="failed",
                    level="error",
                )
        finally:
            db.close()
    finally:
        action_classifier_task_handles.pop(task_id, None)


def _behavior_training_resource_blocker() -> Optional[str]:
    try:
        gpus = _get_gpu_info()
    except Exception:
        gpus = []
    realtime_active = any(item.get("is_active") for item in realtime_stats.values()) or any(
        item.get("is_active") for item in grid_sessions.values()
    )
    default_minimum = 3072 if realtime_active else 1536
    env_name = (
        "BEHAVIOR_TRAINING_MIN_FREE_GPU_MB_REALTIME"
        if realtime_active
        else "BEHAVIOR_TRAINING_MIN_FREE_GPU_MB"
    )
    minimum_free_mb = _safe_int(
        os.getenv(env_name, str(default_minimum)),
        default=default_minimum,
        min_value=256,
        max_value=65536,
    )
    if gpus:
        maximum_free = max(
            float(item.get("memory_total") or 0) - float(item.get("memory_used") or 0)
            for item in gpus
        )
        if maximum_free < minimum_free_mb:
            priority = "实时推理优先" if realtime_active else "GPU 资源不足"
            return f"{priority}：GPU 可用显存 {maximum_free:.0f} MB，等待达到 {minimum_free_mb} MB"
    return None


@app.get("/behavior/action-classifier/info")
@app.get("/api/behavior/action-classifier/info")
@app.get("/behavior/models/active")
@app.get("/api/behavior/models/active")
async def get_action_classifier_info(
    model_path: str = ACTION_CLASSIFIER_CANONICAL_MODEL_PATH,
    _current_user: Dict[str, Any] = Depends(require_algorithm_training_menu),
    db=Depends(get_db),
):
    """查看当前已激活的人工反馈验证器信息。"""
    try:
        if get_action_classifier_model_info is None:
            raise RuntimeError("动作复核分类器模块不可用")
        records = db.query(ModelRegistry).filter(
            ModelRegistry.source == "behavior_verifier",
            ModelRegistry.status == "active",
        ).order_by(ModelRegistry.updated_at.desc()).all()
        record = records[0] if records else None
        resolved_path = record.path if record and model_path == ACTION_CLASSIFIER_CANONICAL_MODEL_PATH else resolve_action_classifier_model_path(model_path)
        payload = get_action_classifier_model_info(resolved_path)
        payload["model"] = record.to_dict() if record and record.path == resolved_path else None
        payload["active_models"] = [item.to_dict() for item in records]
        payload["mode"] = "active" if record else "off"
        return payload
    except Exception as e:
        logger.exception("获取动作复核分类器信息失败")
        raise HTTPException(status_code=500, detail=f"获取动作复核分类器信息失败: {str(e)}")


@app.post("/behavior/action-classifier/train")
@app.post("/api/behavior/action-classifier/train")
@app.post("/behavior/models/train")
@app.post("/api/behavior/models/train")
async def start_action_classifier_training(
    req: ActionClassifierTrainRequest,
    http_request: Request,
    current_user: Dict[str, Any] = Depends(require_algorithm_training_menu),
    db=Depends(get_db),
):
    """启动动作复核分类器训练任务。"""
    try:
        if req.include_uncertain:
            raise HTTPException(status_code=400, detail="不确定样本不能参与验证器训练")
        behavior_type = str(req.behavior_type or "").strip()
        if behavior_type and behavior_type not in {
            "phone_use", "missing_badge", "fall_down",
        }:
            raise HTTPException(status_code=400, detail="不支持的行为类型")
        if not behavior_type and len(req.labels or []) < 2:
            raise HTTPException(status_code=400, detail="旧分类器接口至少需要选择两个标签")
        owner_username = str(current_user.get("username") or "unknown")
        task_id = generate_task_id()
        config = req.dict()
        requested_path = Path(resolve_action_classifier_model_path(config.get("model_path"), for_training=True))
        config["model_path"] = str(requested_path.parent / "behavior_verifiers" / f"{task_id}.pkl")
        config["owner_username"] = owner_username
        config["include_all_reviewers"] = True
        task = create_job(
            db,
            job_type="behavior_verifier_training",
            owner_username=owner_username,
            owner_id=current_user.get("id"),
            input_data=config,
            task_id=task_id,
            message="人工反馈验证器训练任务已创建",
        )
        action_classifier_task_handles[task_id] = asyncio.create_task(
            _run_action_classifier_training_task(task_id, config)
        )
        return task.to_dict()
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("启动动作复核分类器训练失败")
        raise HTTPException(status_code=500, detail=f"启动训练失败: {str(e)}")


@app.get("/behavior/action-classifier/status/{task_id}")
@app.get("/api/behavior/action-classifier/status/{task_id}")
@app.get("/behavior/models/training/{task_id}")
@app.get("/api/behavior/models/training/{task_id}")
async def get_action_classifier_training_status(
    task_id: str,
    http_request: Request,
    _current_user: Dict[str, Any] = Depends(require_algorithm_training_menu),
    db=Depends(get_db),
):
    """查询动作复核分类器训练任务状态。"""
    return _build_action_classifier_task_payload(db, task_id, http_request, include_result=False)


@app.get("/behavior/action-classifier/result/{task_id}")
@app.get("/api/behavior/action-classifier/result/{task_id}")
@app.get("/behavior/models/training/{task_id}/result")
@app.get("/api/behavior/models/training/{task_id}/result")
async def get_action_classifier_training_result(
    task_id: str,
    http_request: Request,
    _current_user: Dict[str, Any] = Depends(require_algorithm_training_menu),
    db=Depends(get_db),
):
    """查询动作复核分类器训练结果。"""
    return _build_action_classifier_task_payload(db, task_id, http_request, include_result=True)


@app.get("/behavior/action-classifier/models")
@app.get("/api/behavior/action-classifier/models")
@app.get("/behavior/models")
@app.get("/api/behavior/models")
async def list_action_verifier_models(
    http_request: Request,
    current_user: Dict[str, Any] = Depends(require_algorithm_training_menu),
    db=Depends(get_db),
):
    query = db.query(ModelRegistry).filter(ModelRegistry.source == "behavior_verifier")
    records = query.order_by(ModelRegistry.created_at.desc()).all()
    return {"models": [record.to_dict() for record in records]}


@app.get("/behavior/models/{model_id}/evaluation")
@app.get("/api/behavior/models/{model_id}/evaluation")
async def get_behavior_model_evaluation(
    model_id: str,
    _current_user: Dict[str, Any] = Depends(require_algorithm_training_menu),
    db=Depends(get_db),
):
    record = db.query(ModelRegistry).filter(
        ModelRegistry.id == model_id,
        ModelRegistry.source == "behavior_verifier",
    ).first()
    if record is None:
        raise HTTPException(status_code=404, detail="行为验证器版本不存在")
    metadata = dict(record.metadata_json or {})
    return {
        "model": record.to_dict(),
        "train_metrics": metadata.get("train_metrics") or {},
        "grouped_validation": metadata.get("grouped_validation") or {},
        "failed_samples": metadata.get("failed_samples") or [],
    }


@app.get("/behavior/models/{model_id}/shadow-comparison")
@app.get("/api/behavior/models/{model_id}/shadow-comparison")
async def get_behavior_model_shadow_comparison(
    model_id: str,
    limit: int = 1000,
    _current_user: Dict[str, Any] = Depends(require_algorithm_training_menu),
    db=Depends(get_db),
):
    record = db.query(ModelRegistry).filter(
        ModelRegistry.id == model_id,
        ModelRegistry.source == "behavior_verifier",
    ).first()
    if record is None:
        raise HTTPException(status_code=404, detail="行为验证器版本不存在")
    rows = db.query(BehaviorEvent).order_by(BehaviorEvent.confirmed_at.desc()).limit(
        _safe_int(limit, default=1000, min_value=1, max_value=10000)
    ).all()
    comparisons = []
    for row in rows:
        decision = row.decision_json if isinstance(row.decision_json, dict) else {}
        shadow = decision.get("shadow") if isinstance(decision.get("shadow"), dict) else {}
        if str(shadow.get("model_id") or "") != model_id:
            continue
        active = decision.get("active") if isinstance(decision.get("active"), dict) else {}
        comparisons.append((bool(active.get("accepted", True)), bool(shadow.get("accepted")), shadow))
    agreement = sum(1 for active, shadow, _ in comparisons if active == shadow)
    return {
        "model_id": model_id,
        "total": len(comparisons),
        "agreement_rate": round(agreement / len(comparisons), 4) if comparisons else None,
        "would_add_alerts": sum(1 for active, shadow, _ in comparisons if not active and shadow),
        "would_suppress_alerts": sum(1 for active, shadow, _ in comparisons if active and not shadow),
        "average_shadow_confidence": round(
            sum(float(item.get("confidence") or 0.0) for _, _, item in comparisons) / len(comparisons), 4
        ) if comparisons else None,
    }


def _activate_behavior_model_record(
    db,
    record: ModelRegistry,
    current_user: Dict[str, Any],
    reason: str,
    *,
    rollback: bool,
) -> Dict[str, Any]:
    model_source = str(record.source or "behavior_verifier")
    locked_rows = db.query(ModelRegistry).filter(
        ModelRegistry.source == model_source,
    ).with_for_update().all()
    record = next((item for item in locked_rows if item.id == record.id), None)
    if record is None:
        raise HTTPException(status_code=404, detail="行为验证器版本不存在")
    if rollback and record.status != "archived":
        raise HTTPException(status_code=409, detail="该版本当前不可回滚")
    if not rollback and record.status not in {"candidate", "active"}:
        raise HTTPException(status_code=409, detail="该版本当前不可激活")
    if not Path(record.path).exists():
        raise HTTPException(status_code=409, detail="行为验证器模型文件不存在")
    metadata = dict(record.metadata_json or {})
    grouped_validation = metadata.get("grouped_validation") or {}
    if record.status == "candidate" and not grouped_validation.get("available"):
        raise HTTPException(status_code=409, detail="候选模型尚未完成独立来源分组验证")
    if record.status == "candidate" and model_source == "behavior_temporal":
        per_event = grouped_validation.get("per_event") or {}
        missing_heads = [
            event_type for event_type in TEMPORAL_EVENT_TYPES
            if not isinstance(per_event.get(event_type), dict)
            or int(per_event.get(event_type, {}).get("support") or 0) <= 0
        ]
        if missing_heads:
            raise HTTPException(status_code=409, detail=f"以下事件头尚无有效分组评测，不能激活: {missing_heads}")
    if rollback and not str(reason or "").strip():
        raise HTTPException(status_code=400, detail="回滚必须填写原因")
    try:
        if model_source == "behavior_temporal":
            load_temporal_model(record.path)
        else:
            preload_behavior_verifier(record)
    except Exception as exc:
        raise HTTPException(status_code=409, detail=f"模型预加载校验失败: {exc}")

    behavior_type = str(metadata.get("behavior_type") or "")
    now = datetime.now().isoformat()
    actor = str(current_user.get("username") or "unknown")
    normalized_reason = str(reason or "").strip()
    active_rows = [
        item for item in locked_rows
        if item.status == "active" and item.id != record.id
    ]
    for active in active_rows:
        active_type = str((active.metadata_json or {}).get("behavior_type") or "")
        same_scope = model_source == "behavior_temporal" or not active_type or active_type == behavior_type
        if same_scope:
            active.status = "archived"
            active_metadata = dict(active.metadata_json or {})
            archive_history = list(active_metadata.get("archive_history") or [])
            archive_history.append({
                "at": now,
                "by": actor,
                "reason": normalized_reason,
                "replaced_by": record.id,
            })
            active_metadata["archive_history"] = archive_history
            active_metadata["verifier_mode"] = "archived"
            active.metadata_json = active_metadata

    action = "rollback" if rollback else "activate"
    history = list(metadata.get("activation_history") or [])
    history.append({
        "action": action,
        "at": now,
        "by": actor,
        "reason": normalized_reason,
    })
    metadata.update({
        "activated_at": now,
        "activated_by": actor,
        "activation_reason": normalized_reason,
        "activation_history": history,
        "verifier_mode": "active",
    })
    record.metadata_json = metadata
    record.status = "active"
    db.commit()
    db.refresh(record)
    return {"success": True, "action": action, "model": record.to_dict()}


@app.post("/behavior/action-classifier/models/{model_id}/activate")
@app.post("/api/behavior/action-classifier/models/{model_id}/activate")
@app.post("/behavior/models/{model_id}/activate")
@app.post("/api/behavior/models/{model_id}/activate")
async def activate_action_verifier_model(
    model_id: str,
    payload: Optional[BehaviorModelActivationPayload] = None,
    current_user: Dict[str, Any] = Depends(require_algorithm_training_menu),
    db=Depends(get_db),
):
    record = db.query(ModelRegistry).filter(
        ModelRegistry.id == model_id,
        ModelRegistry.source == "behavior_verifier",
        ModelRegistry.status.in_(("candidate", "active")),
    ).first()
    if record is None:
        raise HTTPException(status_code=404, detail="行为验证器候选版本不存在")
    return _activate_behavior_model_record(
        db,
        record,
        current_user,
        payload.reason if payload else "",
        rollback=False,
    )


@app.post("/behavior/models/{model_id}/rollback")
@app.post("/api/behavior/models/{model_id}/rollback")
async def rollback_behavior_model(
    model_id: str,
    payload: BehaviorModelActivationPayload,
    current_user: Dict[str, Any] = Depends(require_algorithm_training_menu),
    db=Depends(get_db),
):
    record = db.query(ModelRegistry).filter(
        ModelRegistry.id == model_id,
        ModelRegistry.source == "behavior_verifier",
        ModelRegistry.status == "archived",
    ).first()
    if record is None:
        raise HTTPException(status_code=404, detail="可回滚的归档版本不存在")
    return _activate_behavior_model_record(
        db,
        record,
        current_user,
        payload.reason,
        rollback=True,
    )


@app.post("/behavior/action-classifier/predict")
@app.post("/api/behavior/action-classifier/predict")
@gpu_http_guard
async def predict_action_clip(req: ActionClassifierPredictRequest):
    """用已训练的动作复核分类器预测单个 clip。"""
    try:
        if load_action_verifier is None:
            raise RuntimeError("动作复核分类器模块不可用")
        clf = load_action_verifier(resolve_action_classifier_model_path(req.model_path))
        return clf.predict_clip(req.clip_path, max_frames=_safe_int(req.max_frames_per_clip, default=96, min_value=8, max_value=512))
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e))
    except Exception as e:
        logger.exception("动作复核分类器预测失败")
        raise HTTPException(status_code=500, detail=f"预测失败: {str(e)}")


@app.get('/system/resources')
async def get_system_resources():
    """
    获取当前后端可见的系统资源状态。
    说明：如果后端运行在 Docker 中，这里显示的是后端容器/WSL 可见的 CPU、内存和 GPU 状态。
    """
    memory_info = _get_memory_info()
    gpu_probe = _get_gpu_probe()
    gpus = gpu_probe.get("gpus", [])
    first_gpu = gpus[0] if gpus else None

    return {
        'cpu': _get_cpu_percent(),
        'memory': memory_info['percent'],
        'memory_used': memory_info['used'],
        'memory_total': memory_info['total'],
        'gpu': first_gpu['utilization'] if first_gpu else 0.0,
        'gpu_memory': first_gpu['memory_percent'] if first_gpu else 0.0,
        'gpu_memory_used': first_gpu['memory_used'] if first_gpu else 0.0,
        'gpu_memory_total': first_gpu['memory_total'] if first_gpu else 0.0,
        'gpu_temperature': first_gpu['temperature'] if first_gpu else None,
        'gpu_name': first_gpu['name'] if first_gpu else '未检测到GPU',
        'gpus': gpus,
        'gpu_status': gpu_probe.get('gpu_status'),
        'gpu_reason': gpu_probe.get('gpu_reason'),
        'gpu_error': gpu_probe.get('gpu_error'),
        'cuda_available': gpu_probe.get('cuda_available', False),
        'cuda_device_count': gpu_probe.get('cuda_device_count', 0),
        'cuda_version': gpu_probe.get('cuda_version'),
        'gpu_uuid': gpu_probe.get('gpu_uuid'),
        'probe_time': gpu_probe.get('probe_time'),
        'timestamp': datetime.now().isoformat()
    }

def build_realtime_health_summary(stale_after_ms: float = 5000.0) -> Dict[str, Any]:
    captures = video_processor.get_capture_health()
    stale_sessions = []
    reconnecting_sessions = []
    for session_id, health in captures.items():
        frame_age_ms = health.get("frame_age_ms")
        status = health.get("status")
        if status in {"connecting", "reconnecting"}:
            reconnecting_sessions.append(session_id)
        if frame_age_ms is None or frame_age_ms > stale_after_ms:
            stale_sessions.append(session_id)
    return {
        "capture_count": len(captures),
        "stale_after_ms": stale_after_ms,
        "stale_sessions": stale_sessions,
        "reconnecting_sessions": reconnecting_sessions,
        "captures": captures,
    }


@app.get("/health/live")
async def liveness_check():
    return {"status": "alive", "timestamp": datetime.now().isoformat()}


@app.get("/health/ready")
async def readiness_check():
    realtime_health = build_realtime_health_summary()
    gpu_probe = _get_gpu_probe()
    return {
        "status": "ready" if not realtime_health["stale_sessions"] and gpu_probe["gpu_status"] == "ready" else "degraded",
        "timestamp": datetime.now().isoformat(),
        "realtime": realtime_health,
        "gpu_runtime": gpu_probe,
    }


@app.get("/health")
async def health_check(db=Depends(get_db)):
    """健康检查"""
    job_health = get_job_health_snapshot(db)
    training_worker = read_worker_heartbeat()
    behavior_worker = read_behavior_worker_heartbeat()
    gpu_probe = _get_gpu_probe()
    realtime_session_count = len([s for s in realtime_stats.values() if s.get("is_active", False)])
    training_active = db.query(JobTask).filter(
        JobTask.type == "training",
        JobTask.status.in_({"running", "cancelling"}),
    ).count() > 0
    annotation_active = db.query(JobTask).filter(
        JobTask.type == "annotation",
        JobTask.status.in_(ACTIVE_STATUSES),
    ).count() > 0
    contention_targets = []
    if training_active and realtime_session_count:
        contention_targets.append("实时推理")
    if training_active and annotation_active:
        contention_targets.append("批量标注")
    contention_warning = (
        f"训练与{'、'.join(contention_targets)}同时运行，存在显存和性能风险"
        if contention_targets
        else None
    )
    return {
        "status": "healthy" if gpu_probe["gpu_status"] == "ready" else "degraded",
        "timestamp": datetime.now().isoformat(),
        "active_tasks": job_health["active_tasks"],
        "total_tasks": db.query(JobTask).count(),
        "task_status_counts": job_health["status_counts"],
        "stale_task_count": job_health["stale_task_count"],
        "heartbeat_timeout_seconds": job_health["heartbeat_timeout_seconds"],
        "cached_models": len(inference_manager.get_cached_models()),
        "realtime_sessions": realtime_session_count,
        "grid_sessions": len([g for g in grid_sessions.values() if g.get("is_active", False)]),
        "realtime": build_realtime_health_summary(),
        "training_worker": training_worker,
        "behavior_worker": behavior_worker,
        "gpu_runtime": gpu_probe,
        "gpu_contention": {
            "training_active": training_active,
            "realtime_sessions": realtime_session_count,
            "annotation_active": annotation_active,
            "warning": contention_warning,
        },
    }



@app.post("/behavior/image-dataset/upload-and-create-review-task")
@app.post("/api/behavior/image-dataset/upload-and-create-review-task")
async def upload_image_dataset_and_create_review_task(
    file: UploadFile = File(...),
    default_label: str = Form("uncertain")
):
    """上传图片数据集并生成行为复核任务。"""
    try:
        filename = file.filename or "image_dataset.zip"
        suffix = Path(filename).suffix.lower()
        allowed_image_suffixes = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

        if suffix != ".zip" and suffix not in allowed_image_suffixes:
            raise HTTPException(status_code=400, detail="只支持 zip 图片数据集或 jpg/jpeg/png/bmp/webp 图片")

        task_id = make_behavior_time_task_id()
        dataset_root = Path("data") / "action_image_datasets" / task_id
        dataset_root.mkdir(parents=True, exist_ok=True)

        saved_path = await asyncio.to_thread(
            save_upload_file_keep_name,
            file,
            dataset_root,
            fallback=sanitize_original_filename(filename, "image_dataset.zip"),
            max_bytes=ArchiveLimits.from_env().max_upload_bytes if suffix == ".zip" else IMAGE_UPLOAD_MAX_BYTES,
        )

        if suffix == ".zip":
            try:
                safe_extract_zip(saved_path, dataset_root, ArchiveLimits.from_env())
            except ArchiveSafetyError as exc:
                file_manager.cleanup_dir(str(dataset_root))
                raise HTTPException(status_code=413, detail=str(exc))
            except zipfile.BadZipFile as exc:
                file_manager.cleanup_dir(str(dataset_root))
                raise HTTPException(status_code=400, detail=f"上传文件不是有效 zip 压缩包: {exc}")
            try:
                saved_path.unlink()
            except Exception:
                pass

        manager = _get_behavior_sample_manager()
        if not hasattr(manager, "import_image_dataset_as_review_task"):
            raise HTTPException(
                status_code=500,
                detail="backend/behavior/sample_manager.py 缺少 import_image_dataset_as_review_task，请覆盖最新版本"
            )

        result = manager.import_image_dataset_as_review_task(
            dataset_dir=str(dataset_root),
            task_id=task_id,
            default_label=str(default_label or "uncertain")
        )

        return {
            "success": True,
            "message": "图片数据集已生成复核任务",
            "task_id": task_id,
            "review_task_id": task_id,
            "data": result,
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.exception("上传图片数据集并生成复核任务失败")
        raise HTTPException(status_code=500, detail=f"上传图片数据集失败: {str(e)}")


from continuous_monitoring import install as install_continuous_monitoring
from continuous_monitoring_grid import GridMonitorAdapter
import sys as _monitor_sys

install_continuous_monitoring(app, GridMonitorAdapter(_monitor_sys.modules[__name__]))


if __name__ == "__main__":
    import uvicorn
    
    # 创建必要目录
    for directory in [constants.DATASETS_DIR, constants.EXPORTS_DIR, constants.TEMP_DIR]:
        file_manager.ensure_dir(directory)
    
    print("🚀 启动YOLO API服务...")
    print("📊 API文档: http://localhost:8000/docs")
    print("🔧 健康检查: http://localhost:8000/health")
    
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
