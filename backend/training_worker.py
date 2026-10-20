#!/usr/bin/env python3
"""
独立的训练工作进程
将训练任务从主API服务中分离，避免服务中断
"""

import os
import sys
import gc
import json
import time
import logging
import multiprocessing as mp
import threading
from pathlib import Path
from datetime import datetime
from typing import Dict, Any
import traceback
import faulthandler
os.environ.setdefault("PYTHONUNBUFFERED", "1")
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("YOLO_CONFIG_DIR", "/app/.ultralytics")

import torch
from gpu_runtime import gpu_runtime_is_ready, probe_gpu, training_device_requires_gpu
from gpu_probe_service import GPUUnavailable, report_gpu_fault
from dataset_config_utils import read_dataset_class_names, requested_classes_match_dataset
from job_store import is_cancel_requested, record_job_update, wait_for_gpu_job, requeue_gpu_job
from model_registry import build_model_lineage, register_model_file_external
from training_optimization import extract_grouped_metrics, normalize_training_config, resolve_training_profile
from training_validation import resolve_training_dataset_config_path
from training_workspace import cleanup_training_workspace

# Monkeypatch torch.load to fix the mmap bug in torch 2.11.0
original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    if 'mmap' not in kwargs or kwargs['mmap'] is None:
        kwargs['mmap'] = False
    return original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

# Fix for legacy torch unpickling
try:
    import torch.utils.serialization
except ImportError:
    import types
    import sys
    m = types.ModuleType("torch.utils.serialization")
    sys.modules["torch.utils.serialization"] = m

# YOLOTrainer 不在父进程顶层导入，避免 fork/重载时继承 Ultralytics 与 Torch 状态。

# 设置日志
TRAINING_WORKER_LOG_PATH = Path(os.getenv("TRAINING_WORKER_LOG_FILE", "temp/training_worker.log"))
TRAINING_WORKER_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(TRAINING_WORKER_LOG_PATH),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def flush_logger_handlers(log=None):
    """立即刷新日志处理器，避免训练子进程卡住时最后几行日志丢失。"""
    log = log or logger
    for handler in list(log.handlers):
        try:
            handler.flush()
        except Exception:
            pass


def cleanup_torch_runtime(training_logger=None):
    """释放 PyTorch/CUDA 运行时资源，降低二次训练卡住概率。"""
    log = training_logger or logger
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception as e:
        log.warning(f"清理CUDA缓存失败: {e}")
    finally:
        gc.collect()


def cleanup_task_training_workspace(config_data: Dict, training_logger=None) -> bool:
    log = training_logger or logger
    workspace_path = str(config_data.get("_training_workspace_root") or "").strip()
    if not workspace_path:
        return False
    try:
        removed = cleanup_training_workspace(workspace_path)
        if removed:
            log.info(f"训练临时数据工作区已清理: {workspace_path}")
        return removed
    except Exception as exc:
        log.error(f"训练临时数据工作区清理失败: {workspace_path}, error={exc}")
        return False


def resolve_training_device(device_value=None) -> str:
    """Resolve the device without silently falling back from GPU to CPU."""
    device = str(device_value or "auto").strip().lower()
    if training_device_requires_gpu(device):
        probe = probe_gpu()
        if not gpu_runtime_is_ready(probe):
            reason = probe.get("gpu_reason", "cuda_unavailable")
            raise GPUUnavailable(
                f"GPU_RUNTIME_UNAVAILABLE: {reason}: "
                f"{probe.get('gpu_error') or 'CUDA runtime is unavailable'}"
            )
        return "0" if device in {"auto", "cuda"} else device
    return str(device_value)

# 全局训练进程字典
training_processes = {}
task_status_cache = {}
monitor_thread = None
monitor_thread_lock = threading.Lock()


def get_training_mp_context():
    """训练进程统一使用 spawn，避免 Linux 默认 fork 继承 PyTorch/CUDA/Ultralytics 状态导致二次训练卡死。"""
    try:
        return mp.get_context("spawn")
    except RuntimeError:
        return mp

def update_task_status_file(task_id: str, status: str, progress: float, message: str, results: Dict = None):
    """更新 PostgreSQL 任务状态，并保留兼容状态文件供运维排障。"""
    try:
        stage = {
            "pending": "queued",
            "running": "training",
            "cancelling": "cancelling",
            "cancelled": "cancelled",
            "completed": "completed",
            "failed": "failed",
        }.get(status, status)
        record_job_update(
            task_id,
            status=status,
            stage=stage,
            progress=progress,
            message=message,
            result=results if results is not None else None,
            event_type="status" if status != "running" else "progress",
            level="error" if status == "failed" else "info",
            error_code="training_failed" if status == "failed" else None,
            error_detail=message if status == "failed" else None,
        )
        status_dir = Path("temp/training_status")
        status_dir.mkdir(parents=True, exist_ok=True)
        
        status_file = status_dir / f"{task_id}.json"
        
        # 尝试读取现有数据以获取 created_at
        created_at = datetime.now().isoformat()
        if status_file.exists():
            try:
                with open(status_file, 'r', encoding='utf-8') as f:
                    old_data = json.load(f)
                    if "created_at" in old_data:
                        created_at = old_data["created_at"]
            except:
                pass
        elif task_id in task_status_cache and "created_at" in task_status_cache[task_id]:
            created_at = task_status_cache[task_id]["created_at"]
        
        status_data = {
            "task_id": task_id,
            "status": status,
            "progress": progress,
            "message": message,
            "created_at": created_at,
            "updated_at": datetime.now().isoformat(),
            "results": results
        }
        
        with open(status_file, 'w', encoding='utf-8') as f:
            json.dump(status_data, f, ensure_ascii=False, indent=2)
            
        # 同时更新内存缓存
        task_status_cache[task_id] = status_data
        
        logger.info(f"任务 {task_id} 状态更新: {status} - {progress:.1f}% - {message}")
        
    except Exception as e:
        logger.error(f"更新任务状态失败: {e}")

def keep_only_best_pt_from_model_path(model_path: str, training_logger=None):
    """
    根据 best.pt 路径清理同目录下的其他 .pt 文件，仅保留 best.pt。
    """
    log = training_logger or logger
    try:
        best_pt = Path(model_path)
        weights_dir = best_pt.parent

        if not weights_dir.exists():
            log.warning(f"weights目录不存在，跳过模型清理: {weights_dir}")
            return

        if best_pt.name != "best.pt" or not best_pt.exists():
            log.warning(f"best.pt不存在，跳过模型清理，避免误删: {best_pt}")
            return

        removed_count = 0
        for pt_file in weights_dir.glob("*.pt"):
            if pt_file.name != "best.pt":
                pt_file.unlink()
                removed_count += 1
                log.info(f"已删除非最佳模型文件: {pt_file}")

        log.info(f"模型清理完成，仅保留 best.pt: {best_pt}，删除数量: {removed_count}")

    except Exception as e:
        log.warning(f"清理模型文件失败: {e}")


def resolve_training_model_path(model_name: str, training_logger=None) -> str:
    """
    解析训练使用的模型：
    1. 官方预训练模型名，例如 yolov8n.pt、yolo11n.pt，保持原样交给 Ultralytics；
    2. 前端传入的本地 .pt 路径，例如 runs/.../weights/best.pt，直接使用；
    3. 只传文件名时，优先在 models/ 与 runs/ 下搜索；
    4. 支持继续使用训练得到的 best.pt 作为下一次训练的初始权重。
    """
    log = training_logger or logger

    if not model_name:
        return "yolov8n.pt"

    raw_model_name = str(model_name).strip().strip('"').strip("'")
    normalized_model_name = raw_model_name.replace("\\", "/")

    official_models = {
        "yolo11n.pt", "yolo11s.pt", "yolo11m.pt", "yolo11l.pt", "yolo11x.pt",
        "yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt",
        "yolo26n.pt", "yolo26s.pt", "yolo26m.pt", "yolo26l.pt", "yolo26x.pt",
    }

    # 官方模型名优先使用本地文件，避免二次训练时 Ultralytics 尝试联网下载或卡在模型初始化。
    if normalized_model_name in official_models:
        local_candidates = [
            Path(normalized_model_name),
            Path("models") / normalized_model_name,
            Path("/app/models") / normalized_model_name,
            Path("/app") / normalized_model_name,
            Path.home() / ".cache" / "ultralytics" / normalized_model_name,
        ]
        for candidate in local_candidates:
            if candidate.exists() and candidate.is_file():
                resolved = str(candidate)
                log.info(f"使用本地官方预训练模型: {resolved}")
                return resolved

        log.warning(
            f"未在本地找到官方预训练模型 {normalized_model_name}，将交给 Ultralytics 处理。"
            "如容器无法联网，可能会卡在 YOLO(model) 初始化阶段；建议将模型放到 /app/models/。"
        )
        return normalized_model_name

    candidate_paths = []

    # 1. 前端直接传 path，例如 runs/detect/runs/train/xxx/weights/best.pt
    direct_path = Path(normalized_model_name)
    candidate_paths.append(direct_path)

    # 2. 兼容前端传 /app/xxx 的容器绝对路径
    if normalized_model_name.startswith("/app/"):
        candidate_paths.append(Path(normalized_model_name.replace("/app/", "", 1)))

    # 3. 兼容只传文件名，例如 best.pt、my_model.pt、xxx/best.pt
    candidate_paths.extend([
        Path("models") / normalized_model_name,
        Path("runs") / normalized_model_name,
        Path("runs/train") / normalized_model_name,
        Path("runs/detect") / normalized_model_name,
    ])

    for candidate in candidate_paths:
        if candidate.exists() and candidate.is_file() and candidate.suffix.lower() == ".pt":
            resolved = str(candidate)
            log.info(f"使用本地模型文件: {resolved}")
            return resolved

    # 4. 在常见模型目录中递归搜索。优先使用最新修改的匹配项。
    search_roots = [Path("models"), Path("runs")]
    matched_files = []
    target_name = Path(normalized_model_name).name

    for root in search_roots:
        if not root.exists():
            continue
        try:
            for pt_file in root.rglob("*.pt"):
                pt_path = str(pt_file).replace("\\", "/")
                if (
                    pt_file.name == target_name
                    or pt_path.endswith(normalized_model_name)
                    or pt_path.endswith(f"/{normalized_model_name}")
                ):
                    matched_files.append(pt_file)
        except Exception as e:
            log.warning(f"搜索模型目录失败: {root}, 错误: {e}")

    if matched_files:
        matched_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
        resolved = str(matched_files[0])
        log.info(f"通过搜索找到模型文件: {resolved}")
        return resolved

    raise FileNotFoundError(
        "模型文件不存在或不可用: "
        f"{raw_model_name}。请确认 /models 接口返回的 path 字段可访问，"
        "或者确认 best.pt 位于 models/ 或 runs/ 目录下。"
    )


def training_worker_process(task_id: str, config_data: Dict, dataset_path: str):
    """独立进程中的训练工作函数"""
    gpu_waiting = False
    try:
        try:
            faulthandler.enable()
            # 如果模型加载或训练长时间无输出，Docker logs 会打印 Python 调用栈，方便定位卡点。
        except Exception:
            pass

        # 子进程内部再导入 YOLOTrainer，避免父进程预加载 Ultralytics/PyTorch 状态。
        from yolo_trainer import YOLOTrainer

        # 重新配置日志 - 确保训练进程的日志能够正确输出
        log_file = f"temp/training_logs/{task_id}.log"
        os.makedirs("temp/training_logs", exist_ok=True)
        
        # 配置训练进程专用日志
        training_logger = logging.getLogger(f"training_{task_id}")
        training_logger.setLevel(logging.INFO)
        training_logger.propagate = False
        
        # 清除已有的处理器
        training_logger.handlers.clear()
        
        # 添加文件处理器
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(file_formatter)
        training_logger.addHandler(file_handler)
        
        # 添加控制台处理器（输出到标准输出）
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = logging.Formatter(f'[TRAINING-{task_id}] %(asctime)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(console_formatter)
        training_logger.addHandler(console_handler)
        
        training_logger.info(f"启动训练进程: {task_id}")
        flush_logger_handlers(training_logger)
        update_task_status_file(task_id, "running", 0.0, "初始化训练环境...")
        cleanup_torch_runtime(training_logger)
        
        # 解析配置
        raw_model_name = config_data.get('model_name', 'yolov8n.pt')
        model_name = resolve_training_model_path(raw_model_name, training_logger)
        model_lineage = dict(config_data.get("_model_lineage") or {})
        preflight = dict(config_data.get("_training_preflight") or {})
        training_profile = resolve_training_profile(
            config_data.get("training_profile", "auto"),
            raw_model_name,
            model_name,
            config_data.get("base_model"),
            model_lineage.get("model_family"),
            model_lineage.get("lineage_root_model"),
            preflight.get("model_family"),
        )
        if not model_lineage:
            model_lineage = build_model_lineage(
                raw_model_name,
                {},
                training_profile,
            )
        effective_config = normalize_training_config(config_data, training_profile)
        if Path(str(model_name)).exists():
            try:
                training_logger.info(f"模型文件大小: {Path(str(model_name)).stat().st_size / 1024 / 1024:.2f} MB, path={model_name}")
            except Exception:
                pass
        epochs = effective_config["epochs"]
        img_size = effective_config["img_size"]
        batch_size = effective_config["batch_size"]
        device = resolve_training_device(config_data.get('device', 'auto'))
        project_name = config_data.get('project_name', f'training_{task_id}')
        learning_rate = effective_config["learning_rate"]
        patience = effective_config["patience"]
        optimizer_name = effective_config["optimizer"]
        class_names = config_data.get('class_names', [])

        requested_config_path = resolve_training_dataset_config_path(dataset_path)
        if class_names:
            if not requested_classes_match_dataset(requested_config_path, class_names):
                known_classes = read_dataset_class_names(requested_config_path)
                error_msg = (
                    "当前版本禁止在训练时筛选或重映射类别，以防修改源标签。"
                    f" 数据集类别={known_classes or '未配置'}，请求类别={class_names}。"
                    " 请使用完整类别训练，或等待派生数据集版本功能。"
                )
                training_logger.error(error_msg)
                update_task_status_file(task_id, "failed", 0.0, error_msg)
                return
            # 与 dataset.yaml 完全一致时直接使用原配置，不做任何标签重映射。
            class_names = []
        
        # 初始化训练器
        training_logger.info(f"初始化YOLO训练器，模型: {model_name}")
        training_logger.info("STEP 1/4: 准备调用 YOLOTrainer.load_model()")
        flush_logger_handlers(training_logger)
        trainer = YOLOTrainer(model_name=model_name, training_profile=training_profile)
        trainer.load_model()

        callback_state = {
            "last_heartbeat": 0.0,
            "latest_grouped_metrics": {},
            "best_epoch": 0,
            "best_fitness": None,
        }

        def on_train_batch_end(ultra_trainer):
            now = time.monotonic()
            cancel = is_cancel_requested(task_id)
            if cancel:
                ultra_trainer.stop = True
            if cancel or now - callback_state["last_heartbeat"] >= 10:
                callback_state["last_heartbeat"] = now
                epoch = int(getattr(ultra_trainer, "epoch", 0) or 0) + 1
                total_epochs = int(getattr(ultra_trainer, "epochs", epochs) or epochs)
                progress = 15.0 + min(epoch / max(total_epochs, 1), 1.0) * 75.0
                record_job_update(
                    task_id,
                    status="cancelling" if cancel else "running",
                    stage="cancelling" if cancel else "training",
                    progress=progress,
                    message="正在协作停止训练..." if cancel else f"正在训练：epoch {epoch}/{total_epochs}",
                    event_type="heartbeat",
                    event_payload={"epoch": epoch, "total_epochs": total_epochs},
                )

        def on_train_epoch_end(ultra_trainer):
            epoch = int(getattr(ultra_trainer, "epoch", 0) or 0) + 1
            total_epochs = int(getattr(ultra_trainer, "epochs", epochs) or epochs)
            progress = 15.0 + min(epoch / max(total_epochs, 1), 1.0) * 75.0
            metrics = dict(getattr(ultra_trainer, "metrics", {}) or {})
            serializable_metrics = {}
            for key, value in metrics.items():
                try:
                    serializable_metrics[str(key)] = float(value)
                except (TypeError, ValueError):
                    continue
            try:
                train_losses = ultra_trainer.label_loss_items(
                    getattr(ultra_trainer, "tloss", None),
                    prefix="train",
                )
                for key, value in dict(train_losses or {}).items():
                    serializable_metrics[str(key)] = float(value)
            except (AttributeError, TypeError, ValueError):
                pass
            current_fitness = getattr(ultra_trainer, "fitness", None)
            try:
                current_fitness = float(current_fitness)
            except (TypeError, ValueError):
                current_fitness = None
            if current_fitness is not None and (
                callback_state["best_fitness"] is None
                or current_fitness >= callback_state["best_fitness"]
            ):
                callback_state["best_fitness"] = current_fitness
                callback_state["best_epoch"] = epoch
            optimizer = getattr(ultra_trainer, "optimizer", None)
            learning_rate = None
            if optimizer and getattr(optimizer, "param_groups", None):
                learning_rate = optimizer.param_groups[0].get("lr")
            payload = {
                "epoch": epoch,
                "total_epochs": total_epochs,
                "learning_rate": learning_rate,
                "best_epoch": callback_state["best_epoch"],
                "metrics": serializable_metrics,
            }
            grouped_metrics = extract_grouped_metrics(
                getattr(getattr(ultra_trainer, "validator", None), "metrics", None),
                config_data.get("_training_preflight") or {},
            )
            if grouped_metrics:
                callback_state["latest_grouped_metrics"] = grouped_metrics
                payload["grouped_metrics"] = grouped_metrics
            cancel = is_cancel_requested(task_id)
            if cancel:
                ultra_trainer.stop = True
            record_job_update(
                task_id,
                status="cancelling" if cancel else "running",
                stage="cancelling" if cancel else "training",
                progress=progress,
                message="正在协作停止训练..." if cancel else f"epoch {epoch}/{total_epochs} 完成",
                event_type="epoch",
                event_payload=payload,
            )

        trainer.model.add_callback("on_train_batch_end", on_train_batch_end)
        trainer.model.add_callback("on_train_epoch_end", on_train_epoch_end)
        
        training_logger.info("STEP 2/4: 模型加载完成")
        update_task_status_file(task_id, "running", 5.0, "模型加载完成")
        
        # 处理数据集配置
        if requested_config_path.is_file():
            dataset_config_path = str(requested_config_path)
            trainer.data_config = dataset_config_path
            training_logger.info(f"使用数据集配置文件: {dataset_config_path}")
        else:
            if not class_names:
                class_names = ["object"]
                training_logger.warning("未提供类别信息，使用默认类别: ['object']")

            training_logger.info(f"创建数据集配置文件，数据集路径: {dataset_path}")
            dataset_config_path = trainer.create_dataset_config(
                dataset_path=dataset_path,
                class_names=class_names
            )
            training_logger.info(f"数据集配置文件已创建: {dataset_config_path}")

        dataset_config_path = str(dataset_config_path)
        dataset_root = Path(dataset_config_path).parent
        # 验证数据集类别配置
        training_logger.info("验证数据集类别配置...")
        is_valid, issues = trainer.validate_dataset_classes(dataset_root, dataset_config_path)
        
        if not is_valid:
            error_msg = "数据集类别验证失败:\n" + "\n".join(issues)
            training_logger.error(error_msg)
            training_logger.error("提示: 标注文件中的类别ID必须在 0 到 (nc-1) 范围内")
            training_logger.error("      如果使用AI辅助标注，请确保类别映射正确")
            update_task_status_file(task_id, "failed", 0.0, error_msg)
            return
        
        training_logger.info("数据集类别验证通过")
        
        update_task_status_file(task_id, "running", 10.0, "数据集配置完成")
        
        # 设置训练参数
        train_kwargs = {
            'epochs': epochs,
            'imgsz': img_size,
            'batch': batch_size,
            'device': device,
            'name': project_name,
            'lr0': learning_rate,
            'optimizer': optimizer_name,
            'save_period': -1,  # 不保存epoch过程模型
            'patience': patience,
            'cache': False,  # 不缓存数据集到内存，节省内存
            'workers': 0,  # 减少数据加载器工作进程数
            'verbose': True,
            'plots': True,
            'save': True,
            'exist_ok': True,
            'resume': False,
            'amp': False,  # 关闭AMP检查，避免二次训练触发额外模型检查/下载
            'deterministic': False
        }
        
        training_logger.info(f"训练配置: {train_kwargs}")
        flush_logger_handlers(training_logger)
        update_task_status_file(task_id, "running", 15.0, "开始训练...")

        if is_cancel_requested(task_id):
            update_task_status_file(task_id, "cancelled", 0.0, "训练在启动前已取消")
            return
        
        # 开始训练
        training_logger.info("STEP 3/4: 准备调用 trainer.train()")
        training_logger.info("=== 开始YOLO模型训练 ===")
        flush_logger_handlers(training_logger)
        results = trainer.train(**train_kwargs)
        if is_cancel_requested(task_id):
            update_task_status_file(task_id, "cancelled", 0.0, "训练已取消")
            return
        training_logger.info("STEP 4/4: trainer.train() 已返回")
        training_logger.info("=== YOLO模型训练完成 ===")
        
        update_task_status_file(task_id, "running", 90.0, "训练完成，正在生成结果...")
        
        # 生成训练结果图表
        try:
            trainer.plot_results(f'runs/train/{project_name}')
        except Exception as e:
            logger.warning(f"生成结果图表失败: {e}")
        
        # 保存结果信息
        # 尝试寻找模型文件，因为 YOLO 可能会改变保存路径（例如 detect/runs/train/...）
        default_model_path = f"runs/train/{project_name}/weights/best.pt"
        actual_model_path = default_model_path
        
        # 检查多个可能的路径
        possible_paths = [
            default_model_path,
            f"runs/detect/runs/train/{project_name}/weights/best.pt",
            f"runs/detect/{project_name}/weights/best.pt",
            f"runs/{project_name}/weights/best.pt"
        ]
        
        for p in possible_paths:
            if os.path.exists(p):
                actual_model_path = p
                training_logger.info(f"找到模型文件: {p}")
                break
        else:
            # 如果没找到，尝试在 runs 目录下搜索
            training_logger.info("未在预期位置找到模型文件，开始全局搜索...")
            try:
                for path in Path("runs").rglob("best.pt"):
                    if project_name in str(path):
                        actual_model_path = str(path)
                        training_logger.info(f"通过搜索找到模型文件: {actual_model_path}")
                        break
            except Exception as e:
                training_logger.warning(f"搜索模型文件时出错: {e}")

        best_path = Path(actual_model_path)
        if best_path.name != "best.pt" or not best_path.is_file() or best_path.stat().st_size <= 0:
            raise FileNotFoundError(f"训练未生成可读取的 best.pt: {actual_model_path}")

        registry_record = register_model_file_external(
            best_path,
            source="trained",
            status="active",
            task_id=task_id,
            dataset_version_id=config_data.get("_dataset_version_id"),
            created_by="training-worker",
            metadata={
                "display_name": f"{project_name} - best",
                "experiment_name": project_name,
                "weight_type": "best",
                "base_model": model_lineage["lineage_root_model"],
                **model_lineage,
                "training_profile": training_profile,
                "activation_status": "active",
                "major_class_threshold": effective_config["major_class_threshold"],
                "grouped_metrics": callback_state["latest_grouped_metrics"],
            },
        )

        # 再次清理输出目录，确保最终只留下best.pt
        keep_only_best_pt_from_model_path(actual_model_path, training_logger)

        results_info = {
            "model_path": actual_model_path,
            "config_path": dataset_config_path,
            "training_time": "完成",
            "final_metrics": "已保存",
            "project_name": project_name,
            "model_registry_id": registry_record["id"],
            "training_profile": training_profile,
            "dataset_version_id": config_data.get("_dataset_version_id"),
            "major_class_threshold": effective_config["major_class_threshold"],
            "grouped_metrics": callback_state["latest_grouped_metrics"],
            "best_epoch": callback_state["best_epoch"],
            "preflight": config_data.get("_training_preflight") or {},
        }
        
        update_task_status_file(task_id, "completed", 100.0, "训练成功完成！", results_info)
        training_logger.info(f"训练任务 {task_id} 成功完成！")
        training_logger.info(f"模型保存路径: {results_info['model_path']}")
        
    except Exception as e:
        error_msg = f"训练失败: {str(e)}"
        try:
            training_logger.error(f"训练任务 {task_id} 失败: {e}")
            training_logger.error(traceback.format_exc())
            flush_logger_handlers(training_logger)
        except Exception:
            logger.error(f"训练任务 {task_id} 失败: {e}")
            logger.error(traceback.format_exc())
        if training_device_requires_gpu(config_data.get("device")) and report_gpu_fault(e):
            gpu_waiting = True
            wait_for_gpu_job(task_id, str(e))
            # CUDA cleanup/destructors can block in a damaged context. The owner
            # exits; only the scheduler may requeue after observing this exit.
            flush_logger_handlers(training_logger if 'training_logger' in locals() else logger)
            os._exit(75)
        else:
            update_task_status_file(task_id, "failed", 0.0, error_msg)
    finally:
        try:
            if 'trainer' in locals():
                try:
                    trainer.model = None
                    trainer.results = None
                except Exception:
                    pass
                del trainer
            if 'results' in locals():
                del results
        except Exception:
            pass
        if not gpu_waiting:
            cleanup_torch_runtime(training_logger if 'training_logger' in locals() else logger)
            cleanup_task_training_workspace(
                config_data,
                training_logger if 'training_logger' in locals() else logger,
            )
        try:
            flush_logger_handlers(training_logger)
            for handler in list(training_logger.handlers):
                handler.close()
                training_logger.removeHandler(handler)
        except Exception:
            pass

def start_training_process(task_id: str, config_data: Dict, dataset_path: str):
    """启动训练进程"""
    try:
        # 创建新的训练进程：必须使用 spawn，不能使用 Linux 默认 fork。
        # fork 会继承父进程中的 PyTorch/CUDA/Ultralytics 状态，二次训练时容易卡在 YOLO(.pt) 或 torch.load。
        ctx = get_training_mp_context()
        process = ctx.Process(
            target=training_worker_process,
            args=(task_id, config_data, dataset_path),
            name=f"training_{task_id}"
        )
        
        # 启动进程
        process.start()
        
        record_job_update(
            task_id,
            stage="queued",
            progress=0.0,
            message="训练进程已创建",
            worker_pid=process.pid,
            event_type="worker_started",
            event_payload={"pid": process.pid},
        )
        
        # 存储进程引用
        training_processes[task_id] = {
            'process': process,
            'started_at': datetime.now(),
            'config': config_data,
            'dataset_path': dataset_path
        }
        logger.info(f"训练进程已启动: {task_id}, PID: {process.pid}")
        return True
        
    except Exception as e:
        logger.error(f"启动训练进程失败: {e}")
        update_task_status_file(task_id, "failed", 0.0, f"启动训练进程失败: {str(e)}")
        cleanup_task_training_workspace(config_data, logger)
        return False

def get_task_status(task_id: str) -> Dict:
    """获取任务状态"""
    from database import SessionLocal, JobTask
    db = SessionLocal()
    try:
        task = db.query(JobTask).filter(JobTask.id == task_id).first()
        if task is not None:
            return task.to_dict()
    finally:
        db.close()
    # 从文件读取状态（优先从文件读取，因为训练在独立进程中运行）
    try:
        status_file = Path(f"temp/training_status/{task_id}.json")
        if status_file.exists():
            # 获取文件的最后修改时间，如果文件很旧且内存中有更新的状态，则可能使用内存
            # 但在这里，文件是由训练进程更新的唯一真实来源
            with open(status_file, 'r', encoding='utf-8') as f:
                status_data = json.load(f)
                task_status_cache[task_id] = status_data
                return status_data
    except Exception as e:
        logger.error(f"读取任务状态失败: {e}")
    
    # 如果文件不存在，检查内存缓存（可能是刚创建但还没写入文件）
    if task_id in task_status_cache:
        return task_status_cache[task_id]
    
    # 默认状态
    return {
        "task_id": task_id,
        "status": "unknown",
        "progress": 0.0,
        "message": "任务状态未知",
        "updated_at": datetime.now().isoformat(),
        "results": None
    }

def _force_stop_after_grace(task_id: str, grace_seconds: int, require_cancel: bool = True):
    time.sleep(max(1, grace_seconds))
    if task_id in training_processes:
        process_info = training_processes[task_id]
        process = process_info['process']
        
        if process.is_alive():
            if require_cancel and not is_cancel_requested(task_id):
                return
            logger.warning(f"协作取消超时，终止训练进程: {task_id}, PID: {process.pid}")
            process.terminate()
            
            # 等待进程结束
            process.join(timeout=10)
            
            if process.is_alive():
                logger.warning(f"进程 {task_id} 未能正常终止，强制杀死")
                process.kill()
                process.join()

        cleanup_task_training_workspace(process_info.get("config") or {}, logger)
        
        # 从字典中移除
        del training_processes[task_id]
        
        current = get_task_status(task_id)
        if current.get("status") not in {"completed", "failed", "cancelled"}:
            update_task_status_file(task_id, "cancelled", 0.0, "训练已被取消")
        
        logger.info(f"训练进程已停止: {task_id}")


def stop_training_process(task_id: str, grace_seconds: int = 20):
    """先请求协作取消，超时后由守护线程终止进程。"""
    if task_id not in training_processes:
        return False
    thread = threading.Thread(
        target=_force_stop_after_grace,
        args=(task_id, grace_seconds),
        name=f"cancel_{task_id}",
        daemon=True,
    )
    thread.start()
    return True

def cleanup_all_processes():
    """清理所有训练进程"""
    logger.info("清理所有训练进程...")
    
    for task_id in list(training_processes.keys()):
        _force_stop_after_grace(task_id, 0, False)
    
    logger.info("所有训练进程已清理完成")


def terminate_all_processes_for_worker_shutdown():
    """Stop owned children without rewriting job state during worker shutdown.

    The next worker instance is responsible for marking those rows as
    failed/worker_lost. This keeps an intentional worker restart distinct from
    a user cancellation and never pretends that the job completed.
    """
    logger.info("训练 Worker 退出，终止其拥有的训练子进程")
    for task_id, process_info in list(training_processes.items()):
        process = process_info["process"]
        if process.is_alive():
            process.terminate()
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)
        cleanup_task_training_workspace(process_info.get("config") or {}, logger)
        training_processes.pop(task_id, None)

def cleanup_finished_processes():
    """清理已完成的进程"""
    poll_training_processes()

def get_active_training_tasks():
    """获取活跃的训练任务列表"""
    cleanup_finished_processes()
    return list(training_processes.keys())


def poll_training_processes():
    """Reap children once and persist an explicit failure for bad exit codes."""
    finished_tasks = []
    for task_id, process_info in list(training_processes.items()):
        process = process_info["process"]
        if process.is_alive() or process.exitcode is None:
            continue
        finished_tasks.append(task_id)
        current = get_task_status(task_id)
        if current.get("status") == "interrupted" and current.get("stage") == "waiting_gpu":
            process.join(timeout=0)
            requeue_gpu_job(task_id)
            training_processes.pop(task_id, None)
            continue
        if process.exitcode != 0:
            logger.error(f"训练进程异常退出: {task_id}, 退出码: {process.exitcode}")
            if current.get("status") not in {"completed", "failed", "cancelled"}:
                update_task_status_file(
                    task_id,
                    "failed",
                    0.0,
                    f"训练进程异常退出，退出码: {process.exitcode}",
                )
        elif current.get("status") not in {"completed", "failed", "cancelled"}:
            update_task_status_file(
                task_id,
                "failed",
                0.0,
                "训练进程已退出，但未写入终态",
            )
        cleanup_task_training_workspace(process_info.get("config") or {}, logger)
        training_processes.pop(task_id, None)
    return finished_tasks


def ensure_training_monitor():
    global monitor_thread
    with monitor_thread_lock:
        if monitor_thread and monitor_thread.is_alive():
            return
        monitor_thread = threading.Thread(
            target=monitor_training_processes,
            name="training_process_monitor",
            daemon=True,
        )
        monitor_thread.start()

def monitor_training_processes():
    """监控训练进程状态"""
    while True:
        try:
            poll_training_processes()
            time.sleep(5)  # 每5秒检查一次
            
        except KeyboardInterrupt:
            logger.info("监控进程收到中断信号，正在退出...")
            break
        except Exception as e:
            logger.error(f"监控进程出错: {e}")
            time.sleep(10)

if __name__ == "__main__":
    logger.info("训练工作进程管理器启动")
    
    # 创建必要的目录
    Path("temp/training_status").mkdir(parents=True, exist_ok=True)
    
    # 启动监控
    try:
        monitor_training_processes()
    except KeyboardInterrupt:
        logger.info("收到退出信号")
    finally:
        cleanup_all_processes()
        logger.info("训练工作进程管理器退出")
