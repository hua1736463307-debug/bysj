"""训练域：预检、排队、数据集描述与训练执行。"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

from sqlalchemy.orm import Session

from app.api.deps import CurrentUser
from app.core.config import Settings
from app.core.errors import ApiError
from app.core.storage import ensure_dir
from app.db.models import DatasetAsset, DatasetVersion, JobTask, ModelRegistry
from app.domains import job_service
from app.domains.datasets import service as dataset_service
from app.domains.models import service as model_service
from app.ml.yolo_adapter import compute_sha256, environment_snapshot, get_adapter, write_json
from app.schemas.training import TrainingParams
from pydantic import ValidationError

DEFAULT_PARAMS = {
    "epochs": 80,
    "batch": 8,
    "imgsz": 640,
    "seed": 42,
    "deterministic": True,
    "device": "0",
}


class TrainingPreflightError(ApiError):
    def __init__(self, checks: list[dict]) -> None:
        super().__init__(422, "preflight_failed", "训练预检未通过", {"checks": checks})
        self.checks = checks


def _normalize_params(payload: dict, checks: list[dict]) -> dict:
    try:
        normalized = TrainingParams.model_validate(payload.get("params") or {}).model_dump()
        checks.append({"name": "params", "status": "pass", "detail": "参数白名单与范围校验通过"})
        return normalized
    except ValidationError as exc:
        checks.append({"name": "params", "status": "fail", "detail": str(exc)})
        return {}


def preflight(db: Session, settings: Settings, user: CurrentUser, payload: dict) -> dict:
    checks: list[dict] = []
    version_id = payload.get("dataset_version_id")
    version: DatasetVersion | None = None
    if not version_id:
        checks.append({"name": "dataset_version", "status": "fail", "detail": "必须提供 dataset_version_id"})
    else:
        try:
            version = dataset_service.get_version(db, version_id, user)
        except ApiError:
            checks.append({"name": "dataset_version", "status": "fail", "detail": "版本不存在或无权访问"})
        else:
            checks.append(
                {
                    "name": "dataset_version",
                    "status": "pass" if version.state == "frozen" else "fail",
                    "detail": f"state={version.state}",
                }
            )
            if version.state != "frozen":
                checks.append({"name": "frozen", "status": "fail", "detail": "训练只接受冻结版本"})

    assets = []
    if version is not None:
        assets = (
            db.query(DatasetAsset)
            .filter(DatasetAsset.dataset_version_id == version.id, DatasetAsset.deleted_at.is_(None))
            .all()
        )
        annotated = sum(1 for asset in assets if asset.annotations_json)
        checks.append(
            {
                "name": "assets",
                "status": "pass" if assets and annotated else "fail",
                "detail": f"assets={len(assets)} annotated={annotated}",
            }
        )
        checks.append(
            {
                "name": "class_map",
                "status": "pass" if (version.class_map_json or {}) else "fail",
                "detail": f"classes={len(version.class_map_json or {})}",
            }
        )

    model = None
    model_id = payload.get("model_id") or payload.get("model_ref")
    if model_id and model_id not in {"none", "scratch"}:
        try:
            model = model_service.get_model(db, model_id, user)
            path = model_service.verify_model_file(model, settings)
            checks.append({"name": "model", "status": "pass", "detail": f"sha256={model.sha256[:12]} size={path.stat().st_size}"})
        except ApiError as exc:
            checks.append({"name": "model", "status": "fail", "detail": exc.message})
    else:
        status = "pass" if (settings.ml_backend or "").lower() == "dummy" else "fail"
        checks.append(
            {
                "name": "model",
                "status": status,
                "detail": "dummy backend 允许无基础权重" if status == "pass" else "必须提供已登记的基础权重",
            }
        )

    params = _normalize_params(payload, checks)
    task = str(payload.get("task") or "detect")
    checks.append({"name": "task", "status": "pass" if task == "detect" else "fail", "detail": f"task={task}"})

    ok = all(item["status"] != "fail" for item in checks)
    return {
        "ok": ok,
        "checks": checks,
        "params": params,
        "dataset_version_id": version.id if version else None,
        "model_id": model.id if model else None,
    }


def create_training_job(
    db: Session,
    settings: Settings,
    user: CurrentUser,
    payload: dict,
    *,
    idempotency_key: str | None,
) -> tuple[JobTask, bool]:
    report = preflight(db, settings, user, payload)
    if not report["ok"]:
        raise TrainingPreflightError(report["checks"])

    active = (
        db.query(JobTask)
        .filter(
            JobTask.owner_id == str(user.id),
            JobTask.type == "training",
            JobTask.status.in_(["pending", "running", "cancelling"]),
        )
        .count()
    )
    if active >= settings.max_pending_training_per_user:
        raise ApiError(429, "queue_limit", "待执行训练任务已达上限")

    job_payload = {
        "dataset_version_id": report["dataset_version_id"],
        "model_id": report["model_id"],
        "task": "detect",
        "params": report["params"],
    }
    if payload.get("name"):
        job_payload["name"] = str(payload["name"])[:128]
    return job_service.create_job(
        db,
        job_type="training",
        owner_id=user.id,
        owner_username=user.username,
        payload=job_payload,
        idempotency_key=idempotency_key,
    )


def build_dataset_yaml(db: Session, version: DatasetVersion, workdir: Path) -> Path:
    assets = (
        db.query(DatasetAsset)
        .filter(DatasetAsset.dataset_version_id == version.id, DatasetAsset.deleted_at.is_(None))
        .all()
    )
    root = Path(version.root_path)
    lists_dir = ensure_dir(workdir / "splits")
    split_files = {}
    for split in ("train", "val", "test"):
        paths = [str((root / "images" / asset.relative_path).resolve()) for asset in assets if asset.split == split]
        target = lists_dir / f"{split}.txt"
        target.write_text("\n".join(paths) + ("\n" if paths else ""), encoding="utf-8")
        split_files[split] = target

    names = {str(k): str(v) for k, v in (version.class_map_json or {}).items()}
    dataset_yaml = workdir / "dataset.yaml"
    lines = [
        f"path: {root.resolve().as_posix()}",
        f"train: {(split_files['train']).as_posix()}",
        f"val: {(split_files['val']).as_posix()}",
        f"test: {(split_files['test']).as_posix()}",
        f"nc: {len(names)}",
        "names:",
    ]
    for key in sorted(names, key=lambda item: int(item)):
        lines.append(f"  {key}: {names[key]}")
    dataset_yaml.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return dataset_yaml


def _run_dummy(request: dict, workdir: Path, progress, should_stop=None) -> dict:
    target = workdir / "dummy-best.pt"
    payload = {
        "backend": "dummy",
        "job_id": request["job_id"],
        "class_map": request["class_map"],
        "note": "contract-test artifact; not a trainable neural network",
    }
    target.write_text(json.dumps(payload, sort_keys=True), encoding="utf-8")
    for step in range(1, 6):
        if should_stop and should_stop():
            raise JobStopped("训练收到取消请求")
        if progress:
            progress(step / 5, "training", f"dummy 训练 {step}/5")
        time.sleep(0.01)
    return {
        "best_path": str(target),
        "metrics": {
            "precision": None,
            "recall": None,
            "map50": None,
            "map50_95": None,
            "per_class": {},
            "reason": "dummy backend 不产生真实训练指标",
        },
        "epochs_completed": 0,
        "backend": "dummy",
    }


def _run_ultralytics(request: dict, workdir: Path, should_stop, progress) -> dict:
    request_file = workdir / "train-request.json"
    output_file = workdir / "train-result.json"
    request_file.write_text(json.dumps(request, ensure_ascii=False), encoding="utf-8")
    import app as app_package

    backend_dir = str(Path(app_package.__file__).resolve().parents[1])
    child_env = os.environ.copy()
    child_env["PYTHONPATH"] = backend_dir + os.pathsep + child_env.get("PYTHONPATH", "")
    log_path = workdir / "training.log"
    # 必须重定向到文件：PIPE 缓冲区写满会让训练子进程阻塞。
    with log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
        process = subprocess.Popen(
            [sys.executable, "-m", "app.ml.train_cli", "--request", str(request_file), "--output", str(output_file), "--parent-pid", str(os.getpid())],
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            cwd=str(workdir),
            env=child_env,
            start_new_session=os.name == "posix",
        )
        try:
            while process.poll() is None:
                if should_stop and should_stop():
                    _signal_training(process, hard=False)
                    try:
                        process.wait(timeout=30)
                    except subprocess.TimeoutExpired:
                        _signal_training(process, hard=True)
                        process.wait(timeout=10)
                    raise JobStopped("训练收到取消请求")
                if progress:
                    progress(None, "training", "训练进行中")
                time.sleep(2)
        finally:
            if process.poll() is None:
                _signal_training(process, hard=True)
            process.wait(timeout=10)
    if process.returncode != 0:
        stderr = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise RuntimeError(f"训练子进程退出码 {process.returncode}: {stderr}")
    if not output_file.exists():
        raise RuntimeError("训练子进程没有产出结果文件")
    return json.loads(output_file.read_text(encoding="utf-8"))


def _signal_training(process, *, hard: bool) -> None:
    """Signal only the session created for this exact training child."""
    if os.name == "posix":
        import signal
        try:
            if os.getpgid(process.pid) == process.pid:
                os.killpg(process.pid, signal.SIGKILL if hard else signal.SIGTERM)
        except ProcessLookupError:
            pass
    elif hard:
        process.kill()
    else:
        process.terminate()


from app.workers.dispatcher import JobCancelled


class JobStopped(JobCancelled):
    """训练被取消。"""


def train(
    settings: Settings,
    db: Session,
    *,
    job_id: str,
    version: DatasetVersion,
    model: ModelRegistry | None,
    class_map: dict,
    params: dict,
    workdir: Path,
    should_stop=None,
    progress=None,
) -> dict:
    adapter = get_adapter(settings)
    dataset_yaml = build_dataset_yaml(db, version, workdir)
    request = {
        "job_id": job_id,
        "dataset_yaml": str(dataset_yaml),
        "base_model_path": model.path if model else None,
        "class_map": class_map,
        "params": params,
        "workdir": str(workdir),
        "env": environment_snapshot(),
    }
    if adapter.backend == "dummy":
        result = _run_dummy(request, workdir, progress, should_stop)
    else:
        result = _run_ultralytics(request, workdir, should_stop, progress)
    result["dataset_manifest_sha256"] = version.manifest_sha256
    result["dataset_yaml"] = str(dataset_yaml)
    result["dataset_yaml_sha256"] = compute_sha256(dataset_yaml)
    result["request_sha256"] = write_json(workdir / "train-request-record.json", request)
    return result
