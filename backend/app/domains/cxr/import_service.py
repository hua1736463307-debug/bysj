"""CXR 导入：受控只读来源 → 受控数据集目录 → CxrSample。

安全边界：
- API 只接受**已配置的 source_id**，不接受任意服务器绝对路径或网络 URL；
- 源目录只读，选中的图像**复制**到新版本目录（不使用可能被源修改污染的硬链接）；
- 相对路径规范化后必须仍在源根/目标根之内，改名绕过与路径穿越一律拒绝；
- 患者/检查只保存稳定伪标识，原始相对路径只进受限选择报告，不进 API 响应。
"""
from __future__ import annotations

import csv
import hashlib
import re
import shutil
from pathlib import Path, PurePosixPath

from PIL import Image
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import ApiError
from app.core.storage import atomic_write_bytes, ensure_dir, sha256_file
from app.db.models import CxrSample, DatasetAsset, DatasetResource, DatasetVersion
from app.schemas.cxr import (
    CXR_GROUP_LEVEL,
    CXR_LABEL_SCHEMA,
    CXR_TASK,
    CxrLabelError,
    normalize_raw_labels,
    RAW_STATES,
)
from app.domains.cxr.split_service import canonical_json, pseudo_key, sha256_text

SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_SOURCE_CSV_BYTES = 64 * 1024 * 1024
PATIENT_PATTERN = re.compile(r"patient(\d+)", re.IGNORECASE)
STUDY_PATTERN = re.compile(r"study(\d+)", re.IGNORECASE)
FRONTAL_COLUMNS = ("Frontal/Lateral", "Frontal_Lateral", "View")
SYNTHETIC_MARKER = "SYNTHETIC_DATASET.txt"


class CxrImportError(ApiError):
    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(422, "cxr_import_invalid", message, details or {})


def is_synthetic_source(root: Path) -> bool:
    """显式标记的合成 fixture：默认拒绝，避免用假数据冒充研究结果。"""
    return (Path(root) / SYNTHETIC_MARKER).is_file()


def assert_source_allowed(root: Path, settings: Settings) -> bool:
    synthetic = is_synthetic_source(root)
    if synthetic and not getattr(settings, "cxr_allow_synthetic", False):
        raise ApiError(
            409,
            "cxr_synthetic_source_rejected",
            "该来源被标记为合成 fixture，需显式开启 CXR_ALLOW_SYNTHETIC 才能用于工程验证（不得用于研究结论）",
        )
    return synthetic


def resolve_source_root(settings: Settings, source_id: str) -> Path:
    """只解析已配置的 source_id；未配置即拒绝，不回落到任意路径。"""
    sources = getattr(settings, "cxr_sources", None) or {}
    configured = sources.get(source_id)
    if configured:
        root = Path(configured).expanduser()
    else:
        base = getattr(settings, "cxr_import_root", None)
        if not base:
            raise ApiError(
                409,
                "cxr_source_not_configured",
                "未配置 CXR 数据来源；管理员需先设置 CXR_IMPORT_ROOT 或 CXR_SOURCES",
            )
        root = Path(base).expanduser() / source_id
    root = root.resolve()
    if not root.is_dir():
        raise ApiError(404, "cxr_source_missing", "配置的 CXR 来源目录不存在或不可读")
    return root


def _resolve_inside(root: Path, relative: str) -> Path:
    text = str(relative or "").replace("\\", "/").strip()
    if not text:
        raise CxrImportError("图像路径为空")
    pure = PurePosixPath(text)
    if pure.is_absolute() or (len(text) >= 2 and text[1] == ":"):
        raise CxrImportError("源 CSV 不允许绝对路径", {"path": relative})
    if any(part == ".." for part in pure.parts):
        raise CxrImportError("源 CSV 不允许上级目录", {"path": relative})
    candidate = (root / pure).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise CxrImportError("图像路径越界", {"path": relative}) from exc
    return candidate


def parse_identity(relative_path: str) -> tuple[str, str]:
    """从经验证的路径结构解析 patient/study；解析失败直接拒绝。"""
    text = str(relative_path).replace("\\", "/")
    patient = PATIENT_PATTERN.search(text)
    study = STUDY_PATTERN.search(text)
    if patient is None or study is None:
        raise CxrImportError(
            "无法从路径解析患者/检查标识",
            {"path": relative_path, "expected": "…/patientXXXXX/studyN/…"},
        )
    return patient.group(1), study.group(1)


def parse_projection(relative_path: str) -> str:
    name = PurePosixPath(str(relative_path).replace("\\", "/")).name.lower()
    stem = name.rsplit(".", 1)[0]
    for token in ("ap", "pa"):
        if re.search(rf"(^|[_\-\s]){token}($|[_\-\s])", stem):
            return token.upper()
    return "unknown"


def read_source_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise CxrImportError("来源 CSV 不存在", {"csv": path.name})
    if path.stat().st_size > MAX_SOURCE_CSV_BYTES:
        raise CxrImportError("来源 CSV 超过大小上限")
    rows: list[dict] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise CxrImportError("来源 CSV 没有表头")
        header = [name.strip() for name in reader.fieldnames]
        if "Path" not in header:
            raise CxrImportError("来源 CSV 缺少 Path 列", {"header": header})
        for index, raw in enumerate(reader, start=2):
            row = {(key or "").strip(): value for key, value in raw.items()}
            row["_row"] = index
            rows.append(row)
    if not rows:
        raise CxrImportError("来源 CSV 没有数据行")
    return rows


def pixel_sha256(path: Path) -> tuple[str, int, int, str]:
    """解码像素 hash + 原始尺寸 + 格式；8bit JPG/PNG 之外明确拒绝。"""
    try:
        with Image.open(path) as image:
            image.load()
            fmt = (image.format or "").upper()
            width, height = image.size
            mode = image.mode
            digest = hashlib.sha256()
            digest.update(f"{width}x{height}:{mode}:".encode("ascii"))
            digest.update(image.tobytes())
            return digest.hexdigest(), width, height, fmt
    except Exception as exc:  # 损坏/非图像
        raise CxrImportError("源图像不可解码", {"file": path.name, "error": type(exc).__name__}) from exc


def plan_import(
    *,
    source_root: Path,
    rows: list[dict],
    max_patients: int,
    namespace: str,
    verify_pixels: bool = True,
) -> tuple[list[dict], dict]:
    """取样 + 审计；不看标签或模型得分选图。

    规则：按 patient_key 的固定 hash 排序取前 N 位患者；患者内按规范化
    路径排序取首张正位图。
    """
    by_patient: dict[str, list[dict]] = {}
    rejected: list[dict] = []
    duplicates_in_csv: dict[str, list[int]] = {}

    for row in rows:
        raw_path = row.get("Path")
        if not raw_path:
            rejected.append({"row": row.get("_row"), "reason": "missing_path"})
            continue
        view = ""
        for column in FRONTAL_COLUMNS:
            if column in row:
                view = str(row.get(column) or "").strip()
                break
        if view and view.lower() != "frontal":
            continue  # 首期只接正位；侧位不入库，不混入后再说明
        try:
            patient_id, study_id = parse_identity(str(raw_path))
            raw_labels = normalize_raw_labels(row)
        except CxrLabelError as exc:
            rejected.append({"row": row.get("_row"), "reason": "label_or_identity_invalid", "detail": exc.details})
            continue
        except CxrImportError as exc:
            rejected.append({"row": row.get("_row"), "reason": "identity_unparsable", "detail": exc.data})
            continue
        normalized = str(raw_path).replace("\\", "/")
        duplicates_in_csv.setdefault(normalized.lower(), []).append(int(row.get("_row") or 0))
        patient_key = pseudo_key(namespace, f"patient:{patient_id}")
        by_patient.setdefault(patient_key, []).append(
            {
                "source_path": normalized,
                "patient_key": patient_key,
                "study_key": pseudo_key(namespace, f"study:{patient_id}:{study_id}"),
                "view": "Frontal",
                "projection": parse_projection(normalized),
                "raw_labels": raw_labels,
                "row": row.get("_row"),
            }
        )

    conflicting = {
        path: rows_seen for path, rows_seen in duplicates_in_csv.items() if len(rows_seen) > 1
    }
    if conflicting:
        raise CxrImportError("同一路径在 CSV 中重复出现", {"paths": list(conflicting)[:10]})

    ordered_patients = sorted(
        by_patient, key=lambda key: (sha256_text(f"{namespace}:sample-order:{key}"), key)
    )
    selected_patients = ordered_patients[: max(1, int(max_patients))]
    selected: list[dict] = []
    for patient_key in selected_patients:
        candidates = sorted(by_patient[patient_key], key=lambda item: item["source_path"])
        chosen = candidates[0]
        chosen["alternatives"] = [item["source_path"] for item in candidates[1:]]
        if verify_pixels:
            absolute = _resolve_inside(source_root, chosen["source_path"])
            if absolute.suffix.lower() not in SUPPORTED_SUFFIXES:
                raise CxrImportError(
                    "不支持的图像格式（首期仅 8bit JPG/PNG）",
                    {"path": chosen["source_path"], "suffix": absolute.suffix},
                )
            digest, width, height, fmt = pixel_sha256(absolute)
            if fmt not in {"JPEG", "PNG"}:
                raise CxrImportError("不支持的图像格式", {"format": fmt})
            chosen.update(
                {
                    "sha256": sha256_file(absolute),
                    "pixel_sha256": digest,
                    "width": width,
                    "height": height,
                    "format": fmt,
                    "size": absolute.stat().st_size,
                }
            )
        selected.append(chosen)

    if not selected:
        raise CxrImportError("没有可导入的正位胸片")

    report = {
        "requested_max_patients": int(max_patients),
        "available_patients": len(by_patient),
        "selected_patients": len(selected_patients),
        "selected_images": len(selected),
        "truncated": len(by_patient) > len(selected_patients),
        "rejected_rows": rejected[:200],
        "rejected_count": len(rejected),
        "lateral_skipped": sum(1 for row in rows if str(row.get("Frontal/Lateral") or "").strip().lower() == "lateral"),
        "raw_state_contract": list(RAW_STATES),
        "selection_rule": "patient_key hash order, first frontal image per patient by normalized path",
    }
    return selected, report


def dataset_layout(settings: Settings, dataset_id: str, version_id: str) -> tuple[Path, Path]:
    root = ensure_dir(Path(settings.data_root) / "datasets" / dataset_id / version_id)
    images = ensure_dir(root / "images")
    return root, images


def _target_relative(entry: dict) -> str:
    # 目标相对路径只含伪标识，不暴露原始患者目录名。
    return f"{entry['patient_key']}/{entry['study_key']}/{entry['sha256'][:12]}.jpg"


def materialize(
    db: Session,
    settings: Settings,
    *,
    source_root: Path,
    source_csv: Path,
    source_id: str,
    selected: list[dict],
    report: dict,
    name: str,
    license_note: str | None,
    owner_id: int,
    owner_username: str,
    dataset_id: str | None = None,
    synthetic: bool = False,
) -> dict:
    """在受控目录落地图像与 CxrSample；本函数不 commit 事务边界以外的状态。"""
    dataset_id = dataset_id or f"cxr-{source_id}-{sha256_text(str(owner_id) + name)[:8]}"
    resource = (
        db.query(DatasetResource)
        .filter(DatasetResource.dataset_id == dataset_id, DatasetResource.dataset_role == "medical")
        .one_or_none()
    )
    if resource is None:
        resource = DatasetResource(
            dataset_id=dataset_id,
            dataset_role="medical",
            owner_user_id=owner_id,
            owner_username=owner_username,
            name=(("[SYNTHETIC] " if synthetic else "") + (name or f"CXR {source_id}"))[:255],
            description=(
                "胸片五标签辅助分析数据集（cxr_multilabel / cxr-five-v1）"
                + ("；合成 fixture，仅用于工程验证" if synthetic else "")
            ),
            source_url=None,
            source_version=source_id,
            license=license_note,
            modality="X-ray",
            task_type=CXR_TASK,
            group_level=CXR_GROUP_LEVEL,
        )
        db.add(resource)
        db.flush()

    csv_sha = sha256_file(source_csv)
    version = DatasetVersion(
        dataset_id=resource.id,
        dataset_role="medical",
        status="ready",
        state="draft",
        created_by=owner_username,
        created_by_user_id=owner_id,
        class_map_json={},
        metadata_json={
            "task": CXR_TASK,
            "label_schema": CXR_LABEL_SCHEMA,
            "group_level": CXR_GROUP_LEVEL,
            "synthetic": bool(synthetic),
            "source": {"source_id": source_id, "csv_sha256": csv_sha},
        },
    )
    db.add(version)
    db.flush()

    root, images_dir = dataset_layout(settings, resource.dataset_id, version.id)
    version.root_path = str(root)
    samples = 0
    for entry in selected:
        source_file = _resolve_inside(source_root, entry["source_path"])
        relative = _target_relative(entry)
        target = images_dir / relative
        ensure_dir(target.parent)
        shutil.copyfile(source_file, target)
        copied = sha256_file(target)
        if copied != entry["sha256"]:
            raise ApiError(500, "cxr_copy_hash_mismatch", "复制后文件 hash 与源不一致")
        asset = DatasetAsset(
            dataset_version_id=version.id,
            relative_path=relative,
            basename=target.name,
            size=target.stat().st_size,
            width=int(entry["width"]),
            height=int(entry["height"]),
            content_hash=copied,
            annotations_json=[],  # CXR 不使用 BBox 标注
            group_level=CXR_GROUP_LEVEL,
            source_group=entry["patient_key"],
            transform_json={},
        )
        db.add(asset)
        db.flush()
        db.add(
            CxrSample(
                asset_id=asset.id,
                dataset_version_id=version.id,
                patient_key=entry["patient_key"],
                study_key=entry["study_key"],
                component_key=entry["patient_key"],  # 冻结时按重复候选合并
                view=entry["view"],
                raw_labels_json=entry["raw_labels"],
                schema_version=CXR_LABEL_SCHEMA,
                source_hash=csv_sha,
                pixel_hash=entry.get("pixel_sha256") or "",
            )
        )
        samples += 1

    version.asset_count = samples
    selection_payload = {
        "schema": "cxr-selection-v1",
        "source_id": source_id,
        "source_csv_sha256": csv_sha,
        "dataset_id": resource.dataset_id,
        "version_id": version.id,
        "report": report,
        "selected": [
            {
                "source_path": entry["source_path"],
                "patient_key": entry["patient_key"],
                "study_key": entry["study_key"],
                "projection": entry["projection"],
                "sha256": entry.get("sha256"),
                "pixel_sha256": entry.get("pixel_sha256"),
                "raw_labels": entry["raw_labels"],
                "row": entry.get("row"),
            }
            for entry in selected
        ],
    }
    selection_path = root / "selection.json"
    atomic_write_bytes(selection_path, canonical_json(selection_payload).encode("utf-8"))
    version.metadata_json = {
        **(version.metadata_json or {}),
        "selection_path": str(selection_path),
        "selection_sha256": sha256_text(canonical_json(selection_payload)),
    }
    db.flush()
    return {
        "dataset_id": resource.id,
        "dataset_code": resource.dataset_id,
        "version_id": version.id,
        "asset_count": samples,
        "report": report,
        "selection_sha256": version.metadata_json["selection_sha256"],
    }
