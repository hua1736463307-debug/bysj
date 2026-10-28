"""ISIC 2018 Task 3 导入：官方目录布局 → 受控数据集目录 → IsicSample。

与 CXR 的差异（有意为之，不是复用不足）：
- 标签是**单标签互斥**（softmax），不是四态多标签；本模块不读取也不产生 CXR 四态。
- 分组级别是 `lesion`（官方仅对 training 集提供 lesion 分组表）；
  validation/test 没有分组表，只能按单图分组，因此**患者级隔离不可证明**，必须如实标注。
- 官方 validation/test 作为外部留出集单独记录，不参与四套拟合/阈值选择。

来源布局（官方 zip 解压后）：
    <root>/ISIC2018_Task3_<Split>_Input/ISIC_xxxxxxx.jpg
    <root>/ISIC2018_Task3_<Split>_GroundTruth/ISIC2018_Task3_<Split>_GroundTruth.csv
    <root>/ISIC2018_Task3_Training_LesionGroupings.csv
也接受这些文件位于 `<root>/extracted/` 之下。
"""
from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path

from PIL import Image
from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import ApiError
from app.core.sources import resolve_inside, resolve_source_root
from app.core.storage import atomic_write_bytes, ensure_dir, sha256_file
from app.db.models import DatasetAsset, DatasetResource, DatasetVersion, IsicSample
from app.directions.isic import ISIC_DIRECTION_KEY, ISIC_LABEL_SCHEMA, ISIC_LABELS, ISIC_TASK
from app.domains.cxr.split_service import canonical_json, sha256_text

SPLIT_LABEL = {"train": "Training", "val": "Validation", "test": "Test"}
SUPPORTED_SUFFIXES = {".jpg", ".jpeg", ".png"}
MAX_LABEL_CSV_BYTES = 8 * 1024 * 1024
SOURCE_COLUMNS = tuple(label.source_column for label in ISIC_LABELS)
CLASS_INDEX = {label.source_column: label.index for label in ISIC_LABELS}
CLASS_KEY_BY_COLUMN = {label.source_column: label.key for label in ISIC_LABELS}


class IsicImportError(ApiError):
    def __init__(self, message: str, details: dict | None = None) -> None:
        super().__init__(422, "isic_import_invalid", message, details or {})


def _first_existing(root: Path, candidates: list[str]) -> Path | None:
    for relative in candidates:
        path = root / relative
        if path.exists():
            return path
    return None


def find_ground_truth(root: Path, split: str) -> Path:
    name = f"ISIC2018_Task3_{SPLIT_LABEL[split]}_GroundTruth"
    path = _first_existing(
        root,
        [
            f"{name}/{name}.csv",
            f"extracted/{name}/{name}.csv",
            f"{name}.csv",
        ],
    )
    if path is None or not path.is_file():
        raise IsicImportError(
            "缺少官方 GroundTruth CSV",
            {"split": split, "expected": f"{name}/{name}.csv"},
        )
    if path.stat().st_size > MAX_LABEL_CSV_BYTES:
        raise IsicImportError("GroundTruth CSV 超过大小上限", {"path": path.name})
    return path


def find_images_dir(root: Path, split: str) -> Path:
    name = f"ISIC2018_Task3_{SPLIT_LABEL[split]}_Input"
    path = _first_existing(root, [name, f"extracted/{name}"])
    if path is None or not path.is_dir():
        raise IsicImportError(
            "缺少官方图像目录",
            {
                "split": split,
                "expected": f"{name}/",
                "hint": "官方 Input zip 需解压到来源根目录；未下载时不得导入该 split",
            },
        )
    return path


def find_lesion_groupings(root: Path) -> Path | None:
    return _first_existing(
        root,
        [
            "ISIC2018_Task3_Training_LesionGroupings.csv",
            "extracted/ISIC2018_Task3_Training_LesionGroupings.csv",
        ],
    )


def load_ground_truth(path: Path) -> list[dict]:
    """读取官方 one-hot 表；列名必须与官方 7 列一致，且每行恰好一个 1。"""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = [name.strip() for name in (reader.fieldnames or [])]
        if "image" not in header:
            raise IsicImportError("GroundTruth 缺少 image 列", {"header": header})
        missing = [column for column in SOURCE_COLUMNS if column not in header]
        extra = [column for column in header if column not in {"image", *SOURCE_COLUMNS}]
        if missing or extra:
            raise IsicImportError(
                "GroundTruth 列与官方 ISIC 2018 Task3 不一致",
                {"missing": missing, "unexpected": extra, "header": header},
            )
        rows: list[dict] = []
        for index, raw in enumerate(reader, start=2):
            image = (raw.get("image") or "").strip()
            if not image:
                raise IsicImportError("GroundTruth 存在空 image", {"row": index})
            positives = []
            for column in SOURCE_COLUMNS:
                value = str(raw.get(column) or "").strip()
                try:
                    numeric = float(value)
                except ValueError as exc:
                    raise IsicImportError("标签取值必须是 0/1", {"row": index, "column": column, "value": value}) from exc
                if numeric not in (0.0, 1.0):
                    raise IsicImportError("标签取值必须是 0/1", {"row": index, "column": column, "value": value})
                if numeric == 1.0:
                    positives.append(column)
            if len(positives) != 1:
                raise IsicImportError(
                    "单标签任务要求每行恰好一个正类",
                    {"row": index, "image": image, "positives": positives},
                )
            column = positives[0]
            rows.append(
                {
                    "image": image,
                    "class_column": column,
                    "class_key": CLASS_KEY_BY_COLUMN[column],
                    "class_index": CLASS_INDEX[column],
                }
            )
    if not rows:
        raise IsicImportError("GroundTruth 没有数据行")
    duplicates = len(rows) - len({row["image"] for row in rows})
    if duplicates:
        raise IsicImportError("GroundTruth 存在重复 image", {"duplicates": duplicates})
    return rows


def load_lesion_groupings(path: Path | None) -> dict[str, dict]:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        header = [name.strip() for name in (reader.fieldnames or [])]
        if "image" not in header or "lesion_id" not in header:
            raise IsicImportError("LesionGroupings 缺少 image/lesion_id 列", {"header": header})
        mapping: dict[str, dict] = {}
        for raw in reader:
            image = (raw.get("image") or "").strip()
            lesion = (raw.get("lesion_id") or "").strip()
            if not image or not lesion:
                continue
            mapping[image] = {
                "lesion_id": lesion,
                "diagnosis_confirm_type": (raw.get("diagnosis_confirm_type") or "").strip() or None,
            }
        return mapping


def pixel_sha256(path: Path) -> tuple[str, int, int, str]:
    try:
        with Image.open(path) as image:
            image.load()
            width, height = image.size
            mode = image.mode
            digest = hashlib.sha256()
            digest.update(f"{width}x{height}:{mode}:".encode("ascii"))
            digest.update(image.tobytes())
            return digest.hexdigest(), width, height, (image.format or "").upper()
    except Exception as exc:
        raise IsicImportError("源图像不可解码", {"file": path.name, "error": type(exc).__name__}) from exc


def plan_import(
    *,
    root: Path,
    split: str,
    max_images: int | None = None,
) -> tuple[list[dict], dict]:
    """校验标签与图像的一致性，产出可导入清单（不做任何写入）。"""
    if split not in SPLIT_LABEL:
        raise IsicImportError("split 只支持 train/val/test", {"split": split})
    gt_path = find_ground_truth(root, split)
    images_dir = find_images_dir(root, split)
    rows = load_ground_truth(gt_path)
    groupings = load_lesion_groupings(find_lesion_groupings(root)) if split == "train" else {}

    available = {path.stem: path for path in images_dir.iterdir() if path.suffix.lower() in SUPPORTED_SUFFIXES}
    if not available:
        raise IsicImportError("图像目录中没有可用图像", {"dir": images_dir.name})

    missing = [row["image"] for row in rows if row["image"] not in available]
    if missing:
        raise IsicImportError(
            "GroundTruth 中的图像在目录中缺失",
            {"missing_count": len(missing), "examples": missing[:5], "split": split},
        )
    unlabeled = sorted(set(available) - {row["image"] for row in rows})

    entries: list[dict] = []
    for row in rows:
        path = available[row["image"]]
        digest, width, height, fmt = pixel_sha256(path)
        group = groupings.get(row["image"])
        lesion_id = group["lesion_id"] if group else None
        entries.append(
            {
                "image": row["image"],
                "relative_source": f"{images_dir.name}/{path.name}",
                "source_path": str(path),
                "class_key": row["class_key"],
                "class_index": row["class_index"],
                "class_column": row["class_column"],
                "lesion_id": lesion_id,
                "diagnosis_confirm_type": group["diagnosis_confirm_type"] if group else None,
                "sha256": sha256_file(path),
                "pixel_sha256": digest,
                "width": width,
                "height": height,
                "format": fmt,
                "size": path.stat().st_size,
            }
        )
    if max_images is not None:
        entries = entries[: max(1, int(max_images))]

    counts = {label.key: 0 for label in ISIC_LABELS}
    for entry in entries:
        counts[entry["class_key"]] += 1
    report = {
        "split": split,
        "ground_truth": str(gt_path.relative_to(root)) if gt_path.is_relative_to(root) else gt_path.name,
        "images_dir": images_dir.name,
        "labeled_rows": len(rows),
        "imported": len(entries),
        "unlabeled_images_in_dir": len(unlabeled),
        "lesion_groupings_available": bool(groupings),
        "lesion_groups": len({entry["lesion_id"] for entry in entries if entry["lesion_id"]}),
        "class_counts": counts,
        "class_share": {key: round(value / max(1, len(entries)), 6) for key, value in counts.items()},
        "confirm_type_counts": _count_by(entries, "diagnosis_confirm_type"),
        "limitations": _limitations(split, bool(groupings)),
    }
    return entries, report


def _count_by(entries: list[dict], field: str) -> dict[str, int]:
    counts: dict[str, int] = {}
    for entry in entries:
        key = str(entry.get(field) or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items(), key=lambda item: -item[1]))


def _limitations(split: str, has_groupings: bool) -> list[str]:
    limits: list[str] = []
    if split == "train" and has_groupings:
        limits.append("training 集按官方 lesion_id 分组，可做 lesion 级隔离；患者级隔离仍不可证明。")
    else:
        limits.append(
            "该 split 无官方 lesion 分组表，只能按单图分组；同一患者/同一病变可能跨集，"
            "不得声称患者级隔离已证明。"
        )
    limits.append("官方 validation/test 为外部留出集，不参与训练、校准与阈值选择。")
    return limits


def dataset_layout(settings: Settings, dataset_code: str, version_id: str) -> tuple[Path, Path]:
    root = ensure_dir(Path(settings.data_root) / "datasets" / dataset_code / version_id)
    images = ensure_dir(root / "images")
    return root, images


def materialize(
    db: Session,
    settings: Settings,
    *,
    root: Path,
    split: str,
    entries: list[dict],
    report: dict,
    name: str,
    license_note: str | None,
    owner_id: int,
    owner_username: str,
    dataset_code: str | None = None,
) -> dict:
    dataset_code = dataset_code or f"isic2018-{split}-{sha256_text(str(owner_id) + name)[:6]}"
    resource = (
        db.query(DatasetResource)
        .filter(DatasetResource.dataset_id == dataset_code, DatasetResource.dataset_role == "medical")
        .one_or_none()
    )
    if resource is None:
        resource = DatasetResource(
            dataset_id=dataset_code,
            dataset_role="medical",
            owner_user_id=owner_id,
            owner_username=owner_username,
            name=(name or f"ISIC 2018 Task3 {split}")[:255],
            description=(
                "皮肤镜七类单标签数据集（ISIC 2018 Task 3 / HAM10000，CC BY-NC 4.0）；"
                "仅研究用途，非临床诊断"
            ),
            source_url="https://challenge.isic-archive.com/data/",
            source_version=f"ISIC2018-Task3-{split}",
            license=license_note,
            modality="Dermoscopy",
            task_type=ISIC_TASK,
            group_level="lesion" if report["lesion_groupings_available"] else "image",
        )
        db.add(resource)
        db.flush()

    version = DatasetVersion(
        dataset_id=resource.id,
        dataset_role="medical",
        status="ready",
        state="draft",
        created_by=owner_username,
        created_by_user_id=owner_id,
        class_map_json={label.key: label.display_name for label in ISIC_LABELS},
        metadata_json={
            "direction": ISIC_DIRECTION_KEY,
            "task": ISIC_TASK,
            "label_schema": ISIC_LABEL_SCHEMA,
            "source_split": split,
            "source_report": report,
        },
    )
    db.add(version)
    db.flush()

    version_dir, images_dir = dataset_layout(settings, resource.dataset_id, version.id)
    version.root_path = str(version_dir)
    for entry in entries:
        relative = f"{entry['image']}{Path(entry['source_path']).suffix.lower()}"
        target = images_dir / relative
        ensure_dir(target.parent)
        shutil.copyfile(entry["source_path"], target)
        if sha256_file(target) != entry["sha256"]:
            raise ApiError(500, "isic_copy_hash_mismatch", "复制后文件 hash 与源不一致")
        asset = DatasetAsset(
            dataset_version_id=version.id,
            relative_path=relative,
            basename=target.name,
            size=target.stat().st_size,
            width=int(entry["width"]),
            height=int(entry["height"]),
            content_hash=entry["sha256"],
            annotations_json=[],  # 单标签分类不使用 bbox
            group_level=resource.group_level,
            source_group=entry["lesion_id"],
            transform_json={},
        )
        db.add(asset)
        db.flush()
        db.add(
            IsicSample(
                asset_id=asset.id,
                dataset_version_id=version.id,
                lesion_key=entry["lesion_id"],
                group_key=entry["lesion_id"] or f"image:{entry['image']}",
                class_key=entry["class_key"],
                class_index=int(entry["class_index"]),
                schema_version=ISIC_LABEL_SCHEMA,
                pixel_hash=entry["pixel_sha256"],
                source_hash=entry["sha256"],
            )
        )
    version.asset_count = len(entries)
    selection = {
        "schema": "isic-selection-v1",
        "direction": ISIC_DIRECTION_KEY,
        "split": split,
        "report": report,
        "entries": [
            {
                "image": entry["image"],
                "class_key": entry["class_key"],
                "lesion_id": entry["lesion_id"],
                "sha256": entry["sha256"],
                "pixel_sha256": entry["pixel_sha256"],
            }
            for entry in entries
        ],
    }
    selection_path = version_dir / "selection.json"
    atomic_write_bytes(selection_path, canonical_json(selection).encode("utf-8"))
    version.metadata_json = {
        **(version.metadata_json or {}),
        "selection_sha256": sha256_text(canonical_json(selection)),
    }
    db.flush()
    return {
        "dataset_id": resource.id,
        "dataset_code": resource.dataset_id,
        "version_id": version.id,
        "asset_count": len(entries),
        "report": report,
        "selection_sha256": version.metadata_json["selection_sha256"],
    }


def resolve_source(settings: Settings, source_id: str) -> Path:
    return resolve_source_root(
        settings,
        source_id,
        legacy_sources=getattr(settings, "cxr_sources", None),
        legacy_root=getattr(settings, "cxr_import_root", None),
        direction="ISIC",
    )


__all__ = [
    "IsicImportError",
    "dataset_layout",
    "find_ground_truth",
    "find_images_dir",
    "find_lesion_groupings",
    "load_ground_truth",
    "load_lesion_groupings",
    "materialize",
    "plan_import",
    "resolve_inside",
    "resolve_source",
]
