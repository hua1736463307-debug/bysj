"""数据集导入：staging → 校验 → 原子发布。

安全边界（ACCEPTANCE A04）：
- 上传与解压二次计数，不信任 header / ZIP 声明；
- ZIP 拒绝绝对路径、盘符、``..``、符号链接、超量、超高压缩比；
- 校验失败只清理本次 staging，不触碰已有数据。
"""
from __future__ import annotations

import json
import os
import shutil
import zipfile
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

from sqlalchemy.orm import Session

from app.core.config import Settings
from app.core.errors import ApiError
from app.core.storage import (
    ensure_dir,
    normalize_relative,
    resolve_under,
    probe_image,
    remove_tree,
    sha256_file,
    walk_files,
)
from app.db.base import new_uuid, utcnow
from app.db.models import DatasetAsset, DatasetResource, DatasetVersion

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
LABEL_SUFFIX = ".txt"


@dataclass
class ImportReport:
    errors: list[dict] = field(default_factory=list)
    warnings: list[dict] = field(default_factory=list)
    duplicates: list[dict] = field(default_factory=list)
    counts: dict = field(default_factory=dict)

    def add_error(self, code: str, message: str, file: str | None = None, line: int | None = None) -> None:
        item = {"code": code, "message": message}
        if file:
            item["file"] = file
        if line is not None:
            item["line"] = line
        self.errors.append(item)

    def to_dict(self) -> dict:
        return {
            "errors": self.errors,
            "warnings": self.warnings,
            "duplicates": self.duplicates,
            "counts": self.counts,
        }


class DatasetImportError(ApiError):
    def __init__(self, report: ImportReport) -> None:
        super().__init__(
            422,
            "dataset_invalid",
            "数据集校验失败",
            {"report": report.to_dict()},
        )
        self.report = report


def staging_root(settings: Settings, token: str) -> Path:
    return ensure_dir(settings.data_root / "staging" / normalize_relative(token))


def save_upload(settings: Settings, staging: Path, filename: str, stream, declared_size: int | None = None) -> Path:
    relative = normalize_relative(filename)
    target = staging / "uploads" / relative
    ensure_dir(target.parent)
    written = 0
    with target.open("wb") as handle:
        while True:
            chunk = stream.read(1024 * 1024)
            if not chunk:
                break
            written += len(chunk)
            if written > settings.max_image_bytes and target.suffix.lower() != ".zip":
                target.unlink(missing_ok=True)
                raise ApiError(413, "file_too_large", f"单文件超过上限: {filename}")
            if written > settings.max_upload_bytes:
                target.unlink(missing_ok=True)
                raise ApiError(413, "upload_too_large", "本次上传总量超过上限")
            handle.write(chunk)
    if declared_size is not None and written != declared_size:
        pass  # 以实际写入字节为准，不信任声明大小
    return target


def safe_extract_zip(archive: Path, destination: Path, settings: Settings) -> list[str]:
    extracted: list[str] = []
    ensure_dir(destination)
    with zipfile.ZipFile(archive) as bundle:
        infos = bundle.infolist()
        if len(infos) > settings.max_zip_files:
            raise ApiError(413, "zip_too_many_files", "ZIP 文件数量超过上限")
        total_declared = sum(max(0, info.file_size) for info in infos)
        compressed = max(1, sum(max(0, info.compress_size) for info in infos))
        if total_declared > settings.max_extract_bytes:
            raise ApiError(413, "zip_too_large", "ZIP 解压总量超过上限")
        if total_declared / compressed > settings.max_compression_ratio:
            raise ApiError(413, "zip_ratio", "ZIP 压缩比异常，疑似压缩炸弹")

        # Validate the whole central directory before creating any output.
        # Never overwrite duplicate members or follow pre-existing symlinks.
        targets = set()
        validated = []
        for info in infos:
            mode = (info.external_attr >> 16) & 0o170000
            if mode == 0o120000:
                raise ApiError(422, "zip_symlink", f"ZIP 含符号链接: {info.filename}")
            relative = normalize_relative(info.filename)
            target = resolve_under(destination, relative.as_posix())
            key = os.path.normcase(str(target))
            if key in targets or (target.exists() and not (info.is_dir() and target.is_dir())):
                raise ApiError(422, "zip_duplicate", f"ZIP 目标路径重复或已存在: {info.filename}")
            targets.add(key)
            if info.file_size > settings.max_image_bytes:
                raise ApiError(413, "zip_file_too_large", "ZIP 单文件超过图像大小上限")
            if info.file_size / max(1, info.compress_size) > settings.max_compression_ratio:
                raise ApiError(413, "zip_ratio", "ZIP 单文件压缩比异常")
            validated.append((info, relative, target))
        if shutil.disk_usage(destination).free < total_declared:
            raise ApiError(413, "storage_full", "磁盘空间不足以解压")
        written_total = 0
        for info, relative, target in validated:
            if info.is_dir():
                ensure_dir(target)
                continue
            ensure_dir(target.parent)
            with bundle.open(info) as source, target.open("xb") as sink:
                while True:
                    chunk = source.read(1024 * 1024)
                    if not chunk:
                        break
                    written_total += len(chunk)
                    if written_total > settings.max_extract_bytes:
                        raise ApiError(413, "zip_too_large", "ZIP 实际解压总量超过上限")
                    sink.write(chunk)
            extracted.append(relative.as_posix())
    return extracted


def parse_yolo_label(text: str, report: ImportReport, file_label: str, class_map: dict[str, str]) -> list[dict]:
    annotations: list[dict] = []
    for line_no, raw in enumerate(text.splitlines(), start=1):
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) >= 7 and len(parts) % 2 == 1:
            from app.ml.geometry import normalize_polygon
            try:
                class_id = int(parts[0])
                if str(class_id) not in class_map:
                    raise ValueError("多边形类别不在类别映射中")
                coordinates = [float(value) for value in parts[1:]]
                polygon, bounds = normalize_polygon(list(zip(coordinates[::2], coordinates[1::2])))
                annotations.append({"class_id": class_id, **bounds, "polygon": polygon})
            except (ValueError, TypeError) as exc:
                report.add_error("label_polygon", str(exc), file_label, line_no)
            continue
        if len(parts) != 5:
            report.add_error("label_format", "YOLO 标签必须为 5 列: class cx cy w h", file_label, line_no)
            continue
        try:
            class_id = int(parts[0])
            cx, cy, w, h = (float(value) for value in parts[1:])
        except ValueError:
            report.add_error("label_format", "标签含非数值字段", file_label, line_no)
            continue
        if class_id < 0:
            report.add_error("label_class", "class_id 不能为负", file_label, line_no)
            continue
        if str(class_id) not in class_map:
            report.add_error("label_class", f"未知类别 {class_id}", file_label, line_no)
            continue
        if any(value != value for value in (cx, cy, w, h)):
            report.add_error("label_nan", "标签含 NaN", file_label, line_no)
            continue
        if w <= 0 or h <= 0:
            report.add_error("label_zero_area", "bbox 宽高必须大于 0", file_label, line_no)
            continue
        if not (-1e-6 <= cx <= 1 + 1e-6 and -1e-6 <= cy <= 1 + 1e-6):
            report.add_error("label_out_of_range", "bbox 中心点越界", file_label, line_no)
            continue
        if not (0 < w <= 1 + 1e-6 and 0 < h <= 1 + 1e-6):
            report.add_error("label_out_of_range", "bbox 宽高越界", file_label, line_no)
            continue
        if cx - w / 2 < -1e-6 or cy - h / 2 < -1e-6 or cx + w / 2 > 1 + 1e-6 or cy + h / 2 > 1 + 1e-6:
            report.add_error("label_out_of_range", "bbox 超出图像范围", file_label, line_no)
            continue
        annotations.append({"class_id": class_id, "cx": cx, "cy": cy, "w": w, "h": h})
    return annotations


def _load_class_map(staging: Path, payload: dict) -> dict[str, str]:
    provided = payload.get("class_map")
    if provided:
        return {str(k): str(v) for k, v in provided.items()}
    for directory in ("uploads", "extracted"):
        candidate = staging / directory / "class_map.json"
        if candidate.exists():
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
                if isinstance(data, dict):
                    return {str(k): str(v) for k, v in data.items()}
            except (json.JSONDecodeError, OSError):
                pass
    return {}


def _load_annotation_json(staging: Path) -> dict[str, list[dict]]:
    candidate = staging / "uploads" / "annotations.json"
    if not candidate.exists():
        return {}
    try:
        data = json.loads(candidate.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        raise ApiError(422, "annotation_json_invalid", f"annotations.json 无法解析: {exc}") from exc
    return data if isinstance(data, dict) else {}


def _iter_source_files(staging: Path) -> list[tuple[str, Path]]:
    roots = [staging / "uploads", staging / "extracted"]
    collected: dict[str, Path] = {}
    for root in roots:
        if not root.exists():
            continue
        for path in walk_files(root):
            if path.suffix.lower() == ".zip":
                continue
            if "__MACOSX" in path.parts:
                continue
            relative = path.relative_to(root).as_posix()
            collected.setdefault(relative, path)
    return sorted(collected.items())


def import_dataset(
    db: Session,
    settings: Settings,
    *,
    staging_token: str,
    payload: dict,
    owner_id: int,
    owner_username: str,
    progress=None,
) -> dict:
    staging = staging_root(settings, staging_token)
    if not staging.exists():
        raise ApiError(404, "staging_missing", "上传暂存目录不存在或已清理")

    report = ImportReport()
    for archive in sorted((staging / "uploads").rglob("*.zip")):
        try:
            safe_extract_zip(archive, staging / "extracted", settings)
        except ApiError as exc:
            report.add_error(exc.code, exc.message, archive.name)
            db_raise(report)

    files = _iter_source_files(staging)
    if not files:
        report.add_error("no_files", "没有找到可导入的文件")
        db_raise(report)

    class_map = _load_class_map(staging, payload)
    annotation_map = _load_annotation_json(staging)
    group_level = str(payload.get("group_level") or "original_image")
    groups = payload.get('source_groups')
    if group_level not in {'original_image', 'image_cluster'}:
        report.add_error("group_level_unsupported", "来源组只支持 original_image 或 image_cluster")
        db_raise(report)
    if group_level == 'original_image' and groups is not None:
        report.add_error('source_groups_unexpected', '显式分组必须使用 image_cluster，不能静默忽略分组')
        db_raise(report)
    if group_level == 'image_cluster':
        image_paths = {relative for relative,path in files if path.suffix.lower() in IMAGE_SUFFIXES}
        if not isinstance(groups, dict) or set(groups) != image_paths:
            report.add_error('source_groups_coverage', '分组映射必须完整且精确覆盖本次图像相对路径')
            db_raise(report)
        for relative,group in groups.items():
            if normalize_relative(relative).as_posix() != relative or not isinstance(group,str) or not group.strip() or len(group)>128:
                report.add_error('source_groups_invalid', '组名必须是非空且至多 128 字符，路径必须规范化', relative)
        if report.errors:db_raise(report)

    label_index: dict[str, Path] = {}
    for relative, path in files:
        if path.suffix.lower() == LABEL_SUFFIX and path.parent.name != "":
            label_index[relative] = path

    prepared: list[dict] = []
    seen_hashes: dict[str, str] = {}
    for relative, path in files:
        suffix = path.suffix.lower()
        if suffix == LABEL_SUFFIX or suffix == ".json" or suffix == ".yaml" or suffix == ".yml":
            continue
        if suffix not in IMAGE_SUFFIXES:
            report.warnings.append({"code": "skipped_file", "file": relative, "message": "非图像文件已跳过"})
            continue
        if path.stat().st_size > settings.max_image_bytes:
            report.add_error("file_too_large", "图像超过单文件上限", relative)
            continue
        try:
            width, height, _fmt = probe_image(path)
        except Exception:
            report.add_error("image_decode", "图像无法解码", relative)
            continue
        if width * height > settings.max_image_pixels:
            report.add_error("image_pixels", "图像像素数超过上限", relative)
            continue
        digest = sha256_file(path)
        if digest in seen_hashes:
            if groups is not None and groups[relative] != groups[seen_hashes[digest]]:
                report.add_error('source_groups_content_conflict', '完全相同内容被分配到不同来源组', relative)
            report.duplicates.append({"file": relative, "duplicate_of": seen_hashes[digest], "sha256": digest})
            continue
        seen_hashes[digest] = relative

        annotations = []
        raw_annotations = annotation_map.get(relative) or []
        if not isinstance(raw_annotations, list):
            report.add_error("annotation_json_invalid", "图像标注必须是数组", relative)
        else:
            from app.domains.annotation.service import _validate_box
            try:
                annotations = [{**annotation, **_validate_box(annotation, class_map)} for annotation in raw_annotations]
            except (ApiError, TypeError, AttributeError) as exc:
                report.add_error("annotation_json_invalid", str(exc), relative)
        label_rel = None
        candidate = label_index.get(str(Path(relative).with_suffix(LABEL_SUFFIX).as_posix()))
        if candidate is None and Path(relative).parts[0] == "images":
            label_path = Path("labels", *Path(relative).parts[1:]).with_suffix(LABEL_SUFFIX).as_posix()
            candidate = label_index.get(label_path)
        if candidate is None:
            candidate = label_index.get(f"labels/{Path(relative).stem}{LABEL_SUFFIX}")
        if candidate is not None and not annotations:
            label_rel = candidate.relative_to(staging / "extracted" if "extracted" in candidate.parts else staging / "uploads").as_posix()
            text = candidate.read_text(encoding="utf-8", errors="replace")
            annotations = parse_yolo_label(text, report, label_rel or candidate.name, class_map)
        if not annotations and not payload.get("allow_empty_labels", True):
            report.add_error("label_empty", "空标签且配置不允许", relative)

        prepared.append(
            {
                "relative_path": relative,
                "source_path": path,
                "sha256": digest,
                "size": path.stat().st_size,
                "width": width,
                "height": height,
                "annotations": annotations,
                "source_group": groups[relative] if groups is not None else f"img:{digest[:16]}",
                "label_relative": label_rel,
            }
        )

    if report.errors:
        db_raise(report)
    if not prepared:
        report.add_error("no_valid_images", "没有通过校验的图像")
        db_raise(report)

    if not class_map:
        inferred = sorted({int(ann["class_id"]) for item in prepared for ann in item["annotations"]})
        class_map = {str(class_id): f"class_{class_id}" for class_id in inferred} or {"0": "object"}

    dataset_id = new_uuid()
    version_id = new_uuid()
    version_dir = ensure_dir(settings.data_root / "datasets" / dataset_id / version_id)
    images_dir = ensure_dir(version_dir / "images")
    labels_dir = ensure_dir(version_dir / "labels")

    resource = DatasetResource(
        id=dataset_id,
        dataset_id=dataset_id,
        dataset_role="medical",
        owner_user_id=owner_id,
        owner_username=owner_username,
        visibility_mode="private",
        name=str(payload.get("name") or "导入数据集")[:255],
        description=payload.get("description"),
        source_url=payload.get("source_url"),
        source_version=payload.get("source_version"),
        license=payload.get("license"),
        modality=payload.get("modality"),
        task_type="detect",
        group_level=group_level,
    )
    version = DatasetVersion(
        id=version_id,
        dataset_id=dataset_id,
        dataset_role="medical",
        status="draft",
        state="draft",
        root_path=str(version_dir),
        manifest_path="",
        class_map_json=class_map,
        split_config_json={},
        metadata_json={"source": payload, "group_level": group_level},
        content_hash="",
        created_by=owner_username,
        created_by_user_id=owner_id,
        asset_count=len(prepared),
    )
    db.add(resource)
    db.add(version)
    db.commit()

    try:
        for index, item in enumerate(prepared, start=1):
            target = images_dir / item["relative_path"]
            ensure_dir(target.parent)
            shutil.copy2(item["source_path"], target)
            label_value = None
            if item["annotations"]:
                label_rel = Path(item["relative_path"]).with_suffix(".txt").as_posix()
                label_target = labels_dir / label_rel
                ensure_dir(label_target.parent)
                lines = []
                for ann in item["annotations"]:
                    lines.append(
                        f"{ann['class_id']} {ann['cx']:.6f} {ann['cy']:.6f} {ann['w']:.6f} {ann['h']:.6f}"
                    )
                label_target.write_text("\n".join(lines) + "\n", encoding="utf-8")
                label_value = f"labels/{label_rel}"
            db.add(
                DatasetAsset(
                    dataset_version_id=version_id,
                    relative_path=item["relative_path"],
                    basename=Path(item["relative_path"]).name,
                    size=item["size"],
                    width=item["width"],
                    height=item["height"],
                    label_path=label_value,
                    source_group=item["source_group"],
                    group_level=group_level,
                    content_hash=item["sha256"],
                    annotations_json=item["annotations"],
                )
            )
            if progress:
                progress(index / len(prepared) * 0.9, "copying", f"写入 {index}/{len(prepared)}")
        db.commit()
    except Exception:
        db.rollback()
        remove_tree(version_dir, allowed_root=settings.data_root / "datasets" / dataset_id)
        raise

    report.counts = {
        "files_scanned": len(files),
        "images_imported": len(prepared),
        "duplicates_skipped": len(report.duplicates),
        "annotations": sum(len(item["annotations"]) for item in prepared),
        "classes": dict(Counter(ann["class_id"] for item in prepared for ann in item["annotations"])),
    }

    # 成功后只清理本次 staging
    remove_tree(staging, allowed_root=settings.data_root / "staging")
    return {
        "dataset_id": dataset_id,
        "version_id": version_id,
        "version_state": "draft",
        "report": report.to_dict(),
    }


def db_raise(report: ImportReport) -> None:
    raise DatasetImportError(report)
