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


def _load_dataset_package(files: list[tuple[str, Path]], class_map: dict, payload: dict) -> dict | None:
    """Validate platform exports before preserving their grouping and split layout."""
    indexed = dict(files)
    if "manifest.json" not in indexed:
        return None

    def invalid(message):
        raise ApiError(422, "dataset_package_invalid", message)

    try:
        manifest = json.loads(indexed["manifest.json"].read_text(encoding="utf-8"))
        if "source_groups.json" not in indexed:
            if isinstance(manifest, dict) and (manifest.get("package_type") == "bysj-detection-dataset" or
                    {"assets", "dataset_id", "version_id", "class_map"}.issubset(manifest)):
                invalid("平台导出包缺少 source_groups.json")
            return None
        groups = json.loads(indexed["source_groups.json"].read_text(encoding="utf-8"))
    except (ValueError, OSError) as exc:
        invalid(f"导出包清单无法解析: {exc}")
    if not isinstance(manifest, dict) or manifest.get("schema_version") != 1 or not isinstance(manifest.get("assets"), list):
        invalid("导出包 manifest 格式无效")
    if manifest.get("class_map") != class_map or not isinstance(groups, dict):
        invalid("导出包类别映射或来源分组无效")
    if payload.get("source_groups") is not None and payload["source_groups"] != groups:
        invalid("表单来源分组与导出包不一致")
    level = manifest.get("group_level", "image_cluster")
    if level not in {"original_image", "image_cluster"}:
        invalid("导出包分组级别无效")
    from app.domains.annotation.service import _validate_box
    from app.domains.datasets.split_service import integrity_report

    def geometry(annotations, polygons):
        normalized = []
        for annotation in annotations:
            box = _validate_box(annotation, class_map)
            if polygons:
                x, y, w, h = (box[key] for key in ("cx", "cy", "w", "h"))
                points = box.get("polygon") or [[x-w/2, y-h/2], [x+w/2, y-h/2], [x+w/2, y+h/2], [x-w/2, y+h/2]]
                normalized.append({"class_id": box["class_id"], "polygon": [[round(value, 6) for value in point] for point in points]})
            else:
                normalized.append({key: box[key] for key in ("class_id", "cx", "cy", "w", "h")})
        return sorted(normalized, key=lambda box: json.dumps(box, sort_keys=True))

    splits, labels, entries = {}, set(), []
    for item in manifest["assets"]:
        if not isinstance(item, dict) or item.get("split") not in {"train", "val", "test"}:
            invalid("导出包包含无效划分")
        relative = item.get("relative_path")
        if not isinstance(relative, str) or normalize_relative(relative).as_posix() != relative:
            invalid("导出包图像路径无效")
        image_path = f"images/{item['split']}/{relative}"
        label_path = "labels/" + item["split"] + "/" + Path(relative).with_suffix(".txt").as_posix()
        if image_path in splits or label_path in labels or image_path not in indexed or label_path not in indexed:
            invalid("导出包图像或标签缺失、重名")
        if sha256_file(indexed[image_path]) != item.get("sha256"):
            invalid("导出包图像摘要不匹配")
        group = groups.get(image_path)
        if not isinstance(group, str) or not group.strip() or len(group) > 255:
            invalid("导出包来源组缺失或无效")
        report = ImportReport()
        annotations = parse_yolo_label(indexed[label_path].read_text(encoding="utf-8"), report, label_path, class_map)
        try:
            polygons = any(annotation.get("polygon") for annotation in annotations)
            if report.errors or geometry(annotations, polygons) != geometry(item.get("annotations", []), polygons):
                invalid("导出包标签与清单不一致")
        except (TypeError, ValueError, AttributeError) as exc:
            invalid(f"导出包标注无效: {exc}")
        splits[image_path] = item["split"]
        labels.add(label_path)
        entries.append({"sha256": item["sha256"], "source_group": group, "root_asset": group, "split": item["split"]})
    images = {relative for relative, path in files if path.suffix.lower() in IMAGE_SUFFIXES}
    if set(splits) != images or set(groups) != images or not images:
        invalid("导出包清单和来源组必须精确覆盖全部图像")
    if {relative for relative, path in files if relative.startswith("labels/") and path.suffix == ".txt"} != labels:
        invalid("导出包含未登记标签")
    if not integrity_report({"assets": entries})["ok"]:
        invalid("导出包来源组或相同图像跨越不同划分")
    config = manifest.get("split_config") or {"seed": 42, "ratios": {"train": .8, "val": .1, "test": .1}}
    try:
        import math
        seed = int(config["seed"])
        ratios = {key: float(config["ratios"][key]) for key in ("train", "val", "test")}
        if seed < 0 or not all(math.isfinite(value) and value > 0 for value in ratios.values()):
            invalid("导出包划分配置无效")
    except (KeyError, TypeError, ValueError) as exc:
        invalid(f"导出包划分配置无效: {exc}")
    return {"groups": groups, "splits": splits, "group_level": level,
            "config": {"seed": seed, "ratios": ratios, "group_level": level},
            # Preserve restrictive provenance only. Uploaded claims cannot grant research eligibility.
            "synthetic": manifest.get("synthetic") is True}


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
    package = _load_dataset_package(files, class_map, payload)
    if package:
        annotation_map = {}  # Native sidecars were validated against the package manifest.
    group_level = package["group_level"] if package else str(payload.get("group_level") or "original_image")
    groups = package["groups"] if package else payload.get('source_groups')
    if group_level not in {'original_image', 'image_cluster'}:
        report.add_error("group_level_unsupported", "来源组只支持 original_image 或 image_cluster")
        db_raise(report)
    if group_level == 'original_image' and groups is not None and package is None:
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
        if digest in seen_hashes and package is None:
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
                "split": package["splits"][relative] if package else None,
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
        split_config_json=package["config"] if package else {},
        metadata_json={"source": {**payload, "group_level": group_level}, "group_level": group_level, **({
            "preserve_parent_splits": True, "synthetic": package["synthetic"],
            "research_valid": False, "imported_package": True,
        } if package else {})},
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
                    split=item["split"],
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
