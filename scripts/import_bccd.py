#!/usr/bin/env python3
"""把 BCCD VOC 数据集转换为平台 manifest（PR-07）。

用法：
    python scripts/import_bccd.py --source data/raw/bccd --output data/prepared/bccd \
        --config configs/experiments/bccd-baseline.json

退出码：0 成功；2 参数/数据不合规；1 执行失败。
不会隐式下载数据；输出目录非空时拒绝覆盖。

VOC 坐标约定：BCCD 的 xmin/ymin 为 1-based 左闭，xmax/ymax 为闭区间。
这里统一转换为像素 xyxy（x2/y2 为开区间）后再归一化；不盲目对所有字段减 1。
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "backend"))

from app.domains.datasets.split_service import (  # noqa: E402
    SplitError,
    build_manifest,
    integrity_report,
)
from app.ml.geometry import GeometryError, xyxy_to_cxcywh_norm  # noqa: E402

DEFAULT_CLASS_MAP = {"0": "RBC", "1": "WBC", "2": "Platelets"}
XML_NAME_TO_ID = {"RBC": 0, "WBC": 1, "Platelets": 2}
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_voc(xml_path: Path, width: int, height: int, report: dict) -> list[dict]:
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as exc:
        report["errors"].append({"file": xml_path.name, "code": "xml_parse", "message": str(exc)})
        return []
    root = tree.getroot()
    annotations = []
    for index, obj in enumerate(root.findall("object"), start=1):
        name = (obj.findtext("name") or "").strip()
        class_id = XML_NAME_TO_ID.get(name)
        if class_id is None:
            report["errors"].append(
                {"file": xml_path.name, "line": index, "code": "unknown_class", "message": f"未知类别 {name}"}
            )
            continue
        box = obj.find("bndbox")
        if box is None:
            report["errors"].append(
                {"file": xml_path.name, "line": index, "code": "missing_bbox", "message": "缺少 bndbox"}
            )
            continue
        try:
            xmin = float(box.findtext("xmin"))
            ymin = float(box.findtext("ymin"))
            xmax = float(box.findtext("xmax"))
            ymax = float(box.findtext("ymax"))
        except (TypeError, ValueError):
            report["errors"].append(
                {"file": xml_path.name, "line": index, "code": "bbox_numeric", "message": "坐标不是数值"}
            )
            continue
        x1 = max(0.0, xmin - 1.0)
        y1 = max(0.0, ymin - 1.0)
        x2 = min(float(width), xmax)
        y2 = min(float(height), ymax)
        if x2 <= x1 or y2 <= y1:
            report["errors"].append(
                {"file": xml_path.name, "line": index, "code": "zero_area", "message": "bbox 面积小于等于 0"}
            )
            continue
        try:
            cx, cy, bw, bh = xyxy_to_cxcywh_norm([x1, y1, x2, y2], width, height)
        except GeometryError as exc:
            report["errors"].append(
                {"file": xml_path.name, "line": index, "code": "geometry", "message": str(exc)}
            )
            continue
        annotations.append(
            {
                "class_id": class_id,
                "cx": round(cx, 6),
                "cy": round(cy, 6),
                "w": round(bw, 6),
                "h": round(bh, 6),
            }
        )
    return annotations


def load_config(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(json.dumps({"ok": False, "error": f"配置不存在: {path}"}, ensure_ascii=False))
    config = json.loads(path.read_text(encoding="utf-8"))
    for key in ("split_seed", "train_ratio", "val_ratio", "test_ratio"):
        if key not in config:
            raise SystemExit(json.dumps({"ok": False, "error": f"配置缺少字段 {key}"}, ensure_ascii=False))
    return config


def main() -> int:
    parser = argparse.ArgumentParser(description="BCCD VOC → 平台 manifest")
    parser.add_argument("--source", required=True, help="已下载的 BCCD VOC 目录")
    parser.add_argument("--output", required=True, help="受控输出目录（必须为空或不存在）")
    parser.add_argument("--config", required=True, help="实验配置 JSON")
    args = parser.parse_args()

    source = Path(args.source)
    output = Path(args.output)
    if not source.is_dir():
        print(json.dumps({"ok": False, "error": "源目录不存在"}, ensure_ascii=False))
        return 2
    annotations_dir = source / "Annotations"
    images_dir = source / "JPEGImages"
    if not annotations_dir.is_dir() or not images_dir.is_dir():
        print(json.dumps({"ok": False, "error": "源目录必须包含 Annotations/ 与 JPEGImages/"}, ensure_ascii=False))
        return 2
    if output.exists() and any(output.iterdir()):
        print(json.dumps({"ok": False, "error": "输出目录非空，拒绝覆盖"}, ensure_ascii=False))
        return 2

    config = load_config(Path(args.config))
    report: dict = {
        "source": str(source),
        "source_revision": config.get("source_revision"),
        "license": config.get("source_license_ref"),
        "errors": [],
        "warnings": [],
        "duplicates": [],
    }

    from PIL import Image

    images = sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    if not images:
        print(json.dumps({"ok": False, "error": "没有找到图像"}, ensure_ascii=False))
        return 2

    output.mkdir(parents=True, exist_ok=True)
    out_images = output / "images"
    out_labels = output / "labels"
    out_images.mkdir(exist_ok=True)
    out_labels.mkdir(exist_ok=True)

    entries = []
    seen_hashes: dict[str, str] = {}
    for image in images:
        try:
            with Image.open(image) as handle:
                width, height = handle.size
                handle.verify()
        except Exception as exc:
            report["errors"].append({"file": image.name, "code": "image_decode", "message": str(exc)})
            continue
        digest = sha256_file(image)
        if digest in seen_hashes:
            report["duplicates"].append({"file": image.name, "duplicate_of": seen_hashes[digest], "sha256": digest})
            continue
        seen_hashes[digest] = image.name

        xml_path = annotations_dir / f"{image.stem}.xml"
        annotations = []
        if xml_path.exists():
            annotations = parse_voc(xml_path, width, height, report)
        else:
            report["warnings"].append({"file": image.name, "code": "missing_xml", "message": "没有对应 XML"})

        shutil.copy2(image, out_images / image.name)
        label_target = out_labels / f"{image.stem}.txt"
        label_target.write_text(
            "\n".join(
                f"{ann['class_id']} {ann['cx']:.6f} {ann['cy']:.6f} {ann['w']:.6f} {ann['h']:.6f}"
                for ann in annotations
            )
            + ("\n" if annotations else ""),
            encoding="utf-8",
        )
        entries.append(
            {
                "relative_path": image.name,
                "sha256": digest,
                "size": image.stat().st_size,
                "width": width,
                "height": height,
                "source_group": f"img:{digest[:16]}",
                "root_asset": image.name,
                "annotations": annotations,
            }
        )

    if report["errors"]:
        (output / "validation-report.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(json.dumps({"ok": False, "error": "转换存在错误", "report": "validation-report.json"}, ensure_ascii=False))
        return 2

    try:
        manifest, stats = build_manifest(
            entries,
            class_map=DEFAULT_CLASS_MAP,
            seed=int(config["split_seed"]),
            ratios=(config["train_ratio"], config["val_ratio"], config["test_ratio"]),
            dataset_id="bccd",
            version_id="bccd-v1",
            group_level="original_image",
        )
    except SplitError as exc:
        print(json.dumps({"ok": False, "error": str(exc), "details": exc.details}, ensure_ascii=False))
        return 2

    manifest["manifest_sha256"] = stats["manifest_sha256"]
    manifest["split_manifest_hash"] = stats["split_manifest_hash"]
    (output / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8"
    )

    dataset_yaml = [
        f"path: {output.resolve().as_posix()}",
        "train: images/train.txt",
        "val: images/val.txt",
        "test: images/test.txt",
        f"nc: {len(DEFAULT_CLASS_MAP)}",
        "names:",
    ]
    for key in sorted(DEFAULT_CLASS_MAP, key=int):
        dataset_yaml.append(f"  {key}: {DEFAULT_CLASS_MAP[key]}")
    (output / "dataset.yaml").write_text("\n".join(dataset_yaml) + "\n", encoding="utf-8")
    for split in ("train", "val", "test"):
        names = [item["relative_path"] for item in manifest["assets"] if item["split"] == split]
        (out_images / f"{split}.txt").write_text("\n".join(names) + ("\n" if names else ""), encoding="utf-8")

    integrity = integrity_report(manifest)
    report.update(
        {
            "counts": {
                "images": len(entries),
                "annotations": sum(len(item["annotations"]) for item in entries),
                "classes": dict(Counter(ann["class_id"] for item in entries for ann in item["annotations"])),
                **stats["split_counts"],
            },
            "integrity": integrity,
            "manifest_sha256": stats["manifest_sha256"],
            "split_manifest_hash": stats["split_manifest_hash"],
            "limitations": [
                "BCCD 没有患者/载玻片元数据，来源组退化为 original_image 级",
                "近重复图像只能人工复核，SHA 不能代替",
            ],
        }
    )
    (output / "validation-report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    license_src = source / "LICENSE"
    if license_src.exists():
        (output / "LICENSE.source").write_bytes(license_src.read_bytes())

    if not integrity["ok"]:
        print(json.dumps({"ok": False, "error": "split 完整性检查失败", "integrity": integrity}, ensure_ascii=False))
        return 2

    print(
        json.dumps(
            {
                "ok": True,
                "output": str(output),
                "counts": report["counts"],
                "manifest_sha256": stats["manifest_sha256"],
                "split_manifest_hash": stats["split_manifest_hash"],
            },
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as exc:  # 执行失败
        print(json.dumps({"ok": False, "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        raise SystemExit(1)
