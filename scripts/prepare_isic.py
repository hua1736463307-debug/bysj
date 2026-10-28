"""ISIC 2018 Task 3 离线审计（只读，不写数据库、不发布版本）。

核对：官方标签表结构、每行单标签、各 split 类别分布、lesion 分组覆盖、
跨 split 图像重叠、图像目录是否就位、标签与图像一致性（目录存在时）。

用法：
    python scripts/prepare_isic.py --source-root D:/毕业设计/datasets/isic2018 \
        --output artifacts/isic-audit --dry-run
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.directions.isic import ISIC_LABELS  # noqa: E402
from app.domains.isic import import_service  # noqa: E402

SPLITS = {"train": "Training", "val": "Validation", "test": "Test"}


def audit_split(root: Path, split: str) -> dict:
    result: dict = {"split": split, "ground_truth_present": False, "images_present": False}
    try:
        gt_path = import_service.find_ground_truth(root, split)
        rows = import_service.load_ground_truth(gt_path)
    except Exception as exc:  # 缺文件/格式错都要如实记录
        result["error"] = f"{type(exc).__name__}: {exc}"
        return result

    result["ground_truth_present"] = True
    result["rows"] = len(rows)
    counts = {label.key: 0 for label in ISIC_LABELS}
    for row in rows:
        counts[row["class_key"]] += 1
    result["class_counts"] = counts
    result["class_share"] = {key: round(value / len(rows), 6) for key, value in counts.items()}
    result["multi_label_rows"] = 0  # 解析器已强校验，能走到这里说明全为单标签

    try:
        images_dir = import_service.find_images_dir(root, split)
        available = {p.stem for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}}
        result["images_present"] = True
        result["images_in_dir"] = len(available)
        result["images_missing_for_labels"] = len({row["image"] for row in rows} - available)
        result["unlabeled_images"] = len(available - {row["image"] for row in rows})
    except Exception as exc:
        result["images_error"] = f"{type(exc).__name__}: {exc}"

    if split == "train":
        path = import_service.find_lesion_groupings(root)
        groupings = import_service.load_lesion_groupings(path)
        result["lesion_groupings_present"] = path is not None
        result["lesion_groups"] = len({value["lesion_id"] for value in groupings.values()})
        result["grouping_coverage"] = round(len(groupings) / len(rows), 6)
        confirm: dict[str, int] = {}
        for value in groupings.values():
            key = value["diagnosis_confirm_type"] or "unknown"
            confirm[key] = confirm.get(key, 0) + 1
        result["confirm_type_counts"] = dict(sorted(confirm.items(), key=lambda item: -item[1]))
        images_per_lesion: dict[str, int] = {}
        for value in groupings.values():
            images_per_lesion[value["lesion_id"]] = images_per_lesion.get(value["lesion_id"], 0) + 1
        if images_per_lesion:
            counts_per_lesion = sorted(images_per_lesion.values(), reverse=True)
            result["images_per_lesion"] = {
                "max": counts_per_lesion[0],
                "mean": round(sum(counts_per_lesion) / len(counts_per_lesion), 3),
                "lesions_with_multiple_images": sum(1 for value in counts_per_lesion if value > 1),
            }
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--dry-run", action="store_true", help="只审计，不写文件")
    args = parser.parse_args()

    root = Path(args.source_root).expanduser().resolve()
    if not root.is_dir():
        print(json.dumps({"status": "FAILED", "reason": "source_root_missing"}, ensure_ascii=False))
        return 2

    splits = {split: audit_split(root, split) for split in SPLITS}
    images = {row["image"] for row in []}
    overlap = {}
    image_sets = {}
    for split, payload in splits.items():
        if not payload.get("ground_truth_present"):
            continue
        gt_path = import_service.find_ground_truth(root, split)
        image_sets[split] = {row["image"] for row in import_service.load_ground_truth(gt_path)}
    names = list(image_sets)
    for left in range(len(names)):
        for right in range(left + 1, len(names)):
            common = image_sets[names[left]] & image_sets[names[right]]
            overlap[f"{names[left]}&{names[right]}"] = len(common)

    total_rows = sum(payload.get("rows", 0) for payload in splits.values())
    report = {
        "status": "AUDITED",
        "source_root": str(root),
        "label_schema": "isic2018-task3-v1",
        "labels": [label.key for label in ISIC_LABELS],
        "splits": splits,
        "cross_split_image_overlap": overlap,
        "total_labeled_rows": total_rows,
        "images_downloaded": {split: payload.get("images_present", False) for split, payload in splits.items()},
        "limitations": [
            "validation/test 无官方 lesion 分组表，患者级隔离不可证明。",
            "本审计只读标签与目录，不计算任何模型指标。",
            "训练/测试图像未全部下载时，不得声称已完成真实训练。",
        ],
    }
    if not args.dry_run:
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=True)
        (output / "isic-audit.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        report["written_to"] = str(output / "isic-audit.json")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
