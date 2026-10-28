"""Audit ISIC 2018 Task 3 subset grouping and duplicate risk.

Answers three questions that decide whether the four splits may be frozen:

1. Are the official train / validation / test image lists disjoint?
2. Is there official patient/lesion grouping for each subset, and how good is
   the coverage of the training groupings file?
3. Are there byte-identical duplicate images, and do any cross a subset
   boundary (which would leak the same object into train and test)?

Only byte-level equality is checked. Re-encoded or cropped copies of the same
lesion are **not** detected by this script; that limitation is reported in the
output so it cannot be silently upgraded into a claim of duplicate-free data.

Usage:
    python scripts/audit_isic_groups.py \
        --source-root D:/毕业设计/datasets/isic2018 \
        --output artifacts/isic-groups
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

SUBSETS = ("training", "validation", "test")
ZIP_NAMES = {
    "training": "ISIC2018_Task3_Training_Input.zip",
    "validation": "ISIC2018_Task3_Validation_Input.zip",
    "test": "ISIC2018_Task3_Test_Input.zip",
}
GT_NAMES = {
    "training": "ISIC2018_Task3_Training_GroundTruth.csv",
    "validation": "ISIC2018_Task3_Validation_GroundTruth.csv",
    "test": "ISIC2018_Task3_Test_GroundTruth.csv",
}


def read_labels(root: Path) -> dict[str, dict[str, str]]:
    """image id -> class key, per subset; also validates the one-hot contract."""
    out: dict[str, dict[str, str]] = {}
    classes = ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"]
    for subset in SUBSETS:
        path = root / "extracted" / f"ISIC2018_Task3_{subset.capitalize()}_GroundTruth" / GT_NAMES[subset]
        if not path.exists():
            path = root / "extracted" / GT_NAMES[subset]
        rows: dict[str, str] = {}
        with path.open(encoding="utf-8") as stream:
            reader = csv.DictReader(stream)
            if reader.fieldnames != ["image", *classes]:
                raise SystemExit(f"{path.name} has unexpected header {reader.fieldnames}")
            for row in reader:
                positives = [key for key in classes if float(row[key]) == 1.0]
                if len(positives) != 1:
                    raise SystemExit(f"{row['image']} in {subset} has {len(positives)} positives")
                rows[row["image"]] = positives[0].lower()
        out[subset] = rows
    return out


def hash_zip(path: Path) -> tuple[dict[str, str], dict[str, int]]:
    """image id -> sha256 of the stored bytes, plus file count."""
    digests: dict[str, str] = {}
    sizes: dict[str, int] = {}
    with zipfile.ZipFile(path) as archive:
        for info in archive.infolist():
            if info.is_dir() or not info.filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue
            stem = Path(info.filename).stem
            digest = hashlib.sha256()
            with archive.open(info) as stream:
                for chunk in iter(lambda: stream.read(1 << 20), b""):
                    digest.update(chunk)
            digests[stem] = digest.hexdigest()
            sizes[stem] = info.file_size
    return digests, sizes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = args.source_root.resolve()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    labels = read_labels(root)
    groupings_path = root / "ISIC2018_Task3_Training_LesionGroupings.csv"
    lesion_of: dict[str, str] = {}
    confirm_of: dict[str, str] = {}
    with groupings_path.open(encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            lesion_of[row["image"]] = row["lesion_id"]
            confirm_of[row["image"]] = row["diagnosis_confirm_type"]

    report: dict = {
        "source_root": str(root),
        "note": "byte-level duplicate detection only; re-encoded or cropped copies are not detected",
        "subsets": {},
    }

    digests: dict[str, dict[str, str]] = {}
    for subset in SUBSETS:
        zip_path = root / ZIP_NAMES[subset]
        entry = {
            "labels": len(labels[subset]),
            "class_counts": dict(sorted(Counter(labels[subset].values()).items())),
        }
        if zip_path.exists() and zip_path.stat().st_size > 0:
            try:
                digests[subset], sizes = hash_zip(zip_path)
            except zipfile.BadZipFile:
                # 下载未完成或损坏：如实记录，不当作"无重复"。
                entry["images"] = 0
                entry["status"] = "incomplete_or_corrupt"
                entry["zip_bytes"] = zip_path.stat().st_size
                report["subsets"][subset] = entry
                continue
            entry["images"] = len(digests[subset])
            entry["zip_bytes"] = zip_path.stat().st_size
            entry["sha256"] = file_sha256(zip_path)
            missing = set(labels[subset]) - set(digests[subset])
            extra = set(digests[subset]) - set(labels[subset])
            entry["labels_without_image"] = len(missing)
            entry["images_without_label"] = len(extra)
        else:
            entry["images"] = 0
            entry["status"] = "missing"
        report["subsets"][subset] = entry

    # 1) image id overlap between subsets
    overlap = {}
    for i, first in enumerate(SUBSETS):
        for second in SUBSETS[i + 1:]:
            shared = set(labels[first]) & set(labels[second])
            overlap[f"{first}&{second}"] = len(shared)
    report["label_id_overlap"] = overlap

    # 2) grouping coverage
    train_ids = set(labels["training"])
    grouped = set(lesion_of) & train_ids
    lesion_sizes = Counter(lesion_of[image] for image in grouped)
    report["grouping"] = {
        "grouping_rows": len(lesion_of),
        "training_images": len(train_ids),
        "training_images_with_lesion": len(grouped),
        "training_coverage": round(len(grouped) / max(len(train_ids), 1), 6),
        "lesions": len(lesion_sizes),
        "images_per_lesion": dict(sorted(Counter(lesion_sizes.values()).items())),
        "largest_lesion_images": max(lesion_sizes.values()) if lesion_sizes else 0,
        "validation_grouping": "absent (official file covers training only)",
        "test_grouping": "absent (official file covers training only)",
        "confirm_type_counts": dict(sorted(Counter(confirm_of.values()).items())),
    }
    # any validation/test image id that appears in the training groupings file
    for subset in ("validation", "test"):
        leaked = set(labels[subset]) & set(lesion_of)
        report["grouping"][f"{subset}_ids_in_training_groupings"] = len(leaked)

    # 3) byte-identical duplicates
    by_hash: dict[str, list[str]] = defaultdict(list)
    owner: dict[str, str] = {}
    for subset, mapping in digests.items():
        for image, digest in mapping.items():
            by_hash[digest].append(f"{subset}:{image}")
            owner[f"{subset}:{image}"] = subset
    duplicate_groups = {h: items for h, items in by_hash.items() if len(items) > 1}
    cross_subset = {}
    for items in duplicate_groups.values():
        subsets = {item.split(":", 1)[0] for item in items}
        if len(subsets) > 1:
            key = "&".join(sorted(subsets))
            cross_subset[key] = cross_subset.get(key, 0) + 1
    report["duplicates"] = {
        "hashed_images": sum(len(v) for v in digests.values()),
        "duplicate_groups": len(duplicate_groups),
        "duplicate_images": sum(len(v) for v in duplicate_groups.values()),
        "cross_subset_groups": cross_subset,
        "largest_group": max((len(v) for v in duplicate_groups.values()), default=1),
        "examples": [sorted(v)[:4] for v in list(duplicate_groups.values())[:5]],
    }

    (output / "isic-group-audit.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps({
        "subsets": {k: {"labels": v["labels"], "images": v.get("images", 0)} for k, v in report["subsets"].items()},
        "label_id_overlap": overlap,
        "grouping": {k: report["grouping"][k] for k in
                     ("lesions", "training_coverage", "images_per_lesion", "largest_lesion_images")},
        "duplicates": {k: report["duplicates"][k] for k in
                       ("hashed_images", "duplicate_groups", "duplicate_images", "cross_subset_groups")},
    }, ensure_ascii=False, indent=2))
    return 0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())
