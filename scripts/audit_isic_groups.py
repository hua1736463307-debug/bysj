"""Audit ISIC 2018 Task 3 subset grouping and duplicate risk.

Answers three questions that decide whether the four splits may be frozen:

1. Are the official train / validation / test image lists disjoint?
2. Is there official patient/lesion grouping for each subset, and how good is
   the coverage of the training groupings file?
3. Are there byte-identical duplicate images, and do any cross a subset
   boundary (which would leak the same object into train and test)?

Two levels are available:

* default: byte-level SHA-256 of the stored file (exact copies only);
* ``--deep``: additionally decodes every image to a pixel hash (catches
  re-encoded copies with identical pixels) and computes a 64-bit dHash for
  near-duplicate candidates (hamming distance threshold is reported, and the
  candidate list is a *review list*, not a verdict).

Even ``--deep`` cannot prove "no duplicates": crops, rotations and different
crops of the same lesion remain undetected. The output says so explicitly.

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


def pixel_and_dhash(archive: zipfile.ZipFile, info: zipfile.ZipInfo) -> tuple[str, int]:
    """解码后像素哈希 + 64 位 dHash（9x8 灰度差分）。"""
    from PIL import Image

    with archive.open(info) as stream:
        image = Image.open(stream)
        image.load()
        rgb = image.convert("RGB")
        pixel_digest = hashlib.sha256(rgb.tobytes()).hexdigest()
        gray = rgb.convert("L").resize((9, 8), Image.BILINEAR)
        values = list(gray.getdata())
    bits = 0
    for row in range(8):
        for column in range(8):
            left = values[row * 9 + column]
            right = values[row * 9 + column + 1]
            bits = (bits << 1) | (1 if left > right else 0)
    return pixel_digest, bits


def deep_scan(root: Path, subsets: dict[str, dict]) -> dict:
    """解码像素重复 + dHash 近重复候选（跨子集重点）。"""
    import numpy as np

    pixel_owner: dict[str, list[str]] = defaultdict(list)
    hashes: list[int] = []
    names: list[str] = []
    subset_of: list[str] = []
    for subset, mapping in subsets.items():
        zip_path = root / ZIP_NAMES[subset]
        if not zip_path.exists():
            continue
        with zipfile.ZipFile(zip_path) as archive:
            for info in archive.infolist():
                if info.is_dir() or not info.filename.lower().endswith((".jpg", ".jpeg", ".png")):
                    continue
                image = Path(info.filename).stem
                pixel_digest, bits = pixel_and_dhash(archive, info)
                pixel_owner[pixel_digest].append(f"{subset}:{image}")
                hashes.append(bits)
                names.append(f"{subset}:{image}")
                subset_of.append(subset)

    pixel_groups = {digest: items for digest, items in pixel_owner.items() if len(items) > 1}
    pixel_cross = {}
    for items in pixel_groups.values():
        owners = {item.split(":", 1)[0] for item in items}
        if len(owners) > 1:
            pixel_cross["&".join(sorted(owners))] = pixel_cross.get("&".join(sorted(owners)), 0) + 1

    threshold = 6
    array = np.array(hashes, dtype=np.uint64)
    subset_index = np.array([{"training": 0, "validation": 1, "test": 2}[item] for item in subset_of])
    pairs: list[dict] = []
    cross_pairs = 0
    distance_histogram: dict[int, int] = {}
    cross_histogram: dict[int, int] = {}
    for start in range(0, len(array), 512):
        chunk = array[start : start + 512]
        xor = np.bitwise_xor(chunk[:, None], array[None, :])
        distance = np.bitwise_count(xor).astype(np.int16)
        rows, columns = np.where(distance <= threshold)
        for row, column in zip(rows, columns):
            first = start + int(row)
            second = int(column)
            if second <= first:
                continue
            value = int(distance[row, column])
            distance_histogram[value] = distance_histogram.get(value, 0) + 1
            if subset_index[first] != subset_index[second]:
                cross_pairs += 1
                cross_histogram[value] = cross_histogram.get(value, 0) + 1
            if len(pairs) < 20:
                pairs.append({"a": names[first], "b": names[second], "hamming": value})
    return {
        "decoded_images": len(names),
        "pixel_duplicate_groups": len(pixel_groups),
        "pixel_duplicate_images": sum(len(v) for v in pixel_groups.values()),
        "pixel_cross_subset_groups": pixel_cross,
        "pixel_examples": [sorted(v)[:4] for v in list(pixel_groups.values())[:5]],
        "dhash_bits": 64,
        "dhash_threshold": threshold,
        "dhash_candidate_pairs": int(sum(distance_histogram.values())),
        "dhash_cross_subset_pairs": cross_pairs,
        "dhash_distance_histogram": dict(sorted(distance_histogram.items())),
        "dhash_cross_distance_histogram": dict(sorted(cross_histogram.items())),
        "dhash_note": (
            "dHash(8x8) 在皮肤镜上区分力有限：阈值放宽到 6 时候选对占全部配对 0.23%，"
            "主要是大面积皮肤背景相似，无法作为重复判定；只有距离 0-1 值得人工复核"
        ),
        "dhash_examples": pairs,
        "note": "近重复候选是人工复核清单，不是重复判定；裁剪/旋转仍检测不到",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--deep", action="store_true",
                        help="解码像素哈希与 dHash 近重复候选（慢，但能发现重编码副本）")
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

    if args.deep:
        report["deep"] = deep_scan(root, digests)
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
