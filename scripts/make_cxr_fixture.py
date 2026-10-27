"""生成**合成**胸片 fixture，只用于工程契约与流水线验证。

这不是 CheXpert 数据，也不产生任何可引用的研究指标。输出目录带
`SYNTHETIC_DATASET.txt` 标记，导入服务会在数据集元数据里保留 synthetic 标记。

用法：
    python scripts/make_cxr_fixture.py --output .runtime/cxr-fixture --patients 12
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

LABEL_COLUMNS = ("Atelectasis", "Cardiomegaly", "Consolidation", "Edema", "Pleural Effusion")
MARKER = """SYNTHETIC_DATASET
本目录由 scripts/make_cxr_fixture.py 生成，仅用于工程契约与流水线验证。
不是 CheXpert 数据，不得用于训练真实模型，不得引用任何指标。
"""


def _render(seed: int, size: int) -> Image.Image:
    rng = np.random.default_rng(seed)
    base = np.zeros((size, size), dtype=np.float32)
    yy, xx = np.mgrid[0:size, 0:size]
    base += 40 * np.exp(-(((xx - size * 0.5) ** 2) / (2 * (size * 0.28) ** 2)
                          + ((yy - size * 0.55) ** 2) / (2 * (size * 0.42) ** 2)))
    base += rng.normal(0, 12, size=(size, size))
    base = np.clip(base, 0, 255).astype(np.uint8)
    image = Image.fromarray(base, mode="L").convert("RGB")
    draw = ImageDraw.Draw(image)
    draw.ellipse([size * 0.22, size * 0.18, size * 0.78, size * 0.92], outline=(210, 210, 210), width=2)
    draw.line([size * 0.5, size * 0.15, size * 0.5, size * 0.9], fill=(190, 190, 190), width=2)
    return image


def build(output: Path, patients: int, studies: int, size: int, seed: int, duplicate_pairs: int) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    (output / "SYNTHETIC_DATASET.txt").write_text(MARKER, encoding="utf-8")
    rng = random.Random(seed)
    rows: list[dict] = []
    # 每个标签都保证出现正例与负例，避免冻结门禁因缺类失败。
    label_cycle = []
    for index in range(patients):
        pattern = [0] * len(LABEL_COLUMNS)
        pattern[index % len(LABEL_COLUMNS)] = 1
        if index % 3 == 0:
            pattern[(index + 2) % len(LABEL_COLUMNS)] = -1  # uncertain
        if index % 7 == 0:
            pattern[(index + 1) % len(LABEL_COLUMNS)] = ""  # unmentioned
        label_cycle.append(pattern)

    duplicate_sources: list[str] = []
    duplicates_created = 0
    for patient_index in range(patients):
        patient = f"patient{patient_index + 1:05d}"
        for study_index in range(studies):
            study = f"study{study_index + 1}"
            relative = f"{patient}/{study}/view1_frontal.jpg"
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            if (
                duplicate_sources
                and study_index == 0
                and patient_index % 3 == 0
                and duplicates_created < duplicate_pairs
            ):
                # 跨患者同像素重复：用于验证重复候选合并，不参与真实训练语义。
                source_path = output / duplicate_sources.pop(0)
                # 直接复制字节，保证解码像素完全一致（重新编码会改变像素）。
                shutil.copyfile(source_path, target)
                duplicates_created += 1
            else:
                image_seed = seed * 1000 + patient_index * 10 + study_index
                _render(image_seed, size).save(target, format="JPEG", quality=92)
                if study_index == 0:
                    duplicate_sources.append(relative)
            labels = label_cycle[patient_index]
            rows.append(
                {
                    "Path": relative,
                    "Frontal/Lateral": "Frontal",
                    **{column: labels[position] for position, column in enumerate(LABEL_COLUMNS)},
                }
            )
            if study_index == 0:
                lateral_relative = f"{patient}/{study}/view2_lateral.jpg"
                Image.new("RGB", (size // 2, size // 2), (30, 30, 30)).save(output / lateral_relative, format="JPEG")
                rows.append(
                    {
                        "Path": lateral_relative,
                        "Frontal/Lateral": "Lateral",
                        **{column: "" for column in LABEL_COLUMNS},
                    }
                )

    csv_path = output / "train.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=("Path", "Frontal/Lateral", *LABEL_COLUMNS))
        writer.writeheader()
        for row in rows:
            writer.writerow(row)

    summary = {
        "synthetic": True,
        "patients": patients,
        "studies_per_patient": studies,
        "csv": csv_path.name,
        "rows": len(rows),
        "frontal_rows": sum(1 for row in rows if row["Frontal/Lateral"] == "Frontal"),
        "lateral_rows": sum(1 for row in rows if row["Frontal/Lateral"] == "Lateral"),
        "duplicate_pairs": duplicate_pairs,
        "note": "合成 fixture；不得用于真实训练或研究结论",
    }
    (output / "fixture.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True)
    parser.add_argument("--patients", type=int, default=12)
    parser.add_argument("--studies", type=int, default=1)
    parser.add_argument("--size", type=int, default=320)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--duplicate-pairs", type=int, default=2)
    args = parser.parse_args()
    summary = build(Path(args.output), args.patients, args.studies, args.size, args.seed, args.duplicate_pairs)
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
