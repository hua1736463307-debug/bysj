"""CheXpert 本地副本审计与选择规划（离线 CLI）。

只读：不修改来源、不写数据库、不发布数据版本。真正入库走
`POST /api/v1/cxr/datasets/import`（管理员 + Idempotency-Key）。

用法：
    python scripts/prepare_chexpert.py --source-root D:/data/chexpert \
        --source-csv train.csv --output artifacts/cxr-prep --max-patients 10000 \
        --split-seed 42 --dry-run
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

BACKEND = Path(__file__).resolve().parents[1] / "backend"
if str(BACKEND) not in sys.path:
    sys.path.insert(0, str(BACKEND))

from app.domains.cxr import import_service  # noqa: E402
from app.domains.cxr.split_service import (  # noqa: E402
    CxrSplitError,
    build_cxr_manifest,
    integrity_report,
    summarize_version,
    training_gate,
)
from app.schemas.cxr import CXR_LABEL_SCHEMA, RAW_STATES, SOURCE_COLUMNS  # noqa: E402


def audit_csv(rows: list[dict]) -> dict:
    header = sorted({key for row in rows[:1] for key in row if not key.startswith("_")})
    missing = [column for column in SOURCE_COLUMNS.values() if column not in header]
    value_counts = {column: {state: 0 for state in RAW_STATES} for column in SOURCE_COLUMNS.values()}
    parse_failures: list[dict] = []
    from app.schemas.cxr import CxrLabelError, parse_raw_label

    for row in rows:
        for column in SOURCE_COLUMNS.values():
            if column not in row:
                continue
            try:
                value_counts[column][parse_raw_label(row[column])] += 1
            except CxrLabelError:
                parse_failures.append({"row": row.get("_row"), "column": column, "value": row[column]})
    views: dict[str, int] = {}
    for row in rows:
        view = str(row.get("Frontal/Lateral") or "").strip() or "(空)"
        views[view] = views.get(view, 0) + 1
    return {
        "header": header,
        "missing_label_columns": missing,
        "label_value_counts": value_counts,
        "unparseable_values": parse_failures[:50],
        "unparseable_count": len(parse_failures),
        "view_counts": views,
        "rows": len(rows),
        "label_schema": CXR_LABEL_SCHEMA,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", required=True, help="CheXpert 本地副本根目录（只读）")
    parser.add_argument("--source-csv", required=True, help="相对 source-root 的 CSV 路径")
    parser.add_argument("--output", required=True, help="报告输出目录")
    parser.add_argument("--max-patients", type=int, default=10000)
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--namespace", default="chexpert-local")
    parser.add_argument("--ratios", default="0.7,0.1,0.1,0.1")
    parser.add_argument("--dry-run", action="store_true", help="只检查与规划，不落地任何选择文件")
    parser.add_argument("--verify-pixels", action="store_true", default=True)
    args = parser.parse_args()

    source_root = Path(args.source_root).expanduser().resolve()
    if not source_root.is_dir():
        print(json.dumps({"status": "FAILED", "reason": "source_root_missing"}, ensure_ascii=False))
        return 2
    csv_path = (source_root / args.source_csv).resolve()
    try:
        csv_path.relative_to(source_root)
    except ValueError:
        print(json.dumps({"status": "FAILED", "reason": "csv_outside_source_root"}, ensure_ascii=False))
        return 2

    rows = import_service.read_source_csv(csv_path)
    audit = audit_csv(rows)
    if audit["missing_label_columns"]:
        print(json.dumps({"status": "REJECTED", "reason": "missing_label_columns", "audit": audit}, ensure_ascii=False))
        return 2

    selected, report = import_service.plan_import(
        source_root=source_root,
        rows=rows,
        max_patients=args.max_patients,
        namespace=args.namespace,
        verify_pixels=bool(args.verify_pixels),
    )
    ratios = [float(item) for item in args.ratios.split(",")]
    try:
        manifest, stats = build_cxr_manifest(
            [
                {
                    "relative_path": entry["source_path"],
                    "sha256": entry.get("sha256") or "",
                    "pixel_sha256": entry.get("pixel_sha256") or entry.get("sha256") or "",
                    "size": entry.get("size") or 0,
                    "width": entry.get("width") or 0,
                    "height": entry.get("height") or 0,
                    "patient_key": entry["patient_key"],
                    "study_key": entry["study_key"],
                    "view": entry["view"],
                    "raw_labels": entry["raw_labels"],
                }
                for entry in selected
            ],
            namespace=f"cxr:{args.namespace}",
            dataset_id=args.namespace,
            version_id="plan",
            split_seed=args.split_seed,
            ratios=ratios,
            source={"source_csv_sha256": import_service.sha256_file(csv_path), "csv": args.source_csv},
        )
    except CxrSplitError as exc:
        print(json.dumps({"status": "REJECTED", "reason": "split_invalid", "detail": str(exc), "details": exc.details}, ensure_ascii=False))
        return 2

    payload = {
        "status": "PLAN_READY" if args.dry_run else "PLAN_WRITTEN",
        "dry_run": bool(args.dry_run),
        "source_root": str(source_root),
        "source_csv": args.source_csv,
        "csv_sha256": import_service.sha256_file(csv_path),
        "audit": audit,
        "selection": report,
        "summary": summarize_version(manifest),
        "integrity": integrity_report(manifest),
        "training_gate": training_gate(manifest),
        "stats": {
            "manifest_sha256": stats["manifest_sha256"],
            "split_manifest_hash": stats["split_manifest_hash"],
            "label_schema_hash": stats["label_schema_hash"],
        },
        "limits": [
            "本工具不写数据库、不发布版本；入库需管理员调用 POST /api/v1/cxr/datasets/import",
            "选择规则不看标签或模型得分；取样在四集划分前完成",
        ],
    }
    output = Path(args.output)
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
        (output / "prepare-report.json").write_text(
            json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        with (output / "selection.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["source_path", "patient_key", "study_key", "split", "sha256", "pixel_sha256"])
            for entry in manifest["assets"]:
                writer.writerow([
                    entry["relative_path"], entry["patient_key"], entry["study_key"],
                    entry["split"], entry.get("sha256", ""), entry.get("pixel_sha256", ""),
                ])
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
