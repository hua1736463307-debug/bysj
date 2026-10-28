"""Insert explicitly synthetic CXR/ISIC records into an isolated restore drill.

This exists only to prove that a restored f004 database *and* its file volumes
are readable end to end by the current code. Everything written here carries
``synthetic=True`` / ``research_valid=False``; no number derived from these rows
may be reported as a research result.

The script refuses to run unless every guard passes, because pointing it at a
real database would pollute it with fake clinical records:

1. ``BYSJ_DRILL_PROJECT`` must be set and equal ``--expect-project``;
2. the database must already be at revision ``4a6b1222f004``;
3. no dataset named ``SYNTHETIC-DRILL`` may exist yet (idempotency);
4. ``--confirm`` must literally be ``SYNTHETIC-DRILL``.

Usage (inside the drill stack, never against the running project):

    docker run --rm --network <drill>_default \
      -v <drill>-data:/data -v <drill>-models:/models -v <drill>-artifacts:/artifacts \
      -e DATABASE_URL=... -e DATA_ROOT=/data -e MODEL_ROOT=/models \
      -e ARTIFACT_ROOT=/artifacts -e BYSJ_DRILL_PROJECT=<drill> \
      -v <repo>:/repo:ro -e PYTHONPATH=/repo/backend:/app/backend \
      <api-image> python /repo/scripts/seed_drill_synthetic.py \
      --expect-project <drill> --confirm SYNTHETIC-DRILL --report /tmp/seed.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from app.core.config import get_settings
from app.db.models import (
    CxrAnalysis,
    CxrAnalysisBatch,
    CxrReport,
    CxrReview,
    CxrSample,
    DatasetAsset,
    DatasetResource,
    DatasetVersion,
    IsicSample,
    ModelRegistry,
    User,
)
from app.db.session import get_session_factory
from app.directions.isic import ISIC_LABEL_SCHEMA, ISIC_LABELS, ISIC_TASK
from app.schemas.cxr import (
    CXR_GROUP_LEVEL,
    CXR_LABEL_SCHEMA,
    CXR_TASK,
    LABEL_DISPLAY,
    LABEL_KEYS,
    RAW_NEGATIVE,
    RAW_POSITIVE,
    RAW_UNCERTAIN,
    RAW_UNMENTIONED,
)

DRILL_MARK = "SYNTHETIC-DRILL"
EXPECTED_REVISION = "4a6b1222f004"
SEED = 20261027


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_json(payload: dict) -> str:
    text = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def draw(path: Path, index: int, label: str) -> None:
    """Deterministic small PNG; the marker is in the filename, not the pixels."""
    path.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED + index)
    canvas = (rng.random((64, 64, 3)) * 90 + 40).astype("uint8")
    canvas[8 + index * 3 : 20 + index * 3, 10:54] = 200
    Image.fromarray(canvas).save(path, format="PNG")
    assert label  # keep the argument meaningful for callers


def guard(expect_project: str, db) -> None:
    if os.environ.get("BYSJ_DRILL_PROJECT") != expect_project:
        raise SystemExit("BYSJ_DRILL_PROJECT does not match --expect-project; refusing to seed")
    revision = db.execute(__import__("sqlalchemy").text("SELECT version_num FROM alembic_version")).scalar()
    if revision != EXPECTED_REVISION:
        raise SystemExit(f"database is at {revision}, expected {EXPECTED_REVISION}; refusing to seed")
    existing = (
        db.query(DatasetResource)
        .filter(DatasetResource.name.like(DRILL_MARK + "%"))
        .count()
    )
    if existing:
        raise SystemExit("SYNTHETIC-DRILL datasets already exist; this drill has already been seeded")


def seed(expect_project: str, report_path: Path) -> dict:
    settings = get_settings()
    factory = get_session_factory()
    data_root = Path(settings.data_root).resolve()
    model_root = Path(settings.model_root).resolve()
    report: dict = {
        "synthetic": True,
        "research_valid": False,
        "note": "restore-drill fixture; never a research result",
        "project": expect_project,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "data_root": str(data_root),
        "cxr": {},
        "isic": {},
        "files": [],
    }
    with factory() as db:
        guard(expect_project, db)
        admin = db.query(User).filter(User.role == "admin").order_by(User.id.asc()).first()
        if admin is None:
            raise SystemExit("no admin user in the restored database; cannot attribute synthetic records")
        report["owner"] = {"id": admin.id, "username": admin.username}

        # ---------------- CXR: 10 synthetic samples, four states covered -------
        cxr_dataset = DatasetResource(
            dataset_id=f"{DRILL_MARK}-CXR",
            name=f"{DRILL_MARK} CXR (synthetic, drill only)",
            description="Synthetic chest X-ray records created to verify a restored f004 database.",
            dataset_role="medical",
            owner_user_id=admin.id,
            owner_username=admin.username,
            visibility_mode="private",
            modality="Chest X-ray",
            task_type=CXR_TASK,
            group_level=CXR_GROUP_LEVEL,
            license="synthetic; not a real dataset; no clinical validity",
            source_url=None,
            source_version="synthetic-drill-v1",
        )
        db.add(cxr_dataset)
        db.flush()

        cxr_version = DatasetVersion(
            dataset_id=cxr_dataset.id,
            dataset_role="medical",
            status="ready",
            state="frozen",
            created_by=admin.username,
            created_by_user_id=admin.id,
            metadata_json={
                "synthetic": True,
                "research_valid": False,
                "label_schema": CXR_LABEL_SCHEMA,
                "group_level": CXR_GROUP_LEVEL,
                "note": "drill fixture",
            },
            split_config_json={"seed": 42, "ratios": {"train": 0.6, "val": 0.1, "calibration": 0.2, "test": 0.1}},
        )
        db.add(cxr_version)
        db.flush()
        cxr_root = data_root / "datasets" / cxr_dataset.id / cxr_version.id
        cxr_version.root_path = str(cxr_root)
        cxr_version.manifest_path = str(cxr_root / "cxr-manifest.json")
        cxr_version.frozen_at = datetime.now(timezone.utc).replace(tzinfo=None)

        # (split, patient, positive label or None, raw state override)
        cxr_plan = [
            ("train", "SYN-P01", LABEL_KEYS[0], None),
            ("train", "SYN-P02", LABEL_KEYS[1], None),
            ("train", "SYN-P03", LABEL_KEYS[2], None),
            ("train", "SYN-P04", LABEL_KEYS[3], None),
            ("train", "SYN-P05", LABEL_KEYS[4], None),
            ("train", "SYN-P06", None, None),
            ("val", "SYN-P07", LABEL_KEYS[1], None),
            ("calibration", "SYN-P08", None, (LABEL_KEYS[3], RAW_UNCERTAIN)),
            ("calibration", "SYN-P09", LABEL_KEYS[0], None),
            ("test", "SYN-P10", None, (LABEL_KEYS[4], RAW_UNMENTIONED)),
        ]
        cxr_assets: list[DatasetAsset] = []
        cxr_samples: list[CxrSample] = []
        manifest_assets: list[dict] = []
        support = {split: {key: {"positive": 0, "negative": 0} for key in LABEL_KEYS} for split in
                   ("train", "val", "calibration", "test")}
        for index, (split, patient, positive, override) in enumerate(cxr_plan, start=1):
            name = f"syn_drill_cxr_{index:02d}.png"
            draw(cxr_root / "images" / name, index, name)
            path = cxr_root / "images" / name
            digest = sha256_file(path)
            asset = DatasetAsset(
                dataset_version_id=cxr_version.id,
                relative_path=name,
                basename=name,
                size=path.stat().st_size,
                width=64,
                height=64,
                source_group=patient,
                content_hash=digest,
                group_level=CXR_GROUP_LEVEL,
                split=split,
                transform_json={"synthetic": True},
            )
            db.add(asset)
            db.flush()
            raw = {key: RAW_NEGATIVE for key in LABEL_KEYS}
            if positive:
                raw[positive] = RAW_POSITIVE
            if override:
                key, state = override
                raw[key] = state
            for key, state in raw.items():
                if state in (RAW_POSITIVE, RAW_NEGATIVE):
                    support[split][key][state] += 1
            sample = CxrSample(
                asset_id=asset.id,
                dataset_version_id=cxr_version.id,
                patient_key=patient,
                study_key=f"{patient}-S1",
                component_key=f"{patient}-S1-C1",
                view="PA",
                raw_labels_json=raw,
                schema_version=CXR_LABEL_SCHEMA,
                source_hash=digest,
                pixel_hash=digest,
            )
            db.add(sample)
            cxr_assets.append(asset)
            cxr_samples.append(sample)
            manifest_assets.append(
                {
                    "asset_id": asset.id,
                    "relative_path": name,
                    "split": split,
                    "sha256": digest,
                    "patient_key": patient,
                    "study_key": f"{patient}-S1",
                    "component_key": f"{patient}-S1-C1",
                    "raw_labels": raw,
                    "synthetic": True,
                }
            )
            report["files"].append({"path": str(path), "sha256": digest, "size": path.stat().st_size})

        counts = {split: sum(1 for item in manifest_assets if item["split"] == split) for split in support}
        manifest = {
            "schema": "cxr-manifest-v1",
            "task": CXR_TASK,
            "label_schema": CXR_LABEL_SCHEMA,
            "group_level": CXR_GROUP_LEVEL,
            "split_seed": 42,
            "counts": counts,
            "patient_counts": counts,
            "component_counts": counts,
            "masked_support": support,
            "label_coverage": {key: {"positive": sum(1 for a in manifest_assets if a["raw_labels"][key] == RAW_POSITIVE)}
                               for key in LABEL_KEYS},
            "duplicate_candidates": {"pairs": 0, "note": "synthetic drill fixture"},
            "source": {"source_id": f"{DRILL_MARK}-CXR", "source_version": "synthetic-drill-v1"},
            "synthetic": True,
            "research_valid": False,
            "assets": manifest_assets,
        }
        manifest_path = Path(cxr_version.manifest_path)
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        cxr_version.asset_count = len(manifest_assets)
        cxr_version.split_counts_json = counts
        cxr_version.manifest_sha256 = sha256_file(manifest_path)
        cxr_version.split_manifest_hash = cxr_version.manifest_sha256
        cxr_version.content_hash = cxr_version.manifest_sha256
        db.flush()

        # ---------------- CXR model + analysis + review + report --------------
        model_dir = model_root / "registry" / "synthetic-drill"
        model_dir.mkdir(parents=True, exist_ok=True)
        model_file = model_dir / "synthetic-dummy.json"
        model_file.write_text(
            json.dumps({"backend": "dummy", "label_schema": CXR_LABEL_SCHEMA,
                        "synthetic": True, "research_valid": False}, ensure_ascii=False),
            encoding="utf-8",
        )
        model = ModelRegistry(
            name=f"{DRILL_MARK} dummy cxr (synthetic)",
            path=str(model_file),
            sha256=sha256_file(model_file),
            size=model_file.stat().st_size,
            source="training",
            status="active",
            format="json",
            dataset_version_id=cxr_version.id,
            created_by=admin.username,
            owner_id=admin.id,
            model_task=CXR_TASK,
            trusted_source="platform",
            available=True,
            class_map_json={key: LABEL_DISPLAY[key] for key in LABEL_KEYS},
            metadata_json={"backend": "dummy", "label_schema": CXR_LABEL_SCHEMA,
                           "arch": "densenet121", "image_size": 224,
                           "synthetic": True, "research_valid": False},
        )
        db.add(model)
        db.flush()
        report["files"].append(
            {"path": str(model_file), "sha256": model.sha256, "size": model.size}
        )

        batch = CxrAnalysisBatch(
            owner_id=admin.id,
            owner_username=admin.username,
            model_id=model.id,
            model_sha256=model.sha256,
            request_hash=sha256_json({"items": [cxr_assets[0].id], "synthetic": True}),
            idempotency_key=f"{DRILL_MARK}-batch-1",
            status="completed",
            total=1,
            completed=1,
            failed=0,
            finished_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        db.add(batch)
        db.flush()

        scores = {key: round(0.1 + 0.1 * i, 4) for i, key in enumerate(LABEL_KEYS)}
        analysis = CxrAnalysis(
            batch_id=batch.id,
            owner_id=admin.id,
            asset_id=cxr_assets[0].id,
            item_key=f"asset:{cxr_assets[0].id}",
            image_sha256=cxr_assets[0].content_hash,
            image_width=64,
            image_height=64,
            model_id=model.id,
            model_sha256=model.sha256,
            task=CXR_TASK,
            label_schema=CXR_LABEL_SCHEMA,
            raw_scores_json=scores,
            thresholds_json={key: 0.5 for key in LABEL_KEYS},
            calibration_status="not_calibrated",
            threshold_source="default_no_calibration",
            quality_json={"flags": [], "synthetic": True},
            explanation_json={"note": "synthetic drill fixture; no Grad-CAM produced",
                              "synthetic": True, "research_valid": False},
            backend="dummy",
            status="completed",
            attempt=1,
            finished_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        db.add(analysis)
        db.flush()

        review = CxrReview(
            analysis_id=analysis.id,
            owner_id=admin.id,
            reviewer_id=admin.id,
            reviewer_username=admin.username,
            state="approved",
            revision=1,
            items_json=[{"label_key": key, "flag": "above_threshold" if scores[key] >= 0.5 else "below_threshold",
                         "synthetic": True} for key in LABEL_KEYS],
            model_snapshot_json={"model_id": model.id, "model_sha256": model.sha256,
                                 "raw_scores": scores, "synthetic": True, "research_valid": False},
            reason="synthetic drill approval; not a real review",
            submitted_at=datetime.now(timezone.utc).replace(tzinfo=None),
            decided_at=datetime.now(timezone.utc).replace(tzinfo=None),
            decided_by_id=admin.id,
            decided_by_username=admin.username,
        )
        db.add(review)
        db.flush()

        snapshot = {
            "synthetic": True,
            "research_valid": False,
            "analysis_id": analysis.id,
            "review_id": review.id,
            "review_revision": review.revision,
            "labels": [{"label_key": key, "raw_score": scores[key], "threshold": 0.5} for key in LABEL_KEYS],
            "disclaimer": "synthetic drill fixture; model scores are not clinical findings",
        }
        report_row = CxrReport(
            analysis_id=analysis.id,
            review_id=review.id,
            review_revision=review.revision,
            state="final",
            version=1,
            snapshot_json=snapshot,
            snapshot_sha256=sha256_json(snapshot),
            owner_id=admin.id,
            issued_by_id=admin.id,
            issued_by_username=admin.username,
            issued_at=datetime.now(timezone.utc).replace(tzinfo=None),
        )
        db.add(report_row)
        db.flush()

        # ---------------- ISIC: 4 synthetic single-label samples ---------------
        isic_dataset = DatasetResource(
            dataset_id=f"{DRILL_MARK}-ISIC",
            name=f"{DRILL_MARK} ISIC (synthetic, drill only)",
            description="Synthetic dermoscopy records created to verify a restored f004 database.",
            dataset_role="medical",
            owner_user_id=admin.id,
            owner_username=admin.username,
            visibility_mode="private",
            modality="Dermoscopy",
            task_type=ISIC_TASK,
            group_level="lesion",
            license="synthetic; not a real dataset; no clinical validity",
            source_version="synthetic-drill-v1",
        )
        db.add(isic_dataset)
        db.flush()
        isic_version = DatasetVersion(
            dataset_id=isic_dataset.id,
            dataset_role="medical",
            status="ready",
            state="frozen",
            created_by=admin.username,
            created_by_user_id=admin.id,
            metadata_json={"synthetic": True, "research_valid": False,
                           "label_schema": ISIC_LABEL_SCHEMA, "group_level": "lesion",
                           "class_gate": {"passed": True, "note": "synthetic drill fixture"},
                           "rare_class_warnings": []},
            split_config_json={"seed": 42, "ratios": {"train": 0.5, "val": 0.25, "calibration": 0.0, "test": 0.25}},
        )
        db.add(isic_version)
        db.flush()
        isic_root = data_root / "datasets" / isic_dataset.id / isic_version.id
        isic_version.root_path = str(isic_root)
        isic_version.manifest_path = str(isic_root / "isic-manifest.json")
        isic_version.frozen_at = datetime.now(timezone.utc).replace(tzinfo=None)

        by_key = {label.key: label for label in ISIC_LABELS}
        isic_plan = [("train", "nv"), ("train", "mel"), ("val", "bcc"), ("test", "akiec")]
        class_counts: dict[str, int] = {}
        group_counts: dict[str, int] = {}
        isic_files = []
        for index, (split, class_key) in enumerate(isic_plan, start=1):
            name = f"syn_drill_isic_{index:02d}.png"
            draw(isic_root / "images" / name, 100 + index, name)
            path = isic_root / "images" / name
            digest = sha256_file(path)
            asset = DatasetAsset(
                dataset_version_id=isic_version.id,
                relative_path=name,
                basename=name,
                size=path.stat().st_size,
                width=64,
                height=64,
                source_group=f"SYN-L{index}",
                content_hash=digest,
                group_level="lesion",
                split=split,
                transform_json={"synthetic": True},
            )
            db.add(asset)
            db.flush()
            db.add(
                IsicSample(
                    asset_id=asset.id,
                    dataset_version_id=isic_version.id,
                    lesion_key=f"SYN-L{index}",
                    group_key=f"SYN-L{index}",
                    class_key=class_key,
                    class_index=by_key[class_key].index,
                    schema_version=ISIC_LABEL_SCHEMA,
                    pixel_hash=digest,
                    source_hash=digest,
                )
            )
            class_counts[class_key] = class_counts.get(class_key, 0) + 1
            group_counts[split] = group_counts.get(split, 0) + 1
            isic_files.append({"relative_path": name, "split": split, "class_key": class_key,
                               "lesion_key": f"SYN-L{index}", "sha256": digest})
            report["files"].append({"path": str(path), "sha256": digest, "size": path.stat().st_size})

        isic_counts = {split: sum(1 for item in isic_files if item["split"] == split)
                       for split in ("train", "val", "calibration", "test")}
        isic_manifest = {
            "schema": "isic-manifest-v1",
            "label_schema": ISIC_LABEL_SCHEMA,
            "task": ISIC_TASK,
            "group_level": "lesion",
            "counts": isic_counts,
            "class_counts": class_counts,
            "group_counts": isic_counts,
            "source": {"source_id": f"{DRILL_MARK}-ISIC", "source_version": "synthetic-drill-v1"},
            "synthetic": True,
            "research_valid": False,
            "assets": isic_files,
        }
        isic_manifest_path = Path(isic_version.manifest_path)
        isic_manifest_path.write_text(json.dumps(isic_manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        isic_version.asset_count = len(isic_files)
        isic_version.split_counts_json = isic_counts
        isic_version.manifest_sha256 = sha256_file(isic_manifest_path)
        isic_version.content_hash = isic_version.manifest_sha256
        db.flush()

        db.commit()

        report["cxr"] = {
            "dataset_id": cxr_dataset.id,
            "version_id": cxr_version.id,
            "asset_ids": [asset.id for asset in cxr_assets],
            "sample_ids": [sample.id for sample in cxr_samples],
            "model_id": model.id,
            "batch_id": batch.id,
            "analysis_id": analysis.id,
            "review_id": review.id,
            "report_id": report_row.id,
            "manifest_sha256": cxr_version.manifest_sha256,
            "split_counts": counts,
            "masked_support_train": support["train"],
            "training_gate": "computed by the API; see verification report",
        }
        report["isic"] = {
            "dataset_id": isic_dataset.id,
            "version_id": isic_version.id,
            "class_counts": class_counts,
            "split_counts": isic_counts,
            "manifest_sha256": isic_version.manifest_sha256,
        }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--expect-project", required=True)
    parser.add_argument("--confirm", required=True)
    parser.add_argument("--report", type=Path, required=True)
    args = parser.parse_args()
    if args.confirm != DRILL_MARK:
        print("refusing: --confirm must be " + DRILL_MARK, file=sys.stderr)
        return 2
    report = seed(args.expect_project, args.report)
    print(json.dumps({"status": "seeded", "synthetic": True, "research_valid": False,
                      "cxr_analysis": report["cxr"]["analysis_id"],
                      "isic_version": report["isic"]["version_id"],
                      "files": len(report["files"])}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
