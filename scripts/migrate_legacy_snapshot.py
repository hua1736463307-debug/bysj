"""Offline conversion of a recognized legacy snapshot into an EMPTY migrated DB.

This module is exercised by the isolated PostgreSQL rehearsal. Callers must use
a read-only, repeatable-read source transaction and a newly migrated destination.
It never stamps an old database, modifies source rows, or loads model pickle data.
Unsupported geometry, ambiguous ownership, active jobs and missing files fail
closed. A real legacy dump still needs its own reviewed schema/data assessment.
"""
from __future__ import annotations
from collections import Counter
from pathlib import Path
import hashlib
import json
import shutil
import uuid

import sqlalchemy as sa
from app.core.storage import normalize_relative, resolve_under, sha256_file, atomic_write_bytes
from app.db import models
from app.db.base import Base
from app.domains.annotation.service import _validate_box
from app.domains.datasets.split_service import build_manifest, canonical_json, integrity_report
from legacy_schema import structure

TABLE_MODELS = {model.__tablename__: model for model in (models.User, models.DatasetResource,
    models.DatasetVersion, models.DatasetAsset, models.AnnotationDraft, models.AnnotationDraftAsset,
    models.JobTask, models.JobEvent, models.ModelRegistry)}


def digest(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()


def verify_schema(source, expected_metadata):
    actual = structure(source, TABLE_MODELS)
    for name, expected in expected_metadata.tables.items():
        fields = {c["name"]: (c["type"], c["nullable"]) for c in actual[name]["columns"]}
        wanted = {c.name: (str(c.type.compile(dialect=source.dialect)), c.nullable) for c in expected.columns}
        # PostgreSQL reflects unqualified FLOAT as its DOUBLE PRECISION storage.
        wanted = {key: ("DOUBLE PRECISION" if value[0] == "FLOAT" else value[0], value[1]) for key, value in wanted.items()}
        if fields != wanted:
            difference = {key: {"actual": fields.get(key), "expected": wanted.get(key)} for key in fields.keys() | wanted.keys() if fields.get(key) != wanted.get(key)}
            raise ValueError(f"unrecognized legacy columns/types/nullability: {name}: {difference}")
        if actual[name]["primary_key"] != [c.name for c in expected.primary_key.columns]:
            raise ValueError(f"unrecognized legacy primary key: {name}")
        uniques = sorted([list(c.columns.keys()) for c in expected.constraints if isinstance(c, sa.UniqueConstraint)])
        if actual[name]["unique_constraints"] != uniques:
            raise ValueError(f"unrecognized legacy unique constraints: {name}")
        expected_fk = sorted((tuple(c.columns.keys()), tuple(e.target_fullname for e in c.elements), c.ondelete, c.onupdate)
                             for c in expected.constraints if isinstance(c, sa.ForeignKeyConstraint))
        actual_fk = sorted((tuple(f['constrained_columns']), tuple(f['referred_table']+'.'+column for column in f['referred_columns']),
                            f['options'].get('ondelete'), f['options'].get('onupdate')) for f in actual[name]['foreign_keys'])
        if expected_fk != actual_fk:
            raise ValueError(f"unrecognized legacy foreign keys: {name}")
        expected_indexes = sorted((i.name, tuple(i.columns.keys()), bool(i.unique)) for i in expected.indexes)
        actual_indexes = sorted((i['name'], tuple(i['columns']), bool(i['unique'])) for i in actual[name]['indexes'] if not i['duplicates_constraint'])
        if expected_indexes != actual_indexes:
            raise ValueError(f"unrecognized legacy indexes: {name}")
        for reflected in actual[name]['columns']:
            column = expected.columns[reflected['name']]
            default = reflected['default']
            serial = column.primary_key and isinstance(column.type, sa.Integer) and column.autoincrement in (True, 'auto')
            if column.server_default is None and default is not None and not (serial and default.startswith('nextval(')):
                raise ValueError(f"unrecognized legacy server default: {name}.{column.name}")
    return actual


def safe_existing(path, root):
    candidate = Path(path).resolve()
    candidate.relative_to(Path(root).resolve())
    if not candidate.is_file():
        raise ValueError(f"missing controlled source file: {candidate.name}")
    return candidate


def convert_boxes(annotations, class_map, identity):
    result = []
    for index, annotation in enumerate(annotations or []):
        if annotation.get("type", "bbox") not in {"bbox", "rect"}:
            raise ValueError("legacy polygon/pose geometry requires a separate reviewed converter")
        coords = {key: annotation.get(key, annotation.get(old)) for key, old in (("cx", "x"), ("cy", "y"), ("w", "width"), ("h", "height"))}
        box = {**coords, "class_id": annotation["class_id"], "client_id": annotation.get("client_id") or str(uuid.uuid5(uuid.NAMESPACE_URL, f"{identity}:{index}")), "source": "legacy"}
        result.append(_validate_box(box, class_map))
    return result


def copy_snapshot(source, target, settings, *, expected_metadata, source_data_root, source_model_root):
    schema = verify_schema(source, expected_metadata)
    if source.dialect.name != "postgresql" or target.bind.dialect.name != "postgresql":
        raise ValueError("rehearsed conversion requires PostgreSQL")
    # A new schema already upgraded by Alembic; never overwrite existing rows.
    for table in Base.metadata.sorted_tables:
        if target.execute(sa.select(sa.func.count()).select_from(table)).scalar_one():
            raise ValueError("destination is not empty")
    source_tables = sa.MetaData()
    rows = {}
    for name in TABLE_MODELS:
        table = sa.Table(name, source_tables, autoload_with=source)
        rows[name] = [dict(row) for row in source.execute(sa.select(table)).mappings()]
    if any(row["status"] not in {"completed", "failed", "cancelled"} for row in rows["job_tasks"]):
        raise ValueError("legacy jobs must be quiesced to terminal states before offline conversion")
    if source.execute(sa.text('SHOW transaction_read_only')).scalar_one() != 'on' or source.get_isolation_level() != 'REPEATABLE READ':
        raise ValueError('source transaction must be read-only and REPEATABLE READ')
    users = {row["id"]: row for row in rows["users"]}
    by_username = {row["username"]: row["id"] for row in rows["users"]}
    if any(row["role"] not in {"user", "admin", "researcher", "reviewer"} for row in users.values()):
        raise ValueError("unknown legacy role; provide an explicit reviewed role mapping")
    resources = {(row["dataset_id"], row["dataset_role"]): row for row in rows["dataset_resources"]}
    versions = {row["id"]: row for row in rows["dataset_versions"]}
    drafts = {row["id"]: row for row in rows["annotation_drafts"]}
    assets_by_version = {}
    for row in rows["dataset_assets"]:
        if row["dataset_version_id"] not in versions:
            raise ValueError("orphan legacy asset")
        assets_by_version.setdefault(row["dataset_version_id"], []).append(row)
    converted = {name: [] for name in TABLE_MODELS}
    for name, values in rows.items():
        allowed = set(TABLE_MODELS[name].__table__.columns.keys())
        converted[name] = [{key: value for key, value in row.items() if key in allowed} for row in values]
    report = {"mode": "offline_copy", "source_schema_sha256": digest(schema), "source_structure": schema,
              "source_counts": {name: len(values) for name, values in rows.items()},
              "source_rows_sha256": {name: digest(values) for name, values in rows.items()},
              "excluded_legacy_columns": {name: sorted(set(expected_metadata.tables[name].columns.keys()) - set(TABLE_MODELS[name].__table__.columns.keys())) for name in TABLE_MODELS},
              "versions": [], "files": []}
    migration_id = str(uuid.uuid4())
    data_destination = Path(settings.data_root) / "legacy-import" / migration_id
    model_destination = Path(settings.model_root) / "legacy-import" / migration_id
    if data_destination.exists() or model_destination.exists():
        raise ValueError("migration destination already exists")

    def owner(username, explicit=None):
        candidate = explicit if explicit is not None else by_username.get(username)
        if candidate not in users or users[candidate]["username"] != username:
            raise ValueError("legacy ownership does not resolve to an existing user")
        return candidate

    def copy_file(src, dst, expected=None):
        actual = sha256_file(src)
        if expected and actual != expected:
            raise ValueError("source file hash differs from legacy record")
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        if sha256_file(dst) != actual:
            raise ValueError("copied file hash mismatch")
        report["files"].append({"destination": str(dst), "sha256": actual, "size": dst.stat().st_size})

    for row in converted["dataset_resources"]:
        row.update(owner_user_id=owner(row["owner_username"], row["owner_user_id"]), name=row["dataset_id"], visibility_mode="private")
    for row in converted["dataset_versions"]:
        old = versions[row["id"]]
        resource = resources.get((old["dataset_id"], old["dataset_role"]))
        if not resource:
            raise ValueError("legacy version has no matching dataset resource")
        if old["parent_version_id"] and old["parent_version_id"] not in versions:
            raise ValueError("orphan parent version")
        root = Path(old["root_path"]).resolve()
        root.relative_to(Path(source_data_root).resolve())
        manifest_path = safe_existing(old["manifest_path"], source_data_root)
        legacy_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest_assets = {a["relative_path"]: a for a in legacy_manifest.get("assets", [])}
        destination = data_destination / row["id"]
        entries, fixed, label_names = [], {}, set()
        for asset in assets_by_version.get(row["id"], []):
            relative = normalize_relative(asset["relative_path"]).as_posix()
            entry = manifest_assets.get(relative)
            if entry is None:
                raise ValueError("legacy asset missing from its manifest")
            if entry.get("content_hash", entry.get("sha256")) != asset["content_hash"]:
                raise ValueError("legacy manifest/database asset hash mismatch")
            annotations = convert_boxes(asset["annotations_json"], row["class_map_json"], f"asset:{asset['id']}")
            src = resolve_under(root, relative, must_exist=True)
            copy_file(src, resolve_under(destination / "images", relative), asset["content_hash"])
            label = normalize_relative(relative).with_suffix(".txt").as_posix()
            if label in label_names:
                raise ValueError('legacy assets collide on the same YOLO label path')
            label_names.add(label)
            atomic_write_bytes(resolve_under(destination / "labels", label), ''.join(f"{a['class_id']} {a['cx']:.6f} {a['cy']:.6f} {a['w']:.6f} {a['h']:.6f}\n" for a in annotations).encode())
            entries.append({"relative_path": relative, "sha256": asset["content_hash"], "source_group": asset["source_group"] or relative,
                            "width": asset["width"], "height": asset["height"], "size": asset["size"], "annotations": annotations})
            fixed[relative] = entry.get("split", "unsplit")
        copy_file(manifest_path, destination / "legacy-manifest.json")
        row.update(dataset_id=resource["id"], root_path=str(destination), manifest_path=str(destination / "legacy-manifest.json"),
                   created_by_user_id=owner(old["created_by"]), asset_count=len(entries), state="draft",
                   metadata_json={**(old["metadata_json"] or {}), "legacy_content_hash": old["content_hash"], "legacy_manifest_sha256": sha256_file(manifest_path)})
        # Preserve a valid historical split; never manufacture a new test split.
        if set(fixed.values()) == {"train", "val", "test"}:
            split_config = old["split_config_json"] or {}
            ratios = split_config.get("ratios", [.8, .1, .1])
            if isinstance(ratios, dict): ratios = [ratios[s] for s in ("train", "val", "test")]
            manifest, stats = build_manifest(entries, class_map=row["class_map_json"], seed=split_config.get("seed", 42), ratios=ratios,
                dataset_id=row["dataset_id"], version_id=row["id"], parent_version_id=row["parent_version_id"], fixed_splits=fixed)
            if not integrity_report(manifest)["ok"]:
                raise ValueError("legacy train/val/test leakage")
            atomic_write_bytes(destination / "manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2).encode())
            row.update(state="frozen", frozen_at=old["created_at"], manifest_path=str(destination / "manifest.json"),
                       manifest_sha256=stats["manifest_sha256"], split_manifest_hash=stats["split_manifest_hash"],
                       class_map_hash=stats["class_map_hash"], split_counts_json=stats["split_counts"])
        elif set(fixed.values()) - {"unsplit", None}:
            raise ValueError("legacy partial/invalid split needs explicit review")
        report["versions"].append({"version_id": row["id"], "legacy_dataset_id": old["dataset_id"], "resource_id": resource["id"],
                                   "state": row["state"], "legacy_content_hash": old["content_hash"], "manifest_sha256": row.get("manifest_sha256")})
        for asset in converted["dataset_assets"]:
            if asset["dataset_version_id"] != row["id"]: continue
            asset.update(annotations_json=convert_boxes(asset["annotations_json"], row["class_map_json"], f"asset:{asset['id']}"),
                         label_path="labels/" + normalize_relative(asset["relative_path"]).with_suffix(".txt").as_posix(),
                         split=fixed[asset["relative_path"]] if fixed[asset["relative_path"]] in {"train", "val", "test"} else None,
                         root_asset_id=asset["id"], group_level="original_image")
    converted_versions = {row["id"]: row for row in converted["dataset_versions"]}
    for row in converted["annotation_drafts"]:
        version = converted_versions.get(row["source_version_id"])
        if not version or row["annotation_geometry"] != "rect" or row["annotation_kind"] not in {"bbox", "manual", "auto"}:
            raise ValueError("unsupported legacy draft context")
        owner(row["owner_username"], row["owner_user_id"])
        row.update(dataset_id=version["dataset_id"], annotation_kind="bbox", model_path="")
    for row in converted["annotation_draft_assets"]:
        draft = drafts.get(row["draft_id"])
        if draft is None: raise ValueError("orphan draft asset")
        version = converted_versions[draft["source_version_id"]]
        if row["relative_path"] not in {a["relative_path"] for a in assets_by_version.get(version["id"], [])}:
            raise ValueError("draft refers to absent source asset")
        row.update(annotations_json=convert_boxes(row["annotations_json"], version["class_map_json"], f"draft:{row['draft_id']}:{row['relative_path']}"), deleted_json=[])
    for row in converted["job_tasks"]:
        row["owner_user_id"] = owner(row["owner_username"], int(row["owner_id"]) if row["owner_id"] is not None else None)
    jobs = {row["id"] for row in rows["job_tasks"]}
    if any(row["task_id"] not in jobs for row in rows["job_events"]):
        raise ValueError("orphan job event")
    for row in converted["model_registry"]:
        src = safe_existing(row["path"], source_model_root)
        dst = model_destination / row["id"] / src.name
        copy_file(src, dst, row["sha256"])
        if row["dataset_version_id"] and row["dataset_version_id"] not in versions:
            raise ValueError("orphan model version")
        if row["task_id"] and row["task_id"] not in jobs:
            raise ValueError("orphan model training job")
        row.update(path=str(dst), owner_id=owner(row["created_by"]), training_job_id=row["task_id"],
                   class_map_json=(versions.get(row["dataset_version_id"]) or {}).get("class_map_json", {}),
                   trusted_source="platform" if row["task_id"] else "legacy")
    try:
        for name, model in TABLE_MODELS.items():
            target.add_all([model(**row) for row in converted[name]])
        target.flush()
        for name in ("users", "dataset_assets", "annotation_draft_assets", "job_events"):
            # Names are fixed constants above, never user-provided identifiers.
            target.execute(sa.text(f"SELECT setval(pg_get_serial_sequence('{name}','id'), COALESCE(MAX(id),1), MAX(id) IS NOT NULL) FROM {name}"))
        target.commit()
    except Exception:
        target.rollback()
        raise
    report["target_counts"] = {name: target.query(model).count() for name, model in TABLE_MODELS.items()}
    if report["target_counts"] != report["source_counts"]:
        raise RuntimeError("post-copy row count mismatch")
    report["new_tables"] = sorted(set(Base.metadata.tables) - set(TABLE_MODELS))
    report["target_structure"] = structure(target.connection())
    report["shared_structure_differences"] = {name: {
        key: {"source": schema[name][key], "target": report['target_structure'][name][key]}
        for key in schema[name] if schema[name][key] != report['target_structure'][name][key]
    } for name in TABLE_MODELS}
    report["status"] = "copied_requires_acceptance"
    return report
