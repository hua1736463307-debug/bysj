#!/usr/bin/env python3
"""Controlled grouped-revision import / freeze / verify for the BCCD package.

Subcommands:
    inspect              verify package digests and print the plan (no network)
    import               submit (once) and wait for the import job
    freeze               submit (once) and wait for the freeze job
    verify               compare API assets/labels/splits with the package
    prepare-experiments  write post-hoc E1/E2 config skeletons for review

State is written atomically before every submission. A timeout or broken
connection is an *uncertain* submission: the tool refuses to resend without a
job id and asks the operator to inspect the server first.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_experiments import request_data, wait_job, write_json  # noqa: E402

TERMINAL = {"completed", "failed", "cancelled"}
CLASS_MAP = {"0": "RBC", "1": "WBC", "2": "Platelets"}


class ContractError(RuntimeError):
    """A state/identity precondition failed; the tool must stop."""


def now() -> str:
    return datetime.now(timezone.utc).astimezone().isoformat()


def sha256_file(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    os.replace(temporary, path)


def package_paths(package: Path) -> dict[str, Path]:
    paths = {
        "archive": package / "import.zip",
        "groups": package / "source_groups.json",
        "preparation": package / "PREPARATION.json",
    }
    missing = [name for name, path in paths.items() if not path.is_file()]
    if missing:
        raise ContractError(f"package is missing: {', '.join(missing)}")
    return paths


def package_identity(package: Path, base_url: str, run_id: str) -> dict:
    paths = package_paths(package)
    preparation = load_json(paths["preparation"])
    archive_sha = sha256_file(paths["archive"])
    groups_sha = sha256_file(paths["groups"])
    if archive_sha != preparation.get("archive_sha256"):
        raise ContractError("import.zip sha256 does not match PREPARATION.archive_sha256")
    if groups_sha != preparation.get("group_map_sha256"):
        raise ContractError("source_groups.json sha256 does not match PREPARATION.group_map_sha256")
    if preparation.get("group_level") != "image_cluster":
        raise ContractError("PREPARATION.group_level must be image_cluster")
    return {
        "schema_version": 1,
        "captured_at": now(),
        "base_url": base_url.rstrip("/"),
        "run_id": run_id,
        "package": str(package.resolve()),
        "archive_sha256": archive_sha,
        "groups_sha256": groups_sha,
        "preparation_sha256": sha256_file(paths["preparation"]),
        "label_manifest_sha256": hashlib.sha256(
            json.dumps(preparation.get("label_sha256"), sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest(),
        "images": int(preparation.get("images") or 0),
        "groups": int(preparation.get("groups") or 0),
        "split_seed": int(preparation.get("fixed_split_seed") or 42),
        "group_level": preparation.get("group_level"),
        "dataset_name": f"BCCD-image-cluster-{archive_sha[:8]}-{groups_sha[:8]}-{run_id}",
    }


def load_state(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": 1, "phase": "new", "history": []}
    return load_json(path)


def save_state(path: Path, state: dict) -> None:
    state["updated_at"] = now()
    atomic_json(path, state)


def bind_identity(state: dict, identity: dict) -> dict:
    existing = state.get("identity")
    if existing:
        for key in ("base_url", "archive_sha256", "groups_sha256", "preparation_sha256", "dataset_name"):
            if existing.get(key) != identity.get(key):
                raise ContractError(f"state identity mismatch on {key}; refusing to mix two packages/targets")
    else:
        state["identity"] = identity
    return state


def _job_failure(job: dict, phase: str) -> ContractError:
    error = job.get("error") or {}
    return ContractError(f"{phase} job {job.get('job_id')} ended {job.get('status')}: {error.get('code') or ''}")


def import_dataset(client: httpx.Client, api: str, token: str, package: Path, state_path: Path,
                   identity: dict, *, poll_timeout: float) -> dict:
    state = bind_identity(load_state(state_path), identity)
    if state.get("import_job_id"):
        job = wait_job(client, api, token, state["import_job_id"], poll_timeout)
        if job["status"] != "completed":
            raise _job_failure(job, "import")
        state["phase"] = "imported"
        state["import_result"] = job.get("result")
        save_state(state_path, state)
        return job
    if state.get("phase") == "submitting_import":
        raise ContractError(
            "previous import submission has no job id (uncertain). Inspect /jobs and datasets on the "
            "server; do not resend blindly."
        )

    paths = package_paths(package)
    groups_text = paths["groups"].read_text(encoding="utf-8")
    state.update(phase="submitting_import", import_requested_at=now(), uncertain=False)
    save_state(state_path, state)
    try:
        data = request_data(
            client,
            "POST",
            f"{api}/datasets/import",
            headers={"Authorization": f"Bearer {token}"},
            files={"files": ("import.zip", paths["archive"].read_bytes(), "application/zip")},
            data={
                "name": identity["dataset_name"],
                "source_url": "https://github.com/Shenggan/BCCD_Dataset",
                "source_version": "d272fb14cdff6e473fafeeeba32aba5f560e9e43+cluster:1b7c6906bdbad794",
                "license": "MIT",
                "group_level": "image_cluster",
                "source_groups": groups_text,
                "class_map": json.dumps(CLASS_MAP),
            },
            timeout=600,
        )
    except (httpx.TimeoutException, httpx.TransportError) as exc:
        state.update(uncertain=True, phase="submitting_import",
                     uncertain_reason=f"{type(exc).__name__}", uncertain_at=now())
        save_state(state_path, state)
        raise ContractError(
            "import submission is uncertain (no response). The server may have accepted it; "
            "inspect jobs/datasets before any retry."
        ) from exc

    if not data.get("job_id"):
        state.update(uncertain=True, uncertain_reason="missing job_id in accepted response")
        save_state(state_path, state)
        raise ContractError("import response has no job_id; treated as uncertain")
    state.update(import_job_id=data["job_id"], phase="import_submitted", submitted_at=now())
    save_state(state_path, state)

    job = wait_job(client, api, token, data["job_id"], poll_timeout)
    if job["status"] != "completed":
        state.update(phase="import_failed", import_error=job.get("error"))
        save_state(state_path, state)
        raise _job_failure(job, "import")
    state.update(phase="imported", import_result=job.get("result"))
    save_state(state_path, state)
    return job


def _version_config_matches(version: dict, identity: dict) -> bool:
    config = version.get("split_config") or {}
    return (
        version.get("state") == "frozen"
        and int(config.get("seed", -1)) == identity["split_seed"]
        and config.get("group_level") == "image_cluster"
    )


def freeze_version(client: httpx.Client, api: str, token: str, state_path: Path,
                   identity: dict, *, poll_timeout: float, dataset_id: str | None = None,
                   version_id: str | None = None) -> dict:
    state = bind_identity(load_state(state_path), identity)
    dataset_id = dataset_id or (state.get("import_result") or {}).get("dataset_id")
    version_id = version_id or (state.get("import_result") or {}).get("version_id")
    if not dataset_id or not version_id:
        raise ContractError("import must complete before freeze")

    if state.get("freeze_job_id"):
        job = wait_job(client, api, token, state["freeze_job_id"], poll_timeout)
        if job["status"] != "completed":
            raise _job_failure(job, "freeze")
    elif state.get("phase") == "submitting_freeze":
        raise ContractError(
            "previous freeze submission has no job id (uncertain). Inspect the version state first."
        )
    else:
        current = request_data(client, "GET", f"{api}/dataset-versions/{version_id}",
                               headers={"Authorization": f"Bearer {token}"})
        if current.get("state") == "frozen":
            if not _version_config_matches(current, identity):
                raise ContractError("existing frozen version does not match this protocol; refusing to reuse")
            state.update(phase="frozen", version_snapshot=current, reused_frozen=True)
            save_state(state_path, state)
            return {"reused": True, "version": current}
        if current.get("state") not in {"draft", "failed"}:
            raise ContractError(f"version state {current.get('state')} is not freezable; inspect first")

        state.update(phase="submitting_freeze", freeze_requested_at=now(), uncertain=False)
        save_state(state_path, state)
        try:
            data = request_data(
                client,
                "POST",
                f"{api}/datasets/{dataset_id}/versions",
                headers={"Authorization": f"Bearer {token}"},
                json={"base_version_id": version_id, "split_seed": identity["split_seed"], "ratios": [0.8, 0.1, 0.1]},
                timeout=120,
            )
        except (httpx.TimeoutException, httpx.TransportError) as exc:
            state.update(uncertain=True, uncertain_reason=f"{type(exc).__name__}")
            save_state(state_path, state)
            raise ContractError("freeze submission is uncertain; inspect the version before retrying") from exc
        if not data.get("job_id"):
            state.update(uncertain=True, uncertain_reason="missing freeze job_id")
            save_state(state_path, state)
            raise ContractError("freeze response has no job_id; treated as uncertain")
        state.update(freeze_job_id=data["job_id"], phase="freeze_submitted", freeze_submitted_at=now())
        save_state(state_path, state)
        job = wait_job(client, api, token, data["job_id"], poll_timeout)
        if job["status"] != "completed":
            state.update(phase="freeze_failed", freeze_error=job.get("error"))
            save_state(state_path, state)
            raise _job_failure(job, "freeze")

    current = request_data(client, "GET", f"{api}/dataset-versions/{version_id}",
                           headers={"Authorization": f"Bearer {token}"})
    if not _version_config_matches(current, identity):
        raise ContractError("frozen version config/hash does not match the submitted protocol")
    state.update(phase="frozen", version_snapshot=current)
    save_state(state_path, state)
    return {"reused": False, "version": current}


def _package_labels(paths: dict[str, Path]) -> dict[str, list[dict]]:
    import zipfile

    labels: dict[str, list[dict]] = {}
    with zipfile.ZipFile(paths["archive"]) as bundle:
        for name in bundle.namelist():
            if not name.endswith(".txt"):
                continue
            image = name[:-4] + ".jpg"
            rows = []
            for line in bundle.read(name).decode("utf-8").splitlines():
                parts = line.split()
                if len(parts) != 5:
                    continue
                rows.append({
                    "class_id": int(parts[0]),
                    "cx": float(parts[1]), "cy": float(parts[2]),
                    "w": float(parts[3]), "h": float(parts[4]),
                })
            labels[image] = rows
    return labels


def _paged_assets(client: httpx.Client, api: str, token: str, dataset_id: str, version_id: str) -> list[dict]:
    assets: list[dict] = []
    page = 1
    while True:
        result = request_data(
            client, "GET",
            f"{api}/datasets/{dataset_id}/assets",
            headers={"Authorization": f"Bearer {token}"},
            params={"version_id": version_id, "page": page, "page_size": 200},
        )
        items = result.get("items") or []
        assets.extend(items)
        total = int(result.get("total") or 0)
        if not items and len(assets) < total:
            raise ContractError("asset pagination is incomplete (empty page before total)")
        if len(assets) >= total:
            break
        page += 1
        if page > 50:
            raise ContractError("asset pagination exceeded 50 pages")
    if len(assets) != total:
        raise ContractError(f"asset pagination incomplete: {len(assets)}/{total}")
    return assets


def _boxes_equal(left: list[dict], right: list[dict], tolerance: float = 1e-6) -> bool:
    if len(left) != len(right):
        return False
    for a, b in zip(sorted(left, key=lambda x: (x["class_id"], x["cx"], x["cy"])),
                    sorted(right, key=lambda x: (x["class_id"], x["cx"], x["cy"]))):
        if int(a["class_id"]) != int(b["class_id"]):
            return False
        if any(abs(float(a[key]) - float(b[key])) > tolerance for key in ("cx", "cy", "w", "h")):
            return False
    return True


def verify_package(client: httpx.Client, api: str, token: str, package: Path, state_path: Path,
                   identity: dict, output_dir: Path) -> dict:
    import zipfile

    state = bind_identity(load_state(state_path), identity)
    dataset_id = (state.get("import_result") or {}).get("dataset_id")
    version_id = (state.get("import_result") or {}).get("version_id")
    if not dataset_id or not version_id:
        raise ContractError("import must complete before verify")
    paths = package_paths(package)
    version = request_data(client, "GET", f"{api}/dataset-versions/{version_id}",
                           headers={"Authorization": f"Bearer {token}"})
    problems: list[str] = []
    if version.get("state") != "frozen":
        problems.append(f"version state is {version.get('state')}")
    config = version.get("split_config") or {}
    if config.get("group_level") != "image_cluster":
        problems.append("split_config.group_level is not image_cluster")
    if int(config.get("seed", -1)) != identity["split_seed"]:
        problems.append("split seed differs from the frozen protocol")

    assets = _paged_assets(client, api, token, dataset_id, version_id)
    if len(assets) != identity["images"]:
        problems.append(f"asset count {len(assets)} != package {identity['images']}")
    by_path: dict[str, dict] = {}
    for asset in assets:
        path = asset["relative_path"]
        if path in by_path:
            problems.append(f"duplicate asset path {path}")
        by_path[path] = asset

    labels = _package_labels(paths)
    groups = load_json(paths["groups"])
    with zipfile.ZipFile(paths["archive"]) as bundle:
        names = bundle.namelist()
        images = {name for name in names if name.endswith(".jpg")}
        if images != set(groups):
            problems.append("package images, labels and group mapping have different key sets")
        if set(by_path) != images:
            problems.append("server assets and package images differ")
        mismatched_sha = [name for name in images & set(by_path)
                          if hashlib.sha256(bundle.read(name)).hexdigest() != by_path[name]["sha256"]]
        if mismatched_sha:
            problems.append(f"{len(mismatched_sha)} image sha256 mismatches, first={mismatched_sha[0]}")
    for name in images & set(by_path):
        if str(groups.get(name)) != str(by_path[name].get("source_group")):
            problems.append(f"source_group mismatch for {name}")
            break
    for name, rows in labels.items():
        if name in by_path and not _boxes_equal(rows, by_path[name].get("annotations") or []):
            problems.append(f"annotations differ for {name}")
            break
    unique_groups = {str(value) for value in groups.values()}
    if len(unique_groups) != identity["groups"]:
        problems.append(f"group count {len(unique_groups)} != package {identity['groups']}")
    pair = [by_path.get(name) for name in ("BloodImage_00037.jpg", "BloodImage_00038.jpg")]
    if all(pair):
        if pair[0].get("source_group") != pair[1].get("source_group"):
            problems.append("00037/00038 are not in the same source group")
        if pair[0].get("split") != pair[1].get("split"):
            problems.append("00037/00038 are not in the same split")
        if pair[0].get("split") != "train":
            problems.append("00037/00038 are not in train as planned")
    else:
        problems.append("00037/00038 missing from server assets")

    splits: dict[str, set[str]] = {}
    split_groups: dict[str, set[str]] = {}
    split_hashes: dict[str, set[str]] = {}
    split_classes: dict[str, set[int]] = {}
    for asset in assets:
        split = asset.get("split")
        splits.setdefault(split, set()).add(asset["relative_path"])
        split_groups.setdefault(split, set()).add(str(asset.get("source_group")))
        split_hashes.setdefault(split, set()).add(asset["sha256"])
        split_classes.setdefault(split, set()).update(int(a["class_id"]) for a in asset.get("annotations") or [])
    counts = {split: len(splits.get(split, ())) for split in ("train", "val", "test")}
    if counts != {"train": 291, "val": 37, "test": 36}:
        problems.append(f"split counts differ from the planned 291/37/36: {counts}")
    for level, table in (("group", split_groups), ("hash", split_hashes)):
        overlap = (table.get("train", set()) & table.get("val", set())) | \
                  (table.get("train", set()) & table.get("test", set())) | \
                  (table.get("val", set()) & table.get("test", set()))
        if overlap:
            problems.append(f"{level} overlap across splits: {len(overlap)}")
    for split in ("train", "val", "test"):
        if split_classes.get(split, set()) != {0, 1, 2}:
            problems.append(f"{split} misses a class: {sorted(split_classes.get(split, set()))}")

    report = {
        "status": "verified" if not problems else "failed",
        "dataset_id": dataset_id,
        "version_id": version_id,
        "asset_count": len(assets),
        "group_count": len(unique_groups),
        "split_counts": counts,
        "split_manifest_hash": version.get("split_manifest_hash"),
        "class_map": version.get("class_map"),
        "group_level": config.get("group_level"),
        "problems": problems,
    }
    write_json(output_dir / "verification.json", report)
    if problems:
        raise ContractError("verification failed: " + "; ".join(problems[:5]))
    return report


def prepare_experiments(package: Path, state_path: Path, identity: dict, output_dir: Path,
                        base_config: Path) -> dict:
    state = bind_identity(load_state(state_path), identity)
    if state.get("phase") != "verified":
        raise ContractError("verify must pass before preparing experiment configs")
    result = state.get("import_result") or {}
    base = load_json(base_config)
    protocol = {
        "purpose": "posthoc_grouped_development",
        "dataset_version_id": result.get("version_id"),
        "split_seed": identity["split_seed"],
        "test_manifest_sha256": (state.get("version_snapshot") or {}).get("split_manifest_hash"),
        "dataset_name": identity["dataset_name"],
    }
    configs = {}
    for key in ("e1", "e2"):
        config = {**base, **protocol, "protocol_variant": key, "output_root": f"artifacts/grouped-posthoc-experiments/{key.upper()}"}
        config["note"] = "post-hoc development protocol; old test has been used before, not confirmatory"
        path = output_dir / f"bccd-grouped-posthoc-{key}.json"
        atomic_json(path, config)
        configs[key] = str(path)
    return {"status": "prepared", "configs": configs}


def main() -> int:
    parser = argparse.ArgumentParser(description="Grouped BCCD revision: import/freeze/verify with resumable state")
    parser.add_argument("--base-url", default="http://127.0.0.1:18004")
    parser.add_argument("--package", default="artifacts/bccd-grouped-20261022", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--state", type=Path)
    parser.add_argument("--run-id", default=os.environ.get("BYSJ_RUN_ID", "grouped-resume-manual"))
    parser.add_argument("--poll-timeout", type=float, default=1800)
    parser.add_argument("--username", default=os.environ.get("BYSJ_USER"))
    parser.add_argument("--password", default=os.environ.get("BYSJ_PASSWORD"))
    parser.add_argument("--base-config", default="configs/experiments/bccd-baseline.json", type=Path)
    sub = parser.add_subparsers(dest="command", required=True)
    for name in ("inspect", "import", "freeze", "verify", "prepare-experiments"):
        sub.add_parser(name)
    args = parser.parse_args()

    package = args.package
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    state_path = args.state or output / "revision-state.json"
    api = args.base_url.rstrip("/") + "/api/v1"
    identity = package_identity(package, args.base_url, args.run_id)

    if args.command == "inspect":
        state = bind_identity(load_state(state_path), identity)
        save_state(state_path, state)
        print(json.dumps({"status": "inspected", "identity": identity, "phase": state.get("phase")},
                         ensure_ascii=False, indent=2))
        return 0

    if not args.username or not args.password:
        print(json.dumps({"status": "failed", "error": "BYSJ_USER/BYSJ_PASSWORD required"}, ensure_ascii=False))
        return 2
    client = httpx.Client(timeout=120, trust_env=False)
    token = request_data(client, "POST", f"{api}/auth/login",
                         json={"username": args.username, "password": args.password})["token"]

    try:
        if args.command == "import":
            job = import_dataset(client, api, token, package, state_path, identity, poll_timeout=args.poll_timeout)
            print(json.dumps({"status": job["status"], "job_id": job["job_id"], "result": job.get("result")},
                             ensure_ascii=False, indent=2))
        elif args.command == "freeze":
            result = freeze_version(client, api, token, state_path, identity, poll_timeout=args.poll_timeout)
            print(json.dumps({"status": result["version"].get("state"), "reused": result["reused"],
                              "version": result["version"]}, ensure_ascii=False, indent=2))
        elif args.command == "verify":
            report = verify_package(client, api, token, package, state_path, identity, output)
            state = load_state(state_path)
            state["phase"] = "verified"
            save_state(state_path, state)
            print(json.dumps(report, ensure_ascii=False, indent=2))
        else:
            print(json.dumps(prepare_experiments(package, state_path, identity, output, args.base_config),
                             ensure_ascii=False, indent=2))
    except ContractError as exc:
        print(json.dumps({"status": "stopped", "error": str(exc)}, ensure_ascii=False))
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
