#!/usr/bin/env python3
"""Freeze a complete protocol, resume jobs idempotently, then evaluate fixed test.

Credentials are read only from BYSJ_USER/BYSJ_PASSWORD. Each protocol has its
own SHA-256 directory. A failed/interrupted/timed-out seed is never a zero metric.
Use --prepare-only to inspect the frozen protocol before submitting any job.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import sys
import time

import httpx

TERMINAL = {"completed", "failed", "cancelled", "interrupted"}
METRICS = ("precision", "recall", "map50", "map50_95")


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def config_hash(config):
    return hashlib.sha256(canonical(config)).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False).encode())
    os.replace(temporary, path)


def freeze_protocol(root, protocol):
    digest = config_hash(protocol)
    output = Path(root) / digest
    output.mkdir(parents=True, exist_ok=True)
    frozen = output / "config.frozen.json"
    try:
        with frozen.open("xb") as handle:
            handle.write(canonical(protocol))
            handle.flush()
            os.fsync(handle.fileno())
    except FileExistsError:
        if frozen.read_bytes() != canonical(protocol):
            raise ValueError("frozen protocol differs; refusing to overwrite historical evidence")
    return output, digest


def validate_config(config):
    for key in ("dataset_version_id", "approved_model_id", "base_weight_sha256", "split_seed", "epochs", "batch", "imgsz", "seeds"):
        if key not in config or config[key] is None or config[key] == "":
            raise ValueError(f"missing required config: {key}")
    seeds = config["seeds"]
    if not isinstance(seeds, list) or not seeds or any(type(s) is not int or s < 0 for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError("seeds must be a nonempty list of unique nonnegative integers")
    if type(config["split_seed"]) is not int or config["split_seed"] < 0:
        raise ValueError("split_seed must be a nonnegative integer")
    if not re.fullmatch(r"[0-9a-f]{64}", config["base_weight_sha256"]):
        raise ValueError("base_weight_sha256 must identify the registered trusted base")
    if not str(config.get("purpose", "")).strip():
        raise ValueError("purpose is required before reading test results")
    if config.get("profile", "formal") not in {"formal", "engineering_smoke"}:
        raise ValueError("unknown experiment profile")


def request_data(client, method, url, **kwargs):
    response = client.request(method, url, **kwargs)
    # Report status/request id only, never request headers or credentials.
    if response.is_error:
        try:
            body = response.json()
        except ValueError:
            body = {}
        raise RuntimeError(f"HTTP {response.status_code}: {body.get('code', 'request_failed')}; request_id={body.get('request_id')}")
    body = response.json()
    if not isinstance(body, dict) or "data" not in body:
        raise ValueError("invalid v1 response envelope")
    return body["data"]


def wait_job(client, api, token, job_id, timeout):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = request_data(client, "GET", f"{api}/jobs/{job_id}", headers={"Authorization": f"Bearer {token}"})
        if job["status"] in TERMINAL:
            return job
        time.sleep(2)
    raise TimeoutError(f"job {job_id} still active; no cancellation or resubmission performed")


def training_body(config, seed):
    params = {key: config[key] for key in ("epochs", "batch", "imgsz", "augmentation", "early_stopping",
        "optimizer", "lr0", "lrf", "momentum", "weight_decay", "warmup_epochs") if key in config}
    params.update(seed=seed, deterministic=config.get("deterministic", True), device=config.get("device", "0"))
    return {"dataset_version_id": config["dataset_version_id"], "model_id": config["approved_model_id"], "task": "detect", "params": params}


def snapshot_protocol(client, api, headers, config):
    validate_config(config)
    get = lambda path: request_data(client, "GET", api + path, headers=headers)
    version = get("/dataset-versions/" + config["dataset_version_id"])
    model = get("/models/" + config["approved_model_id"])
    build = get("/health/live")
    if version["state"] != "frozen" or model.get("trusted_source") != "base" or model["sha256"] != config["base_weight_sha256"]:
        raise ValueError("frozen dataset or trusted base hash mismatch")
    if version["split_config"].get("seed") != config["split_seed"]:
        raise ValueError("split seed differs from frozen dataset")
    if config.get("class_map") is not None and version["class_map"] != config["class_map"]:
        raise ValueError("class map mismatch")
    provenance = {k: build.get(k) for k in ("code_sha", "requirements_lock_sha256", "ml_lock_sha256")}
    if config.get("profile", "formal") == "formal":
        if not re.fullmatch(r"[0-9a-f]{40}", provenance.get("code_sha") or ""):
            raise ValueError("formal protocol requires an exact clean Git SHA")
        if any(not re.fullmatch(r"[0-9a-f]{64}", provenance.get(k) or "") for k in ("requirements_lock_sha256", "ml_lock_sha256")):
            raise ValueError("formal protocol requires core and ML lock hashes")
    assets, page = [], 1
    while True:
        result = get(f"/datasets/{version['dataset_id']}/assets?version_id={version['version_id']}&page={page}&page_size=200")
        assets.extend(result["items"])
        if len(assets) >= result["total"]:
            break
        if not result["items"]:
            raise ValueError("incomplete asset pagination")
        page += 1
    split_hashes = {}
    for split in ("train", "val", "test"):
        records = [{k: a[k] for k in ("relative_path", "sha256", "source_group", "annotations")} for a in assets if a["split"] == split]
        if not records or len(records) != version["split_counts"].get(split):
            raise ValueError(f"empty or inconsistent {split} split")
        split_hashes[split] = config_hash(sorted(records, key=lambda a: a["relative_path"]))
    # This digest includes image content, groups and annotations, not just names.
    if config.get("test_manifest_sha256") and config["test_manifest_sha256"] != split_hashes["test"]:
        raise ValueError("test snapshot hash mismatch")
    preflights = []
    for seed in config["seeds"]:
        report = request_data(client, "POST", api + "/training/preflight", headers=headers, json=training_body(config, seed))
        if not report["ok"]:
            raise ValueError(f"training preflight failed for seed {seed}: {report['checks']}")
        preflights.append({"seed": seed, "params": report["params"]})
    return {"config": config, "version": version, "base": {"model_id": model["model_id"], "sha256": model["sha256"]},
            "build": provenance, "split_asset_sha256": split_hashes, "effective_requests": preflights,
            "test_digest_definition": "canonical JSON sorted by relative_path: relative_path, sha256, source_group, annotations"}


def require_metrics(metrics):
    if any(type(metrics.get(key)) not in (int, float) or not math.isfinite(metrics[key]) for key in METRICS):
        raise ValueError("missing/nonfinite required metrics; result is not successful")
    if metrics.get("confusion_matrix") is None or not metrics.get("per_class"):
        raise ValueError("missing real confusion matrix or per-class metrics")


def summarize(runs, seeds):
    complete = [r for r in runs if r.get("status") == "completed"]
    metrics = {}
    for key in METRICS:
        values = [r["evaluation"]["metrics"][key] for r in complete]
        metrics[key] = {"n": len(values), "values": values, "mean": statistics.mean(values) if values else None,
                        "std": statistics.stdev(values) if len(values) > 1 else None}
    return {"status": "completed" if len(complete) == len(seeds) and seeds else "partial" if complete else "failed",
            "expected_seeds": seeds, "completed_seeds": [r["seed"] for r in complete], "runs": runs,
            "test_metrics": metrics, "notes": ["Independent fixed test evaluations; sample standard deviation uses n-1.",
            "Interrupted/timeout tasks require explicit recovery; rerun reuses the same idempotency keys."]}


def execute(client, api, token, protocol, output, digest, timeout, retry_cancelled=False):
    config = protocol["config"]
    headers = {"Authorization": f"Bearer {token}"}
    runs = []
    for seed in config["seeds"]:
        record = {"seed": seed, "status": "pending"}
        target = output / f"seed-{seed}.json"
        try:
            saved = json.loads(target.read_text(encoding='utf-8')) if target.exists() else {}
            base_key = f"exp-{digest}-seed{seed}-train"
            key = saved.get('training_idempotency_key', base_key)
            submit = lambda request_key: request_data(client, "POST", api + "/training/jobs", headers={**headers, "Idempotency-Key": request_key}, json=training_body(config, seed))
            job = submit(key)
            if job['status']=='cancelled' and retry_cancelled:
                # Explicit operator action only. A new job keeps old logs and
                # checkpoints intact; the frozen experiment parameters do not
                # change. Never retry failed, interrupted or ambiguous jobs here.
                archive=output/f"seed-{seed}.cancelled-{job['job_id']}.json"
                if not archive.exists():write_json(archive,{'previous_record':saved,'cancelled_job':job,'reason':'explicit cancelled-seed retry'})
                record['retry_of_training_job_id']=job['job_id']
                key=base_key+'-retry-'+job['job_id'].replace('-','')[:16]
                record.update(status='submitting_retry',training_idempotency_key=key)
                write_json(target,record)  # an ambiguous POST reuses this key
                job=submit(key)
            record['training_idempotency_key']=key
            record.update(training_job_id=job["job_id"], status=job["status"])
            write_json(target, record)
            job = wait_job(client, api, token, job["job_id"], timeout)
            record["training"] = job
            if job["status"] != "completed":
                record["status"] = job["status"]
                raise ValueError("training did not complete")
            result = job["result"]
            if result.get("backend") != "ultralytics":
                raise ValueError("real experiments require ultralytics backend")
            environment = result.get("environment") or {}
            for key, expected in protocol["build"].items():
                actual = environment.get(key) if key == "code_sha" else environment.get("locks", {}).get(key)
                if expected != actual:
                    raise ValueError(f"worker provenance differs from frozen protocol: {key}")
            evaluation = request_data(client, "POST", api + "/evaluations", headers=headers, json={
                "model_id": result["model_id"], "dataset_version_id": config["dataset_version_id"], "split": "test",
                "params": {"imgsz": config["imgsz"], "batch": config["batch"], "device": config.get("device", "0")},
                "purpose": config["purpose"] + "; protocol=" + digest,
                "idempotency_key": f"exp-{digest}-seed{seed}-test"})
            record.update(evaluation_id=evaluation["evaluation"]["evaluation_id"], evaluation_job_id=evaluation["job"]["job_id"])
            write_json(target, record)
            evaluation_job = wait_job(client, api, token, record["evaluation_job_id"], timeout)
            record["evaluation_job"] = evaluation_job
            if evaluation_job["status"] != "completed":
                record["status"] = evaluation_job["status"]
                raise ValueError("test evaluation did not complete")
            record["evaluation"] = request_data(client, "GET", api + "/evaluations/" + record["evaluation_id"], headers=headers)
            require_metrics(record["evaluation"]["metrics"])
            record["status"] = "completed"
        except TimeoutError as exc:
            record.update(status="timeout", error=str(exc))
        except Exception as exc:
            if record["status"] not in {"failed", "cancelled", "interrupted"}:
                record["status"] = "failed"
            record["error"] = f"{type(exc).__name__}: {exc}"
        write_json(target, record)
        runs.append(record)
        write_json(output / "summary.json", {"protocol_sha256": digest, **summarize(runs, config["seeds"])})
        # Do not stack additional long jobs behind an unresolved owner.
        if record["status"] in {"timeout", "interrupted", "cancelled", "failed"}:
            break
    return summarize(runs, config["seeds"])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default=os.environ.get("BYSJ_BASE_URL", "http://127.0.0.1:8000"))
    parser.add_argument("--timeout", type=float, default=7200)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--retry-cancelled", action="store_true", help="Explicitly start fresh jobs only for server-confirmed cancelled seeds; retain old evidence")
    args = parser.parse_args()
    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        validate_config(config)
        username, password = os.environ.get("BYSJ_USER"), os.environ.get("BYSJ_PASSWORD")
        if not username or not password:
            raise ValueError("BYSJ_USER and BYSJ_PASSWORD are required")
        api = args.base_url.rstrip("/") + "/api/v1"
        with httpx.Client(timeout=120, trust_env=False) as client:
            token = request_data(client, "POST", api + "/auth/login", json={"username": username, "password": password})["token"]
            protocol = snapshot_protocol(client, api, {"Authorization": f"Bearer {token}"}, config)
            output, digest = freeze_protocol(args.output, protocol)
            if args.prepare_only:
                print(json.dumps({"status": "prepared", "protocol_sha256": digest, "output": str(output)}))
                return 0
            summary = execute(client, api, token, protocol, output, digest, args.timeout, retry_cancelled=args.retry_cancelled)
            print(json.dumps({"status": summary["status"], "protocol_sha256": digest, "output": str(output), "completed_seeds": summary["completed_seeds"]}))
            return 0 if summary["status"] == "completed" else 1
    except Exception as exc:
        print(json.dumps({"status": "rejected", "error": f"{type(exc).__name__}: {exc}"}, ensure_ascii=False))
        return 2


if __name__ == "__main__":
    sys.exit(main())
