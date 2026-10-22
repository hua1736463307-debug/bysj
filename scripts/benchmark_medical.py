#!/usr/bin/env python3
"""Measure fixed-input inference with separate cold-model and warm-model samples.

Warmup images are prepended to the SAME run/model handle and excluded from warm
statistics. This is not an OS-cache or process cold-start measurement. Every
requested input must exist and every response page is collected.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
import sys
import time

import httpx
from run_experiments import request_data, write_json, freeze_protocol


def percentile(values, ratio):
    if not values:
        return None
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, max(0, math.ceil(ratio * len(ordered)) - 1))]


def describe(values):
    values = [v for v in values if type(v) in (int, float) and math.isfinite(v)]
    return {"n": len(values), "mean": statistics.mean(values) if values else None,
            "p50": percentile(values, .5), "p95": percentile(values, .95), "max": max(values) if values else None}


def select_inputs(config):
    manifest_path = Path(config["input_manifest"]).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    root = (manifest_path.parent / "images").resolve()
    count = config.get("count", 100)
    if type(count) is not int or count < 1 or count > len(manifest["assets"]):
        raise ValueError("requested count must be positive and available; no silent truncation")
    paths, hashes = [], []
    for item in manifest["assets"][:count]:
        path = (root / item["relative_path"]).resolve()
        path.relative_to(root)
        if not path.is_file():
            raise ValueError(f"missing input: {item['relative_path']}")
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        expected = item.get("sha256") or item.get("content_hash")
        if expected and expected != digest:
            raise ValueError(f"input content changed: {item['relative_path']}")
        paths.append(path)
        hashes.append({"relative_path": item["relative_path"], "sha256": digest})
    return paths, hashes


def wait_run(client, api, token, run_id, timeout):
    headers = {"Authorization": f"Bearer {token}"}
    url = api + "/inference/runs/" + run_id
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        run = request_data(client, "GET", url, headers=headers, params={"page": 1, "page_size": 200})
        if run["status"] in {"completed", "partial", "failed", "cancelled", "interrupted"}:
            items = list(run["items"])
            page = 2
            while len(items) < run["total"]:
                result = request_data(client, "GET", url, headers=headers, params={"page": page, "page_size": 200})
                if not result["items"]:
                    raise ValueError("incomplete run pagination")
                items.extend(result["items"])
                page += 1
            if len(items) != run["total"] or len({i["item_id"] for i in items}) != len(items):
                raise ValueError("duplicate or missing inference items")
            return {**run, "items": items}
        time.sleep(.5)
    raise TimeoutError(f"inference run {run_id}; active job was not cancelled")


def benchmark(client, api, token, config, paths, output):
    headers = {"Authorization": f"Bearer {token}"}
    raw, batches = [], []
    warmup_count = config.get("warmup_count", 5)
    repeats = config.get("repeat_count", 3)
    if type(warmup_count) is not int or not 1 <= warmup_count <= len(paths) or type(repeats) is not int or repeats < 1:
        raise ValueError("warmup_count must be 1..count and repeat_count must be positive")
    max_batch = config.get("max_batch", 100)
    if type(max_batch) is not int or not warmup_count < max_batch <= 100:
        raise ValueError("max_batch must fit warmup plus samples and the default API limit of 100")
    capacity = max_batch - warmup_count
    params = {"device": config.get("device", "0"), "imgsz": config.get("imgsz", 640),
              "confidence": config.get("confidence", .25), "iou": config.get("iou", .45)}
    for repeat in range(repeats):
        # One fresh handle for the cold sample; a separate handle warms within its
        # own batch. Warmup from a previous request would not warm this handle.
        batches_to_run = [("cold_model", paths[:1], 0)] + [
            ("warm_model", paths[:warmup_count] + paths[start:start+capacity], warmup_count)
            for start in range(0, len(paths), capacity)]
        for batch_index, (phase, batch, skip) in enumerate(batches_to_run):
            files = [("files", (f"{i:04d}-{p.name}", p.read_bytes(), "application/octet-stream")) for i, p in enumerate(batch)]
            start = time.perf_counter()
            accepted = request_data(client, "POST", api + "/inference/images", headers=headers,
                                    data={"model_id": config["model_id"], "params": json.dumps(params)}, files=files)
            accepted_ms = (time.perf_counter() - start) * 1000
            run = wait_run(client, api, token, accepted["run_id"], config.get("timeout_seconds", 300))
            wall_ms = (time.perf_counter() - start) * 1000
            write_json(output / f"{phase}-{repeat}-{batch_index}.json", run)
            if len(run["items"]) != len(batch):
                raise ValueError("requested input count does not match stored results")
            batches.append({"repeat": repeat, "phase": phase, "run_id": run["run_id"], "wall_ms": wall_ms,
                            "accepted_ms": accepted_ms, "requested": len(batch), "warmup_excluded": skip,
                            "completed": run["completed"], "failed": run["failed"], "status": run["status"]})
            for index, item in enumerate(run["items"]):
                raw.append({"repeat": repeat, "phase": phase, "warmup": index < skip, "item_id": item["item_id"],
                            "run_id": run["run_id"], "status": item["status"], **item.get("timing", {})})
            write_json(output / "raw-timings.json", raw)
            write_json(output / "batches.json", batches)
            if run["status"] in {"cancelled", "interrupted"}:
                raise RuntimeError(f"run {run['status']}; stop issuing more work")
    measured = [r for r in raw if not r["warmup"]]
    fields = sorted({k for r in raw for k in r})
    with (output / "raw-timings.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(raw)
    summaries = {}
    for phase in ("cold_model", "warm_model"):
        rows = [r for r in measured if r["phase"] == phase]
        good = [r for r in rows if r["status"] == "completed"]
        summaries[phase] = {"samples": len(rows), "success": len(good), "failure_rate": 1-len(good)/len(rows) if rows else None,
            "latency_ms": {key: describe([r.get(key) for r in good]) for key in
                           ("queue_ms", "gpu_wait_ms", "load_ms", "preprocess_ms", "model_ms", "postprocess_ms", "storage_ms", "processing_ms", "total_ms")},
            "peak_gpu_allocated_bytes": max((r.get("peak_gpu_allocated_bytes") or 0 for r in good), default=0) or None,
            "peak_gpu_reserved_bytes": max((r.get("peak_gpu_reserved_bytes") or 0 for r in good), default=0) or None}
    summary = {"status": "completed" if measured and all(r["status"] == "completed" for r in raw) else "partial",
        "samples": len(measured), "phases": summaries, "concurrency": 1,
        "api_acceptance_ms": describe([b["accepted_ms"] for b in batches]),
        "batch_throughput_including_warmup_per_second": sum(b["completed"] for b in batches)/(sum(b["wall_ms"] for b in batches)/1000),
        "notes": ["Warmup excluded from sample statistics but included in explicitly labelled batch throughput.",
                  "total_ms is cumulative accepted-run to item persistence-start latency, not kernel time.",
                  "Cold model means a fresh model handle; process and OS caches are not reset.",
                  "CUDA allocated/reserved peaks are allocator measurements, not device-wide nvidia-smi memory."],
        "not_measured": {"peak_rss_bytes": "no process sampler", "gpu_utilization": "no device sampler",
                         "device_memory_peak": "no device sampler", "process_cold_start": "worker remains running"}}
    write_json(output / "summary.json", summary)
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True); parser.add_argument("--output", required=True)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    args = parser.parse_args()
    try:
        config = json.loads(Path(args.config).read_text(encoding="utf-8"))
        if not config.get("model_id") or not config.get("input_manifest"):
            raise ValueError("model_id and input_manifest are required")
        paths, hashes = select_inputs(config)
        user, password = os.environ.get("BYSJ_USER"), os.environ.get("BYSJ_PASSWORD")
        if not user or not password:
            raise ValueError("BYSJ_USER/BYSJ_PASSWORD are required")
        api = args.base_url.rstrip("/") + "/api/v1"
        with httpx.Client(timeout=120, trust_env=False) as client:
            token = request_data(client, "POST", api + "/auth/login", json={"username": user, "password": password})["token"]
            model = request_data(client, "GET", api + "/models/" + config["model_id"], headers={"Authorization": f"Bearer {token}"})
            output, digest = freeze_protocol(args.output, {"config": config, "inputs": hashes, "model_sha256": model["sha256"]})
            # Measurements are append-only executions under the frozen protocol.
            output = output / str(time.time_ns()); output.mkdir()
            summary = benchmark(client, api, token, config, paths, output)
            print(json.dumps({"status": summary["status"], "output": str(output), "protocol_sha256": digest}))
            return 0 if summary["status"] == "completed" else 1
    except Exception as exc:
        print(json.dumps({"status": "failed", "error": f"{type(exc).__name__}: {exc}"}))
        return 2


if __name__ == "__main__":
    sys.exit(main())
