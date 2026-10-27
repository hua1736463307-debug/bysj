"""CXR 离线工具：prepare_chexpert 审计与 run_cxr_experiments 协议冻结。"""
from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
PYTHON = sys.executable


def _fixture(root: Path, patients: int = 12) -> Path:
    subprocess.run(
        [PYTHON, str(REPO_ROOT / "scripts" / "make_cxr_fixture.py"),
         "--output", str(root), "--patients", str(patients), "--size", "128"],
        check=True, capture_output=True, text=True,
    )
    return root


def _run_prepare(source: Path, output: Path, *extra: str) -> dict:
    result = subprocess.run(
        [PYTHON, str(REPO_ROOT / "scripts" / "prepare_chexpert.py"),
         "--source-root", str(source), "--source-csv", "train.csv",
         "--output", str(output), "--max-patients", "12", *extra],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    payload = json.loads(result.stdout)
    return payload


def test_dry_run_plans_without_writing(tmp_path):
    source = _fixture(tmp_path / "src")
    output = tmp_path / "out"
    payload = _run_prepare(source, output, "--dry-run")
    assert payload["status"] == "PLAN_READY"
    assert payload["dry_run"] is True
    assert not (output / "prepare-report.json").exists()
    assert payload["audit"]["missing_label_columns"] == []
    assert payload["integrity"]["ok"] is True
    assert payload["selection"]["selected_patients"] == 12
    # 侧位被跳过而不是混入
    assert payload["selection"]["lateral_skipped"] >= 1


def test_full_run_writes_selection_and_report(tmp_path):
    source = _fixture(tmp_path / "src")
    output = tmp_path / "out"
    payload = _run_prepare(source, output)
    assert payload["status"] == "PLAN_WRITTEN"
    report = json.loads((output / "prepare-report.json").read_text(encoding="utf-8"))
    assert report["stats"]["manifest_sha256"]
    selection = (output / "selection.csv").read_text(encoding="utf-8").strip().splitlines()
    assert len(selection) == payload["selection"]["selected_images"] + 1
    header = selection[0].split(",")
    assert header[:4] == ["source_path", "patient_key", "study_key", "split"]


def test_missing_label_column_is_rejected(tmp_path):
    source = _fixture(tmp_path / "src")
    csv_path = source / "train.csv"
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    lines = [line.replace(",Edema", "") for line in lines]
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = _run_prepare(source, tmp_path / "out", "--dry-run")
    assert payload["status"] == "REJECTED"
    assert payload["reason"] == "missing_label_columns"
    assert "Edema" in payload["audit"]["missing_label_columns"]


def test_unparsable_identity_rows_are_rejected_not_guessed(tmp_path):
    source = _fixture(tmp_path / "src")
    csv_path = source / "train.csv"
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    lines.append("weird/unknown/path.jpg,Frontal,1,0,0,0,0")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    payload = _run_prepare(source, tmp_path / "out", "--dry-run")
    assert payload["status"] == "PLAN_READY"
    reasons = {item["reason"] for item in payload["selection"]["rejected_rows"]}
    assert "identity_unparsable" in reasons


def test_path_traversal_in_csv_is_refused(tmp_path):
    source = _fixture(tmp_path / "src")
    csv_path = source / "train.csv"
    lines = csv_path.read_text(encoding="utf-8").splitlines()
    lines.append("../../escape/patient99999/study1/view1_frontal.jpg,Frontal,1,0,0,0,0")
    csv_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    result = subprocess.run(
        [PYTHON, str(REPO_ROOT / "scripts" / "prepare_chexpert.py"),
         "--source-root", str(source), "--source-csv", "train.csv",
         "--output", str(tmp_path / "out"), "--dry-run"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert result.returncode != 0
    assert "invalid_path" in result.stderr or "越界" in result.stderr or "上级目录" in result.stderr


def test_runner_prepare_only_freezes_protocol_without_http(tmp_path):
    config = json.loads((REPO_ROOT / "configs" / "cxr" / "pilot.json").read_text(encoding="utf-8"))
    config["dataset_version_id"] = "11111111-1111-1111-1111-111111111111"
    config["base_model_id"] = "22222222-2222-2222-2222-222222222222"
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    output = tmp_path / "run"
    result = subprocess.run(
        [PYTHON, str(REPO_ROOT / "scripts" / "run_cxr_experiments.py"),
         "--config", str(config_path), "--output", str(output), "--prepare-only",
         "--base-url", "http://127.0.0.1:9"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 0, result.stderr
    payload = json.loads(result.stdout)
    assert payload["status"] == "PROTOCOL_FROZEN"
    assert payload["combinations"] == 6
    protocol_files = list(output.rglob("protocol.json"))
    assert len(protocol_files) == 1
    protocol = json.loads(protocol_files[0].read_text(encoding="utf-8"))
    assert protocol["label_schema"] == "cxr-five-v1"
    assert protocol["task"] == "cxr_multilabel"
    assert len(protocol["protocol_hash"]) == 64


def test_runner_rejects_invalid_config(tmp_path):
    config_path = tmp_path / "bad.json"
    config_path.write_text(json.dumps({"name": "bad", "seeds": [42, 42]}), encoding="utf-8")
    result = subprocess.run(
        [PYTHON, str(REPO_ROOT / "scripts" / "run_cxr_experiments.py"),
         "--config", str(config_path), "--output", str(tmp_path / "out"), "--prepare-only"],
        capture_output=True, text=True, cwd=str(REPO_ROOT),
    )
    assert result.returncode == 2
    payload = json.loads(result.stdout)
    assert payload["status"] == "CONFIG_INVALID"
    assert any("seeds" in item for item in payload["problems"])
