import importlib.util
import json
from pathlib import Path
import sys

import httpx
import pytest

sys.path.insert(0, str(Path(__file__).parents[1]))
import run_experiments as runner
import benchmark_medical as benchmark


def config():
    return {"dataset_version_id": "v", "approved_model_id": "m", "base_weight_sha256": "a"*64,
            "split_seed": 0, "epochs": 1, "batch": 2, "imgsz": 64, "seeds": [0], "purpose": "engineering",
            "augmentation": {"mosaic": 0}, "early_stopping": {"patience": 0, "monitor": "val/mAP50-95"}}


def test_seed_zero_and_all_parameters_survive():
    value = {**config(), "optimizer":"SGD", "lr0":.002, "weight_decay":.0001}; runner.validate_config(value)
    params = runner.training_body(value, 0)["params"]
    assert params["seed"] == 0
    assert params["augmentation"] == value["augmentation"]
    assert params["early_stopping"] == value["early_stopping"]
    assert params['optimizer']=='SGD' and params['lr0']==.002 and params['weight_decay']==.0001


@pytest.mark.parametrize("change", [{"seeds": []}, {"seeds": [1, 1]}, {"approved_model_id": None}, {"base_weight_sha256": "bad"}])
def test_invalid_protocol_rejected(change):
    with pytest.raises(ValueError): runner.validate_config({**config(), **change})


def test_frozen_protocol_cannot_be_overwritten(tmp_path):
    first, digest = runner.freeze_protocol(tmp_path, config())
    assert runner.freeze_protocol(tmp_path, config()) == (first, digest)
    second, _ = runner.freeze_protocol(tmp_path, {**config(), "epochs": 2})
    assert first != second
    (first / "config.frozen.json").write_text("tampered")
    with pytest.raises(ValueError): runner.freeze_protocol(tmp_path, config())


def test_interrupted_wait_stops_immediately():
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(200, json={"data": {"status": "interrupted"}}))) as client:
        assert runner.wait_job(client, "http://test/api", "secret", "job", 5)["status"] == "interrupted"


def test_expired_auth_fails_without_leaking_credentials():
    with httpx.Client(transport=httpx.MockTransport(lambda _: httpx.Response(401, json={"code": "unauthorized"}))) as client:
        with pytest.raises(RuntimeError, match="HTTP 401") as error:
            runner.wait_job(client, "http://test/api", "never-log-this", "job", 5)
        assert "never-log-this" not in str(error.value)


@pytest.mark.parametrize("metrics", [{}, {"precision": float("nan"), "recall": .1, "map50": .2, "map50_95": .1},
                                      {"precision": .1, "recall": .1, "map50": .2, "map50_95": .1}])
def test_missing_metrics_never_pass(metrics):
    with pytest.raises(ValueError): runner.require_metrics(metrics)


def test_empty_runs_are_failed():
    report = runner.summarize([], [0])
    assert report["status"] == "failed"
    assert report["test_metrics"]["map50"]["mean"] is None


def test_data_quality_block_prevents_even_cancelled_retry(tmp_path):
    (tmp_path/'BLOCKED.json').write_text('{"reason":"cross-split similarity"}')
    with pytest.raises(ValueError,match='protocol blocked'):
        runner.execute(None,'http://unused','token',{},tmp_path,'digest',1,retry_cancelled=True)


def test_replay_reuses_training_and_test_keys(tmp_path):
    keys = []; calls = []
    metrics = {**{k: .5 for k in runner.METRICS}, "per_class": {"0": {}}, "confusion_matrix": [[1, 0], [0, 0]]}
    def transport(request):
        calls.append(request.url.path)
        body = json.loads(request.content) if request.content else {}
        if request.url.path.endswith("/training/jobs"):
            keys.append(request.headers["Idempotency-Key"]); data = {"job_id": "train", "status": "completed"}
        elif request.url.path.endswith("/jobs/train"):
            data = {"job_id": "train", "status": "completed", "result": {"model_id": "trained", "backend": "ultralytics", "environment": {}}}
        elif request.url.path.endswith("/evaluations"):
            assert body["split"] == "test"; keys.append(body["idempotency_key"])
            data = {"evaluation": {"evaluation_id": "evaluation"}, "job": {"job_id": "test"}}
        elif request.url.path.endswith("/jobs/test"):
            data = {"job_id": "test", "status": "completed"}
        else: data = {"metrics": metrics}
        return httpx.Response(200, json={"data": data})
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        for _ in range(2):
            result = runner.execute(client, "http://test/api", "token", {"config": config(), "build": {}}, tmp_path, "digest", 10)
            assert result["status"] == "completed"
    assert keys[0:2] == keys[2:4]
    assert keys[0] != keys[1]


def test_benchmark_fetches_all_pages():
    def transport(request):
        page = int(request.url.params["page"])
        return httpx.Response(200, json={"data": {"status": "completed", "total": 3,
                    "items": [{"item_id": str(i)} for i in ([1, 2] if page == 1 else [3])]}})
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        assert len(benchmark.wait_run(client, "http://test/api", "token", "run", 10)["items"]) == 3


def test_missing_input_not_silently_dropped(tmp_path):
    manifest = tmp_path / "manifest.json"
    manifest.write_text(json.dumps({"assets": [{"relative_path": "missing.jpg"}]}))
    with pytest.raises(ValueError, match="missing input"):
        benchmark.select_inputs({"input_manifest": str(manifest), "count": 1})


@pytest.mark.parametrize('explicit',[False,True])
def test_cancelled_seed_requires_explicit_retry_and_preserves_prior_evidence(tmp_path,explicit):
    keys=[]
    metrics={**{k:.5 for k in runner.METRICS},'per_class':{'0':{}},'confusion_matrix':[[1,0],[0,0]]}
    def transport(request):
        body=json.loads(request.content) if request.content else {}
        if request.url.path.endswith('/training/jobs'):
            key=request.headers['Idempotency-Key'];keys.append(key)
            data={'job_id':'new' if '-retry-' in key else 'old-cancelled','status':'completed' if '-retry-' in key else 'cancelled'}
        elif request.url.path.endswith('/jobs/old-cancelled'):data={'job_id':'old-cancelled','status':'cancelled'}
        elif request.url.path.endswith('/jobs/new'):data={'job_id':'new','status':'completed','result':{'model_id':'trained','backend':'ultralytics','environment':{}}}
        elif request.url.path.endswith('/evaluations'):data={'evaluation':{'evaluation_id':'evaluation'},'job':{'job_id':'test'}}
        elif request.url.path.endswith('/jobs/test'):data={'job_id':'test','status':'completed'}
        else:data={'metrics':metrics}
        return httpx.Response(200,json={'data':data})
    with httpx.Client(transport=httpx.MockTransport(transport)) as client:
        result=runner.execute(client,'http://test/api','token',{'config':config(),'build':{}},tmp_path,'digest',10,retry_cancelled=explicit)
        assert result['status']==('completed' if explicit else 'failed')
        if explicit:
            assert list(tmp_path.glob('seed-0.cancelled-*.json'))
            assert len(keys)==2 and '-retry-' in keys[-1]
            runner.execute(client,'http://test/api','token',{'config':config(),'build':{}},tmp_path,'digest',10)
            assert keys[-1]==keys[-2]  # resumed execution reuses new job key
        else:assert len(keys)==1
