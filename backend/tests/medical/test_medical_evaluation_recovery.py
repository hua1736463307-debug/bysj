from datetime import timedelta
import uuid

import pytest

from app.core.errors import ApiError
from app.db.base import utcnow
from app.db.models import EvaluationRun, JobTask
from app.domains import job_service


def evaluation(owner, purpose=None, identifier=None):
    return EvaluationRun(id=identifier, owner_id=owner, model_id="model", dataset_version_id="version",
                         split="val", purpose=purpose, params_json={"task": "fixture"}, context_hash="hash",
                         status="pending")


@pytest.mark.parametrize("job_type", ["cxr_evaluate", "isic_evaluate"])
def test_legacy_random_evaluation_id_is_reused(db, researcher, job_type):
    legacy = evaluation(researcher.id, identifier=str(uuid.uuid4()))
    db.add(legacy)
    db.commit()
    payload = {"model_id": "model", "dataset_version_id": "version", "split": "val"}
    original, _ = job_service.create_job(db, job_type=job_type, owner_id=researcher.id,
        owner_username=researcher.username, payload={**payload, "evaluation_id": legacy.id},
        idempotency_key="legacy-request")
    retried, created = job_service.create_evaluation_job(db, job_type=job_type, owner_id=researcher.id,
        owner_username=researcher.username, evaluation=evaluation(researcher.id), payload=payload,
        idempotency_key="legacy-request")
    assert not created and retried.id == original.id
    assert db.query(EvaluationRun).count() == 1
    db.refresh(legacy)
    assert legacy.job_id == original.id
    with pytest.raises(ApiError, match="不同评价请求"):
        job_service.create_evaluation_job(db, job_type=job_type, owner_id=researcher.id,
            owner_username=researcher.username, evaluation=evaluation(researcher.id),
            payload={**payload, "split": "test"}, idempotency_key="legacy-request")
    assert db.query(EvaluationRun).count() == 1


def test_isic_evaluation_state_tracks_pending_cancel_and_stale_resume(db, researcher):
    payload = {"model_id": "model", "dataset_version_id": "version", "split": "val"}
    def create(key):
        return job_service.create_evaluation_job(db, job_type="isic_evaluate", owner_id=researcher.id,
            owner_username=researcher.username, evaluation=evaluation(researcher.id), payload=payload,
            idempotency_key=key)[0]
    cancelled = create("pending-cancel")
    job_service.cancel_job(db, cancelled)
    assert db.get(EvaluationRun, cancelled.input_json['evaluation_id']).status == 'cancelled'
    interrupted = create("stale")
    interrupted.status = 'running'
    interrupted.lease_owner = 'stale-worker'
    interrupted.lease_expires_at = utcnow() - timedelta(hours=1)
    interrupted.heartbeat_at = utcnow() - timedelta(hours=1)
    db.commit()
    assert job_service.mark_stale_interrupted(db, job_id=interrupted.id) == 1
    run = db.get(EvaluationRun, interrupted.input_json['evaluation_id'])
    assert run.status == 'interrupted' and run.finished_at is not None
    job_service.pause_for_resume(db, interrupted)
    db.refresh(run)
    assert run.status == 'pending' and run.finished_at is None


def test_retry_discards_pending_run_when_peer_wins_between_lookups(db, researcher, monkeypatch):
    from app.db.session import get_session_factory
    original = job_service.create_job
    def committed_peer(session, **kwargs):
        with get_session_factory()() as peer:
            peer.add(evaluation(researcher.id, identifier=kwargs['payload']['evaluation_id']))
            original(peer, **kwargs)
        return original(session, **kwargs)
    monkeypatch.setattr(job_service, 'create_job', committed_peer)
    db.autoflush = True
    job, created = job_service.create_evaluation_job(db, job_type='isic_evaluate', owner_id=researcher.id,
        owner_username=researcher.username, evaluation=evaluation(researcher.id),
        payload={'model_id': 'model', 'dataset_version_id': 'version', 'split': 'val'},
        idempotency_key='peer-wins')
    assert not created
    assert db.query(EvaluationRun).count() == db.query(JobTask).count() == 1
    assert db.query(EvaluationRun).one().job_id == job.id
