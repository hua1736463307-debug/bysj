from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest

from app.core.errors import ApiError
from app.db.models import EvaluationRun, JobTask
from app.domains import job_service, auth_service


@pytest.mark.parametrize("job_type", ["cxr_evaluate", "isic_evaluate"])
@pytest.mark.parametrize("different", [False, True])
def test_concurrent_medical_evaluation_retries_are_atomic(sessions, monkeypatch, job_type, different):
    with sessions() as db:
        owner = auth_service.create_user(db, username='evaluator', password='integration-password').id
    original = job_service.create_job
    barrier = Barrier(2)
    def race(*args, **kwargs):
        barrier.wait(timeout=15)
        return original(*args, **kwargs)
    monkeypatch.setattr(job_service, 'create_job', race)
    def create(purpose):
        with sessions() as db:
            run = EvaluationRun(owner_id=owner, model_id='model', dataset_version_id='version', split='val',
                                purpose=purpose, params_json={}, context_hash='hash', status='pending')
            payload = {'model_id': 'model', 'dataset_version_id': 'version', 'split': 'val', 'purpose': purpose}
            try:
                job, created = job_service.create_evaluation_job(db, job_type=job_type, owner_id=owner,
                    owner_username='evaluator', evaluation=run, payload=payload, idempotency_key='same-key')
                return job.id, created
            except ApiError as error:
                return error.code, False
    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(create, ['first', 'second' if different else 'first']))
    with sessions() as db:
        assert db.query(EvaluationRun).count() == db.query(JobTask).count() == 1
        run, job = db.query(EvaluationRun).one(), db.query(JobTask).one()
        assert run.job_id == job.id and job.input_json['evaluation_id'] == run.id
    assert sum(created for _, created in outcomes) == 1
    if different:
        assert any(identifier == 'idempotency_conflict' for identifier, _ in outcomes)
    else:
        assert outcomes[0][0] == outcomes[1][0]
