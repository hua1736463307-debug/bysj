import copy
from pathlib import Path
from PIL import Image
import support
from app.db.models import DatasetVersion, DatasetAsset, ModelRegistry
from app.domains.datasets.derivative_service import transform


def frozen(client, headers, prefix="image"):
    dataset, version = support.make_dataset(client, headers, count=9, seed_prefix=prefix)
    response = support.freeze(client, headers, dataset, version_id=version)
    assert response.json()['data']['status'] == 'completed'
    return dataset, version


def test_augmentation_preserves_holdouts_and_is_idempotent(client, researcher_headers, db):
    dataset, version_id = frozen(client, researcher_headers)
    before = db.query(DatasetAsset).filter_by(dataset_version_id=version_id).all()
    snapshots = {row.id: (row.content_hash, copy.deepcopy(row.annotations_json)) for row in before}
    body = {'operation': 'augment', 'version_ids': [version_id], 'rotations': [90], 'flip': True}
    headers = {**researcher_headers, 'Idempotency-Key': 'augment-once'}
    response = client.post('/api/v1/dataset-derivatives', headers=headers, json=body)
    assert response.status_code == 202, response.text
    job = response.json()['data']
    assert job['status'] == 'completed', job
    child_id = job['result']['version_id']
    rows = db.query(DatasetAsset).filter_by(dataset_version_id=child_id).all()
    train_count = sum(row.split == 'train' for row in before)
    assert len(rows) == len(before) + 2 * train_count
    for row in rows:
        if row.split in {'val', 'test'}:
            assert (row.content_hash, row.annotations_json) == snapshots[row.original_asset_id]
    retry = client.post('/api/v1/dataset-derivatives', headers=headers, json=body).json()['data']
    assert retry['job_id'] == job['job_id']
    db.expire_all()
    assert db.get(DatasetVersion, version_id).state == 'frozen'


def test_merge_rejects_draft_and_keeps_sources(client, researcher_headers, db):
    dataset, version = frozen(client, researcher_headers)
    from app.domains.datasets.service import create_annotation_version
    from app.api.deps import CurrentUser
    from app.core.config import get_settings
    user = client.get('/api/v1/auth/me', headers=researcher_headers).json()['data']
    draft = create_annotation_version(db, get_settings(), dataset, version, CurrentUser(user['id'], user['username'], user['role'], user['role']))
    response = client.post('/api/v1/dataset-derivatives', headers={**researcher_headers, 'Idempotency-Key': 'bad-merge'},
                           json={'operation': 'merge', 'version_ids': [version, draft.id]})
    assert response.status_code == 409
    assert db.get(DatasetVersion, version).state == 'frozen'


def test_ai_annotation_creates_reviewable_draft(client, researcher_headers, admin_headers, db, settings):
    _, version = frozen(client, researcher_headers)
    model_file = settings.model_root / 'test-ai.pt'
    model_file.parent.mkdir(parents=True, exist_ok=True)
    model_file.write_bytes(b'dummy-model')
    from app.core.storage import sha256_file
    user = client.get('/api/v1/auth/me', headers=researcher_headers).json()['data']
    model = ModelRegistry(id='ai-model', name='AI fixture', path=str(model_file), sha256=sha256_file(model_file),
                          owner_id=user['id'], trusted_source='platform', model_task='detect', class_map_json={'0': 'RBC'})
    db.add(model)
    db.commit()
    response = client.post('/api/v1/dataset-derivatives', headers={**researcher_headers, 'Idempotency-Key': 'ai-once'},
                           json={'operation': 'preannotate', 'version_ids': [version], 'model_id': model.id})
    assert response.status_code == 202, response.text
    job = response.json()['data']
    assert job['status'] == 'completed', job
    child = db.get(DatasetVersion, job['result']['version_id'])
    assert child.state == 'draft'
    assert child.metadata_json['requires_annotation_review']
    assert child.metadata_json['ai_backend'] == 'dummy'
    assert child.metadata_json['synthetic'] is True and child.metadata_json['research_valid'] is False
    freeze = support.freeze(client, researcher_headers, child.dataset_id, version_id=child.id)
    assert freeze.json()['data']['status'] == 'failed'
    assert freeze.json()['data']['error']['code'] == 'annotation_review_required'
    assert client.post('/api/v1/dataset-versions/' + child.id + '/annotation-review', headers=admin_headers).status_code == 200
    reviewed = support.freeze(client, researcher_headers, child.dataset_id, version_id=child.id, seed=99).json()['data']
    assert reviewed['status'] == 'completed', reviewed
    db.expire_all()
    parent_splits = {asset.id: asset.split for asset in db.query(DatasetAsset).filter_by(dataset_version_id=version).all()}
    for asset in db.query(DatasetAsset).filter_by(dataset_version_id=child.id).all():
        assert asset.split == parent_splits[asset.original_asset_id]
    assert db.get(DatasetVersion, version).state == 'frozen'


def test_rotation_and_mirror_update_boxes():
    image = Image.new('RGB', (100, 200))
    boxes = [{'class_id': 0, 'cx': .2, 'cy': .3, 'w': .1, 'h': .2}]
    rotated, output = transform(image, boxes, 'rotate', 90)
    assert rotated.size == (200, 100)
    assert output[0] == {'class_id': 0, 'cx': .3, 'cy': .8, 'w': .2, 'h': .1}
    _, mirrored = transform(image, boxes, 'flip', True)
    assert mirrored[0]['cx'] == .8 and boxes[0]['cx'] == .2


def test_merge_two_frozen_versions(client, researcher_headers, db):
    first_dataset, first = frozen(client, researcher_headers)
    files = []
    for index in range(9):
        files.extend([(f"second-{index}.jpg", support.image_bytes((index * 17 + 3, 100, 40))),
                      (f"second-{index}.txt", b"0 .5 .5 .3 .3\n1 .5 .5 .3 .3\n2 .5 .5 .3 .3")])
    imported = support.import_dataset(client, researcher_headers, files=files).json()['data']
    second_dataset, second = imported['result']['dataset_id'], imported['result']['version_id']
    assert support.freeze(client, researcher_headers, second_dataset, version_id=second).json()['data']['status'] == 'completed'
    response = client.post('/api/v1/dataset-derivatives', headers={**researcher_headers, 'Idempotency-Key': 'merge-two'},
                           json={'operation': 'merge', 'version_ids': [first, second], 'name': 'merged'})
    job = response.json()['data']
    assert job['status'] == 'completed', job
    merged = db.get(DatasetVersion, job['result']['version_id'])
    assert merged.asset_count == 18
    assert merged.metadata_json['parent_version_ids'] == [first, second]
    assert client.delete('/api/v1/datasets/' + second_dataset, headers=researcher_headers).status_code == 409
    assert client.delete('/api/v1/datasets/' + first_dataset, headers=researcher_headers).status_code == 409
