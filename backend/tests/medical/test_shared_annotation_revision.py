import copy

import support
from app.db.models import AnnotationDraft, AnnotationDraftAsset, DatasetAsset, DatasetVersion


def test_owner_and_admin_share_revision_and_authoritative_labels(client, researcher_headers, admin_headers):
    dataset, version = support.make_dataset(client, researcher_headers, count=3)
    asset = client.get(f'/api/v1/datasets/{dataset}/assets', headers=researcher_headers).json()['data']['items'][0]
    url = f"/api/v1/annotations/{asset['asset_id']}"
    owner = client.get(url, headers=researcher_headers).json()['data']
    admin = client.get(url, headers=admin_headers).json()['data']
    first = client.put(url, headers=researcher_headers, json={
        'expected_revision': owner['revision'], 'boxes': [], 'deleted_ids': ['owner-deleted'],
    })
    assert first.status_code == 200
    stale = client.put(url, headers=admin_headers, json={
        'expected_revision': admin['revision'], 'boxes': admin['annotations'],
    })
    assert stale.status_code == 409
    fresh = client.get(url, headers=admin_headers).json()['data']
    assert fresh['annotations'] == []
    assert fresh['revision'] == first.json()['data']['revision']
    restored = client.put(url, headers=admin_headers, json={
        'expected_revision': fresh['revision'], 'boxes': admin['annotations'],
    })
    assert restored.status_code == 200
    final = client.get(url, headers=researcher_headers).json()['data']
    assert final['annotations'] == restored.json()['data']['annotations']
    assert final['revision'] == restored.json()['data']['revision']
    assert final['deleted_ids'] == []


def test_legacy_drafts_require_refresh_and_do_not_hide_current_labels(client, researcher_headers, admin_headers, db):
    dataset, version_id = support.make_dataset(client, researcher_headers, count=3)
    user = client.get('/api/v1/auth/me', headers=researcher_headers).json()['data']
    admin = client.get('/api/v1/auth/me', headers=admin_headers).json()['data']
    asset = db.query(DatasetAsset).filter_by(dataset_version_id=version_id).first()
    authoritative = copy.deepcopy(asset.annotations_json)
    for owner, revision in ((user, 4), (admin, 7)):
        draft = AnnotationDraft(owner_user_id=owner['id'], owner_username=owner['username'],
                                dataset_id=dataset, source_version_id=version_id, revision=revision)
        db.add(draft)
        db.flush()
        db.add(AnnotationDraftAsset(draft_id=draft.id, relative_path=asset.relative_path,
                                    annotations_json=[], deleted_json=['stale']))
    db.commit()
    url = f'/api/v1/annotations/{asset.id}'
    for headers, old_revision in ((researcher_headers, 4), (admin_headers, 7)):
        response = client.put(url, headers=headers, json={'expected_revision': old_revision, 'boxes': []})
        assert response.status_code == 409
        assert response.json()['data']['server_revision'] == 8
    loaded = client.get(url, headers=researcher_headers).json()['data']
    assert loaded['revision'] == 8 and loaded['annotations'] == authoritative
    assert loaded['deleted_ids'] == []
    assert client.get(url, headers=admin_headers).json()['data']['revision'] == 8
    saved = client.put(url, headers=researcher_headers, json={'expected_revision': 8, 'boxes': []})
    assert saved.status_code == 200 and saved.json()['data']['revision'] == 9


def test_fork_uses_its_own_revision_namespace(client, researcher_headers, db):
    dataset, version_id = support.make_dataset(client, researcher_headers, count=3)
    parent = db.get(DatasetVersion, version_id)
    parent.metadata_json = {'annotation_revision': {'version_id': version_id, 'value': 25}}
    db.commit()
    assert support.freeze(client, researcher_headers, dataset, version_id=version_id).json()['data']['status'] == 'completed'
    response = client.post(f'/api/v1/datasets/{dataset}/annotation-versions', headers=researcher_headers,
                           json={'base_version_id': version_id})
    child = response.json()['data']['version_id']
    asset = client.get(f'/api/v1/datasets/{dataset}/assets', headers=researcher_headers,
                       params={'version_id': child}).json()['data']['items'][0]
    state = client.get(f"/api/v1/annotations/{asset['asset_id']}", headers=researcher_headers).json()['data']
    assert state['revision'] == 0
