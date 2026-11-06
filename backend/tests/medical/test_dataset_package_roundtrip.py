import io
import json
import zipfile
from collections import defaultdict

import pytest
import support
from app.db.models import DatasetAsset, DatasetVersion


def exported_package(client, headers, fmt='yolo-dataset'):
    dataset, version = support.make_dataset(client, headers, count=9)
    assets = client.get(f'/api/v1/datasets/{dataset}/assets', headers=headers).json()['data']['items']
    revision = 0
    for asset in assets:
        annotations = asset['annotations']
        annotations[0]['polygon'] = [[.123456789, .187654321], [.786543219, .234567891], [.345678912, .876543219]]
        annotations[1].update({'cx': .456789, 'cy': .567891, 'w': .234567, 'h': .345679})
        saved = client.put(f"/api/v1/annotations/{asset['asset_id']}", headers=headers,
                           json={'expected_revision': revision, 'boxes': annotations})
        assert saved.status_code == 200, saved.text
        revision = saved.json()['data']['revision']
    assert support.freeze(client, headers, dataset, version_id=version).json()['data']['status'] == 'completed'
    augmented = client.post('/api/v1/dataset-derivatives', headers={**headers, 'Idempotency-Key': 'package-augment'},
                            json={'operation': 'augment', 'version_ids': [version], 'brightness': [1.2]}).json()['data']
    assert augmented['status'] == 'completed', augmented
    child = augmented['result']['version_id']
    exported = client.post('/api/v1/annotations/export', headers=headers,
                           json={'dataset_id': dataset, 'version_id': child, 'format': fmt}).json()['data']
    assert exported['status'] == 'completed', exported
    content = client.get(exported['result']['download_url'], headers=headers).content
    with zipfile.ZipFile(io.BytesIO(content)) as archive:
        return {name: archive.read(name) for name in archive.namelist()}


def import_package(client, headers, entries):
    response = client.post('/api/v1/datasets/import', headers=headers,
                           files=[('files', ('training.zip', support.build_zip(entries), 'application/zip'))],
                           data={'name': 'Imported training data'})
    assert response.status_code == 202, response.text
    return response.json()['data']


@pytest.mark.parametrize('fmt', ['yolo-dataset', 'yolo-seg-dataset'])
@pytest.mark.parametrize('legacy', [False, True])
def test_export_import_freeze_preserves_groups_and_holdouts(client, researcher_headers, db, fmt, legacy):
    entries = exported_package(client, researcher_headers, fmt)
    manifest = json.loads(entries['manifest.json'])
    original_groups = json.loads(entries['source_groups.json'])
    original_splits = {f"images/{item['split']}/{item['relative_path']}": item['split'] for item in manifest['assets']}
    if legacy:
        for key in ('package_type', 'group_level', 'split_config', 'synthetic', 'research_valid'):
            manifest.pop(key, None)
    # Uploaded metadata must never confer research eligibility.
    manifest['research_valid'] = True
    entries['manifest.json'] = json.dumps(manifest).encode()
    job = import_package(client, researcher_headers, entries)
    assert job['status'] == 'completed', job
    dataset, version = job['result']['dataset_id'], job['result']['version_id']
    frozen = support.freeze(client, researcher_headers, dataset, version_id=version, seed=1234).json()['data']
    assert frozen['status'] == 'completed', frozen
    db.expire_all()
    rows = db.query(DatasetAsset).filter_by(dataset_version_id=version).all()
    assert len(rows) == len(original_groups)
    by_group = defaultdict(set)
    for row in rows:
        assert row.source_group == original_groups[row.relative_path]
        assert row.split == original_splits[row.relative_path]
        by_group[row.source_group].add(row.split)
    assert all(len(splits) == 1 for splits in by_group.values())
    assert db.get(DatasetVersion, version).metadata_json['research_valid'] is False


@pytest.mark.parametrize('damage', ['image', 'label', 'groups_missing', 'group_cross_split', 'extra_image', 'manifest_path'])
def test_native_package_rejects_tampering_and_incomplete_provenance(client, researcher_headers, damage):
    entries = exported_package(client, researcher_headers)
    manifest = json.loads(entries['manifest.json'])
    if damage == 'image':
        name = next(name for name in entries if name.startswith('images/'))
        entries[name] = support.image_bytes((123, 13, 41))
    elif damage == 'label':
        name = next(name for name in entries if name.startswith('labels/'))
        entries[name] = b'0 .4 .4 .1 .1\n'
    elif damage == 'groups_missing':
        del entries['source_groups.json']
    elif damage == 'group_cross_split':
        groups = json.loads(entries['source_groups.json'])
        train = next(key for key in groups if key.startswith('images/train/'))
        test = next(key for key in groups if key.startswith('images/test/'))
        groups[test] = groups[train]
        entries['source_groups.json'] = json.dumps(groups).encode()
    elif damage == 'extra_image':
        entries['images/train/extra.jpg'] = support.image_bytes((12, 34, 56))
    else:
        manifest['assets'][0]['relative_path'] = '../outside.jpg'
        entries['manifest.json'] = json.dumps(manifest).encode()
    job = import_package(client, researcher_headers, entries)
    assert job['status'] == 'failed', job
    assert job['error']['code'] in {'dataset_package_invalid', 'invalid_path'}
