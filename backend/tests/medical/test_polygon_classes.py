import io
import zipfile
import support


def test_polygon_roundtrip_and_segmentation_export(client, researcher_headers):
    dataset, version = support.make_dataset(client, researcher_headers, count=3)
    asset = client.get(f'/api/v1/datasets/{dataset}/assets', headers=researcher_headers,
                       params={'version_id': version}).json()['data']['items'][0]
    polygon = [[.1, .1], [.8, .1], [.5, .7]]
    response = client.put(f"/api/v1/annotations/{asset['asset_id']}", headers=researcher_headers, json={
        'expected_revision': 0, 'boxes': [{'class_id': 0, 'cx': .5, 'cy': .5, 'w': .2, 'h': .2, 'polygon': polygon}],
    })
    assert response.status_code == 200, response.text
    box = response.json()['data']['annotations'][0]
    assert box['polygon'] == polygon
    assert box['cx'] == .45 and box['w'] == .7
    job = client.post('/api/v1/annotations/export', headers=researcher_headers,
                      json={'dataset_id': dataset, 'version_id': version, 'format': 'yolo-seg'}).json()['data']
    assert job['status'] == 'completed', job
    archive = client.get('/api/v1/artifacts/' + job['result']['artifact_id'], headers=researcher_headers)
    with zipfile.ZipFile(io.BytesIO(archive.content)) as bundle:
        label = 'labels/' + asset['relative_path'].rsplit('.', 1)[0] + '.txt'
        assert len(bundle.read(label).decode().split()) == 7


def test_class_changes_preserve_ids_and_invalidate_old_drafts(client, researcher_headers):
    dataset, version = support.make_dataset(client, researcher_headers, count=3)
    asset = client.get(f'/api/v1/datasets/{dataset}/assets', headers=researcher_headers,
                       params={'version_id': version}).json()['data']['items'][0]
    current = client.get(f"/api/v1/annotations/{asset['asset_id']}", headers=researcher_headers).json()['data']
    classes = {**current['class_map'], '3': 'Other'}
    result = client.put('/api/v1/dataset-versions/' + version + '/classes', headers=researcher_headers,
                        json={'expected_class_map': current['class_map'], 'class_map': classes})
    assert result.status_code == 200, result.text
    stale = client.put(f"/api/v1/annotations/{asset['asset_id']}", headers=researcher_headers,
                       json={'expected_revision': current['revision'], 'boxes': []})
    assert stale.status_code == 409
    removed = client.put('/api/v1/dataset-versions/' + version + '/classes', headers=researcher_headers,
                         json={'expected_class_map': classes, 'class_map': {'0': 'RBC'}})
    assert removed.status_code == 409


def test_full_training_archive_contains_images_labels_and_groups(client, researcher_headers):
    dataset, version = support.make_dataset(client, researcher_headers, count=9)
    assert support.freeze(client, researcher_headers, dataset, version_id=version).json()['data']['status'] == 'completed'
    job = client.post('/api/v1/annotations/export', headers=researcher_headers,
                      json={'dataset_id': dataset, 'version_id': version, 'format': 'yolo-dataset'}).json()['data']
    assert job['status'] == 'completed', job
    response = client.get('/api/v1/artifacts/' + job['result']['artifact_id'], headers=researcher_headers)
    with zipfile.ZipFile(io.BytesIO(response.content)) as archive:
        names = archive.namelist()
        assert {'data.yaml', 'class_map.json', 'source_groups.json', 'manifest.json'} <= set(names)
        assert len([name for name in names if name.startswith('images/')]) == 9
        assert len([name for name in names if name.startswith('labels/')]) == 9
