from app.db.models import DatasetResource, ModelRegistry
import support


def test_dataset_search_is_paginated_and_owner_scoped(client, admin_headers, researcher_headers, db):
    dataset_id, _ = support.make_dataset(client, admin_headers, count=1)
    resource = db.get(DatasetResource, dataset_id)
    resource.name = 'sample% literal'
    db.commit()
    params = {'search': '%', 'owner': 'admin', 'page_size': 1}
    result = client.get('/api/v1/datasets', headers=admin_headers, params=params).json()['data']
    assert result['total'] == 1 and result['items'][0]['dataset_id'] == dataset_id
    assert client.get('/api/v1/datasets', headers=researcher_headers, params=params).json()['data']['total'] == 0
    assert client.get('/api/v1/datasets', headers=admin_headers, params={'search': 'missing'}).json()['data']['total'] == 0


def test_model_search_respects_visibility(client, admin_headers, researcher_headers, db):
    db.add(ModelRegistry(id='private-model', name='model%literal', path='/unused', sha256='a' * 64,
                         owner_id=999, trusted_source='training', model_task='detect', format='pt'))
    db.commit()
    params = {'search': '%', 'file_format': 'pt', 'task': 'detect'}
    assert client.get('/api/v1/models', headers=admin_headers, params=params).json()['data']['total'] == 1
    assert client.get('/api/v1/models', headers=researcher_headers, params=params).json()['data']['total'] == 0
    assert client.get('/api/v1/models', headers=admin_headers, params={'search': '_'}).json()['data']['total'] == 0
