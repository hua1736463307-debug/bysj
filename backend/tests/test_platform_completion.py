import hashlib
from datetime import timedelta
from app.db.base import utcnow
from app.db.models import JobTask, ResourceLease


def test_model_upload_hash_and_authorization(client, admin_headers, researcher_headers):
    content = b'fixture-model'
    form = {'name': 'uploaded', 'sha256': hashlib.sha256(content).hexdigest(), 'class_map': '{"0":"cell"}'}
    assert client.post('/api/v1/models/upload', headers=researcher_headers, data=form,
                       files={'file': ('model.pt', content)}).status_code == 403
    bad = client.post('/api/v1/models/upload', headers=admin_headers, data={**form, 'sha256': '0' * 64},
                      files={'file': ('model.pt', content)})
    assert bad.status_code == 422
    response = client.post('/api/v1/models/upload', headers=admin_headers, data=form,
                           files={'file': ('model.pt', content)})
    assert response.status_code == 201, response.text
    model = response.json()['data']
    assert model['sha256'] == form['sha256']
    assert client.get('/api/v1/models/' + model['model_id'] + '/download', headers=researcher_headers).content == content


def test_archive_retains_task_and_restore(client, researcher_headers, researcher, db):
    job = JobTask(id='archive-test', type='training', status='completed', owner_id=str(researcher.id),
                  owner_user_id=researcher.id, owner_username=researcher.username, input_json={'name': 'archive%name'})
    db.add(job)
    db.commit()
    assert client.get('/api/v1/jobs', params={'search': '%'}, headers=researcher_headers).json()['data']['total'] == 1
    assert client.delete('/api/v1/jobs/archive-test', headers=researcher_headers).status_code == 200
    assert client.get('/api/v1/jobs', headers=researcher_headers).json()['data']['total'] == 0
    assert client.get('/api/v1/jobs/archive-test', headers=researcher_headers).status_code == 200
    assert client.post('/api/v1/jobs/archive-test/restore', headers=researcher_headers).status_code == 200
    assert client.get('/api/v1/jobs', headers=researcher_headers).json()['data']['total'] == 1


def test_expired_gpu_is_quarantined_in_resource_panel(client, admin_headers, db):
    db.add(ResourceLease(resource='gpu:0', holder='old-worker', expires_at=utcnow() - timedelta(seconds=10)))
    db.commit()
    response = client.get('/api/v1/training/resources', headers=admin_headers)
    assert response.status_code == 200, response.text
    assert response.json()['data']['gpu_lease_state'] == 'recovery_required'


def test_navigation_override_and_unknown_route(client, admin_headers, researcher_headers):
    payload = {'path': '/raw-dataset', 'label': '影像资料', 'sort_order': 7, 'visible': False}
    assert client.put('/api/v1/system/menus', headers=researcher_headers, json=payload).status_code == 403
    assert client.put('/api/v1/system/menus', headers=admin_headers, json=payload).status_code == 200
    items = client.get('/api/v1/auth/navigation', headers=researcher_headers).json()['data']['items']
    assert next(item for item in items if item['path'] == '/raw-dataset') == payload
    assert client.put('/api/v1/system/menus', headers=admin_headers, json={**payload, 'path': 'https://example.com'}).status_code == 422


def test_password_change_revokes_current_session(client, researcher_headers, researcher):
    assert client.post('/api/v1/auth/password', headers=researcher_headers,
                       json={'current_password': 'wrong', 'new_password': 'replacement-password'}).status_code == 403
    assert client.post('/api/v1/auth/password', headers=researcher_headers,
                       json={'current_password': 'researcher-password', 'new_password': 'replacement-password'}).status_code == 200
    assert client.get('/api/v1/auth/me', headers=researcher_headers).status_code == 401
    assert client.post('/api/v1/auth/login', json={'username': researcher.username, 'password': 'replacement-password'}).status_code == 200
