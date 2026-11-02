def test_department_cycle_and_reference_protection(client, admin_headers, researcher, researcher_headers):
    root = client.post('/api/v1/system/departments', headers=admin_headers, json={'name': '研究中心'}).json()['data']
    child = client.post('/api/v1/system/departments', headers=admin_headers,
                        json={'name': '影像组', 'parent_id': root['id']}).json()['data']
    result = client.put('/api/v1/system/departments/' + root['id'], headers=admin_headers,
                        json={'name': '研究中心', 'parent_id': child['id']})
    assert result.status_code == 422
    assert client.delete('/api/v1/system/departments/' + root['id'], headers=admin_headers).status_code == 409
    assert client.get('/api/v1/system/departments', headers=researcher_headers).status_code == 403
    assignment = client.put(f'/api/v1/system/users/{researcher.id}/profile', headers=admin_headers,
                            json={'department_id': child['id']})
    assert assignment.status_code == 200
    assert client.delete('/api/v1/system/departments/' + child['id'], headers=admin_headers).status_code == 409


def test_role_permissions_are_enforced_in_api(client, admin_headers, researcher, researcher_headers):
    payload = {'name': '标注岗位', 'base_role': 'researcher', 'modules': ['annotations']}
    response = client.post('/api/v1/system/roles', headers=admin_headers, json=payload)
    assert response.status_code == 200, response.text
    role = response.json()['data']
    assert set(role['modules']) == {'annotations', 'datasets', 'models'}
    assert client.put(f'/api/v1/system/users/{researcher.id}/profile', headers=admin_headers,
                      json={'role_id': role['id']}).status_code == 200
    assert client.get('/api/v1/auth/me', headers=researcher_headers).status_code == 401
    login = client.post('/api/v1/auth/login', json={'username': researcher.username, 'password': 'researcher-password'})
    headers = {'Authorization': 'Bearer ' + login.json()['data']['token']}
    assert client.get('/api/v1/datasets', headers=headers).status_code == 200
    assert client.get('/api/v1/training/resources', headers=headers).status_code == 403
    assert client.get('/api/v1/cxr/dashboard', headers=headers).status_code == 403
    assert client.delete('/api/v1/system/roles/' + role['id'], headers=admin_headers).status_code == 409
    assert client.put('/api/v1/system/roles/' + role['id'], headers=admin_headers,
                      json={**payload, 'status': 'disabled'}).status_code == 200
    assert client.post('/api/v1/auth/login', json={'username': researcher.username, 'password': 'researcher-password'}).status_code == 403


def test_audit_logs_exclude_credentials(client, admin_headers):
    client.post('/api/v1/auth/login', json={'username': 'admin', 'password': 'deliberately-secret'})
    result = client.get('/api/v1/system/logs', headers=admin_headers, params={'kind': 'login'})
    assert result.status_code == 200
    assert any(row['status'] == 401 for row in result.json()['data']['items'])
    assert 'deliberately-secret' not in result.text
    client.post('/api/v1/system/departments', headers=admin_headers, json={'name': '审计部门'})
    operations = client.get('/api/v1/system/logs', headers=admin_headers, params={'kind': 'operation'}).json()['data']['items']
    assert any(row['action'] == 'POST /api/v1/system/departments' for row in operations)


def test_role_rejects_unknown_modules_and_admin_assignment(client, admin_headers):
    assert client.post('/api/v1/system/roles', headers=admin_headers, json={
        'name': 'bad', 'modules': ['system-admin'],
    }).status_code == 422
    admin_id = client.get('/api/v1/auth/me', headers=admin_headers).json()['data']['id']
    assert client.put(f'/api/v1/system/users/{admin_id}/profile', headers=admin_headers, json={}).status_code == 409


def test_deleting_account_releases_current_organization_links(client, admin_headers, researcher):
    department = client.post('/api/v1/system/departments', headers=admin_headers, json={'name': 'temporary'}).json()['data']
    role = client.post('/api/v1/system/roles', headers=admin_headers, json={'name': 'temporary', 'modules': ['datasets']}).json()['data']
    assert client.put(f'/api/v1/system/users/{researcher.id}/profile', headers=admin_headers,
                      json={'department_id': department['id'], 'role_id': role['id']}).status_code == 200
    assert client.delete(f'/api/v1/auth/users/{researcher.id}', headers=admin_headers).status_code == 200
    assert client.delete('/api/v1/system/roles/' + role['id'], headers=admin_headers).status_code == 200
    assert client.delete('/api/v1/system/departments/' + department['id'], headers=admin_headers).status_code == 200


def test_custom_role_cannot_resume_training_via_generic_jobs(client, admin_headers, researcher, db):
    from app.db.models import JobTask
    role = client.post('/api/v1/system/roles', headers=admin_headers, json={'name': 'data-only', 'modules': ['datasets']}).json()['data']
    client.put(f'/api/v1/system/users/{researcher.id}/profile', headers=admin_headers, json={'role_id': role['id']})
    db.add(JobTask(id='restricted-training', type='training', status='interrupted',
                   owner_id=str(researcher.id), owner_user_id=researcher.id, owner_username=researcher.username))
    db.commit()
    login = client.post('/api/v1/auth/login', json={'username': researcher.username, 'password': 'researcher-password'}).json()['data']
    headers = {'Authorization': 'Bearer ' + login['token']}
    assert client.post('/api/v1/jobs/restricted-training/resume', headers=headers).status_code == 403
    assert client.get('/api/v1/jobs', headers=headers).json()['data']['total'] == 0
