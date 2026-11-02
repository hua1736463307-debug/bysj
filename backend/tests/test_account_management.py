"""Regression coverage for the restored account management entry points."""


def test_account_lifecycle(client, admin_headers, researcher_headers, researcher):
    url = f'/api/v1/auth/users/{researcher.id}'
    changed = client.patch(url, headers=admin_headers, json={
        'nickname': '审核员', 'role': 'reviewer', 'status': 'disabled',
    })
    assert changed.status_code == 200, changed.text
    assert client.get('/api/v1/auth/me', headers=researcher_headers).status_code == 401
    login = {'username': researcher.username, 'password': 'researcher-password'}
    assert client.post('/api/v1/auth/login', json=login).status_code == 403
    assert client.patch(url, headers=admin_headers, json={
        'nickname': '审核员', 'role': 'reviewer', 'status': 'active',
    }).status_code == 200
    # Re-enabling an account must not resurrect its old session.
    assert client.get('/api/v1/auth/me', headers=researcher_headers).status_code == 401
    logged_in = client.post('/api/v1/auth/login', json=login).json()['data']
    assert logged_in['user']['role'] == 'reviewer'
    headers = {'Authorization': f"Bearer {logged_in['token']}"}
    reset = client.post(url + '/reset-password', headers=admin_headers,
                        json={'password': 'replacement-password'})
    assert reset.status_code == 200
    assert client.get('/api/v1/auth/me', headers=headers).status_code == 401
    assert client.post('/api/v1/auth/login', json=login).status_code == 401
    login['password'] = 'replacement-password'
    assert client.post('/api/v1/auth/login', json=login).status_code == 200


def test_accounts_authorization_and_validation(client, admin_headers, researcher_headers, researcher):
    url = f'/api/v1/auth/users/{researcher.id}'
    payload = {'nickname': 'name', 'role': 'reviewer', 'status': 'active'}
    assert client.get('/api/v1/auth/users', headers=researcher_headers).status_code == 403
    assert client.patch(url, json=payload, headers=researcher_headers).status_code == 403
    assert client.post(url + '/reset-password', json={'password': 'new-password'},
                       headers=researcher_headers).status_code == 403
    assert client.patch(url, json={**payload, 'role': 'admin'}, headers=admin_headers).status_code == 422
    assert client.patch(url, json={**payload, 'status': 'bogus'}, headers=admin_headers).status_code == 422
    admin = client.get('/api/v1/auth/me', headers=admin_headers).json()['data']
    assert client.patch(f"/api/v1/auth/users/{admin['id']}", json=payload,
                        headers=admin_headers).status_code == 409
    assert client.post(f"/api/v1/auth/users/{admin['id']}/reset-password",
                       json={'password': 'new-password'}, headers=admin_headers).status_code == 409
    assert client.patch('/api/v1/auth/users/999999', json=payload, headers=admin_headers).status_code == 404


def test_account_search_creation_and_legacy_list(client, admin_headers):
    payload = {'username': 'new-user', 'password': 'new-password', 'role': 'reviewer', 'nickname': '审查'}
    assert client.post('/api/v1/auth/users', json=payload, headers=admin_headers).status_code == 200
    assert client.post('/api/v1/auth/users', json=payload, headers=admin_headers).status_code == 409
    assert client.post('/api/v1/auth/users', json={**payload, 'role': 'bogus'}, headers=admin_headers).status_code == 422
    result = client.get('/api/v1/auth/users', params={'page': 1, 'page_size': 1, 'search': '审查'},
                        headers=admin_headers).json()['data']
    assert result['total'] == 1 and result['items'][0]['username'] == 'new-user'
    legacy = client.get('/api/v1/auth/users', headers=admin_headers).json()['data']
    assert len(legacy['items']) == legacy['total'] == 2
    assert all('password_hash' not in user for user in legacy['items'])


def test_delete_retains_identity_and_revokes_sessions(client, admin_headers, researcher_headers, researcher, db):
    from app.db.models import User
    url = f'/api/v1/auth/users/{researcher.id}'
    assert client.delete(url, headers=researcher_headers).status_code == 403
    assert client.delete(url, headers=admin_headers).status_code == 200
    db.expire_all()
    assert db.get(User, researcher.id).status == 'deleted'
    assert client.get('/api/v1/auth/me', headers=researcher_headers).status_code == 401
    listed = client.get('/api/v1/auth/users', headers=admin_headers).json()['data']['items']
    assert researcher.id not in [user['id'] for user in listed]
    assert client.patch(url, headers=admin_headers, json={
        'nickname': 'restore', 'role': 'researcher', 'status': 'active',
    }).status_code == 404


def test_session_admin_listing_and_revocation(client, admin_headers, researcher_headers, researcher):
    assert client.get('/api/v1/auth/sessions', headers=researcher_headers).status_code == 403
    sessions = client.get('/api/v1/auth/sessions', headers=admin_headers).json()['data']['items']
    session = next(item for item in sessions if item['username'] == researcher.username)
    assert 'jti' not in session
    url = f"/api/v1/auth/sessions/{session['session_id']}"
    assert client.delete(url, headers=researcher_headers).status_code == 403
    assert client.delete(url, headers=admin_headers).status_code == 200
    assert client.delete(url, headers=admin_headers).status_code == 200
    assert client.get('/api/v1/auth/me', headers=researcher_headers).status_code == 401
