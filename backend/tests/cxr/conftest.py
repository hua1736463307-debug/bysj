"""CXR 测试夹具：reviewer / 第二用户 / 已冻结数据集。"""
from __future__ import annotations

import pytest


@pytest.fixture()
def reviewer(db):
    from app.domains.auth_service import create_user

    return create_user(db, username="reviewer1", password="reviewer-password", role="reviewer")


@pytest.fixture()
def reviewer_headers(client, reviewer):
    response = client.post(
        "/api/v1/auth/login", json={"username": "reviewer1", "password": "reviewer-password"}
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['data']['token']}"}


@pytest.fixture()
def other_user(db):
    from app.domains.auth_service import create_user

    return create_user(db, username="other1", password="other-password", role="researcher")


@pytest.fixture()
def other_headers(client, other_user):
    response = client.post(
        "/api/v1/auth/login", json={"username": "other1", "password": "other-password"}
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['data']['token']}"}


@pytest.fixture()
def cxr_dataset(client, admin_headers, settings, tmp_path):
    from cxr_support import ready_dataset

    return ready_dataset(client, admin_headers, settings, tmp_path, patients=40)
