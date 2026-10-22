"""后端测试公共装置。

导入任何 ``app.*`` 之前先固定环境变量；每个用例重建表，保证隔离。
SQLite 只用于契约/单元测试，不作为 PostgreSQL 并发或锁行为的证据。
"""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

_TEST_ROOT = Path(tempfile.mkdtemp(prefix="bysj-tests-"))
os.environ["APP_ENV"] = "test"
os.environ["DATABASE_URL"] = f"sqlite:///{(_TEST_ROOT / 'test.db').as_posix()}"
os.environ["JWT_SECRET_KEY"] = "test-only-secret-key-0123456789abcdef"
os.environ["DATA_ROOT"] = str(_TEST_ROOT / "data")
os.environ["MODEL_ROOT"] = str(_TEST_ROOT / "models")
os.environ["ARTIFACT_ROOT"] = str(_TEST_ROOT / "artifacts")
os.environ["PUBLIC_ORIGIN"] = "http://127.0.0.1:8000"
os.environ["BOOTSTRAP_ADMIN_USERNAME"] = "admin"
os.environ["BOOTSTRAP_ADMIN_PASSWORD"] = "admin-test-password"
os.environ["INLINE_JOBS"] = "true"
os.environ["ML_BACKEND"] = "dummy"

import pytest  # noqa: E402

import app.db.models  # noqa: E402,F401  注册 metadata
from app.core.config import get_settings, reset_settings_cache  # noqa: E402
from app.db.base import Base  # noqa: E402
from app.db.session import get_engine, reset_engine_cache  # noqa: E402


@pytest.fixture(scope="session", autouse=True)
def _prepare_engine():
    reset_settings_cache()
    reset_engine_cache()
    Base.metadata.create_all(get_engine())
    yield


@pytest.fixture(autouse=True)
def clean_database():
    engine = get_engine()
    Base.metadata.drop_all(engine)
    Base.metadata.create_all(engine)
    yield


@pytest.fixture()
def settings():
    return get_settings()


@pytest.fixture()
def db():
    from app.db.session import get_session_factory

    session = get_session_factory()()
    try:
        yield session
    finally:
        session.close()


@pytest.fixture()
def client():
    from fastapi.testclient import TestClient

    from app.main import create_app

    application = create_app(get_settings())
    with TestClient(application) as test_client:
        yield test_client


@pytest.fixture()
def admin_headers(client, db):
    from app.core.auth import hash_password
    from app.db.models import User

    if db.query(User).filter(User.role == "admin").count() == 0:
        db.add(
            User(
                username="admin",
                password_hash=hash_password("admin-test-password"),
                role="admin",
                status="active",
            )
        )
        db.commit()
    response = client.post(
        "/api/v1/auth/login",
        json={"username": "admin", "password": "admin-test-password"},
    )
    assert response.status_code == 200, response.text
    token = response.json()["data"]["token"]
    return {"Authorization": f"Bearer {token}"}


@pytest.fixture()
def researcher(db):
    from app.domains.auth_service import create_user

    return create_user(db, username="researcher1", password="researcher-password", role="researcher")


@pytest.fixture()
def researcher_headers(client, researcher):
    response = client.post(
        "/api/v1/auth/login",
        json={"username": researcher.username, "password": "researcher-password"},
    )
    assert response.status_code == 200, response.text
    return {"Authorization": f"Bearer {response.json()['data']['token']}"}
