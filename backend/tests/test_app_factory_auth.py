"""PR-01：新应用工厂、鉴权与路由白名单验收。"""
from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def test_import_has_no_side_effects(tmp_path):
    """只 import 不应建目录、连数据库。"""
    env = dict(os.environ)
    env.update(
        {
            "PYTHONPATH": str(REPO_ROOT / "backend"),
            "DATABASE_URL": f"sqlite:///{(tmp_path / 'never.db').as_posix()}",
            "DATA_ROOT": str(tmp_path / "never-data"),
            "MODEL_ROOT": str(tmp_path / "never-models"),
            "ARTIFACT_ROOT": str(tmp_path / "never-artifacts"),
            "APP_ENV": "test",
            "JWT_SECRET_KEY": "test-only-secret-key-0123456789abcdef",
        }
    )
    code = "import app.main; print('imported')"
    proc = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, env=env, cwd=str(REPO_ROOT)
    )
    assert proc.returncode == 0, proc.stderr
    assert "imported" in proc.stdout
    assert not (tmp_path / "never.db").exists()
    assert not (tmp_path / "never-data").exists()


def test_live_is_independent_of_database(client):
    response = client.get("/api/v1/health/live")
    assert response.status_code == 200
    body = response.json()
    assert body["code"] == "ok"
    assert body["data"]["status"] == "live"
    assert response.headers["X-Request-ID"]


def test_ready_reports_components(client):
    response = client.get("/api/v1/health/ready")
    assert response.status_code == 200
    data = response.json()["data"]
    assert data["components"]["database"]["status"] == "ok"


def test_login_returns_token_and_user(client, admin_headers):
    me = client.get("/api/v1/auth/me", headers=admin_headers)
    assert me.status_code == 200
    assert me.json()["data"]["role"] == "admin"


def test_login_rejects_wrong_password(client):
    response = client.post(
        "/api/v1/auth/login", json={"username": "admin", "password": "wrong-password"}
    )
    assert response.status_code == 401
    assert response.json()["code"] == "invalid_credentials"


def test_missing_token_is_401_envelope(client):
    response = client.get("/api/v1/auth/me")
    assert response.status_code == 401
    body = response.json()
    assert body["code"] == "unauthenticated"
    assert set(body) == {"code", "message", "data", "request_id"}


def test_logout_revokes_token(client, admin_headers):
    assert client.post("/api/v1/auth/logout", headers=admin_headers).status_code == 200
    after = client.get("/api/v1/auth/me", headers=admin_headers)
    assert after.status_code == 401
    assert after.json()["code"] == "session_revoked"


def test_expired_token_is_401(client, settings):
    from datetime import datetime, timedelta, timezone

    from jose import jwt

    payload = {
        "sub": "1",
        "username": "admin",
        "role": "admin",
        "jti": "expired-jti",
        "iat": int((datetime.now(timezone.utc) - timedelta(hours=2)).timestamp()),
        "exp": int((datetime.now(timezone.utc) - timedelta(hours=1)).timestamp()),
        "iss": settings.jwt_issuer,
    }
    token = jwt.encode(payload, settings.jwt_secret_key, algorithm=settings.jwt_algorithm)
    response = client.get("/api/v1/auth/me", headers={"Authorization": f"Bearer {token}"})
    assert response.status_code == 401
    assert response.json()["code"] == "token_expired"


def test_openapi_has_no_industry_paths(client):
    paths = client.get("/api/openapi.json").json()["paths"]
    forbidden = ("monitor", "behavior", "face", "recorder", "pose", "alarm")
    leaked = [path for path in paths if any(word in path.lower() for word in forbidden)]
    assert leaked == []
    assert "/api/v1/health/live" in paths
    assert "/api/v1/auth/login" in paths


def test_validation_error_is_422_envelope(client):
    response = client.post("/api/v1/auth/login", json={"username": ""})
    assert response.status_code == 422
    body = response.json()
    assert body["code"] == "validation_error"
    assert body["data"]["fields"]
