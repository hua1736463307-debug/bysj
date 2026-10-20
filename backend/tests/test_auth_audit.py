import os

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from sqlalchemy import BigInteger, create_engine
from sqlalchemy.ext.compiler import compiles
from sqlalchemy.orm import sessionmaker
from starlette.requests import Request

from auth_audit import audit_session_consistency
from database import OnlineSession, OperationLog
from log_utils import record_auth_audit


@compiles(BigInteger, "sqlite")
def _compile_big_integer_as_sqlite_integer(_type, _compiler, **_kwargs):
    return "INTEGER"


def _request(request_id="req-auth-test"):
    scope = {
        "type": "http",
        "method": "POST",
        "path": "/api/auth/logout",
        "query_string": b"",
        "headers": [
            (b"user-agent", b"pytest-browser"),
            (b"x-request-id", request_id.encode()),
        ],
        "client": ("127.0.0.1", 1234),
        "scheme": "http",
        "server": ("testserver", 80),
    }
    return Request(scope)


def _session():
    engine = create_engine("sqlite:///:memory:")
    OperationLog.__table__.create(engine)
    OnlineSession.__table__.create(engine)
    return sessionmaker(bind=engine, future=True)()


def test_auth_audit_captures_session_request_and_client_metadata():
    db = _session()

    record_auth_audit(
        db,
        operator="admin",
        session_id="session-1",
        operation_type="退出",
        reason="manual_logout",
        detail="用户主动退出登录",
        request=_request(),
        event_key="session-1:manual_logout",
        commit=True,
    )

    row = db.query(OperationLog).one()
    assert row.title == "认证"
    assert row.session_id == "session-1"
    assert row.request_id == "req-auth-test"
    assert row.user_agent == "pytest-browser"
    assert row.reason == "manual_logout"


def test_auth_audit_event_key_is_idempotent():
    db = _session()
    kwargs = {
        "operator": "admin",
        "session_id": "session-2",
        "operation_type": "过期",
        "reason": "token_expired",
        "detail": "登录令牌已过期，会话自动失效",
        "request": _request("req-expired"),
        "event_key": "session-2:token_expired",
        "commit": True,
    }

    record_auth_audit(db, **kwargs)
    assert record_auth_audit(db, **kwargs) is None
    assert db.query(OperationLog).count() == 1


def test_consistency_check_reports_missing_auth_transition():
    db = _session()
    db.add(OnlineSession(
        id=1,
        session_id="session-3",
        user_id=1,
        username="admin",
        status="revoked",
        expires_at=__import__("datetime").datetime.now(),
        revoked_at=__import__("datetime").datetime.now(),
    ))
    db.commit()

    result = audit_session_consistency(db, since_hours=24)

    assert result["missingAuditTotal"] == 1
    assert result["missingAudit"][0]["sessionId"] == "session-3"
