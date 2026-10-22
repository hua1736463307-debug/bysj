"""Opt-in PostgreSQL tests outside backend/tests' SQLite autouse fixtures.

Only an explicitly named local test database is accepted. Every session uses a
new schema, initialized by Alembic; teardown cannot drop a shared schema.
"""
import os
import re
import uuid
from pathlib import Path
import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url


@pytest.fixture(scope="session")
def pg_engine(tmp_path_factory):
    raw = os.environ.get("BYSJ_TEST_DATABASE_URL")
    if not raw:
        pytest.skip("explicit BYSJ_TEST_DATABASE_URL required for PG integration")
    url = make_url(raw)
    if (url.get_backend_name() != "postgresql" or not (url.database or "").startswith("bysj_test_")
            or url.host not in {"localhost", "127.0.0.1", "bysj-test-pg-20261022"}):
        raise RuntimeError("Refusing integration test against non-local/non-test database")
    schema = "bysj_it_" + uuid.uuid4().hex
    assert re.fullmatch(r"bysj_it_[0-9a-f]{32}", schema)
    admin = create_engine(url)
    with admin.begin() as connection:
        connection.execute(text(f'CREATE SCHEMA "{schema}"'))
    target = url.update_query_dict({"options": f"-csearch_path={schema}"})
    root = tmp_path_factory.mktemp("pg-data")
    updates = {"DATABASE_URL": target.render_as_string(hide_password=False), "APP_ENV": "test",
               "ML_BACKEND": "dummy", "INLINE_JOBS": "false",
               "JWT_SECRET_KEY": "integration-only-secret-0123456789abcdef",
               "DATA_ROOT": str(root / "data"), "MODEL_ROOT": str(root / "models"), "ARTIFACT_ROOT": str(root / "artifacts")}
    before = {key: os.environ.get(key) for key in updates}
    os.environ.update(updates)
    from app.core.config import reset_settings_cache
    from app.db.session import reset_engine_cache, get_engine
    reset_settings_cache()
    reset_engine_cache()
    engine = None
    try:
        from alembic.config import Config
        from alembic import command
        config = Config(str(Path(__file__).parents[2] / "backend/alembic.ini"))
        command.upgrade(config, "head")
        command.upgrade(config, "head")
        engine = get_engine()
        assert engine.dialect.name == "postgresql"
        yield engine
    finally:
        if engine is not None:
            engine.dispose()
        reset_engine_cache()
        for key, value in before.items():
            if value is None: os.environ.pop(key, None)
            else: os.environ[key] = value
        reset_settings_cache()
        with admin.begin() as connection:
            connection.execute(text(f'DROP SCHEMA "{schema}" CASCADE'))
        admin.dispose()


@pytest.fixture(autouse=True)
def empty_tables(pg_engine):
    from app.db.base import Base
    names = ','.join('"' + table.name + '"' for table in Base.metadata.sorted_tables)
    with pg_engine.begin() as connection:
        connection.execute(text(f"TRUNCATE {names} RESTART IDENTITY CASCADE"))


@pytest.fixture()
def sessions(pg_engine):
    from sqlalchemy.orm import sessionmaker
    return sessionmaker(pg_engine, expire_on_commit=False)
