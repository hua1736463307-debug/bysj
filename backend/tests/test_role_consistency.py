import os
from types import SimpleNamespace

os.environ.setdefault("DATABASE_URL", "sqlite:///:memory:")

from auth_utils import effective_user_role


def test_existing_non_admin_relation_overrides_stale_legacy_admin():
    user = SimpleNamespace(role="admin")
    links = [SimpleNamespace(role_id=2)]
    active_roles = [SimpleNamespace(role_key="user")]
    assert effective_user_role(user, links, active_roles) == "user"


def test_active_admin_relation_grants_admin():
    user = SimpleNamespace(role="user")
    links = [SimpleNamespace(role_id=1)]
    active_roles = [SimpleNamespace(role_key="admin")]
    assert effective_user_role(user, links, active_roles) == "admin"


def test_no_relation_temporarily_uses_legacy_role_for_upgrade_compatibility():
    user = SimpleNamespace(role="admin")
    assert effective_user_role(user, [], []) == "admin"
