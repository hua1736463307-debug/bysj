from pathlib import Path


BACKEND = Path(__file__).resolve().parents[1]


def _source(name):
    return (BACKEND / name).read_text(encoding="utf-8")


def test_sensitive_user_mutations_require_admin():
    source = _source("admin_routes.py")
    for signature in (
        "def create_user(",
        "def update_user(",
        "def delete_user(",
        "def reset_password(",
        "def reveal_password(",
    ):
        start = source.index(signature)
        line = source[start:source.index("\n", start)]
        assert "Depends(require_admin)" in line


def test_role_and_menu_mutations_require_admin():
    source = _source("system_admin_routes.py")
    for signature in (
        "def create_role(",
        "def update_role(",
        "def update_role_scope(",
        "def delete_role(",
        "def assign_role_users(",
        "def create_menu(",
        "def update_menu(",
        "def delete_menu(",
        "def sort_menus(",
    ):
        start = source.index(signature)
        line = source[start:source.index("\n", start)]
        assert "Depends(require_admin)" in line


def test_user_password_routes_do_not_encrypt_reversible_copies():
    source = _source("admin_routes.py")
    assert "encrypt_password" not in source
    assert "decrypt_password" not in source
    assert "password_ciphertext=_encrypt" not in source
    assert "user.password_ciphertext = None" in source
    assert "status_code=410" in source
