"""Reversible password display encryption kept separate from bcrypt authentication."""
import os

from cryptography.fernet import Fernet, InvalidToken


PASSWORD_DISPLAY_KEY_ENV = "PASSWORD_DISPLAY_KEY"


class PasswordEncryptionError(RuntimeError):
    """Raised when encrypted password material cannot be safely processed."""


def password_encryption_configured() -> bool:
    return bool(os.getenv(PASSWORD_DISPLAY_KEY_ENV, "").strip())


def _fernet() -> Fernet:
    key = os.getenv(PASSWORD_DISPLAY_KEY_ENV, "").strip()
    if not key:
        raise PasswordEncryptionError("PASSWORD_DISPLAY_KEY is not configured")
    try:
        return Fernet(key.encode("ascii"))
    except (TypeError, ValueError) as exc:
        raise PasswordEncryptionError("PASSWORD_DISPLAY_KEY is invalid") from exc


def encrypt_password(password: str) -> str:
    if not isinstance(password, str) or not password:
        raise PasswordEncryptionError("password is empty")
    return _fernet().encrypt(password.encode("utf-8")).decode("ascii")


def decrypt_password(ciphertext: str) -> str:
    if not ciphertext:
        raise PasswordEncryptionError("password ciphertext is empty")
    try:
        return _fernet().decrypt(ciphertext.encode("ascii")).decode("utf-8")
    except (InvalidToken, UnicodeDecodeError, ValueError) as exc:
        raise PasswordEncryptionError("password ciphertext cannot be decrypted") from exc
