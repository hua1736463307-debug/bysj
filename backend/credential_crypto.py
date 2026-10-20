"""Versioned reversible encryption for external device credentials.

User login passwords must never use this module. New recorder/device credentials
are written with DEVICE_CREDENTIAL_KEY. Unversioned ciphertext is treated as a
legacy PASSWORD_DISPLAY_KEY record only during the controlled migration window.
"""
import os

from cryptography.fernet import Fernet, InvalidToken


DEVICE_CREDENTIAL_KEY_ENV = "DEVICE_CREDENTIAL_KEY"
LEGACY_DEVICE_CREDENTIAL_KEY_ENV = "PASSWORD_DISPLAY_KEY"
DEVICE_CREDENTIAL_PREFIX = "v2:"


class DeviceCredentialError(RuntimeError):
    """Raised when device credential material cannot be safely processed."""


def _fernet_from_env(env_name: str) -> Fernet:
    key = os.getenv(env_name, "").strip()
    if not key:
        raise DeviceCredentialError(f"{env_name} is not configured")
    try:
        return Fernet(key.encode("ascii"))
    except (TypeError, ValueError) as exc:
        raise DeviceCredentialError(f"{env_name} is invalid") from exc


def device_credential_encryption_configured() -> bool:
    try:
        _fernet_from_env(DEVICE_CREDENTIAL_KEY_ENV)
        return True
    except DeviceCredentialError:
        return False


def is_legacy_device_credential(ciphertext: str) -> bool:
    return bool(ciphertext) and not str(ciphertext).startswith(DEVICE_CREDENTIAL_PREFIX)


def encrypt_device_credential(secret: str) -> str:
    if not isinstance(secret, str) or not secret:
        raise DeviceCredentialError("device credential is empty")
    token = _fernet_from_env(DEVICE_CREDENTIAL_KEY_ENV).encrypt(secret.encode("utf-8")).decode("ascii")
    return f"{DEVICE_CREDENTIAL_PREFIX}{token}"


def decrypt_device_credential(ciphertext: str) -> str:
    if not ciphertext:
        raise DeviceCredentialError("device credential ciphertext is empty")

    value = str(ciphertext)
    if value.startswith(DEVICE_CREDENTIAL_PREFIX):
        token = value[len(DEVICE_CREDENTIAL_PREFIX):]
        fernet = _fernet_from_env(DEVICE_CREDENTIAL_KEY_ENV)
    else:
        # Explicit legacy format: do not try arbitrary keys in sequence.
        token = value
        fernet = _fernet_from_env(LEGACY_DEVICE_CREDENTIAL_KEY_ENV)

    try:
        return fernet.decrypt(token.encode("ascii")).decode("utf-8")
    except (InvalidToken, UnicodeDecodeError, ValueError) as exc:
        raise DeviceCredentialError("device credential ciphertext cannot be decrypted") from exc
