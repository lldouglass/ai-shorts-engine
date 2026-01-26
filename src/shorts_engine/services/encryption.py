"""Encryption utilities for secure token storage.

Uses Fernet symmetric encryption with a master key from environment variables.
"""

import base64
import logging
import os
import secrets
from functools import lru_cache

from cryptography.fernet import Fernet, InvalidToken

logger = logging.getLogger(__name__)


class EncryptionError(Exception):
    """Raised when encryption/decryption fails."""

    pass


def _get_master_key() -> bytes:
    """Get the master encryption key from environment.

    The key must be a valid 32-byte base64-encoded Fernet key.
    If not set, raises an error in production or generates a warning key for development.
    """
    key = os.environ.get("ENCRYPTION_MASTER_KEY")

    if not key:
        # Check if we're in development mode
        env = os.environ.get("ENVIRONMENT", "development").lower()
        if env in ("production", "prod"):
            raise EncryptionError(
                "ENCRYPTION_MASTER_KEY environment variable is required in production. "
                "Generate one with: python -c 'from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())'"
            )

        # Development fallback - use a deterministic key for local testing
        # WARNING: This is not secure and should never be used in production
        logger.warning(
            "ENCRYPTION_MASTER_KEY not set. Using development fallback key. "
            "DO NOT USE IN PRODUCTION!"
        )
        # Generate a deterministic key from a known seed for development
        # This allows tokens to persist across restarts in development
        dev_seed = b"shorts-engine-dev-key-not-for-prod!"
        key = base64.urlsafe_b64encode(dev_seed[:32]).decode()

    return key.encode() if isinstance(key, str) else key


@lru_cache(maxsize=1)
def get_fernet() -> Fernet:
    """Get a cached Fernet instance with the master key."""
    try:
        return Fernet(_get_master_key())
    except Exception as e:
        raise EncryptionError(f"Failed to initialize encryption: {e}")


def encrypt_token(token: str) -> str:
    """Encrypt a token for secure storage.

    Args:
        token: The plaintext token to encrypt.

    Returns:
        The encrypted token as a base64-encoded string.

    Raises:
        EncryptionError: If encryption fails.
    """
    if not token:
        raise EncryptionError("Cannot encrypt empty token")

    try:
        fernet = get_fernet()
        encrypted = fernet.encrypt(token.encode())
        return encrypted.decode()
    except Exception as e:
        raise EncryptionError(f"Failed to encrypt token: {e}")


def decrypt_token(encrypted_token: str) -> str:
    """Decrypt a stored token.

    Args:
        encrypted_token: The encrypted token (base64-encoded).

    Returns:
        The decrypted plaintext token.

    Raises:
        EncryptionError: If decryption fails (invalid key or corrupted data).
    """
    if not encrypted_token:
        raise EncryptionError("Cannot decrypt empty token")

    try:
        fernet = get_fernet()
        decrypted = fernet.decrypt(encrypted_token.encode())
        return decrypted.decode()
    except InvalidToken:
        raise EncryptionError(
            "Failed to decrypt token: Invalid key or corrupted data. "
            "This may happen if ENCRYPTION_MASTER_KEY changed."
        )
    except Exception as e:
        raise EncryptionError(f"Failed to decrypt token: {e}")


def generate_master_key() -> str:
    """Generate a new master encryption key.

    Returns:
        A new Fernet-compatible key as a string.
    """
    return Fernet.generate_key().decode()


def rotate_token_encryption(
    encrypted_token: str,
    old_key: str,
    new_key: str,
) -> str:
    """Re-encrypt a token with a new key.

    Useful for key rotation procedures.

    Args:
        encrypted_token: The token encrypted with the old key.
        old_key: The old encryption key.
        new_key: The new encryption key.

    Returns:
        The token encrypted with the new key.

    Raises:
        EncryptionError: If rotation fails.
    """
    try:
        old_fernet = Fernet(old_key.encode())
        new_fernet = Fernet(new_key.encode())

        # Decrypt with old key
        plaintext = old_fernet.decrypt(encrypted_token.encode())

        # Encrypt with new key
        new_encrypted = new_fernet.encrypt(plaintext)

        return new_encrypted.decode()
    except Exception as e:
        raise EncryptionError(f"Failed to rotate token encryption: {e}")
