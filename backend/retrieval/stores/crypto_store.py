"""Small authenticated encryption helper using Python stdlib only."""

from __future__ import annotations

import base64
import hashlib
import hmac
import os
import secrets

VERSION = b"v1"
NONCE_SIZE = 16
TAG_SIZE = 32


import contextvars

# ContextVar to store request-scoped encryption key override
master_key_override_var: contextvars.ContextVar[str | None] = contextvars.ContextVar(
    "master_key_override_var", default=None
)


def has_explicit_app_encryption_key() -> bool:
    override = master_key_override_var.get()
    return bool(override or os.getenv("CODESEEK_APP_ENCRYPTION_KEY", "").strip())


def _master_key() -> bytes:
    override = master_key_override_var.get()
    raw = (
        override
        or os.getenv("CODESEEK_APP_ENCRYPTION_KEY", "").strip()
        or os.getenv("APP_ENCRYPTION_KEY", "").strip()
        or os.getenv("CODESEEK_API_KEY", "").strip()
    )
    if not raw:
        raise RuntimeError(
            "Missing CODESEEK_APP_ENCRYPTION_KEY (or APP_ENCRYPTION_KEY / CODESEEK_API_KEY fallback)"
        )
    return hashlib.sha256(raw.encode("utf-8")).digest()


def _derive(label: bytes) -> bytes:
    return hashlib.sha256(_master_key() + b":" + label).digest()


def _xor_keystream(data: bytes, key: bytes, nonce: bytes) -> bytes:
    out = bytearray()
    counter = 0
    while len(out) < len(data):
        block = hashlib.sha256(key + nonce + counter.to_bytes(8, "big")).digest()
        out.extend(block)
        counter += 1
    return bytes(a ^ b for a, b in zip(data, out[: len(data)]))


def encrypt_secret(value: str) -> str:
    plaintext = value.encode("utf-8")
    nonce = secrets.token_bytes(NONCE_SIZE)
    enc_key = _derive(b"enc")
    mac_key = _derive(b"mac")
    ciphertext = _xor_keystream(plaintext, enc_key, nonce)
    message = VERSION + nonce + ciphertext
    tag = hmac.new(mac_key, message, hashlib.sha256).digest()
    return base64.urlsafe_b64encode(message + tag).decode("ascii")


def decrypt_secret(payload: str) -> str:
    raw = base64.urlsafe_b64decode(payload.encode("ascii"))
    minimum = len(VERSION) + NONCE_SIZE + TAG_SIZE
    if len(raw) < minimum:
        raise ValueError("Encrypted secret payload is invalid")
    version = raw[: len(VERSION)]
    if version != VERSION:
        raise ValueError("Encrypted secret version is unsupported")
    nonce_start = len(VERSION)
    nonce_end = nonce_start + NONCE_SIZE
    nonce = raw[nonce_start:nonce_end]
    ciphertext = raw[nonce_end:-TAG_SIZE]
    tag = raw[-TAG_SIZE:]
    mac_key = _derive(b"mac")
    expected = hmac.new(mac_key, version + nonce + ciphertext, hashlib.sha256).digest()
    if not hmac.compare_digest(tag, expected):
        raise ValueError("Encrypted secret authentication failed")
    enc_key = _derive(b"enc")
    return _xor_keystream(ciphertext, enc_key, nonce).decode("utf-8")
