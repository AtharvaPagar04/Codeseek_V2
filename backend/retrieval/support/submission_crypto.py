"""Public-key encryption helpers for one-time secret submission from browser to backend."""

from __future__ import annotations

import base64
import hashlib
import os
from functools import lru_cache

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding, rsa


def get_submission_public_key_pem() -> str:
    return _key_material().public_pem


def get_submission_key_id() -> str:
    return _key_material().key_id


def decrypt_submission_secret(ciphertext_b64: str, *, key_id: str) -> str:
    material = _key_material()
    if key_id.strip() != material.key_id:
        raise ValueError("Submission key id is invalid or expired")
    try:
        ciphertext = base64.b64decode(ciphertext_b64.encode("ascii"))
    except Exception as exc:  # pragma: no cover - malformed client payload
        raise ValueError("Submission ciphertext is invalid") from exc
    plaintext = material.private_key.decrypt(
        ciphertext,
        padding.OAEP(
            mgf=padding.MGF1(algorithm=hashes.SHA256()),
            algorithm=hashes.SHA256(),
            label=None,
        ),
    )
    return plaintext.decode("utf-8")


@lru_cache(maxsize=1)
def _key_material():
    private_key = _load_private_key()
    public_key = private_key.public_key()
    public_der = public_key.public_bytes(
        encoding=serialization.Encoding.DER,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    )
    public_pem = public_key.public_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PublicFormat.SubjectPublicKeyInfo,
    ).decode("ascii")
    key_id = hashlib.sha256(public_der).hexdigest()[:16]
    return _SubmissionKeyMaterial(private_key, public_pem, key_id)


def _load_private_key():
    pem_inline = os.getenv("CODESEEK_SUBMISSION_PRIVATE_KEY_PEM", "").strip()
    pem_path = os.getenv("CODESEEK_SUBMISSION_PRIVATE_KEY_PATH", "").strip()
    if pem_inline:
        pem_bytes = pem_inline.encode("utf-8")
    elif pem_path:
        with open(pem_path, "rb") as handle:
            pem_bytes = handle.read()
    else:
        return rsa.generate_private_key(public_exponent=65537, key_size=2048)
    return serialization.load_pem_private_key(pem_bytes, password=None)


class _SubmissionKeyMaterial:
    def __init__(self, private_key, public_pem: str, key_id: str):
        self.private_key = private_key
        self.public_pem = public_pem
        self.key_id = key_id
