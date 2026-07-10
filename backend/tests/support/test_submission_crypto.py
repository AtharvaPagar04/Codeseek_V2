import base64
import unittest
from unittest.mock import patch

from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

from retrieval.support import submission_crypto


class SubmissionCryptoTests(unittest.TestCase):
    def test_public_key_and_decrypt_roundtrip(self) -> None:
        with patch.dict(
            submission_crypto.os.environ,
            {"CODESEEK_SUBMISSION_PRIVATE_KEY_PEM": ""},
            clear=False,
        ):
            submission_crypto._key_material.cache_clear()
            public_key = serialization.load_pem_public_key(
                submission_crypto.get_submission_public_key_pem().encode("utf-8")
            )
            plaintext = b"super-secret-token"
            ciphertext = public_key.encrypt(
                plaintext,
                padding.OAEP(
                    mgf=padding.MGF1(algorithm=hashes.SHA256()),
                    algorithm=hashes.SHA256(),
                    label=None,
                ),
            )
            encoded = base64.b64encode(ciphertext).decode("ascii")
            decrypted = submission_crypto.decrypt_submission_secret(
                encoded,
                key_id=submission_crypto.get_submission_key_id(),
            )
            self.assertEqual(decrypted, plaintext.decode("utf-8"))


if __name__ == "__main__":
    unittest.main()
