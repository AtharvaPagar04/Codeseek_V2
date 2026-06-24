import unittest
from unittest.mock import patch

from retrieval.stores.crypto_store import decrypt_secret, encrypt_secret


class CryptoStoreTests(unittest.TestCase):
    def test_encrypt_and_decrypt_round_trip(self) -> None:
        with patch.dict(
            "os.environ",
            {"CODESEEK_APP_ENCRYPTION_KEY": "test-encryption-key"},
            clear=False,
        ):
            encrypted = encrypt_secret("ghp_example_secret")
            self.assertNotEqual(encrypted, "ghp_example_secret")
            self.assertEqual(decrypt_secret(encrypted), "ghp_example_secret")

    def test_decrypt_rejects_tampering(self) -> None:
        with patch.dict(
            "os.environ",
            {"CODESEEK_APP_ENCRYPTION_KEY": "test-encryption-key"},
            clear=False,
        ):
            encrypted = encrypt_secret("ghp_example_secret")
            tampered = encrypted[:-1] + ("A" if encrypted[-1] != "A" else "B")
            with self.assertRaises(ValueError):
                decrypt_secret(tampered)

    def test_master_key_override_var(self) -> None:
        from retrieval.stores.crypto_store import master_key_override_var
        
        # When override is set, encrypt under the override
        token = master_key_override_var.set("override-key")
        try:
            encrypted = encrypt_secret("override_secret")
        finally:
            master_key_override_var.reset(token)
            
        # Decrypting without override (using test-encryption-key) fails
        with patch.dict(
            "os.environ",
            {"CODESEEK_APP_ENCRYPTION_KEY": "test-encryption-key"},
            clear=False,
        ):
            with self.assertRaises(ValueError):
                decrypt_secret(encrypted)
                
            # Decrypting with the override key succeeds
            token = master_key_override_var.set("override-key")
            try:
                self.assertEqual(decrypt_secret(encrypted), "override_secret")
            finally:
                master_key_override_var.reset(token)


if __name__ == "__main__":
    unittest.main()
