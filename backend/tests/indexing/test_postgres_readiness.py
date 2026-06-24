import json
import os
import subprocess
import sys
import unittest
from pathlib import Path


class PostgresReadinessIntegrationTests(unittest.TestCase):
    @unittest.skipUnless(
        os.getenv("CODESEEK_POSTGRES_TEST_URL", "").strip(),
        "CODESEEK_POSTGRES_TEST_URL is not configured",
    )
    def test_postgres_readiness_script(self) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        env = dict(os.environ)
        env["PYTHONPATH"] = str(repo_root)
        env["CODESEEK_DB_BACKEND"] = "postgres"
        env["CODESEEK_DATABASE_URL"] = os.environ["CODESEEK_POSTGRES_TEST_URL"]
        env.setdefault("CODESEEK_APP_ENCRYPTION_KEY", "postgres-test-key")

        proc = subprocess.run(
            [sys.executable, "scripts/validate_postgres_readiness.py"],
            cwd=repo_root,
            env=env,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, proc.stderr or proc.stdout)
        payload = json.loads(proc.stdout)
        self.assertTrue(all(payload.values()), payload)


if __name__ == "__main__":
    unittest.main()
