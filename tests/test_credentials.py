import os
import unittest


class CredentialsTests(unittest.TestCase):
    def setUp(self) -> None:
        self._env_backup = dict(os.environ)

    def tearDown(self) -> None:
        os.environ.clear()
        os.environ.update(self._env_backup)

    def test_normalize_secret_from_env(self) -> None:
        from credentials import get_perps_credentials

        os.environ["API_KEY_PERPS"] = "k"
        os.environ["API_SECRET_PERPS"] = "-----BEGIN TEST KEY-----\\nABC\\n-----END TEST KEY-----\\n"
        key, secret = get_perps_credentials()

        self.assertEqual(key, "k")
        self.assertIn("-----BEGIN TEST KEY-----\nABC\n-----END TEST KEY-----\n", secret)
        self.assertNotIn("\\n", secret)

    def test_normalize_secret_passthrough(self) -> None:
        from credentials import normalize_secret

        raw = "plain-secret"
        self.assertEqual(normalize_secret(raw), raw)

