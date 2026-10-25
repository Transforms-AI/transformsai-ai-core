"""
Tests for redact.py: every secret form is masked before text leaves the process.

Run: uv run python -m unittest tests.test_redact
"""

import unittest

from transformsai_ai_core.redact import add_secret, clear_secrets, redact


class TestRedact(unittest.TestCase):
    def setUp(self):
        clear_secrets()
        self.addCleanup(clear_secrets)

    def test_url_userinfo(self):
        self.assertEqual(redact("open rtsp://admin:Hik12345@10.0.0.5:554/s failed"),
                         "open rtsp://admin:***@10.0.0.5:554/s failed")
        self.assertEqual(redact("https://u:p@x.org/a and http://v:q@y.org"),
                         "https://u:***@x.org/a and http://v:***@y.org")
        # no password: nothing to mask
        self.assertEqual(redact("rtsp://admin@10.0.0.5/s"), "rtsp://admin@10.0.0.5/s")

    def test_header_values_bare_and_quoted(self):
        self.assertEqual(redact("X-Secret-Key: abcdef123"), "X-Secret-Key: ***")
        self.assertEqual(redact("headers={'X-AUTH-KEY': 'abcdef123', 'Accept': 'json'}"),
                         "headers={'X-AUTH-KEY': '***', 'Accept': 'json'}")
        self.assertEqual(redact('{"x-fleet-key": "f00ba4"}'), '{"x-fleet-key": \'***\'}')
        self.assertEqual(redact("x-tai-token=tok123; next"), "x-tai-token=***; next")

    def test_authorization_with_bearer(self):
        self.assertEqual(redact("Authorization: Bearer eyJhbGciOi.payload.sig"), "Authorization: ***")
        self.assertEqual(redact("'authorization': 'Bearer abc.def'"), "'authorization': '***'")

    def test_bare_bearer_token(self):
        self.assertEqual(redact("sent bearer abcdefgh12345 upstream"), "sent bearer *** upstream")
        # too short to be a token
        self.assertEqual(redact("bearer abc"), "bearer abc")

    def test_registered_literal_and_url_encoded_form(self):
        add_secret("p@ss/word")
        self.assertEqual(redact("pw=p@ss/word"), "pw=***")
        self.assertEqual(redact("rtsp-ish p%40ss%2Fword here"), "rtsp-ish *** here")

    def test_short_literals_ignored(self):
        add_secret("abc")
        add_secret(None)
        add_secret("")
        self.assertEqual(redact("abc abc"), "abc abc")

    def test_longest_secret_wins(self):
        add_secret("token")
        add_secret("token-extended")
        self.assertEqual(redact("token-extended"), "***")

    def test_non_strings_pass_through(self):
        self.assertEqual(redact(42), 42)
        self.assertIsNone(redact(None))
        self.assertEqual(redact(""), "")


if __name__ == "__main__":
    unittest.main()
