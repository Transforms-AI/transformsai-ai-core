"""
Credential scrubbing for text leaving the process (log sinks, health reports).

``central_logger`` runs every record through ``redact()`` before any sink sees it.
Three kinds of secret are caught:

- URL userinfo: ``scheme://user:password@`` -> ``scheme://user:***@``;
- auth header values (``x-auth-key``, ``x-secret-key``, ``authorization``,
  ``x-fleet-key``, ``x-tai-token``), quoted or bare, with an optional ``Bearer``;
- bare ``Bearer <token>`` strings;
- every literal registered with ``add_secret()`` (and its URL-encoded form).

Stdlib only; safe to call from any thread.
"""

import re
import threading
from urllib.parse import quote

MASK = "***"

_URL_USERINFO = re.compile(r"(\b[a-zA-Z][a-zA-Z0-9+.-]*://[^\s/:@'\"]+):([^\s@'\"/]+)@")
_HEADER_VALUE = re.compile(
    r"""(?ix)
    (['"]?(?:x-auth-key|x-secret-key|authorization|x-tai-token|x-fleet-key)['"]?\s*[:=]\s*)
    (?:(['"])(?:bearer\s+)?[^'"]*\2|(?:bearer\s+)?[^\s,;}'"]+)
    """
)
_BEARER = re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._~+/=-]{8,}")

# Literals shorter than this are ignored: masking them would mangle ordinary text
MIN_SECRET_LENGTH = 4

_lock = threading.Lock()
_secrets: list[str] = []


def add_secret(value) -> None:
    """Mask `value` (and its URL-encoded form) in every log line from now on."""
    if value is None:
        return
    text = str(value)
    if len(text) < MIN_SECRET_LENGTH:
        return
    with _lock:
        for variant in {text, quote(text, safe="")}:
            if variant not in _secrets:
                _secrets.append(variant)
        _secrets.sort(key=len, reverse=True)  # longest first, so a secret's prefix never wins


def clear_secrets() -> None:
    """Forget every registered literal (the patterns stay)."""
    with _lock:
        _secrets.clear()


def redact(text):
    """`text` with every known secret form replaced by '***'. Non-strings are returned as-is."""
    if not text or not isinstance(text, str):
        return text
    text = _URL_USERINFO.sub(rf"\1:{MASK}@", text)
    text = _HEADER_VALUE.sub(lambda m: m.group(1) + (f"'{MASK}'" if m.group(2) else MASK), text)
    text = _BEARER.sub(rf"\1{MASK}", text)
    with _lock:
        secrets = list(_secrets)
    for secret in secrets:
        if secret in text:
            text = text.replace(secret, MASK)
    return text
