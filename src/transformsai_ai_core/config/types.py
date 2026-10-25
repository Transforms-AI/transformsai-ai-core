"""
Value types shared by the config schema.

Every type here converts only when the conversion is unambiguous (``"10"`` -> 10,
``"3s"`` -> 3.0, ``"800k"`` -> 800000) and constrains only what is physically
impossible (a non-positive period, a port outside 1-65535). There are no policy
ranges: any positive interval is accepted.
"""

import re
from typing import Annotated, Any

from annotated_types import Ge, Gt, Le
from pydantic import (
    BeforeValidator,
    Field,
    PlainSerializer,
    StringConstraints,
    WithJsonSchema,
)

# ==============================================================================
# Durations
# ==============================================================================
_DURATION_RE = re.compile(r"^\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)\s*(ms|s|m|h|d)?\s*$")
_DURATION_UNITS = {None: 1.0, "ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0, "d": 86400.0}


def parse_duration(value: Any) -> Any:
    """Seconds as a number, or a string with a unit: ``"500ms"``, ``"3s"``, ``"2m"``, ``"1h"``, ``"1d"``."""
    if value is None:
        return None
    if isinstance(value, bool):
        raise ValueError("expected a duration: seconds, or e.g. '500ms', '3s', '2m', '1h'")
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        match = _DURATION_RE.match(value)
        if not match:
            raise ValueError("expected a duration: seconds, or e.g. '500ms', '3s', '2m', '1h'")
        return float(match.group(1)) * _DURATION_UNITS[match.group(2)]
    return value


_DURATION_SCHEMA = {
    "anyOf": [
        {"type": "number"},
        {"type": "string", "pattern": r"^\s*\d*\.?\d+\s*(ms|s|m|h|d)?\s*$"},
    ]
}

Duration = Annotated[float, BeforeValidator(parse_duration), Gt(0), WithJsonSchema(_DURATION_SCHEMA)]
"""A period in seconds, > 0."""

DurationOrOff = Annotated[float, BeforeValidator(parse_duration), Ge(0), WithJsonSchema(_DURATION_SCHEMA)]
"""A period in seconds, >= 0 (0 = off)."""

# ==============================================================================
# Bitrates
# ==============================================================================
_BITRATE_RE = re.compile(r"^\s*(\d+\.?\d*|\.\d+)\s*([kKmMgG]?)\s*$")
_BITRATE_UNITS = {"": 1, "k": 1_000, "m": 1_000_000, "g": 1_000_000_000}


def parse_bitrate(value: Any) -> Any:
    """``"800k"``, ``"1.2M"`` or a plain number of bits/s -> int bits/s."""
    if isinstance(value, bool):
        raise ValueError("expected a bitrate such as '800k', '1.2M' or 800000")
    if isinstance(value, (int, float)):
        return round(value)
    if isinstance(value, str):
        match = _BITRATE_RE.match(value)
        if not match:
            raise ValueError("expected a bitrate such as '800k', '1.2M' or 800000")
        return round(float(match.group(1)) * _BITRATE_UNITS[match.group(2).lower()])
    return value


def format_bitrate(bps: int) -> str:
    """FFmpeg form of a bitrate: ``800000`` -> ``"800k"``; anything else verbatim."""
    bps = int(bps)
    if bps and bps % 1000 == 0:
        return f"{bps // 1000}k"
    return str(bps)


Bitrate = Annotated[
    int,
    BeforeValidator(parse_bitrate),
    Gt(0),
    PlainSerializer(format_bitrate, return_type=str),
    WithJsonSchema({"anyOf": [{"type": "integer", "exclusiveMinimum": 0},
                              {"type": "string", "pattern": r"^\s*\d*\.?\d+\s*[kKmMgG]?\s*$"}]}),
]
"""A bitrate in bits/s, > 0. Dumps back as ``"800k"``."""

# ==============================================================================
# Numbers / strings
# ==============================================================================
Port = Annotated[int, Ge(1), Le(65535)]
PositiveInt = Annotated[int, Gt(0)]
NonNegativeInt = Annotated[int, Ge(0)]
PositiveFloat = Annotated[float, Gt(0)]
NonEmptyStr = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


def _lower(value: Any) -> Any:
    return value.strip().lower() if isinstance(value, str) else value


def _upper(value: Any) -> Any:
    return value.strip().upper() if isinstance(value, str) else value


Lowercase = BeforeValidator(_lower)
Uppercase = BeforeValidator(_upper)


def _none_to_dict(value: Any) -> Any:
    return {} if value is None else value


Settings = Annotated[dict[str, Any], BeforeValidator(_none_to_dict)]
"""A project-owned section: any mapping, never inspected by core (``settings:`` with no value is ``{}``)."""


def settings_field(owner: str = "this object"):
    """The native free ``settings: {}`` every core object carries."""
    return Field(default_factory=dict, description=f"Project-owned settings for {owner}; core never inspects them")


# ==============================================================================
# Secrets
# ==============================================================================
SECRET_KEY = "secret"


def secret_field(default: Any = None, **kwargs):
    """A field whose value(s) are secrets: listed by Config.secrets(), masked in dumps and logs."""
    extra = dict(kwargs.pop("json_schema_extra", None) or {})
    extra[SECRET_KEY] = True
    if "default_factory" in kwargs:
        return Field(json_schema_extra=extra, **kwargs)
    return Field(default, json_schema_extra=extra, **kwargs)


def is_secret(field_info) -> bool:
    extra = field_info.json_schema_extra
    return isinstance(extra, dict) and bool(extra.get(SECRET_KEY))
