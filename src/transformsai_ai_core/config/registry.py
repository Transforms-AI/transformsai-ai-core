"""
Project hooks into config validation: extra checks and extra secrets.

Core ships no validators for project data; it only runs what a project registers,
so a single gate (``validate_config`` / ``transformsaicore config validate``)
covers core and project keys alike.

Paths are dotted; ``*`` matches any key (or list index) at that level::

    register_validator("settings.meter", MeterSettings)            # a pydantic model
    register_validator("cameras.*.settings.flash", check_flash)    # a function
    register_secret("settings.modbus.password")
"""

import difflib
import threading
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass
from typing import Any

from .report import Issue

Check = Callable[[Any], Any]


@dataclass(frozen=True)
class RegisteredValidator:
    pattern: str
    check: Any  # a pydantic model class or a callable
    required: bool


_lock = threading.Lock()
_validators: list[RegisteredValidator] = []
_secret_patterns: list[str] = []


def register_validator(path: str, check: Any, *, required: bool = False) -> None:
    """Validate the value(s) at `path` with `check` on every load and validation.

    `check` is a pydantic model class (validated with ``model_validate``; use
    ``extra="forbid"`` to catch typos) or a callable taking the value and returning
    None/True (ok), False, a message, an ``Issue`` (path relative to `path`), or a
    list of those -- or raising ValueError. Paths that are absent are skipped.

    Failures are errors in strict mode. In lenient mode they are warnings, unless
    `required` is true, in which case ``load_config`` refuses the config.
    """
    with _lock:
        _validators.append(RegisteredValidator(path, check, bool(required)))


def register_secret(path: str) -> None:
    """Treat the value(s) at `path` as secrets: in ``Config.secrets()`` and masked in ``masked_dump()``."""
    with _lock:
        if path not in _secret_patterns:
            _secret_patterns.append(path)


def clear_registry() -> None:
    """Forget every registered validator and secret path (for tests)."""
    with _lock:
        _validators.clear()
        _secret_patterns.clear()


def validators() -> list[RegisteredValidator]:
    with _lock:
        return list(_validators)


def secret_patterns() -> list[str]:
    with _lock:
        return list(_secret_patterns)


# ==============================================================================
# Path helpers
# ==============================================================================
def dotted(path: Iterable[Any]) -> str:
    return ".".join(str(part) for part in path)


def iter_matches(data: Any, pattern: str) -> Iterator[tuple[tuple, Any]]:
    """(key path, value) for every value in `data` matching dotted `pattern`."""
    parts = pattern.split(".") if pattern else []

    def walk(node: Any, i: int, path: tuple) -> Iterator[tuple[tuple, Any]]:
        if i == len(parts):
            yield path, node
            return
        key = parts[i]
        if isinstance(node, dict):
            if key == "*":
                for k, v in list(node.items()):
                    yield from walk(v, i + 1, (*path, k))
            elif key in node:
                yield from walk(node[key], i + 1, (*path, key))
        elif isinstance(node, list):
            if key == "*":
                for k, v in enumerate(node):
                    yield from walk(v, i + 1, (*path, k))
            elif key.isdigit() and int(key) < len(node):
                yield from walk(node[int(key)], i + 1, (*path, int(key)))

    yield from walk(data, 0, ())


def set_path(data: Any, path: tuple, value: Any) -> None:
    node = data
    for key in path[:-1]:
        node = node[key]
    node[path[-1]] = value


# ==============================================================================
# Running checks
# ==============================================================================
def _join(base: str, rel: str) -> str:
    return f"{base}.{rel}" if base and rel else (base or rel)


def _model_at(model: Any, loc: tuple) -> Any:
    """The nested pydantic model class reached by following `loc` through model fields, if any."""
    from pydantic import BaseModel

    current = model
    for part in loc:
        fields = getattr(current, "model_fields", None)
        if not fields or part not in fields:
            return current
        annotation = fields[part].annotation
        if isinstance(annotation, type) and issubclass(annotation, BaseModel):
            current = annotation
        else:
            return current
    return current


def _pydantic_issues(model: Any, value: Any, base: str) -> list[Issue]:
    from pydantic import ValidationError

    try:
        model.model_validate(value)
        return []
    except ValidationError as exc:
        issues = []
        for err in exc.errors(include_url=False):
            loc = tuple(err["loc"])
            suggestion = None
            if err["type"] == "extra_forbidden" and loc:
                owner = _model_at(model, loc[:-1])
                close = difflib.get_close_matches(str(loc[-1]), list(getattr(owner, "model_fields", {})), n=1, cutoff=0.6)
                suggestion = close[0] if close else None
                message = "unknown key"
            else:
                message = err["msg"]
            issues.append(Issue(_join(base, dotted(loc)), message, suggestion))
        return issues


def _normalise(result: Any, base: str) -> list[Issue]:
    if result is None or result is True:
        return []
    if result is False:
        return [Issue(base, "failed validation")]
    if isinstance(result, str):
        return [Issue(base, result)]
    if isinstance(result, Issue):
        return [Issue(_join(base, result.path), result.message, result.suggestion)]
    if isinstance(result, Iterable):
        issues: list[Issue] = []
        for item in result:
            issues.extend(_normalise(item, base))
        return issues
    return [Issue(base, f"validator returned an unexpected {type(result).__name__}")]


def run_check(check: Any, value: Any, path: tuple) -> list[Issue]:
    """Run one registered check against one value; never raises."""
    base = dotted(path)
    try:
        if isinstance(check, type) and hasattr(check, "model_validate"):
            return _pydantic_issues(check, value, base)
        return _normalise(check(value), base)
    except (ValueError, TypeError, AssertionError) as exc:
        return [Issue(base, str(exc) or type(exc).__name__)]
    except Exception as exc:  # a broken project validator must not take the loader down
        return [Issue(base, f"validator raised {type(exc).__name__}: {exc}")]
