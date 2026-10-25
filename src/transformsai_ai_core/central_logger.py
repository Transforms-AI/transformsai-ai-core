"""
Central loguru logging: a console sink and a JSONL file sink, with secrets scrubbed.

Ordering::

    from transformsai_ai_core import configure_logging, get_logger

    configure_logging(config.logging, secrets=config.secrets())  # 1. core sinks first
    logger.add(my_journald_sink, level="WARNING")                # 2. then project sinks
    configure_logging(console_level="DEBUG")                     # later calls are safe

``configure_logging`` only ever removes the sinks it added itself, so it may be
called again at any time without dropping project sinks.

Redaction: core installs the one global loguru patcher. It scrubs every record's
message and string ``extra`` values (URL passwords, auth headers, Bearer tokens,
registered secrets) before any sink sees it; core's own sinks also scrub
rendered tracebacks. Don't call ``logger.configure(patcher=...)`` yourself --
that replaces core's patcher. Chain yours with ``add_patcher()``, and wrap
your own sinks with ``redacting_sink()`` if they render tracebacks.
"""

import inspect
import json
import secrets
import sys
import threading
import traceback
from datetime import datetime
from pathlib import Path

from loguru import logger

from .redact import add_secret, clear_secrets, redact

__all__ = [
    "configure_logging", "get_logger", "add_secret", "clear_secrets", "redact",
    "add_patcher", "redacting_sink", "OFF",
]

OFF = "OFF"
_KEEP = object()

# Remove only loguru's own default stderr handler (id 0); sinks a project added stay
try:
    logger.remove(0)
except ValueError:
    pass

_lock = threading.RLock()
_core_sink_ids: list[int] = []
_is_configured = False
_run_id = None
_console_level: str | None = "INFO"
_file_level: str | None = "DEBUG"
_patchers: list = []

_CONSOLE_FORMAT = (
    "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | "
    "<cyan>{extra[logger_name]}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>"
)


# ==============================================================================
# Redaction
# ==============================================================================
def _scrub(value):
    if isinstance(value, str):
        return redact(value)
    if isinstance(value, dict):
        return {k: _scrub(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(_scrub(v) for v in value)
    return value


def _patch(record) -> None:
    record["message"] = redact(record["message"])
    extra = record["extra"]
    for key, value in list(extra.items()):
        extra[key] = _scrub(value)
    for patcher in list(_patchers):
        try:
            patcher(record)
        except Exception:
            pass  # a project patcher must never cost a log line


logger.configure(patcher=_patch)


def add_patcher(fn) -> None:
    """Run `fn(record)` on every record, after core's redaction (loguru allows only one patcher)."""
    with _lock:
        if fn not in _patchers:
            _patchers.append(fn)


def redacting_sink(sink):
    """Wrap a project sink (file-like or callable) so its rendered text -- tracebacks included -- is scrubbed."""
    write = sink.write if hasattr(sink, "write") else sink

    def _sink(message):
        scrubbed = type(message)(redact(str(message)))
        try:
            scrubbed.record = message.record
        except AttributeError:
            pass
        write(scrubbed)

    return _sink


# ==============================================================================
# Sinks
# ==============================================================================
def _generate_run_id() -> str:
    """Generate a 6-character hex hash for run identification."""
    return secrets.token_hex(3)


def _get_log_directory() -> Path:
    """Get the date-based log directory path."""
    date_dir = Path(".core-logs") / datetime.now().strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    return date_dir


def _generate_log_path(run_id: str, directory: Path) -> Path:
    """Generate log file path with hash and timestamp."""
    timestamp = datetime.now().strftime("%H-%M-%S-%f")
    return directory / f"{run_id}_{timestamp}.jsonl"


def _build_payload(record: dict) -> dict:
    """Build a JSON payload from a (already patched) loguru record."""
    payload = {
        "timestamp": record["time"].isoformat(),
        "level": record["level"].name,
        "message": record["message"],
        "logger_name": record["extra"].get("logger_name", record["name"]),
        "file": record["file"].name,
        "line": record["line"],
        "function": record["function"],
        "process_name": record["process"].name,
        "elapsed_sec": record["elapsed"].total_seconds(),
    }

    payload.update(record["extra"])

    if record["exception"]:
        payload["exception"] = redact("".join(traceback.format_exception(*record["exception"])))

    return payload


def _create_rotation_sink(run_id: str, max_size: int = 10 * 1024 * 1024):
    """Create a sink function with rotation logic based on file size."""
    current_path = None
    current_size = 0

    def sink(message):
        nonlocal current_path, current_size

        # Initialize path on first call
        if current_path is None:
            current_path = _generate_log_path(run_id, _get_log_directory())

        # Check if rotation is needed
        if current_size > max_size:
            # Create new file with fresh timestamp
            current_path = _generate_log_path(run_id, _get_log_directory())
            current_size = 0

        # Build and write log entry
        payload = _build_payload(message.record)
        line = json.dumps(payload, default=str) + "\n"

        try:
            f = open(current_path, "a", encoding="utf-8")
        except FileNotFoundError:
            # Log directory removed under us (cleanup job, date rollover tooling): start a new file
            current_path = _generate_log_path(run_id, _get_log_directory())
            current_size = 0
            f = open(current_path, "a", encoding="utf-8")
        with f:
            f.write(line)
            current_size += len(line.encode("utf-8"))

    return sink


def _console_sink(message) -> None:
    # Resolve sys.stderr per call so redirected/captured stderr is honoured
    sys.stderr.write(redact(str(message)))


def _normalise_level(level) -> str | None:
    """None/'OFF' = no sink; anything else is a loguru level name."""
    if level is None:
        return None
    level = str(level).strip().upper()
    return None if level in ("", OFF) else level


def _apply_sinks(console_level, file_level) -> None:
    global _is_configured, _run_id, _console_level, _file_level

    with _lock:
        for sink_id in _core_sink_ids:
            try:
                logger.remove(sink_id)
            except ValueError:
                pass  # already removed by someone else
        _core_sink_ids.clear()

        if _run_id is None:
            _run_id = _generate_run_id()

        console_level, file_level = _normalise_level(console_level), _normalise_level(file_level)
        if console_level is not None:
            _core_sink_ids.append(logger.add(
                _console_sink,
                level=console_level,
                format=_CONSOLE_FORMAT,
                colorize=True,
                backtrace=True,
                diagnose=False,  # variable values in tracebacks could carry secrets
                enqueue=True,
            ))
        if file_level is not None:
            _core_sink_ids.append(logger.add(
                _create_rotation_sink(_run_id),
                level=file_level,
                diagnose=False,
                enqueue=True,
            ))

        _console_level = console_level
        _file_level = file_level
        _is_configured = True


def configure_logging(config=None, *, console_level=_KEEP, file_level=_KEEP, secrets=()) -> None:
    """(Re)configure core's console and JSONL file sinks.

    Args:
        config: a ``LoggingConfig`` (``config.logging``); its levels are used
            unless a keyword below overrides them.
        console_level / file_level: a loguru level name, or ``"OFF"``/None for no
            sink. Omitted (and no `config`) = keep the current level.
        secrets: literals to mask in every log line (``config.secrets()``).

    Only core's own sinks are replaced; sinks added by the project survive.
    """
    for value in secrets or ():
        add_secret(value)

    with _lock:
        console, file = _console_level, _file_level
        if config is not None:
            console, file = config.console_level, config.file_level
        if console_level is not _KEEP:
            console = console_level
        if file_level is not _KEEP:
            file = file_level
        _apply_sinks(console, file)


class _LoggerWrapper:
    """Wraps a loguru bound logger so that .error() auto-includes traceback."""

    def __init__(self, bound_logger):
        self._logger = bound_logger

    def error(self, msg, *args, **kwargs):
        exc = sys.exc_info()[0] is not None
        self._logger.opt(exception=exc).error(msg, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._logger, name)


def get_logger(name=None):
    """A logger bound to `name`.

    Args:
        name: logger name; an object uses its class name; None = the calling module.

    Levels are set only through ``configure_logging``. If nothing configured
    logging yet, the default sinks (console INFO, file DEBUG) are installed.
    """
    if name is not None and not isinstance(name, str):
        name = type(name).__name__
    if name is None:
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get("__name__", "unknown_module")

    if not _is_configured:
        with _lock:
            if not _is_configured:
                _apply_sinks(_console_level, _file_level)

    return _LoggerWrapper(logger.bind(logger_name=name))
