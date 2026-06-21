import sys
import json
import inspect
import secrets
import traceback
from datetime import datetime
from pathlib import Path
from loguru import logger

# Remove default loguru handler
logger.remove()

_is_configured = False
_run_id = None
_cli_sink_level = "INFO"
_file_sink_level = "DEBUG"


def _generate_run_id() -> str:
    """Generate a 6-character hex hash for run identification."""
    return secrets.token_hex(3)


def _get_log_directory() -> Path:
    """Get the date-based log directory path."""
    now = datetime.now()
    base_dir = Path(".core-logs")
    date_dir = base_dir / now.strftime("%Y-%m-%d")
    date_dir.mkdir(parents=True, exist_ok=True)
    return date_dir


def _generate_log_path(run_id: str, directory: Path) -> Path:
    """Generate log file path with hash and timestamp."""
    timestamp = datetime.now().strftime("%H-%M-%S-%f")
    return directory / f"{run_id}_{timestamp}.jsonl"


def _build_payload(record: dict) -> dict:
    """Build a JSON payload from a loguru record."""
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
        payload["exception"] = "".join(traceback.format_exception(*record["exception"]))
    
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
        
        with open(current_path, "a", encoding="utf-8") as f:
            f.write(line)
            current_size += len(line.encode("utf-8"))
    
    return sink


def _apply_sinks(cli_sink_level: str, file_sink_level: str):
    global _is_configured, _run_id, _cli_sink_level, _file_sink_level

    logger.remove()

    if _run_id is None:
        _run_id = _generate_run_id()

    rotation_sink = _create_rotation_sink(_run_id)

    logger.add(
        sys.stderr,
        level=cli_sink_level,
        format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{extra[logger_name]}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
        colorize=True,
        backtrace=True,
        enqueue=True
    )

    logger.add(
        rotation_sink,
        level=file_sink_level,
        enqueue=True
    )

    _cli_sink_level = cli_sink_level
    _file_sink_level = file_sink_level
    _is_configured = True


def configure_logging(cli_sink_level=None, file_sink_level=None):
    cli = cli_sink_level if cli_sink_level is not None else _cli_sink_level
    file = file_sink_level if file_sink_level is not None else _file_sink_level
    _apply_sinks(cli, file)


class _LoggerWrapper:
    """Wraps a loguru bound logger so that .error() auto-includes traceback."""
    def __init__(self, bound_logger):
        self._logger = bound_logger

    def error(self, msg, *args, **kwargs):
        exc = sys.exc_info()[0] is not None
        self._logger.opt(exception=exc).error(msg, *args, **kwargs)

    def __getattr__(self, name):
        return getattr(self._logger, name)


def get_logger(
    name: str = None,
    cli_sink_level: str = None,
    file_sink_level: str = None,
    cli_debug: bool = None,  # DEPRECATED
    module_name: str = None   # DEPRECATED
):
    """
    Get a bound logger instance.

    Call ``configure_logging()`` from your entry script to set sink levels.
    Library code should call ``get_logger()`` without level arguments.

    Args:
        name: Logger name. If None, auto-detected from calling module.
             Can also pass an object (class instance) to use its class name.
        cli_sink_level: Log level for console output. Passing a value triggers
                        (re)configuration — last call wins.
        file_sink_level: Log level for file output. Passing a value triggers
                         (re)configuration — last call wins.
        cli_debug: DEPRECATED - Use cli_sink_level="DEBUG" instead.
        module_name: DEPRECATED - Use name parameter instead.

    Returns:
        A logger instance bound with the given logger_name.

    Example:
        # Entry script (main.py)
        from transformsai_ai_core import configure_logging, get_logger

        configure_logging(cli_sink_level="DEBUG", file_sink_level="TRACE")
        logger = get_logger()

        # Library module
        from transformsai_ai_core import get_logger

        logger = get_logger()  # Uses already-configured levels
        logger = get_logger("MyClass")  # With custom name
    """
    global _is_configured, _run_id

    if module_name is not None and name is None:
        name = module_name
    if cli_debug is not None and cli_sink_level is None:
        cli_sink_level = "DEBUG" if cli_debug else "INFO"

    if name is not None and not isinstance(name, str):
        if hasattr(name, '__class__'):
            name = name.__class__.__name__

    if name is None:
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get("__name__", "unknown_module")

    if cli_sink_level is not None or file_sink_level is not None:
        configure_logging(cli_sink_level, file_sink_level)
    elif not _is_configured:
        _apply_sinks(_cli_sink_level, _file_sink_level)

    return _LoggerWrapper(logger.bind(logger_name=name))