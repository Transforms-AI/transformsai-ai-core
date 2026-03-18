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
    timestamp = datetime.now().strftime("%H-%M-%S")
    return directory / f"{timestamp}_{run_id}.jsonl"


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
    cli_sink_level: str = "INFO",
    file_sink_level: str = "DEBUG",
    cli_debug: bool = None,  # DEPRECATED
    module_name: str = None   # DEPRECATED
):
    """
    Get a bound logger instance.
    
    The logger is configured on first call. If called from the entry script
    (detected by __name__ == "__main__"), the passed level parameters are used.
    Otherwise, default levels are applied if not yet configured.
    
    Args:
        name: Logger name. If None, auto-detected from calling module.
             Can also pass an object (class instance) to use its class name.
        cli_sink_level: Log level for console output (default: "INFO").
                       Only effective if called from entry script or not yet configured.
        file_sink_level: Log level for file output (default: "DEBUG").
                        Only effective if called from entry script or not yet configured.
        cli_debug: DEPRECATED - Use cli_sink_level="DEBUG" instead.
        module_name: DEPRECATED - Use name parameter instead.
    
    Returns:
        A logger instance bound with the given logger_name.
    
    Example:
        # Entry script (main.py) - sets custom levels
        from transformsai_ai_core.central_logger import get_logger
        
        logger = get_logger(cli_sink_level="DEBUG", file_sink_level="TRACE")
        
        # Any other module - just gets the configured logger
        from transformsai_ai_core.central_logger import get_logger
        
        logger = get_logger()  # Uses already-configured levels
        logger = get_logger("MyClass")  # With custom name
    """
    global _is_configured, _run_id
    
    # Handle object as name
    if name is not None and not isinstance(name, str):
        if hasattr(name, '__class__'):
            name = name.__class__.__name__
            
    # Auto-detect name from caller and check if entry script
    caller_is_main = False
    if name is None:
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get("__name__", "unknown_module")
        caller_is_main = frame.f_globals.get("__name__") == "__main__"
    
    # Configure if not yet configured
    if not _is_configured or caller_is_main:
        logger.remove()  # Remove default handler
        
        # Generate run ID once
        _run_id = _generate_run_id()
        
        # Create rotation-aware sink
        rotation_sink = _create_rotation_sink(_run_id)
        
        # Console Sink
        logger.add(
            sys.stderr,
            level=cli_sink_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{extra[logger_name]}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True,
            backtrace=True,
            enqueue=True
        )
        
        # File Sink with rotation
        logger.add(
            rotation_sink,
            level=file_sink_level,
            enqueue=True
        )
        
        _is_configured = True
    
    return _LoggerWrapper(logger.bind(logger_name=name))