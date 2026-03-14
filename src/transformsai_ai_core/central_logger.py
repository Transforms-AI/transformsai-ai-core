import sys
import os
import atexit
import json
import inspect
import traceback
from datetime import datetime
from pathlib import Path
from multiprocessing import current_process
from loguru import logger

# Remove default loguru handler
logger.remove()

_is_configured = False
_current_log_path = None
_start_time_obj = None
_rotation_counter = 0

def _flat_json_sink(message):
    global _current_log_path, _rotation_counter
    record = message.record
    
    # 1. Check for rotation (10MB = 10 * 1024 * 1024 bytes)
    if os.path.exists(_current_log_path) and os.path.getsize(_current_log_path) > 10 * 1024 * 1024:
        _rotation_counter += 1
        directory = os.path.dirname(_current_log_path)
        # Rename the full file to a numbered "part"
        part_path = os.path.join(directory, _current_log_path.name.replace("IN_PROGRESS", f"PART_{_rotation_counter}"))
        try:
            os.rename(_current_log_path, part_path)
        except Exception:
            pass # Avoid crashing if file is locked

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

    with open(_current_log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, default=str) + "\n")

def _rename_log_at_exit():
    global _current_log_path, _start_time_obj
    if _current_log_path and os.path.exists(_current_log_path):
        if current_process().name == 'MainProcess':
            end_time_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            start_time_str = _start_time_obj.strftime("%Y-%m-%d_%H-%M-%S")
            directory = os.path.dirname(_current_log_path)
            final_path = os.path.join(directory, f"run_{start_time_str}_to_{end_time_str}.jsonl")
            try:
                os.rename(_current_log_path, final_path)
            except Exception:
                pass

def get_logger(name: str = None, cli_debug: bool = False, module_name: str = None):
    """
    Get the pre-configured Loguru logger instance.

    Args:
        name: A string or an object (e.g., a class instance) to name the logger.
              If an object is passed, its class name will be used.
        cli_debug: If True, enables DEBUG level for console output. Default is False (INFO level).
                   Note: This parameter was renamed from 'debug' for backwards compatibility.
        module_name: DEPRECATED: This parameter will be removed in a future version.
                     Use 'name' parameter instead. Kept for backwards compatibility.

    Returns:
        A logger instance bound with the given name.
    """
    global _is_configured, _current_log_path, _start_time_obj

    # Handle object as name (for backwards compatibility with code passing 'self')
    if name is not None and not isinstance(name, str):
        if hasattr(name, '__class__'):
            name = name.__class__.__name__

    # Handle deprecated module_name parameter
    if module_name is not None:
        # module_name is deprecated, prefer 'name' if provided
        if name is None:
            name = module_name

    if name is None:
        frame = inspect.currentframe().f_back
        name = frame.f_globals.get("__name__", "unknown_module")

    if not _is_configured:
        console_level = "DEBUG" if cli_debug else "INFO"
        log_dir = Path(".core-logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        _start_time_obj = datetime.now()
        start_time_str = _start_time_obj.strftime("%Y-%m-%d_%H-%M-%S")
        _current_log_path = log_dir / f"run_{start_time_str}_IN_PROGRESS.jsonl"

        # 1. Console Sink
        logger.add(
            sys.stderr,
            level=console_level,
            format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | <level>{level: <8}</level> | <cyan>{extra[logger_name]}</cyan>:<cyan>{line}</cyan> - <level>{message}</level>",
            colorize=True,
            enqueue=True
        )

        # 2. File Sink (Using our flat JSON sink function)
        logger.add(
            _flat_json_sink, 
            level="TRACE", 
            enqueue=True
        )
        
        atexit.register(_rename_log_at_exit)
        _is_configured = True

    return logger.bind(logger_name=name)