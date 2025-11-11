# --- Information About Script ---
__name__ = "Central Logger (Powered by Loguru)"
__version__ = "4.0.0"
__author__ = "TransformsAI"

import logging
import sys
from pathlib import Path
from typing import Union

from loguru import logger

# --- 1. Centralized Configuration ---

# Create logs directory
Path("logs").mkdir(exist_ok=True)

# Remove default handler
logger.remove()

# Patcher to use custom logger names
def patch_record_with_bound_name(record: dict) -> None:
    if record["extra"].get("name"):
        record["name"] = record["extra"]["name"]

logger.configure(patcher=patch_record_with_bound_name)

# Console handler - INFO and above, traceback controlled by record extra
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>",
    colorize=True,
    backtrace=lambda record: record["extra"].get("show_console_traceback", False),
    diagnose=lambda record: record["extra"].get("show_console_traceback", False)
)

# Debug file handler - all levels with traceback, date-based rotation
logger.add(
    "logs/debug_{time:YYYY-MM-DD}.log",
    level="DEBUG",
    rotation="10 MB",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True,
    encoding="utf8"
)

# Error file handler - ERROR and above with full traceback, date-based rotation
logger.add(
    "logs/error_{time:YYYY-MM-DD}.log",
    level="ERROR",
    rotation="5 MB",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True,
    encoding="utf8"
)

# --- 2. Intercept Standard Logging ---
class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


# --- 3. The Public API Function ---

def get_logger(name: Union[str, object] = None, module_name: str = None):
    """
    Get the pre-configured Loguru logger instance.

    Args:
        name: A string or an object (e.g., a class instance) to name the logger.
        module_name: Kept for compatibility, usually `__name__`.

    Returns:
        A logger wrapper with error() and exception() methods.
    """
    logger_name = name or module_name
    
    if logger_name and not isinstance(logger_name, str):
        if hasattr(logger_name, '__class__'):
            logger_name = logger_name.__class__.__name__

    base_logger = logger.bind(name=logger_name) if isinstance(logger_name, str) else logger
    
    class LoggerWrapper:
        def __init__(self, base):
            self._logger = base
        
        def error(self, msg, *args, traceback: bool = True, **kwargs):
            """
            Log an error message.
            
            Args:
                msg: The message to log.
                traceback: If True (default), includes full traceback in file but not stdout.
                           If False, logs message only.
            """
            if traceback:
                # Add traceback to files only (console has backtrace conditional)
                self._logger.opt(exception=True).error(msg, *args, **kwargs)
            else:
                self._logger.error(msg, *args, **kwargs)
        
        def exception(self, msg, *args, **kwargs):
            """
            Log an exception with full traceback to both file and stdout.
            Always captures exception context.
            """
            # Use bind to mark this as needing console traceback
            self._logger.bind(show_console_traceback=True).opt(exception=True, depth=1).error(msg, *args, **kwargs)
        
        def __getattr__(self, name):
            return getattr(self._logger, name)
    
    return LoggerWrapper(base_logger)