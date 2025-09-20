# --- Information About Script ---
__name__ = "Central Logger (Powered by Loguru)"
__version__ = "3.1.0"
__author__ = "TransformsAI"

import logging
import sys
from pathlib import Path
from typing import Union

from loguru import logger

# --- 1. Centralized Configuration ---

# Create logs directory if it doesn't exist
Path("logs").mkdir(exist_ok=True)

# Remove the default handler to prevent duplicate logs
logger.remove()

# Define a "patcher" function to modify the log record.
# This function checks if a custom 'name' was bound using get_logger().
# If so, it replaces the default module name with the custom one.
def patch_record_with_bound_name(record: dict) -> None:
    if record["extra"].get("name"):
        record["name"] = record["extra"]["name"]

# Configure the logger to use our patcher. This is the key to the solution.
logger.configure(patcher=patch_record_with_bound_name)


# Add a rich console logger for development
# The format string now correctly displays the class name because the patcher has modified the record.
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>",
    colorize=True,
    backtrace=True,
    diagnose=False
)

# Add a file logger to capture all levels (from DEBUG upwards)
logger.add(
    "logs/debug.log",
    level="DEBUG",
    rotation="10 MB",
    retention="10 days",
    compression="zip",
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True,
    encoding="utf8"
)

# Add a separate file logger specifically for errors
logger.add(
    "logs/error.log",
    level="ERROR",
    rotation="5 MB",
    retention="30 days",
    compression="zip",
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
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


# --- 3. The Public API Function ---

def get_logger(name: Union[str, object] = None, module_name: str = None) -> "logger":
    """
    Get the pre-configured Loguru logger instance.

    This function can accept a string or a class instance to derive a name.
    - If an object (like a class instance `self`) is passed, the logger name
      will be the class's name (e.g., "DataUploader").
    - If a string is passed, that string will be used as the name.
    - If nothing is passed, Loguru's default behavior (module name) is used.

    Args:
        name: A string or an object (e.g., a class instance) to name the logger.
        module_name: Kept for compatibility, usually `__name__`.

    Returns:
        The configured Loguru logger instance, potentially with a bound name.
    """
    logger_name = name or module_name
    
    if logger_name and not isinstance(logger_name, str):
        # If an object (like a class instance) is passed, get its class name
        if hasattr(logger_name, '__class__'):
            logger_name = logger_name.__class__.__name__

    if isinstance(logger_name, str):
        # .bind() creates a logger with that context attached.
        # Our patcher will then use this bound name for display.
        return logger.bind(name=logger_name)

    return logger


# --- Example Usage ---
if __name__ == "__main__":
    # 1. Logger with a custom string name
    string_log = get_logger(name="MyCustomTask")
    string_log.info("This log record will be named 'MyCustomTask'.")

    # 2. Logger inside a class instance
    class MyService:
        def __init__(self):
            # Pass the instance `self` to get a logger named after the class
            self.logger = get_logger(name=self)

        def do_work(self):
            self.logger.info("This log record will be named 'MyService'.")

    service = MyService()
    service.do_work()

    # 3. Logger with no name (falls back to module name)
    default_log = get_logger()
    default_log.warning("This log record will be named after the module (__main__).")