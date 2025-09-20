# --- Information About Script ---
__name__ = "Central Logger using Loguru"
__version__ = "3.0.0"
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

# Add a rich console logger for development
logger.add(
    sys.stderr,
    level="INFO",
    format="<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
           "<level>{level: <8}</level> | "
           "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
           "<level>{message}</level>",
    colorize=True,
    backtrace=True,  # Show full stack trace on exceptions
    diagnose=False   # Don't add variable values to console logs for security
)

# Add a file logger to capture all levels (from DEBUG upwards)
logger.add(
    "logs/debug.log",
    level="DEBUG",
    rotation="10 MB",   # Rotate the log file when it reaches 10 MB
    retention="10 days",# Keep logs for 10 days
    compression="zip",  # Compress old log files
    format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level: <8} | {name}:{function}:{line} - {message}",
    backtrace=True,
    diagnose=True,      # Add variable values to file logs for easier debugging
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
    backtrace=True,     # Always include the full stack trace for errors
    diagnose=True,      # This is the killer feature: log variable values on error
    encoding="utf8"
)

# --- (Optional) For structured logging with JSON ---
# Comment out the file loggers above and uncomment this to write logs as JSON files.
# This is ideal for log management systems like the ELK Stack, Datadog, or Splunk.
# logger.add(
#     "logs/app.json",
#     level="DEBUG",
#     rotation="10 MB",
#     retention="10 days",
#     compression="zip",
#     serialize=True # This is the key to JSON output
# )

# --- 2. Intercept Standard Logging ---
# This ensures that logs from third-party libraries that use Python's
# standard `logging` module are captured and formatted by Loguru.

class InterceptHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        # Get corresponding Loguru level if it exists
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # Find caller from where originated the logged message
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

# Configure the standard logging module to use our InterceptHandler
logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

# Set lower log levels for noisy third-party libraries
logging.getLogger("urllib3").setLevel(logging.WARNING)
logging.getLogger("requests").setLevel(logging.WARNING)
logging.getLogger("httpx").setLevel(logging.WARNING)


# --- 3. The Public API Function ---

def get_logger(name: Union[str, object] = None, module_name: str = None) -> "logger":
    """
    Get the pre-configured Loguru logger instance.

    This function maintains the original API for backward compatibility.
    It can accept a string or a class instance to derive a name, but with
    Loguru, this is often not necessary as the logger automatically
    identifies the calling module.

    Args:
        name: A string or an object (e.g., a class instance) to name the logger.
        module_name: Kept for compatibility, usually `__name__`.

    Returns:
        The configured Loguru logger instance.
    """
    # In Loguru, there is one primary logger object that is used everywhere.
    # It automatically captures the correct module, function, and line number.
    # This function simply returns that global logger. The name extraction
    # is kept to demonstrate how one might add context if needed, but it's
    # generally not required.
    
    logger_name = name or module_name
    
    if logger_name and not isinstance(logger_name, str):
        # If an object (like a class instance) is passed, get its class name
        if hasattr(logger_name, '__class__'):
            logger_name = logger_name.__class__.__name__

    if isinstance(logger_name, str):
        # .bind() creates a logger with that context attached.
        # This is useful if you want to filter logs based on this name later.
        return logger.bind(name=logger_name)

    return logger


# --- Example Usage ---
# This block demonstrates how to use the logger.
# You can run this file directly (`python logger_config.py`) to test it.
if __name__ == "__main__":

    # Basic usage (same as before)
    log = get_logger(module_name=__name__)
    log.info("Logger configured successfully. This is an info message.")
    log.debug("This debug message will only appear in 'logs/debug.log'.")
    log.warning("This is a warning message.")

    # Usage inside a class
    class MyService:
        def __init__(self):
            # Pass the instance `self` to get a logger named after the class
            self.logger = get_logger(name=self)

        def do_work(self, user_id: int):
            self.logger.info(f"Starting work for user {user_id}...")
            try:
                # Simulate a failure
                result = 100 / 0
            except ZeroDivisionError:
                # Loguru automatically captures the exception details
                self.logger.exception("A critical error occurred during processing.")

    service = MyService()
    service.do_work(user_id=12345)

    log.success("Script finished. Check the 'logs' directory for output files.")