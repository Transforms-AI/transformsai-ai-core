"""
Usage Examples:
# Basic usage in any module
from logger_config import get_logger
logger = get_logger(module_name=__name__)

# Custom logger name
logger = get_logger(name="CustomComponent")

# In classes (recommended pattern)
class MyClass:
    def __init__(self):
        self.logger = get_logger(name=f"{self.__class__.__name__}")
        
# Temporary log level changes
from logger_config import TemporaryLogLevel
with TemporaryLogLevel('DEBUG'):
    logger.debug("This will show even if global level is INFO")
"""

# --- Information About Script ---
__name__ = "Central Logger"
__version__ = "1.1.1" 
__author__ = "TransformsAI"

import logging
import logging.handlers
import sys
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any, Union


class ColoredFormatter(logging.Formatter):
    """Custom formatter with colors for console output"""
    
    # ANSI color codes
    COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[41m', # Red background
        'RESET': '\033[0m'      # Reset
    }

    def format(self, record):
        # Create a copy of the record to avoid modifying the original
        log_record = logging.makeLogRecord(record.__dict__)
        
        # Add color to levelname copy
        if log_record.levelname in self.COLORS:
            log_record.levelname = f"{self.COLORS[log_record.levelname]}{log_record.levelname}{self.COLORS['RESET']}"
        
        return super().format(log_record)


class TracebackHandler(logging.Handler):
    """Custom handler that automatically adds tracebacks for ERROR and CRITICAL levels"""
    
    def __init__(self, base_handler):
        super().__init__()
        self.base_handler = base_handler
        self.setLevel(base_handler.level)
        self.setFormatter(base_handler.formatter)
    
    def emit(self, record):
        # If it's an error/critical and we don't already have exception info
        if record.levelno >= logging.ERROR and record.exc_info is None:
            import traceback
            import sys
            # Get current exception if available
            exc_info = sys.exc_info()
            if exc_info[0] is not None:
                record.exc_info = exc_info
                # Also add traceback to the message for better formatting
                if not hasattr(record, 'exc_text') or record.exc_text is None:
                    record.exc_text = ''.join(traceback.format_exception(*exc_info))
        
        self.base_handler.emit(record)

class LoggerManager:
    """
    Centralized logger management for the entire application.
    
    This singleton class ensures consistent logging configuration across all modules.
    Features:
    - Thread-safe singleton pattern
    - Colored console output
    - Rotating file handlers for all logs and errors
    - Automatic third-party library noise reduction
    - Module-based logger naming for easy identification
    """
    
    _instance = None
    _lock = threading.Lock()
    _loggers: Dict[str, logging.Logger] = {}
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super(LoggerManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            with self._lock:
                if not self._initialized:
                    self._setup_logging()
                    self._initialized = True
    
    def _setup_logging(self):
        """Setup the base logging configuration"""
        # Create logs directory if it doesn't exist
        logs_dir = Path("logs")
        logs_dir.mkdir(exist_ok=True)
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(logging.DEBUG)
        
        # Clear any existing handlers
        root_logger.handlers.clear()
        
        # Console handler with colors and detailed format
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        console_formatter = ColoredFormatter(
            '%(asctime)s | %(levelname)-11s | %(name)-20s | %(funcName)-15s:%(lineno)-4d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(console_formatter)
        
        # File handler for all logs (rotating)
        file_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "all.log",
            maxBytes=10*1024*1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(logging.DEBUG)
        file_formatter = logging.Formatter(
            '%(asctime)s | %(levelname)-11s | %(name)-20s | %(funcName)-15s:%(lineno)-4d | %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(file_formatter)
        
        # Error file handler for errors and above
        error_handler = logging.handlers.RotatingFileHandler(
            logs_dir / "errors.log",
            maxBytes=5*1024*1024,  # 5MB
            backupCount=3
        )
        error_handler.setLevel(logging.ERROR)
        error_handler.setFormatter(file_formatter)
        
        # Wrap handlers with traceback functionality
        console_handler_with_tb = TracebackHandler(console_handler)
        file_handler_with_tb = TracebackHandler(file_handler)
        error_handler_with_tb = TracebackHandler(error_handler)
        
        # Add handlers to root logger
        root_logger.addHandler(console_handler_with_tb)
        root_logger.addHandler(file_handler_with_tb)
        root_logger.addHandler(error_handler_with_tb)
        
        # Set specific library log levels to reduce noise
        logging.getLogger('urllib3').setLevel(logging.WARNING)
        logging.getLogger('requests').setLevel(logging.WARNING)
        logging.getLogger('cv2').setLevel(logging.WARNING)
        logging.getLogger('PIL').setLevel(logging.WARNING)
        logging.getLogger('matplotlib').setLevel(logging.WARNING)
        
    def get_logger(self, name: str = None, module_name: str = None) -> logging.Logger:
        """
        Get a logger instance for a specific module or component.
        
        Args:
            name: Custom name for the logger (e.g., "APIClient", "FaceMatcher")
            module_name: Usually __name__ from the calling module
            
        Returns:
            logging.Logger: Configured logger instance with consistent formatting
            
        Examples:
            logger = get_logger(module_name=__name__)  # Uses module name
            logger = get_logger(name="CustomComponent")  # Uses custom name
        """
        if name is None and module_name is None:
            raise ValueError("Either name or module_name must be provided")
        
        logger_name = name if name else module_name
        
        # Clean up module names for better readability
        if logger_name and logger_name.startswith('__main__'):
            logger_name = 'main'
        elif logger_name and '.' in logger_name:
            logger_name = logger_name.split('.')[-1]
        
        # Ensure logger name doesn't exceed reasonable length
        if logger_name and len(logger_name) > 20:
            logger_name = logger_name[-20:]
            
        # Use a default name if somehow we still don't have one
        logger_name = logger_name or 'unknown'
        
        if logger_name not in self._loggers:
            logger = logging.getLogger(logger_name)
            # Don't set level here - let it inherit from root
            self._loggers[logger_name] = logger
        
        return self._loggers[logger_name]
    
    def set_level(self, level: Union[int, str]):
        """
        Set logging level for all handlers
        
        Args:
            level: Logging level (int or string like 'DEBUG', 'INFO', etc.)
        """
        if isinstance(level, str):
            level = getattr(logging, level.upper(), logging.INFO)
            
        root_logger = logging.getLogger()
        root_logger.setLevel(level)
        
        # Update console handler level
        for handler in root_logger.handlers:
            if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
                handler.setLevel(level)
                
    def add_custom_handler(self, handler: logging.Handler):
        """Add a custom handler to the root logger"""
        logging.getLogger().addHandler(handler)
        
    def get_all_loggers(self) -> Dict[str, logging.Logger]:
        """Get all registered loggers for debugging purposes"""
        return self._loggers.copy()
        
    def reset_loggers(self):
        """Reset all loggers (mainly for testing)"""
        with self._lock:
            self._loggers.clear()
            root_logger = logging.getLogger()
            for handler in root_logger.handlers[:]:
                root_logger.removeHandler(handler)


# Global logger manager instance
_logger_manager = LoggerManager()


def get_logger(name: str = None, module_name: str = None) -> logging.Logger:
    """
    Convenience function to get a logger
    
    Usage:
        # In any module:
        from logger_config import get_logger
        logger = get_logger(module_name=__name__)
        
        # Or with custom name:
        logger = get_logger(name="custom_component")
    
    Args:
        name: Custom name for the logger
        module_name: Usually __name__ from the calling module
        
    Returns:
        logging.Logger: Configured logger instance
    """
    return _logger_manager.get_logger(name=name, module_name=module_name)


def set_log_level(level: str):
    """
    Set the logging level globally
    
    Args:
        level: Log level ('DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL')
    """
    level_map = {
        'DEBUG': logging.DEBUG,
        'INFO': logging.INFO,
        'WARNING': logging.WARNING,
        'ERROR': logging.ERROR,
        'CRITICAL': logging.CRITICAL
    }
    
    if level.upper() in level_map:
        _logger_manager.set_level(level_map[level.upper()])
    else:
        print(f"Invalid log level: {level}. Using INFO.")
        _logger_manager.set_level(logging.INFO)


# Additional convenience functions
def log_function_entry(logger: logging.Logger, func_name: str, **kwargs):
    """
    Log function entry with parameters (useful for debugging)
    
    Usage:
        def my_function(param1, param2):
            log_function_entry(logger, 'my_function', param1=param1, param2=param2)
    """
    if kwargs:
        params_str = ", ".join([f"{k}={v}" for k, v in kwargs.items()])
        logger.debug(f"→ Entering {func_name}({params_str})")
    else:
        logger.debug(f"→ Entering {func_name}()")

def log_function_exit(logger: logging.Logger, func_name: str, result=None, execution_time: float = None):
    """
    Log function exit with optional result and execution time
    
    Usage:
        result = some_computation()
        log_function_exit(logger, 'my_function', result=result, execution_time=0.123)
    """
    msg_parts = [f"← Exiting {func_name}"]
    if result is not None:
        msg_parts.append(f"result={result}")
    if execution_time is not None:
        msg_parts.append(f"time={execution_time:.3f}s")
    logger.debug(" | ".join(msg_parts))

def log_performance(logger: logging.Logger, operation: str, execution_time: float, threshold: float = 1.0):
    """
    Log performance metrics, with warnings for slow operations
    
    Args:
        logger: Logger instance
        operation: Description of the operation
        execution_time: Time taken in seconds
        threshold: Warning threshold in seconds
    """
    if execution_time > threshold:
        logger.warning(f"⚠️  SLOW OPERATION: {operation} took {execution_time:.3f}s (threshold: {threshold}s)")
    else:
        logger.debug(f"⚡ {operation} completed in {execution_time:.3f}s")

def log_error_with_context(logger: logging.Logger, error: Exception, context: Dict[str, Any] = None, include_traceback: bool = True):
    """
    Log errors with additional context information
    
    Args:
        logger: Logger instance
        error: The exception that occurred
        context: Additional context information
        include_traceback: Whether to include full traceback
    """
    error_msg = f"❌ ERROR: {type(error).__name__}: {str(error)}"
    
    if context:
        context_str = ", ".join([f"{k}={v}" for k, v in context.items()])
        error_msg += f" | Context: {context_str}"
    
    logger.error(error_msg)
    
    if include_traceback:
        import traceback
        logger.error(f"Full traceback:\n{traceback.format_exc()}")

def create_class_logger(cls) -> logging.Logger:
    """
    Create a logger for a class instance
    
    Usage:
        class MyClass:
            def __init__(self):
                self.logger = create_class_logger(self)
    """
    class_name = cls.__class__.__name__
    return get_logger(name=class_name)

def configure_third_party_loggers():
    """Configure third-party library loggers to reduce noise"""
    third_party_loggers = [
        'urllib3', 'requests', 'cv2', 'PIL', 'matplotlib',
        'insightface', 'onnxruntime', 'ultralytics', 'torch',
        'transformers', 'tensorflow', 'keras'
    ]
    
    for logger_name in third_party_loggers:
        logging.getLogger(logger_name).setLevel(logging.WARNING)


# Initialize third-party logger configuration
configure_third_party_loggers()
