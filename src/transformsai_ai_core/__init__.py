"""
TransformsAI Core Library
A collection of utilities for AI vision applications.
"""

from importlib.metadata import version, PackageNotFoundError
try:
    __version__ = version("transformsai-ai-core")
except PackageNotFoundError:  # not installed (e.g. raw source tree)
    __version__ = "0.0.0"
__author__ = "TransformsAI"

# Check YOLO availability early (cheap check, no imports)
try:
    import importlib.util
    YOLO_AVAILABLE = importlib.util.find_spec("ultralytics") is not None
except (ImportError, AttributeError):
    YOLO_AVAILABLE = False

__all__ = [
    # Core utilities
    'get_logger',
    'configure_logging',
    'add_secret',
    'redact',
    'ApiClient',
    'Response',
    'MediaMTXStreamer',
    'VideoCaptureAsync',
    'time_to_string',
    'mat_to_response',
    # Config (transformsai_ai_core.config)
    'Config',
    'NodeConfig',
    'CameraConfig',
    'CaptureConfig',
    'RestartConfig',
    'RtspSource',
    'DeviceSource',
    'FileSource',
    'UrlSource',
    'StreamConfig',
    'EncoderConfig',
    'AudioConfig',
    'MediaMTXConfig',
    'ApiConfig',
    'ModelConfig',
    'LoggingConfig',
    'ValidationReport',
    'Issue',
    'ConfigError',
    'load_config',
    'validate_config',
    'save_config',
    'generate_example',
    'register_validator',
    'register_secret',
    # YOLO wrappers (optional)
    'YOLOWrapper',
    'YOLOEWrapper',
    'YOLO_AVAILABLE',
]

# Lazy imports to reduce memory footprint at import time
# All imports deferred until first access (saves ~50-100MB RSS on edge devices)
_LAZY_IMPORTS = {
    # Core utilities
    'get_logger': ('central_logger', 'get_logger'),
    'configure_logging': ('central_logger', 'configure_logging'),
    'add_secret': ('redact', 'add_secret'),
    'redact': ('redact', 'redact'),
    'ApiClient': ('api_client', 'ApiClient'),
    'Response': ('api_client', 'Response'),
    'MediaMTXStreamer': ('mediamtx_streamer', 'MediaMTXStreamer'),
    'VideoCaptureAsync': ('video_capture', 'VideoCaptureAsync'),
    'time_to_string': ('utils', 'time_to_string'),
    'mat_to_response': ('utils', 'mat_to_response'),
    # Config
    'Config': ('config', 'Config'),
    'NodeConfig': ('config', 'NodeConfig'),
    'CameraConfig': ('config', 'CameraConfig'),
    'CaptureConfig': ('config', 'CaptureConfig'),
    'RestartConfig': ('config', 'RestartConfig'),
    'RtspSource': ('config', 'RtspSource'),
    'DeviceSource': ('config', 'DeviceSource'),
    'FileSource': ('config', 'FileSource'),
    'UrlSource': ('config', 'UrlSource'),
    'StreamConfig': ('config', 'StreamConfig'),
    'EncoderConfig': ('config', 'EncoderConfig'),
    'AudioConfig': ('config', 'AudioConfig'),
    'MediaMTXConfig': ('config', 'MediaMTXConfig'),
    'ApiConfig': ('config', 'ApiConfig'),
    'ModelConfig': ('config', 'ModelConfig'),
    'LoggingConfig': ('config', 'LoggingConfig'),
    'ValidationReport': ('config', 'ValidationReport'),
    'Issue': ('config', 'Issue'),
    'ConfigError': ('config', 'ConfigError'),
    'load_config': ('config', 'load_config'),
    'validate_config': ('config', 'validate_config'),
    'save_config': ('config', 'save_config'),
    'generate_example': ('config', 'generate_example'),
    'register_validator': ('config', 'register_validator'),
    'register_secret': ('config', 'register_secret'),
    # YOLO wrappers
    'YOLOWrapper': ('yolo_wrapper', 'YOLOWrapper'),
    'YOLOEWrapper': ('yolo_wrapper', 'YOLOEWrapper'),
}

def __getattr__(name):
    """Lazy import mechanism for deferred module loading."""
    if name in _LAZY_IMPORTS:
        module_name, attr_name = _LAZY_IMPORTS[name]
        from importlib import import_module
        module = import_module(f'.{module_name}', __package__)
        value = getattr(module, attr_name)
        # Cache in module dict for future access
        globals()[name] = value
        return value
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")