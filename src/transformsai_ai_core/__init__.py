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
    'DataUploader',
    'ApiClient',
    'Response',
    'MediaMTXStreamer',
    'VideoCaptureAsync',
    'time_to_string',
    'mat_to_response',
    # Config schema
    'AppConfig',
    'MetaConfig',
    'CameraConfig',
    'RtspSource',
    'CaptureSettings',
    'ModelConfig',
    'LoadOptions',
    'DatasendConfig',
    'ApiConfig',
    'StreamEncoderSettings',
    'LivestreamConfig',
    'AdvancedConfig',
    # Config loader
    'load_config',
    'save_config',
    'process_config',
    'build_rtsp_url',
    'resolve_model_paths',
    'download_model',
    'get_formatted_fields',
    'get_freeform_fields',
    'init_kwargs',
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
    'DataUploader': ('datasend', 'DataUploader'),
    'ApiClient': ('api_client', 'ApiClient'),
    'Response': ('api_client', 'Response'),
    'MediaMTXStreamer': ('mediamtx_streamer', 'MediaMTXStreamer'),
    'VideoCaptureAsync': ('video_capture', 'VideoCaptureAsync'),
    'time_to_string': ('utils', 'time_to_string'),
    'mat_to_response': ('utils', 'mat_to_response'),
    # Config schema
    'AppConfig': ('config_schema', 'AppConfig'),
    'MetaConfig': ('config_schema', 'MetaConfig'),
    'CameraConfig': ('config_schema', 'CameraConfig'),
    'RtspSource': ('config_schema', 'RtspSource'),
    'CaptureSettings': ('config_schema', 'CaptureSettings'),
    'ModelConfig': ('config_schema', 'ModelConfig'),
    'LoadOptions': ('config_schema', 'LoadOptions'),
    'DatasendConfig': ('config_schema', 'DatasendConfig'),
    'ApiConfig': ('config_schema', 'ApiConfig'),
    'StreamEncoderSettings': ('config_schema', 'StreamEncoderSettings'),
    'LivestreamConfig': ('config_schema', 'LivestreamConfig'),
    'AdvancedConfig': ('config_schema', 'AdvancedConfig'),
    # Config loader
    'load_config': ('config_loader', 'load_config'),
    'save_config': ('config_loader', 'save_config'),
    'process_config': ('config_loader', 'process_config'),
    'build_rtsp_url': ('config_loader', 'build_rtsp_url'),
    'resolve_model_paths': ('config_loader', 'resolve_model_paths'),
    'download_model': ('config_loader', 'download_model'),
    'get_formatted_fields': ('config_loader', 'get_formatted_fields'),
    'get_freeform_fields': ('config_loader', 'get_freeform_fields'),
    'init_kwargs': ('config_loader', 'init_kwargs'),
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