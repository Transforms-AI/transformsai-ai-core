"""
TransformsAI Core Library
A collection of utilities for AI vision applications.
"""

__version__ = "1.0.0"
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
    'DataUploader', 
    'MediaMTXStreamer',
    'VideoCaptureAsync',
    'time_to_string',
    'mat_to_response',
    # Config schema
    'AppConfig',
    'MetaConfig',
    'CameraConfig',
    'RtspSource',
    'ModelConfig',
    'DatasendConfig',
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
    'DataUploader': ('datasend', 'DataUploader'),
    'MediaMTXStreamer': ('mediamtx_streamer', 'MediaMTXStreamer'),
    'VideoCaptureAsync': ('video_capture', 'VideoCaptureAsync'),
    'time_to_string': ('utils', 'time_to_string'),
    'mat_to_response': ('utils', 'mat_to_response'),
    # Config schema
    'AppConfig': ('config_schema', 'AppConfig'),
    'MetaConfig': ('config_schema', 'MetaConfig'),
    'CameraConfig': ('config_schema', 'CameraConfig'),
    'RtspSource': ('config_schema', 'RtspSource'),
    'ModelConfig': ('config_schema', 'ModelConfig'),
    'DatasendConfig': ('config_schema', 'DatasendConfig'),
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