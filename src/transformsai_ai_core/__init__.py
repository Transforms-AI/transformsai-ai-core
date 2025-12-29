"""
TransformsAI Core Library
A collection of utilities for AI vision applications.
"""

__version__ = "1.0.0"
__author__ = "TransformsAI"

# Import main classes and functions for easy access
from .central_logger import get_logger
from .datasend import DataUploader
from .mediamtx_streamer import MediaMTXStreamer
from .video_capture import VideoCaptureAsync
from .utils import time_to_string, mat_to_response

# Config management
from .config_schema import (
    AppConfig,
    MetaConfig,
    CameraConfig,
    RtspSource,
    ModelConfig,
    DatasendConfig,
    LivestreamConfig,
    AdvancedConfig,
)
from .config_loader import (
    load_config,
    save_config,
    process_config,
    build_rtsp_url,
    resolve_model_paths,
    download_model,
    get_formatted_fields,
    get_freeform_fields,
)

# Optional YOLO/YOLOE wrappers (requires ultralytics)
try:
    from .yolo_wrapper import YOLOWrapper, YOLOEWrapper
    YOLO_AVAILABLE = True
except ImportError:
    YOLO_AVAILABLE = False
    YOLOWrapper = None
    YOLOEWrapper = None

__all__ = [
    # Existing
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