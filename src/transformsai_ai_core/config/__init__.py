"""
transformsai-ai-core config: typed models, a lenient runtime loader and a strict gate.

    from transformsai_ai_core.config import load_config, validate_config

    config, report = load_config("config.yaml")     # runtime: warnings don't stop you
    for issue in report.warnings:
        logger.warning(str(issue))

    report = validate_config("config.yaml")          # pre-deploy gate: strict
    if not report.ok:
        print(report); sys.exit(1)

Imports only pydantic, PyYAML and the stdlib (no OpenCV or torch).
"""

from .example import generate_example
from .loader import (
    DOWNLOAD_BASE_URL,
    download_model,
    load_config,
    save_config,
    validate_config,
)
from .registry import clear_registry, register_secret, register_validator
from .report import ConfigError, Issue, ValidationReport
from .schema import (
    AUDIO_INPUT_TYPES,
    SOURCE_TYPES,
    AlsaAudioInput,
    ApiConfig,
    AudioConfig,
    AudioFilter,
    AudioRetryConfig,
    CameraAudioInput,
    CameraConfig,
    CaptureConfig,
    Config,
    DemandConfig,
    DeviceSource,
    EncoderConfig,
    EndpointConfig,
    FileSource,
    LoggingConfig,
    MediaMTXConfig,
    ModelConfig,
    NodeConfig,
    OutboxConfig,
    PoolConfig,
    ResolvedAudioFilter,
    RestartConfig,
    RetriesConfig,
    RtspSource,
    SourceBase,
    StreamConfig,
    UrlSource,
    mask_url,
)
from .types import format_bitrate, parse_bitrate, parse_duration
from .urls import StreamUrls, build_stream_urls

__all__ = [
    # loading
    "load_config", "validate_config", "save_config", "generate_example",
    "download_model", "DOWNLOAD_BASE_URL",
    # reports
    "ValidationReport", "Issue", "ConfigError",
    # project hooks
    "register_validator", "register_secret", "clear_registry",
    # models
    "Config", "NodeConfig", "CameraConfig", "CaptureConfig", "RestartConfig",
    "SourceBase", "RtspSource", "DeviceSource", "FileSource", "UrlSource", "SOURCE_TYPES",
    "StreamConfig", "EncoderConfig", "AudioConfig", "AudioFilter", "AudioRetryConfig",
    "AlsaAudioInput", "CameraAudioInput", "AUDIO_INPUT_TYPES", "ResolvedAudioFilter",
    "MediaMTXConfig", "DemandConfig",
    "ApiConfig", "RetriesConfig", "OutboxConfig", "PoolConfig", "EndpointConfig",
    "ModelConfig", "LoggingConfig",
    # helpers
    "StreamUrls", "build_stream_urls", "mask_url",
    "parse_duration", "parse_bitrate", "format_bitrate",
]
