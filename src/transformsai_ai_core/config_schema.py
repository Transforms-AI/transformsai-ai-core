"""
Configuration Schema for TransformsAI Projects.

Defines Pydantic models for validating and structuring config.yaml files.
Formatted sections have strict schemas, freeform sections use dict[str, Any].
"""

from typing import Any, Literal
from pydantic import BaseModel, Field


# ==============================================================================
# Meta Configuration
# ==============================================================================
class MetaConfig(BaseModel):
    """Project metadata. Read-only except token."""

    name: str = Field(..., description="Project identifier (e.g., 'people-count', 'sentiment')")
    version: str = Field(default="1.0.0", description="Config schema version")
    token: str = Field(default="", description="Hardware token, auto-generated on first run")
    extras: dict[str, Any] = Field(default_factory=dict, description="Sanctioned freeform channel for unmodeled fields")


# ==============================================================================
# Camera Configuration
# ==============================================================================
class RtspSource(BaseModel):
    """RTSP connection parameters for network cameras."""

    username: str = Field(default="admin", description="RTSP username")
    password: str = Field(default="", description="RTSP password")
    ip: str = Field(default="", description="Camera IP address")
    port: int = Field(default=554, description="RTSP port")
    path: str = Field(default="/Streaming/Channels/101", description="RTSP stream path")
    extras: dict[str, Any] = Field(default_factory=dict, description="Sanctioned freeform channel for unmodeled fields")


class RestartSettings(BaseModel):
    """
    Reconnect policy for VideoCaptureAsync.

    `enabled` and `delay` default to None so an absent block never outranks the legacy
    flat `capture.auto_restart_on_fail` / `capture.restart_delay` keys — Pydantic
    materializes nested defaults on model_dump(), so a concrete default here would
    silently win. VideoCaptureAsync.__init__ holds the real defaults.
    """

    enabled: bool | None = Field(default=None, description="Auto-restart on failure (default: true; falls back to capture.auto_restart_on_fail)")
    delay: float | None = Field(default=None, description="Backoff ceiling in seconds (default: 30.0; falls back to capture.restart_delay)")
    backoff_start: float = Field(default=1.0, description="First reconnect delay in seconds; doubles up to `delay`")
    backoff_jitter: float = Field(default=0.2, description="Jitter fraction applied to each backoff delay (0 = off)")
    reset_after: float = Field(default=30.0, description="Seconds of healthy streaming before the backoff ramp resets")
    max_attempts: int | None = Field(default=None, description="Consecutive reconnect attempts before giving up (null = forever)")
    stall_timeout: float | None = Field(default=None, description="Seconds without a new frame before forcing a reconnect (null = auto max(5, 20/fps), 0 = off)")
    extras: dict[str, Any] = Field(default_factory=dict, description="Sanctioned freeform channel for unmodeled fields")


class CaptureSettings(BaseModel):
    """Video capture optimization settings."""

    buffer_size: int = Field(default=1, description="OpenCV buffer size (1 = minimal lag)")
    opencv_backend: str | None = Field(default="auto", description="Backend: 'auto', 'ffmpeg', 'gstreamer', None")
    max_frame_age_ms: int | None = Field(default=None, description="Drop frames older than this (ms)")
    width: int | None = Field(default=None, description="Capture width (None = native)")
    height: int | None = Field(default=None, description="Capture height (None = native)")
    driver: str | None = Field(default=None, description="Optional driver hint")
    auto_restart_on_fail: bool | None = Field(default=None, description="LEGACY: use restart.enabled (default: true)")
    restart_delay: float | None = Field(default=None, description="LEGACY: use restart.delay (default: 30.0)")
    auto_resize: bool = Field(default=True, description="Auto-resize if dimensions don't match")
    hw_decode: bool = Field(default=False, description="Enable hardware-accelerated decoding")
    fps: int | None = Field(default=None, description="Target frames per second (None = native)")
    open_timeout: float = Field(default=5.0, description="Network open/read timeout in seconds (applied via FFmpeg)")
    rtsp_transport: str | None = Field(default="tcp", description="RTSP transport: 'tcp', 'udp', or null to let FFmpeg negotiate")
    ffmpeg_options: str | None = Field(default=None, description="Verbatim OPENCV_FFMPEG_CAPTURE_OPTIONS override")
    health_log_interval: float = Field(default=60.0, description="Seconds between periodic capture health log lines (0 = off)")
    restart: RestartSettings = Field(default_factory=RestartSettings, description="Reconnect policy")
    extras: dict[str, Any] = Field(default_factory=dict, description="Sanctioned freeform channel for unmodeled fields")


class CameraConfig(BaseModel):
    """Single camera configuration. SN is auto-generated at runtime."""

    local: bool = Field(default=False, description="Use local file instead of RTSP stream")
    local_source: str = Field(default="", description="Path to local video file")
    rtsp_source: RtspSource = Field(default_factory=RtspSource, description="RTSP connection settings")
    capture: CaptureSettings = Field(default_factory=CaptureSettings, description="Capture optimization settings")
    settings: dict[str, Any] = Field(default_factory=dict, description="Freeform per-camera settings")
    extras: dict[str, Any] = Field(default_factory=dict, description="Sanctioned freeform channel for unmodeled fields")


# ==============================================================================
# Model Configuration
# ==============================================================================
class LoadOptions(BaseModel):
    """Model loading parameters."""

    lib_type: str = Field(default="YOLO", description="Model library type (YOLO, YOLOE, etc.)")
    task: str = Field(default="detect", description="Model task type (detect, classify, segment)")
    extras: dict[str, Any] = Field(default_factory=dict, description="Sanctioned freeform channel for unmodeled fields")


class ModelConfig(BaseModel):
    """AI model definition for Ultralytics wrapper."""

    download_key: str = Field(default="", description="Direct download key from server")
    type: str = Field(..., description="Model type/category (e.g., face-det, person-det)")
    batch: int = Field(default=1, description="Batch size for inference")
    path: str = Field(default="", description="Local path, auto-populated after download")
    load_options: LoadOptions = Field(default_factory=LoadOptions, description="Model loading parameters")
    export: bool = Field(default=False, description="Enable automatic model export")
    export_options: dict[str, Any] = Field(default_factory=dict, description="Freeform export options")
    extras: dict[str, Any] = Field(default_factory=dict, description="Sanctioned freeform channel for unmodeled fields")


# ==============================================================================
# Api Configuration (unified — replaces legacy datasend)
# ==============================================================================
class ApiConfig(BaseModel):
    """ApiClient configuration — mirror of the constructor."""

    enabled: bool = Field(default=True, description="Master switch for the API client")
    base_url: str = Field(default="", description="API base URL")
    headers: dict[str, str] = Field(default_factory=dict, description="Default request headers")
    timeout: int = Field(default=30, description="Request timeout in seconds")
    success_codes: list[int] = Field(default_factory=lambda: [200, 201, 202, 204], description="HTTP status codes considered successful")
    default_content_type: str = Field(default="auto", description="Default content type: 'auto', 'json', 'form'")
    auth_keys: list[str] = Field(default_factory=list, description="API authentication keys (rotated per request)")
    auth_header: str = Field(default="X-Secret-Key", description="Header name used to send the auth key")
    max_retries: int = Field(default=3, description="Maximum retry attempts on failure")
    retry_backoff: float = Field(default=1.0, description="Base backoff delay in seconds")
    retry_backoff_max: float = Field(default=30, description="Maximum backoff delay in seconds")
    retry_on_status: list[int] = Field(default_factory=lambda: [408, 429, 500, 502, 503, 504], description="HTTP status codes that trigger retry")
    max_workers: int | None = Field(default=None, description="Thread pool size (None = auto)")
    cache_enabled: bool = Field(default=True, description="Enable request caching")
    cache_dir: str = Field(default=".core-api-cache", description="Cache directory path")
    cache_retry_interval: int = Field(default=100, description="Cache retry interval in seconds")
    max_cache_items: int = Field(default=300, description="Max cached requests")
    max_cache_age_seconds: int = Field(default=86400, description="Max cache age in seconds (24h)")
    max_cache_retries: int = Field(default=5, description="Max cache retry attempts")
    endpoints: dict[str, Any] = Field(default_factory=dict, description="Freeform endpoint profiles (string path or rich dict)")
    pool_connections: int = Field(default=10, description="HTTP connection pool size")
    pool_maxsize: int = Field(default=10, description="HTTP connection pool max size")
    settings: dict[str, Any] = Field(default_factory=dict, description="Freeform settings (jpeg_quality, etc.)")
    extras: dict[str, Any] = Field(default_factory=dict, description="Sanctioned freeform channel for unmodeled fields")


# ==============================================================================
# Datasend Configuration (legacy — kept for backward compat imports)
# ==============================================================================
class DatasendConfig(ApiConfig):
    """Legacy alias. Use ApiConfig instead."""
    pass


# ==============================================================================
# Livestream Configuration
# ==============================================================================
class StreamEncoderSettings(BaseModel):
    """FFmpeg encoder settings for resource optimization."""

    preset: str = Field(default="ultrafast", description="Encoder preset: ultrafast, fast, medium, slow")
    codec: str = Field(default="copy", description="Codec: 'copy' (no re-encode) or 'libx264'")
    queue_size: int = Field(default=2, description="Async frame queue depth (0 = sync)")
    extras: dict[str, Any] = Field(default_factory=dict, description="Sanctioned freeform channel for unmodeled fields")


class LivestreamConfig(BaseModel):
    """MediaMTX RTSP streaming settings. Mirrors MediaMTXStreamer constructor."""

    enabled: bool = Field(default=True, description="Master switch for livestreaming")
    mediamtx_ip: str = Field(default="localhost", description="MediaMTX server IP")
    rtsp_port: int = Field(default=8554, description="MediaMTX RTSP port")
    camera_sn_id: str = Field(default="", description="Camera serial number (overridable per camera at call time)")
    fps: int = Field(default=30, description="Stream frames per second")
    frame_width: int = Field(default=1920, description="Stream frame width")
    frame_height: int = Field(default=1080, description="Stream frame height")
    bitrate: str = Field(default="1500k", description="Video bitrate")
    hw_encode: bool = Field(default=False, description="Auto-detect hardware encoder")
    debug_log_interval: float = Field(default=60.0, description="Debug log interval in seconds")
    encoder: StreamEncoderSettings = Field(default_factory=StreamEncoderSettings, description="Encoder optimization")
    on_demand: bool = Field(default=False, description="Only push the stream while a viewer is watching. False = always-on (current behavior)")
    demand_url: str = Field(default="", description="Optional override for the demand URL; blank = derive https://{mediamtx_ip}/demand/cam_sn_{camera_sn_id}")
    demand_poll_interval: float = Field(default=3.0, description="Seconds between outbound demand polls")
    demand_grace_period: float = Field(default=10.0, description="Seconds demand must stay OFF before stopping FFmpeg")
    demand_timeout: float = Field(default=5.0, description="HTTP timeout per demand poll (seconds)")
    settings: dict[str, Any] = Field(default_factory=dict, description="Freeform settings (NOT consumed by from_config)")
    extras: dict[str, Any] = Field(default_factory=dict, description="Sanctioned freeform channel for unmodeled fields")


# ==============================================================================
# Advanced Configuration
# ==============================================================================
class AdvancedConfig(BaseModel):
    """Advanced settings section. Password-protected in admin UI."""

    models: dict[str, ModelConfig] = Field(default_factory=dict, description="AI model definitions")
    timings: dict[str, Any] = Field(default_factory=dict, description="Freeform timing values")
    api: ApiConfig = Field(default_factory=ApiConfig, description="Unified API client settings")
    livestream: LivestreamConfig = Field(default_factory=LivestreamConfig, description="Streaming settings")
    pipeline: dict[str, Any] = Field(default_factory=dict, description="Freeform project-specific settings")
    extras: dict[str, Any] = Field(default_factory=dict, description="Sanctioned freeform channel for unmodeled fields")


# ==============================================================================
# Root Configuration
# ==============================================================================
class AppConfig(BaseModel):
    """Root configuration model for all TransformsAI projects."""

    meta: MetaConfig = Field(..., description="Project metadata")
    cameras: list[CameraConfig] = Field(default_factory=list, description="Camera definitions")
    advanced: AdvancedConfig = Field(default_factory=AdvancedConfig, description="Advanced settings")
