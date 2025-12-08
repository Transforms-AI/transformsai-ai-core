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


class CameraConfig(BaseModel):
    """Single camera configuration. SN is auto-generated at runtime."""

    local: bool = Field(default=False, description="Use local file instead of RTSP stream")
    local_source: str = Field(default="", description="Path to local video file")
    rtsp_source: RtspSource = Field(default_factory=RtspSource, description="RTSP connection settings")
    settings: dict[str, Any] = Field(default_factory=dict, description="Freeform per-camera settings")


# ==============================================================================
# Model Configuration
# ==============================================================================
class LoadOptions(BaseModel):
    """Model loading parameters."""

    lib_type: str = Field(default="YOLO", description="Model library type (YOLO, YOLOE, etc.)")
    task: str = Field(default="detect", description="Model task type (detect, classify, segment)")


class ModelConfig(BaseModel):
    """AI model definition for Ultralytics wrapper."""

    download_key: str = Field(default="", description="Direct download key from server")
    type: str = Field(..., description="Model type/category (e.g., face-det, person-det)")
    batch: int = Field(default=1, description="Batch size for inference")
    path: str = Field(default="", description="Local path, auto-populated after download")
    load_options: LoadOptions = Field(default_factory=LoadOptions, description="Model loading parameters")
    export_options: dict[str, Any] = Field(default_factory=dict, description="Freeform export options")


# ==============================================================================
# Datasend Configuration
# ==============================================================================
class DatasendConfig(BaseModel):
    """Backend API communication settings."""

    enabled: bool = Field(default=True, description="Master switch for data sending")
    base_url: str = Field(default="", description="API base URL")
    endpoints: dict[str, str] = Field(default_factory=dict, description="Freeform endpoint paths")
    secret_keys: list[str] = Field(default_factory=list, description="API authentication keys")
    settings: dict[str, Any] = Field(default_factory=dict, description="Freeform settings (jpeg_quality, etc.)")


# ==============================================================================
# Livestream Configuration
# ==============================================================================
class LivestreamConfig(BaseModel):
    """MediaMTX RTSP streaming settings."""

    enabled: bool = Field(default=True, description="Master switch for livestreaming")
    mediamtx_ip: str = Field(default="localhost", description="MediaMTX server IP")
    rtsp_port: int = Field(default=8554, description="MediaMTX RTSP port")
    settings: dict[str, Any] = Field(default_factory=dict, description="Freeform settings (fps, draw_annotations, etc.)")


# ==============================================================================
# Advanced Configuration
# ==============================================================================
class AdvancedConfig(BaseModel):
    """Advanced settings section. Password-protected in admin UI."""

    models: dict[str, ModelConfig] = Field(default_factory=dict, description="AI model definitions")
    timings: dict[str, Any] = Field(default_factory=dict, description="Freeform timing values")
    datasend: DatasendConfig = Field(default_factory=DatasendConfig, description="API settings")
    livestream: LivestreamConfig = Field(default_factory=LivestreamConfig, description="Streaming settings")
    pipeline: dict[str, Any] = Field(default_factory=dict, description="Freeform project-specific settings")


# ==============================================================================
# Root Configuration
# ==============================================================================
class AppConfig(BaseModel):
    """Root configuration model for all TransformsAI projects."""

    meta: MetaConfig = Field(..., description="Project metadata")
    cameras: list[CameraConfig] = Field(default_factory=list, description="Camera definitions")
    advanced: AdvancedConfig = Field(default_factory=AdvancedConfig, description="Advanced settings")
