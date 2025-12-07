"""
Configuration Loader for TransformsAI Projects.

Provides utilities for loading, saving, and processing config.yaml files.
Includes RTSP URL building and model download functionality.
"""

import hashlib
import uuid
from pathlib import Path
from typing import Any
from urllib.parse import quote

import yaml

from .config_schema import AppConfig, CameraConfig


# ==============================================================================
# RTSP URL Builder
# ==============================================================================
def build_rtsp_url(rtsp_source: dict[str, Any]) -> str:
    """
    Build RTSP URL from decomposed components.

    Args:
        rtsp_source: Dict with username, password, ip, port, path

    Returns:
        Full RTSP URL string
    """
    username = rtsp_source.get("username", "")
    password = rtsp_source.get("password", "")
    ip = rtsp_source.get("ip", "")
    port = rtsp_source.get("port", 554)
    path = rtsp_source.get("path", "/Streaming/Channels/101")

    if not ip:
        raise ValueError("RTSP IP address is required")

    # URL-encode password for special characters
    auth = ""
    if username and password:
        auth = f"{username}:{quote(password, safe='')}@"
    elif username:
        auth = f"{username}@"

    return f"rtsp://{auth}{ip}:{port}{path}"


# ==============================================================================
# Model Path Resolver
# ==============================================================================
def get_model_dir(base_dir: Path, model_name: str) -> Path:
    """
    Get the standard model directory path.

    Structure: {base_dir}/models/{model_name}/

    Args:
        base_dir: Project root directory
        model_name: Model identifier

    Returns:
        Path to model directory
    """
    return base_dir / "models" / model_name


def download_model(
    model_name: str,
    base_dir: Path,
    api_url: str = "http://localhost:8000/api/models/download",
) -> Path:
    """
    Download model from model server and return local path.

    Placeholder implementation - actual download logic to be added.

    Args:
        model_name: Model identifier
        base_dir: Project root directory
        api_url: Model server API URL

    Returns:
        Path to downloaded model file
    """
    model_dir = get_model_dir(base_dir, model_name)
    model_dir.mkdir(parents=True, exist_ok=True)

    model_path = model_dir / f"{model_name}"

    # TODO: Implement actual download
    # url = f"{api_url}?name={model_name}&backend={backend}"
    # response = requests.get(url, stream=True)
    # with open(model_path, 'wb') as f:
    #     for chunk in response.iter_content(chunk_size=8192):
    #         f.write(chunk)

    return model_path


def resolve_model_paths(config: dict[str, Any], base_dir: Path, download: bool = False) -> dict[str, Any]:
    """
    Resolve model paths in config, optionally downloading missing models.

    Args:
        config: Full config dict
        base_dir: Project root directory
        download: If True, download missing models

    Returns:
        Updated config with resolved model paths
    """
    models = config.get("advanced", {}).get("models", [])

    for model in models:
        model_name = model.get("name", "")

        if not model_name:
            continue

        model_dir = get_model_dir(base_dir, model_name)
        expected_path = model_dir / f"{model_name}"
        # NOTE: We will assume extenstion is pt if none given, and we will provide extension
        # if its something specific like .rknn
        # TODO: Analyze and find if will cause future issues

        if expected_path.exists():
            model["path"] = str(expected_path)
        elif download:
            downloaded_path = download_model(model_name, base_dir)
            model["path"] = str(downloaded_path)

    return config


# ==============================================================================
# Config Loading / Saving
# ==============================================================================
def load_config(config_path: str | Path, validate: bool = True) -> dict[str, Any]:
    """
    Load and optionally validate config from YAML file.

    Args:
        config_path: Path to config.yaml
        validate: If True, validate against Pydantic schema

    Returns:
        Config as dictionary
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    if validate:
        # Validate and get defaults
        validated = AppConfig(**config)
        config = validated.model_dump()

    return config


def save_config(config_path: str | Path, config: dict[str, Any]) -> None:
    """
    Save config to YAML file.

    Args:
        config_path: Path to config.yaml
        config: Config dictionary
    """
    config_path = Path(config_path)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False, allow_unicode=True)


# ==============================================================================
# Config Processing
# ==============================================================================
def process_config(
    config_path: str | Path,
    base_dir: str | Path | None = None,
    resolve_models: bool = False,
    download_models: bool = False,
) -> dict[str, Any]:
    """
    Load config and process for runtime use.

    Generates camera SNs and optionally resolves model paths.

    Args:
        config_path: Path to config.yaml
        base_dir: Project root directory (defaults to config parent)
        resolve_models: If True, resolve model paths
        download_models: If True, download missing models

    Returns:
        Processed config ready for use
    """
    config_path = Path(config_path)
    base_dir = Path(base_dir) if base_dir else config_path.parent

    config = load_config(config_path, validate=True)

    # Build camera urls
    cameras = config.get("cameras", [])
    for i, camera in enumerate(cameras):
        # Build RTSP URL if not local
        if not camera.get("local", False):
            camera["rtsp_url"] = build_rtsp_url(camera.get("rtsp_source", {}))

    # Resolve model paths
    if resolve_models:
        config = resolve_model_paths(config, base_dir, download=download_models)

    return config


# ==============================================================================
# Schema Helpers for Admin UI
# ==============================================================================
def get_formatted_fields() -> dict[str, list[str]]:
    """
    Return list of formatted (rigid schema) fields for admin UI.

    These fields have strict types and should render as proper widgets.
    Freeform fields should render as key-value editors.

    Returns:
        Dict mapping section to list of formatted field names
    """
    return {
        "meta": ["name", "version", "token"],
        "cameras": ["local", "local_source", "rtsp_source"],
        "cameras.rtsp_source": ["username", "password", "ip", "port", "path"],
        "advanced.models": ["name", "type", "task", "path", "batch", "backend"],
        "advanced.datasend": ["enabled", "base_url", "secret_keys"],
        "advanced.livestream": ["enabled", "mediamtx_ip", "rtsp_port"],
    }


def get_freeform_fields() -> list[str]:
    """
    Return list of freeform fields for admin UI.

    These should render as dynamic key-value editors.

    Returns:
        List of freeform field paths
    """
    return [
        "cameras.*.settings",
        "advanced.models.*.export_options",
        "advanced.timings",
        "advanced.datasend.endpoints",
        "advanced.datasend.settings",
        "advanced.livestream.settings",
        "advanced.pipeline",
    ]
