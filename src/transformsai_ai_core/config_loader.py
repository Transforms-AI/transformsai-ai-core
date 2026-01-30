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
# Model Download Configuration
# ==============================================================================
# Set this to your model download API base URL
DOWNLOAD_BASE_URL = ""

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
    model_config: dict[str, Any],
    base_dir: Path,
    download_base_url: str,
) -> Path:
    """
    Download model from API endpoint using gdown for Google Drive links.

    Attempts to download via model_key or model_type. On failure, returns
    fallback path to allow graceful continuation.

    Args:
        model_name: Model identifier
        model_config: Model configuration dict with download_key/type fields
        base_dir: Project root directory
        download_base_url: API base URL for model downloads

    Returns:
        Path to downloaded model file or directory
    """
    import os
    import shutil
    import zipfile
    import requests
    import gdown
    from .central_logger import get_logger

    logger = get_logger()
    model_dir = get_model_dir(base_dir, model_name)
    model_dir.mkdir(parents=True, exist_ok=True)
    fallback_path = model_dir / f"{model_name}"

    if not download_base_url:
        logger.warning(f"No download_base_url set, skipping download for {model_name}")
        return fallback_path

    try:
        # Determine API endpoint based on download_key or type
        download_key = model_config.get("download_key", "")
        model_type = model_config.get("type", "")

        if download_key:
            endpoint = f"{download_base_url}/models/download/key/{download_key}"
            logger.info(f"Fetching model by key: {download_key}")
            response = requests.get(endpoint, timeout=30)
        elif model_type:
            endpoint = f"{download_base_url}/models/download/latest"
            params = {"model_type_name": model_type}
            logger.info(f"Fetching latest model for type: {model_type}")
            response = requests.get(endpoint, params=params, timeout=30)
        else:
            logger.error(f"No download_key or type specified for model {model_name}")
            return fallback_path

        response.raise_for_status()
        data = response.json()

        # Extract download URL from API response
        download_url = data.get("download_url") or data.get("data", {}).get("download_url")
        if not download_url:
            logger.error(f"No download_url in API response for {model_name}")
            return fallback_path

        logger.info(f"Downloading model '{model_name}' from {download_url}")

        # Download using gdown (handles Google Drive links)
        downloaded_path = gdown.download(download_url, output=None, quiet=False, fuzzy=True)
        if not downloaded_path:
            logger.error(f"gdown failed to download {model_name}")
            return fallback_path

        # Move downloaded file to models directory
        filename = Path(downloaded_path).name
        target_path = model_dir / filename

        if target_path.exists():
            if target_path.is_dir():
                shutil.rmtree(target_path)
            else:
                os.remove(target_path)

        shutil.move(downloaded_path, target_path)
        logger.info(f"File downloaded to: {target_path}")

        final_path = target_path

        # Extract if zip file
        if target_path.suffix.lower() == ".zip":
            logger.info(f"Extracting zip file: {filename}")
            try:
                with zipfile.ZipFile(target_path, "r") as zip_ref:
                    zip_ref.extractall(model_dir)

                os.remove(target_path)

                # Scan for first subfolder in extracted content
                extracted_items = list(model_dir.iterdir())
                subfolders = [item for item in extracted_items if item.is_dir()]

                if subfolders:
                    final_path = subfolders[0]
                    logger.info(f"Extracted to folder: {final_path}")
                else:
                    # No subfolder, files extracted directly
                    final_path = model_dir
                    logger.info(f"Files extracted to: {final_path}")

            except zipfile.BadZipFile:
                logger.error(f"Downloaded file is not a valid zip: {target_path}")
                return fallback_path

        return final_path

    except requests.exceptions.RequestException as e:
        logger.error(f"API request failed for {model_name}: {str(e)}")
        return fallback_path
    except Exception as e:
        logger.error(f"Error downloading model {model_name}: {str(e)}")
        return fallback_path


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
    models = config.get("advanced", {}).get("models", {})

    for model_name, model in models.items():
        if not model_name:
            continue

        model_dir = get_model_dir(base_dir, model_name)
        expected_path = model_dir / f"{model_name}"
        # NOTE: We will assume extension is pt if none given, and we will provide extension
        # if its something specific like .rknn
        # TODO: Analyze and find if will cause future issues

        if expected_path.exists():
            model["path"] = str(expected_path)
        elif download:
            downloaded_path = download_model(model_name, model, base_dir, DOWNLOAD_BASE_URL)
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
        "advanced.models": ["download_key", "type", "batch", "path", "load_options"],
        "advanced.models.*.load_options": ["lib_type", "task"],
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
