"""
Loading, validating and saving configs, and downloading models.

    config, report = load_config("config.yaml")          # lenient: runtime load
    report = validate_config("config.yaml")              # strict: pre-deploy gate

Nothing here writes derived values into the config: the source URL, stream
names/URLs and model paths are properties of the returned ``Config``.
"""

import copy
import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import yaml

from .report import ConfigError, ValidationReport
from .schema import Config, ModelConfig
from .validate import validate_data

# Model download API base URL; downloads are skipped while it is empty.
DOWNLOAD_BASE_URL = ""

ConfigSource = str | os.PathLike | Mapping


def _read(source: Any, report: ValidationReport) -> tuple[Any, Path | None]:
    """Parsed data and the directory it came from (None for a mapping)."""
    if isinstance(source, Config):
        return source.model_dump(mode="json", exclude_unset=True), source.base_dir
    if isinstance(source, (str, os.PathLike)):
        path = Path(source)
        text = path.read_text(encoding="utf-8")  # FileNotFoundError is the caller's to see
        try:
            data = yaml.safe_load(text)
        except yaml.YAMLError as exc:
            report.error("", f"not valid YAML: {exc}")
            return None, path.parent
        return ({} if data is None else data), path.parent
    if isinstance(source, Mapping):
        return copy.deepcopy(dict(source)), None
    raise TypeError(f"expected a path or a mapping, got {type(source).__name__}")


def _merge(into: ValidationReport, other: ValidationReport) -> ValidationReport:
    into.errors.extend(other.errors)
    into.warnings.extend(other.warnings)
    return into


def load_config(source: ConfigSource, *, strict: bool = False,
                base_dir: str | os.PathLike | None = None) -> tuple[Config, ValidationReport]:
    """Load and validate a config file (or an already-parsed mapping).

    Lenient (default): unknown keys and invalid optional values are warnings, and
    invalid optional values fall back to their defaults. Strict: both are errors.
    Raises ``ConfigError`` (with the full report) when the config has errors --
    in lenient mode, only a missing/invalid required key, or a ``required``
    project validator, causes that. The caller decides what to do with warnings.

    Relative model paths resolve against `base_dir` (default: the config file's
    directory, or the working directory for a mapping).
    """
    report = ValidationReport()
    data, origin = _read(source, report)
    if not report.errors:
        config, result = validate_data(data, strict=strict)
        _merge(report, result)
    else:
        config = None
    if report.errors or config is None:
        if not report.errors:
            report.error("", "config could not be built")
        raise ConfigError(report)
    config.set_base_dir(Path(base_dir) if base_dir is not None else (origin or Path.cwd()))
    return config, report


def validate_config(source: ConfigSource) -> ValidationReport:
    """Strict validation with no side effects: the gate a config must pass before it is written or deployed."""
    report = ValidationReport()
    data, _ = _read(source, report)
    if report.errors:
        return report
    _, result = validate_data(data, strict=True)
    return _merge(report, result)


def save_config(path: str | os.PathLike, config: "Config | Mapping", *, validate: bool = True) -> ValidationReport:
    """Write a config as YAML, refusing (``ConfigError``) one that fails strict validation.

    A ``Config`` is written with the keys the user set only, so defaults are never
    materialised into the file.
    """
    if isinstance(config, Config):
        data = config.model_dump(mode="json", exclude_unset=True)
    else:
        data = copy.deepcopy(dict(config))
    report = validate_config(data) if validate else ValidationReport()
    report.raise_for_errors()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)
    return report


# ==============================================================================
# Model download
# ==============================================================================
def download_model(model: ModelConfig, base_url: str) -> Path | None:
    """Fetch one model by `download_key` (or the latest of its `type`) into model.model_dir.

    Uses gdown, so Google Drive links work. Returns the downloaded file or
    extracted directory, or None when the download didn't happen.
    """
    import shutil
    import zipfile

    import gdown
    import requests

    from ..central_logger import get_logger

    logger = get_logger()
    name = model.name
    if not base_url:
        logger.warning(f"No download base URL set, skipping download for {name}")
        return None

    model_dir = model.model_dir
    model_dir.mkdir(parents=True, exist_ok=True)
    try:
        if model.download_key:
            logger.info(f"Fetching model by key: {model.download_key}")
            response = requests.get(f"{base_url}/models/download/key/{model.download_key}", timeout=30)
        elif model.type:
            logger.info(f"Fetching latest model for type: {model.type}")
            response = requests.get(f"{base_url}/models/download/latest",
                                    params={"model_type_name": model.type}, timeout=30)
        else:
            logger.error(f"No download_key or type specified for model {name}")
            return None
        response.raise_for_status()
        data = response.json()

        download_url = data.get("download_url") or (data.get("data") or {}).get("download_url")
        if not download_url:
            logger.error(f"No download_url in API response for {name}")
            return None

        logger.info(f"Downloading model '{name}' from {download_url}")
        downloaded = gdown.download(download_url, output=None, quiet=False, fuzzy=True)
        if not downloaded:
            logger.error(f"gdown failed to download {name}")
            return None

        target = model_dir / Path(downloaded).name
        if target.exists():
            shutil.rmtree(target) if target.is_dir() else os.remove(target)
        shutil.move(downloaded, target)
        logger.info(f"File downloaded to: {target}")

        if target.suffix.lower() != ".zip":
            return target
        logger.info(f"Extracting zip file: {target.name}")
        try:
            with zipfile.ZipFile(target, "r") as archive:
                archive.extractall(model_dir)
        except zipfile.BadZipFile:
            logger.error(f"Downloaded file is not a valid zip: {target}")
            return None
        os.remove(target)
        subfolders = [item for item in model_dir.iterdir() if item.is_dir()]
        final = subfolders[0] if subfolders else model_dir
        logger.info(f"Extracted to: {final}")
        return final
    except requests.exceptions.RequestException as exc:
        logger.error(f"API request failed for {name}: {exc}")
    except Exception as exc:
        logger.error(f"Error downloading model {name}: {exc}")
    return None


def download_models(config: Config, names: list[str] | None = None, *, base_url: str | None = None,
                    force: bool = False) -> dict[str, Path | None]:
    """Download every selected model that has no explicit `path`.

    An explicit `path` always wins and is never downloaded over. A model already
    present in its model directory is reused unless `force`. Returns name ->
    resolved path (None when a download failed).
    """
    base_url = DOWNLOAD_BASE_URL if base_url is None else base_url
    results: dict[str, Path | None] = {}
    for name, model in config.models.items():
        if names is not None and name not in names:
            continue
        if model.path:
            results[name] = model.resolved_path
            continue
        if not force and model.resolved_path.exists():
            results[name] = model.resolved_path
            continue
        downloaded = download_model(model, base_url)
        if downloaded is not None:
            model._downloaded_path = downloaded
        results[name] = downloaded
    return results
