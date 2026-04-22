# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`transformsai-ai-core` is an internal Python library (src layout, `uv` managed) providing thread-safe utilities for computer vision pipelines: video capture, YOLO inference, structured logging, data upload, RTSP streaming, and YAML config management.

## Common Commands

```bash
# Install dependencies
uv sync

# Install with optional extras (mutually exclusive)
uv sync --extra trt    # TensorRT GPU inference
uv sync --extra rknn   # RKNN edge device inference
uv sync --extra onnx   # ONNX Runtime GPU

# Run tests
uv run python tests/test_config_manager.py
uv run python tests/test_logger.py
uv run python tests/test_datasend.py

# Run a single unittest test
uv run python -m unittest tests.test_logger.TestCentralLogger.test_basic_logging

# CLI entry point
uv run transformsaicore download models --config config.yaml
```

## Architecture

All production code is under `src/transformsai_ai_core/`. The `__init__.py` uses `__getattr__`-based lazy imports to defer heavy module loads (saves 50–100 MB RSS on edge devices).

### Core Modules

**`central_logger.py`** — Loguru-based structured logger.
- Output: JSON Lines at `.core-logs/{YYYY-MM-DD}/{HH-MM-SS}_{run_id}.jsonl`
- API: `get_logger(name, cli_debug=False)` — returns a `LoggerWrapper`
- `cli_debug=True` enables DEBUG-level console output for that logger
- `.error()` automatically captures current exception traceback
- Auto-rotation at 10 MB; run ID is 6 chars, auto-generated per process

**`config_schema.py` + `config_loader.py`** — Pydantic-validated YAML config system.
- Root schema: `AppConfig` → `MetaConfig`, `CameraConfig[]`, `AdvancedConfig`
- `AdvancedConfig` contains `ModelConfig`, `DatasendConfig`, `LivestreamConfig` plus arbitrary freeform dicts
- Key distinction: *formatted* (strict Pydantic) vs *freeform* (dict[str, Any]) sections — don't add Pydantic validation to intentionally freeform fields
- Main entry point: `process_config(path, base_dir, resolve_models, download_models)`
- `build_rtsp_url(rtsp_source)` constructs authenticated RTSP URLs

**`video_capture.py`** — `VideoCaptureAsync` background-thread capture.
- Auto-detects source (USB index, RTSP URL, local file)
- Key params: `auto_restart_on_fail`, `restart_delay`, `max_frame_age_ms`, `hw_decode`
- Usage: `.start()` → `.read()` in loop → `.stop()`

**`datasend.py`** — `DataUploader` async HTTP client.
- Non-blocking by default; thread pool auto-tuned to `min(2, cpu_count)`
- Failed uploads are cached and retried with exponential backoff (up to `max_cache_items=300`, TTL 24h)
- `send_data()` is async (fire-and-forget); `send_data_sync()` blocks for response
- Heartbeats skip the cache — they are not retried on failure

**`mediamtx_streamer.py`** — `MediaMTXStreamer` pushes frames to MediaMTX via FFmpeg.
- Queue-based async frame updates; generates RTSP + WebRTC URLs
- Supports hardware encoding (`hw_encode=True` auto-detects GPU encoder)

**`yolo_wrapper.py`** — `YOLOWrapper` / `YOLOEWrapper` around Ultralytics.
- Auto-exports to ONNX/TensorRT/RKNN when `export=True` in model dict
- TensorRT export is skipped on `aarch64` (must use native build)
- `YOLOEWrapper` supports text and visual prompts

**`utils.py`** — `time_to_string()`, `mat_to_response()` (JPEG encode + optional timestamp overlay)

## Testing

Tests use `unittest` with functional test scripts (not pytest). Test fixtures are in `tests/` (`test-config-1.yaml`, `test-config-2.yaml`, `test.png`). Tests clean up after themselves (reset loggers, remove `.core-logs/`).

## Configuration YAML Structure

```yaml
meta:
  name: "project-name"   # required
  version: "1.0.0"
  token: ""

cameras:
  - local: false
    rtsp_source:
      username: admin
      password: ""
      ip: 192.168.1.100
      port: 554
      path: /Streaming/Channels/101
    capture:
      buffer_size: 1
      opencv_backend: auto
    settings: {}  # freeform

advanced:
  models:
    yolo11n:
      type: "person-det"   # required
      download_key: ""
      batch: 1
      path: ""
      export: false
      load_options:
        lib_type: YOLO
        task: detect       # detect|classify|segment
      export_options: {}
  timings: {}      # freeform
  datasend:
    enabled: true
    base_url: https://api.example.com
    endpoints: {}  # freeform
    secret_keys: []
  livestream:
    enabled: true
    mediamtx_ip: localhost
    rtsp_port: 8554
    encoder:
      preset: ultrafast
      codec: copy
      queue_size: 2
    settings: {}   # freeform
  pipeline: {}     # freeform project-specific
```
