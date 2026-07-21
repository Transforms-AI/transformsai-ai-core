# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`transformsai-ai-core` is an internal Python library (src layout, build backend `setuptools.build_meta`, `uv` for dependency management/running, Python pinned `>=3.10,<3.12`) providing thread-safe utilities for computer vision pipelines: video capture, YOLO inference, structured logging, data upload, RTSP streaming, and YAML config management. Build backend is `setuptools.build_meta`; Python is pinned `>=3.10,<3.12`.

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
uv run python -m unittest tests.test_logger
uv run python -m unittest tests.test_fps_crash
uv run --with pytest pytest tests/test_datasend.py

# Run a single unittest test
uv run python -m unittest tests.test_logger.TestCentralLogger.test_basic_logging

# CLI entry point (download models is currently the only subcommand)
uv run transformsaicore download models --config config.yaml
```

## Architecture

All production code is under `src/transformsai_ai_core/`. The `__init__.py` uses `__getattr__`-based lazy imports to defer heavy module loads (saves 50–100 MB RSS on edge devices).

`mediamtx-host/` is not library code: it is the VPS deployment (docker compose + `mediamtx.yml` + demand-flag server) that pairs with `MediaMTXStreamer`'s `on_demand` mode. See [mediamtx-host/README.md](mediamtx-host/README.md).

### Core Modules

**`central_logger.py`** — Loguru-based structured logger.
- Output: JSON Lines at `.core-logs/{YYYY-MM-DD}/{HH-MM-SS}_{run_id}.jsonl`
- API: `get_logger(name=None, cli_sink_level="INFO", file_sink_level="DEBUG")` — returns a `_LoggerWrapper`
- Console level is set via `cli_sink_level`, configured on first call (levels honored when called from the entry script or before first configuration)
- `cli_debug` and `module_name` params still exist but are **DEPRECATED**
- `.error()` automatically captures current exception traceback
- Auto-rotation at 10 MB; run ID is 6 chars, auto-generated per process

**`config_schema.py` + `config_loader.py`** — Pydantic-validated YAML config system.
- Root schema: `AppConfig` → `MetaConfig`, `CameraConfig[]`, `AdvancedConfig`
- `AdvancedConfig` contains `ModelConfig`, `ApiConfig` (unified, replaces legacy `DatasendConfig`), `LivestreamConfig` plus arbitrary freeform dicts
- Key distinction: *formatted* (strict Pydantic) vs *freeform* (dict[str, Any]) sections — don't add Pydantic validation to intentionally freeform fields
- Every strict (formatted) section has an `extras: dict[str, Any]` field — a sanctioned freeform channel so the loader never drops data we haven't modeled yet
- Main entry point: `process_config(path, base_dir, resolve_models, download_models)`
- `build_rtsp_url(rtsp_source)` constructs authenticated RTSP URLs
- `init_kwargs(cls, data)` filters a dict to names accepted by `cls.__init__`, used by `from_config` to drop control/freeform keys (`enabled`, `settings`, `extras` leftovers)

**`video_capture.py`** — `VideoCaptureAsync` background-thread capture.
- Auto-detects source (USB index, RTSP URL, local file)
- Key params: `auto_restart_on_fail`, `restart_delay`, `max_frame_age_ms`, `hw_decode`
- Usage: `.start()` → `.read()` in loop → `.stop()`

**`api_client.py`** — `ApiClient` generic, pooled HTTP client (the v4 client; prefer for new code).
- `Response`/`EndpointProfile`/`ApiClient`; backed by one pooled `requests.Session` (keep-alive, `max_retries=0` so urllib3 doesn't double-retry)
- No transformsai-specific formatting and no special heartbeat path — a heartbeat is just `post(..., cache=False)`
- Generic `request(method, path, *, json/data/params/files/..., async_=False)` + `get/post/put/patch/delete`; any verb works
- `register_endpoint(name, profile)` (string or dict) + `send(name, ...)`; precedence per-call arg > profile > client default
- Returns a frozen `Response` (`status_code/text/.json()/ok/attempts/...`); total failure → `Response(status_code=0, ok=False)`, never raises
- Directory-per-request cache (`cache_dir/`, one folder per pending request — atomic, corruption-tolerant, O(1)); auto-retry policy: caches failed writes only, never GET/HEAD/OPTIONS
- `persist=True` (on `request`/`send`/verb methods, or an endpoint profile's `persist` key) marks a cache entry **non-expiring**: cached and retried exactly like any other failed write, and removed on success, but immune to `max_cache_age_seconds` and `max_cache_retries`. `max_cache_items` still applies (transient entries evicted first). `list_cached()` / `remove_cached(fid)` inspect and hand-drop cached requests

**`datasend.py`** — `DataUploader` async HTTP client (**legacy**, still available; `ApiClient` supersedes it).
- Non-blocking by default; thread pool auto-tuned to `min(2, cpu_count)`
- Failed uploads are cached and retried with exponential backoff (up to `max_cache_items=300`, TTL 24h)
- `send_data()` is async (fire-and-forget); `send_data_sync()` blocks for response
- Heartbeats skip the cache — they are not retried on failure

**`mediamtx_streamer.py`** — `MediaMTXStreamer` pushes frames to MediaMTX via FFmpeg.
- Queue-based async frame updates; generates RTSP + WebRTC URLs
- Supports hardware encoding (`hw_encode=True` auto-detects GPU encoder)

**`yolo_wrapper.py`** — `YOLOWrapper` / `YOLOEWrapper` around Ultralytics.
- Auto-exports to ONNX/TensorRT/RKNN when `export=True` in model dict
- TensorRT is excluded on `aarch64` at the **dependency level** (`pyproject.toml`: `tensorrt-cu12 ...; platform_machine != 'aarch64'`), not by branch logic in `yolo_wrapper.py`
- `YOLOEWrapper` supports text and visual prompts

**`utils.py`** — `time_to_string()`, `mat_to_response()` (JPEG encode + optional timestamp overlay), plus visualization helpers: `resize_frame()`, `draw_boxes()`, `draw_common_elements()`, `hide_camera_timestamp_and_add_current_time()`, `get_legend_layout()`

## Testing

Tests are a **mix** of `unittest` ([test_logger.py](tests/test_logger.py), [test_fps_crash.py](tests/test_fps_crash.py), functional [test_config_manager.py](tests/test_config_manager.py)) and `pytest` ([test_datasend.py](tests/test_datasend.py)). Additional test files: [test_yolo_wrapper.py](tests/test_yolo_wrapper.py), [test_cli_debug.py](tests/test_cli_debug.py). Test fixtures are in `tests/` (`test-config-1.yaml`, `test-config-2.yaml`, `test.png`). Tests clean up after themselves (reset loggers, remove `.core-logs/`). Note: pytest is **not** a declared dependency, so run with `uv run --with pytest pytest tests/test_datasend.py`.

## Configuration YAML Structure

```yaml
meta:
  name: "project-name"   # required
  version: "1.0.0"
  token: ""
  extras: {}           # sanctioned freeform channel

cameras:
  - local: false
    local_source: ""
    rtsp_source:
      username: admin
      password: ""
      ip: 192.168.1.100
      port: 554
      path: /Streaming/Channels/101
      extras: {}
    capture:            # mirrors VideoCaptureAsync.__init__
      buffer_size: 1
      opencv_backend: auto
      max_frame_age_ms: null
      width: null
      height: null
      driver: null
      auto_restart_on_fail: false
      restart_delay: 30.0
      auto_resize: true
      hw_decode: false
      fps: null
      extras: {}
    settings: {}        # freeform per-camera
    extras: {}

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
        extras: {}
      export_options: {}
      extras: {}
  timings: {}      # freeform
  api:              # Unified ApiClient config (replaces legacy datasend)
    enabled: true
    base_url: ""
    headers: {}
    timeout: 30
    success_codes: [200, 201, 202, 204]
    default_content_type: auto
    auth_keys: []
    auth_header: X-Secret-Key
    max_retries: 3
    retry_backoff: 1.0
    retry_backoff_max: 30
    retry_on_status: [408, 429, 500, 502, 503, 504]
    max_workers: null
    cache_enabled: true
    cache_dir: .core-api-cache
    cache_retry_interval: 100
    max_cache_items: 300
    max_cache_age_seconds: 86400
    max_cache_retries: 5
    endpoints: {}  # freeform — auto-registered by from_config
    pool_connections: 10
    pool_maxsize: 10
    settings: {}   # freeform
    extras: {}
  livestream:       # mirrors MediaMTXStreamer.__init__
    enabled: true
    mediamtx_ip: localhost
    rtsp_port: 8554
    camera_sn_id: ""     # config default, overridable per camera
    fps: 30
    frame_width: 1920
    frame_height: 1080
    bitrate: "1500k"
    hw_encode: false
    debug_log_interval: 60.0
    on_demand: false     # true = push only while a viewer is watching (edge polls a VPS demand flag)
    demand_url: ""       # blank = derive https://{mediamtx_ip}/demand/cam_sn_{camera_sn_id}
    demand_poll_interval: 3.0
    demand_grace_period: 10.0
    demand_timeout: 5.0
    encoder:
      preset: ultrafast
      codec: copy
      queue_size: 2
      extras: {}
    settings: {}   # freeform (NOT consumed by from_config)
    extras: {}
  pipeline: {}     # freeform project-specific
```

## Config-Driven Construction (from_config)

Every runtime class now has a `from_config(cls, cfg, **overrides)` classmethod that
constructs the object directly from a config section. Precedence: **overrides > typed
fields > `extras`**.

```python
# ApiClient — near 1:1 mapping, endpoints auto-registered
client = ApiClient.from_config(cfg["advanced"]["api"], base_url="https://override.example.com")

# VideoCaptureAsync — derives src from camera config
cap = VideoCaptureAsync.from_config(cfg["cameras"][0])

# MediaMTXStreamer — flattens encoder nesting
streamer = MediaMTXStreamer.from_config(cfg["advanced"]["livestream"], camera_sn_id="cam1")

# YOLOWrapper / YOLOEWrapper — thin pass-through
model = YOLOWrapper.from_config(cfg["advanced"]["models"]["yolo11n"])
```
