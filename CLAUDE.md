# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

`transformsai-ai-core` is an internal Python library (src layout, build backend `setuptools.build_meta`, `uv` for dependency management/running, Python pinned `>=3.10,<3.12`) providing thread-safe utilities for computer vision pipelines: video capture, YOLO inference, structured logging with secret redaction, HTTP upload, RTSP streaming, and typed YAML config. It is shared by several edge projects — keep it general, never shaped around one project.

## Common Commands

```bash
# Install dependencies
uv sync

# Install with optional extras (mutually exclusive)
uv sync --extra trt    # TensorRT GPU inference
uv sync --extra rknn   # RKNN edge device inference
uv sync --extra onnx   # ONNX Runtime GPU

# Run tests (use --frozen so uv doesn't re-lock uv.lock)
uv run --frozen --with pytest pytest tests/ --ignore tests/test_yolo_wrapper.py
uv run --frozen python -m unittest tests.test_config_schema
uv run --frozen python tests/test_yolo_wrapper.py   # script; downloads ultralytics models

# Run a single unittest test
uv run --frozen python -m unittest tests.test_logger.TestCentralLogger.test_off_disables_a_sink

# CLI
uv run transformsaicore config validate config.yaml      # strict gate, exit 1 on errors
uv run transformsaicore config example -o config.example.yaml   # regenerate after schema changes
uv run transformsaicore config schema
uv run transformsaicore download models --config config.yaml
```

## Architecture

All production code is under `src/transformsai_ai_core/`. The `__init__.py` uses `__getattr__`-based lazy imports to defer heavy module loads (saves 50–100 MB RSS on edge devices).

`mediamtx-host/` is not library code: it is the VPS deployment (docker compose + `mediamtx.yml` + demand-flag server) that pairs with `MediaMTXStreamer`'s `on_demand` mode. Its `mediamtx.yml` raises demand flags only for `live/cam_sn_*` paths, which is why core fixes the `cam_sn_` prefix on every stream name (`urls.STREAM_NAME_PREFIX`). See [mediamtx-host/README.md](mediamtx-host/README.md).

### Core Modules

**`central_logger.py`** + **`redact.py`** — Loguru-based structured logger with secret redaction.
- Output: JSON Lines at `.core-logs/{YYYY-MM-DD}/{run_id}_{HH-MM-SS-ffffff}.jsonl` + a console sink
- API: `configure_logging(config=None, *, console_level=, file_level=, secrets=())`, `get_logger(name=None)` → `_LoggerWrapper`
- `configure_logging` only removes the sink ids it added itself (`_core_sink_ids`) — project sinks added
  after it survive later calls. `OFF`/None = no sink. Levels are set only via `configure_logging`
- Core owns the one global loguru patcher (`_patch`): redacts message + string `extra` values via
  `redact.redact()` (URL userinfo, auth headers, Bearer tokens, `add_secret()` literals + URL-encoded form);
  projects chain with `add_patcher()`, never `logger.configure(patcher=)`. Core sinks also redact
  rendered tracebacks and use `diagnose=False`; `redacting_sink()` wraps project sinks
- `.error()` automatically captures current exception traceback
- Auto-rotation at 10 MB; run ID is 6 chars, auto-generated per process

**`config/`** — the 5.0 config system (pydantic + PyYAML + stdlib only; no cv2/torch imports).
- `schema.py`: typed models for core sections only — `Config` root → `NodeConfig` (`sn` required),
  `cameras: dict[str, CameraConfig]` (a list is normalised to `camera_0…`), `MediaMTXConfig | None`,
  `ApiConfig | None`, `models: dict[str, ModelConfig]`, `LoggingConfig`, free `settings`.
  `CameraConfig.source` is a discriminated union on `type` (`RtspSource`/`DeviceSource`/`FileSource`/
  `UrlSource`; add a type = one class + `SOURCE_TYPES`/`Source` entry). Per-camera `capture`, optional
  `stream` (with optional `audio`). **Defaults live only here**; components map these models onto
  constructors (a drift test in `tests/test_stream_config.py` keeps constructor defaults equal)
- `CoreModel`: `extra="allow"` (unknown keys kept + reported, never dropped), `validate_default=True`,
  `coerce_numbers_to_str=True`, null → `{}`. Every core object has a free `settings: dict[str, Any]`
- `types.py`: `Duration`/`DurationOrOff` (`"500ms"`, `"3s"`, `"2m"`, `"1h"`), `Bitrate` (int bit/s,
  dumps as `"800k"`), `Port`, `secret_field()` (marks secrets for `secrets()`/`masked_dump()`).
  Only physical constraints — **never policy ranges**
- `validate.py`: the engine. Unknown keys (walks raw data against the models; difflib suggestion +
  `MIGRATION_HINTS` for 4.x keys): strict → error, lenient → warning. Invalid optional value: strict →
  error, lenient → warning + key pruned so its default applies. Required problems → error, enclosing
  camera/model/optional block pruned (or a placeholder filled) so the rest is still reported in one pass.
  Then cross-field checks, then project validators (`registry.py`)
- `loader.py`: `load_config(src, strict=False, base_dir=None) -> (Config, ValidationReport)` raises
  `ConfigError(report)` iff errors; `validate_config(src)` strict/no side effects; `save_config` refuses
  invalid configs, writes `exclude_unset`; `download_models` (explicit `path` always wins)
- `urls.py`: `build_stream_urls()` — the only place stream URLs are built (shared by `Config.stream_urls`
  and `MediaMTXStreamer`). Stream name = fixed `cam_sn_` + template `{sn} {camera} {index}` (default
  `{sn}_{camera}`); `stream_path_name()` adds the prefix once, everywhere (config, URLs, streamer ctor)
- `example.py`: `generate_example()` → the committed `config.example.yaml` (a test keeps it fresh)
- Derived values are properties, never written into the config: `source.url`, `config.stream_urls()`,
  `model.resolved_path`. `Config.secrets()`, `Config.masked_dump()`, `Config.schema_json()`

**`video_capture.py`** — `VideoCaptureAsync` background-thread capture.
- Auto-detects source (USB index, RTSP URL, local file) unless a `source_type` hint
  (`device`/`file`/`network`) is given — `from_camera` always passes one
- Key params: `auto_restart_on_fail`, `restart_delay`, `max_frame_age_ms`, `hw_decode`
- Usage: `.start()` → `.read()` in loop → `.stop()`
- **Supervised reconnect, on by default** (`auto_restart_on_fail=True`). Failed open, a
  30-read failure streak, a stall, or an exception all funnel through one backoff policy:
  `restart_delay` is the **ceiling**, `restart_backoff_start` (1.0) the first delay,
  doubling with ±`restart_backoff_jitter`, reset after `restart_reset_after` healthy
  seconds. `max_restart_attempts=None` retries forever
- Stall watchdog: no decoded frame within `stall_timeout` (default `max(5, 20/fps)`, live
  sources only) forces a reconnect. `max_frame_age_ms` is consumer-side only and does *not*
  trigger one
- Network opens set `OPENCV_FFMPEG_CAPTURE_OPTIONS` (`timeout`/`stimeout` from
  `open_timeout`, plus `rtsp_transport` for `rtsp://` only) under a class lock, so a dead
  link can't block `grab()` indefinitely. `ffmpeg_options` overrides the string verbatim
- The capture thread **owns the handle** and always releases it; `release()` abandons a
  thread wedged in OpenCV instead of double-releasing (the old code could segfault here)
- Health: `.state`, `.is_healthy`, `.get_stats()`, `on_state_change(state, stats)` callback,
  plus a `health_log_interval` line — INFO even when healthy, on purpose: it is the
  once-a-minute liveness beacon node watchdogs rely on (degraded lines add `last_error`)
- `read()` blocks up to `timeout` while down (no busy-spin), returns instantly after `stop()`
- Config: `VideoCaptureAsync.from_camera(camera, **overrides)` — source from `camera.source`
  (`target`, `kind` → `source_type` hint, `transport`, file `loop`), the rest from `capture` /
  `capture.restart`. Overrides win
- **`frame_hook`** (`Callable[[ndarray], ndarray | None]`): runs on the capture thread on every
  published frame (in `_publish_frame`); exceptions are caught, counted (`get_stats()["hook_errors"]`),
  logged first-then-rate-limited, and the unhooked frame is published — a hook can never kill capture.
  Project overlays (e.g. `utils.hide_camera_timestamp_and_add_current_time`) go here; core has no
  timestamp config
- Logs, `RuntimeError`s and `get_stats()["src"]` use `self._label` (password masked); `self.src` is real

**`api_client.py`** — `ApiClient` generic, pooled HTTP client.
- `Response`/`EndpointProfile`/`ApiClient`; backed by one pooled `requests.Session` (keep-alive, `max_retries=0` so urllib3 doesn't double-retry)
- No transformsai-specific formatting and no special heartbeat path — a heartbeat is just `post(..., cache=False)`
- Generic `request(method, path, *, json/data/params/files/..., async_=False)` + `get/post/put/patch/delete`; any verb works
- `register_endpoint(name, profile)` (string or dict) + `send(name, ...)`; precedence per-call arg > profile > client default
- Returns a frozen `Response` (`status_code/text/.json()/ok/attempts/...`); total failure → `Response(status_code=0, ok=False)`, never raises
- Directory-per-request cache (`cache_dir/`, one folder per pending request — atomic, corruption-tolerant, O(1)); auto-retry policy: caches failed writes only, never GET/HEAD/OPTIONS
- `from_config(api: ApiConfig, **overrides)` maps `retries.*` → `max_retries`/`retry_*`, `outbox.*` →
  `cache_*`/`max_cache_*`, `pool.*` → `pool_*`/`max_workers`; outbox limits accept `None` = unbounded
- `persist=True` (on `request`/`send`/verb methods, or an endpoint profile's `persist` key) marks a cache entry **non-expiring**: cached and retried exactly like any other failed write, and removed on success, but immune to `max_cache_age_seconds` and `max_cache_retries`. `max_cache_items` still applies (transient entries evicted first). `list_cached()` / `remove_cached(fid)` inspect and hand-drop cached requests

**`mediamtx_streamer.py`** — `MediaMTXStreamer` pushes one camera's frames to MediaMTX via FFmpeg.
- `from_camera(camera, server: MediaMTXConfig, node: NodeConfig, **overrides)`; constructor is
  `(host, rtsp_port, stream_name, ...)` — `stream_name` gets the `cam_sn_` prefix if it lacks it
- Size/fps: `stream.*` > `capture.*`; unknown size → adopted from the first `update_frame` frame
  (FFmpeg start deferred, `_start_pending`); fps fallback `StreamConfig.DEFAULT_FPS` (15)
- `encoder_codec`: `libx264` | `auto` (detect hardware) | an FFmpeg encoder name
- `build_audio_filter()` (module-level, pure): pan (alsa) → highpass (≤20 Hz single stage, >20 Hz two
  cascaded `poles=2` stages) → volume → alimiter → aresample. Unset filter keys use
  `AlsaAudioInput`/`CameraAudioInput.filter_defaults` (4.x behaviour). Golden-argv tests in
  `tests/test_mediamtx_audio_fallback.py` guard every FFmpeg flag — keep its field-measured comments
- **Audio is best-effort, video is not.** If FFmpeg dies at the start check with the audio
  input (ALSA `audio_device` or camera `audio_url`), the same start call relaunches
  video-only (argv identical to a no-audio node; the 5 s rapid-restart guard is not hit).
  Audio is retried at the next fresh start after a backoff (`audio_retry_backoff` 60 s,
  doubling to `audio_retry_backoff_max` 900 s), reset only by a run that kept audio for
  `audio_reset_after` (60 s). A mid-run death whose log blames input #1 (`[in#1`, `[alsa @`,
  or the device/URL) counts too; output (`[out#0`) errors never do. A healthy video-only
  stream is never restarted just to retry audio. The device is never probed from Python;
  instead a launch with audio writes one blank frame during the start check, because
  FFmpeg 9 doesn't open input #1 until input #0 (the pipe) has data.
  Read-only `audio_active` / `audio_error` (credentials scrubbed), also in `get_stats()`

**`yolo_wrapper.py`** — `YOLOWrapper` / `YOLOEWrapper` around Ultralytics.
- `from_config(model: ModelConfig, **overrides)` → model dict `{path: resolved_path, task, batch, export,
  export_options, set_classes}`
- Auto-exports to ONNX/TensorRT/RKNN when `export=True` in model dict
- TensorRT is excluded on `aarch64` at the **dependency level** (`pyproject.toml`: `tensorrt-cu12 ...; platform_machine != 'aarch64'`), not by branch logic in `yolo_wrapper.py`
- `YOLOEWrapper` supports text and visual prompts

**`utils.py`** — `time_to_string()`, `mat_to_response()` (JPEG encode + optional timestamp overlay), plus config-free visualization helpers: `resize_frame()`, `draw_boxes()`, `draw_common_elements()`, `hide_camera_timestamp_and_add_current_time()` (badge caption via `corner_kicker`), `get_legend_layout()`

## Testing

Tests are a **mix** of `unittest` classes and script-style files; run them with pytest:
`uv run --frozen --with pytest pytest tests/ --ignore tests/test_yolo_wrapper.py` (pytest is **not** a
declared dependency). `test_yolo_wrapper.py` is a script that downloads ultralytics models. Config
tests: [test_config_schema.py](tests/test_config_schema.py), [test_stream_config.py](tests/test_stream_config.py)
(incl. the constructor-default drift test), [test_from_config.py](tests/test_from_config.py) (ApiClient,
YOLO mapping, CLI). [test_config_schema.py](tests/test_config_schema.py) fails if `config.example.yaml`
is stale — regenerate with `uv run transformsaicore config example -o config.example.yaml`. Tests clean
up after themselves (reset loggers, remove `.core-logs/`, call `clear_registry()`).

## Configuration

The authoritative, commented reference is [config.example.yaml](config.example.yaml) (generated from
the schema). Migration from 4.x: [MIGRATION-5.0.md](MIGRATION-5.0.md). Principles the schema follows:
validate only what core reads; lenient conversion, physical constraints only; require only what's
strictly necessary; every core object has a free `settings: {}`; project features (overlays, GPIO,
flash, insets) are never core schema; each fact lives in one place (serial = `node.sn`; streams derive
size/fps from their camera); unknown keys are reported, never dropped.

```python
config, report = load_config("config.yaml")
cap = VideoCaptureAsync.from_camera(config.cameras["surveillance"], frame_hook=hook)
streamer = MediaMTXStreamer.from_camera(config.cameras["surveillance"], config.mediamtx, config.node)
client = ApiClient.from_config(config.api)
model = YOLOWrapper.from_config(config.models["person-det"])
configure_logging(config.logging, secrets=config.secrets())
```
