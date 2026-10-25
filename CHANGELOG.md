# Changelog

## 5.0.0

The config system is reworked. **Breaking:** 4.x configs and 4.x `from_config`
calls no longer load. See [MIGRATION-5.0.md](MIGRATION-5.0.md).

### Config
- **New package.** `transformsai_ai_core.config` replaces `config_schema.py` and
  `config_loader.py`. It imports only pydantic, PyYAML and the stdlib.
- **New shape.**
  - `node` carries the node identity in `node.sn`, the only place the serial lives.
  - `cameras` is a named map (a list is still accepted and named `camera_0`, `camera_1`, …).
  - `mediamtx` describes the shared stream server.
  - `cameras.<name>.stream` describes that camera's livestream.
  - `api`, `models` and `logging` sit at the top level, next to a free top-level `settings`.
  - Every core object has its own free `settings: {}`.
  - `advanced`, `meta`, `extras` and the global `livestream` block are gone.
- **Typed camera sources.** `source` is a discriminated union on `type`: `rtsp`
  (`host`), `device` (`path` or `index`), `file` (`path`, `loop`) and `url` (`url`).
- **Validation.**
  - Core validates only the keys it reads.
  - Conversion is lax but unambiguous. Durations accept `"500ms"`, `"3s"`, `"2m"`
    and `"1h"`; bitrates accept `"800k"`, `"1.2M"` or bits/s.
  - The only constraints are physical ones, never policy ranges.
  - Unknown keys outside `settings` are reported with a "did you mean" suggestion
    (a migration hint for 4.x keys): a warning in lenient mode, an error in strict
    mode. They are never dropped.
  - In lenient mode, an invalid optional value falls back to its default with a warning.
  - Cross-field checks: unique stream names, a known template placeholder, MediaMTX
    path characters, `mediamtx.host` when a stream is enabled, `api.base_url` when
    the API is enabled, camera audio only on rtsp/url sources, and one of
    `path` / `download_key` / `type` per model.
- **API.**
  - `load_config(path_or_dict, strict=False) -> (Config, ValidationReport)` raises
    `ConfigError` (with the full report) only on errors.
  - `validate_config()` is the strict, side-effect-free pre-deploy gate.
  - `save_config()` refuses an invalid config and writes only the keys you set.
  - `ValidationReport` holds `errors` and `warnings` as lists of
    `Issue(path, message, suggestion)`.
- **Project hooks.** `register_validator(path, model_or_fn, required=False)` and
  `register_secret(path)` add project checks and secrets to the same gate.
- **Derived values are properties.** `camera.source.url` (credentials URL-encoded),
  `config.stream_name()` / `config.stream_urls()` and `model.resolved_path`.
  Nothing is written back into the config.
- **Secrets and schema.** `Config.secrets()` and `Config.masked_dump()` (secrets as
  `***`). `Config.schema_json()` replaces `get_formatted_fields()` /
  `get_freeform_fields()`, and `generate_example()` produces the commented
  example. The repo's `config.example.yaml` is that generated output.

### Components
- **`VideoCaptureAsync`.**
  - New `from_camera(camera, **overrides)`.
  - New `frame_hook` runs on the capture thread on every published frame and is
    fail-open: an exception is logged and the frame is published unhooked.
  - New `source_type` hint and `loop` parameter.
  - Log lines, errors and `get_stats()["src"]` mask the source password.
  - Removed `timestamp_overlay` / `timestamp_overlay_options`, and the unused
    `driver` / `opencv_backend`.
- **`MediaMTXStreamer`.**
  - New `from_camera(camera, server, node, **overrides)`.
  - The stream name comes from the `stream.name` template (`{sn}`, `{camera}`,
    `{index}`; default `cam_sn_{sn}_{camera}`).
  - The size and fps come from the stream or the camera's capture. An unknown size
    is adopted from the first frame.
  - `webrtc_port` is honoured, and the URL templates come from `mediamtx`.
  - `encoder.codec` is honoured (`libx264`, `auto`, or any encoder name) and
    replaces `hw_encode`.
- **Audio.**
  - The input is `alsa` or `camera`; camera audio always uses the camera's own URL.
  - New configurable `filter`: `highpass_hz` (above 20 Hz it is a 4th-order
    cascade), `gain_db` and `limiter`.
  - Defaults are per input type and reproduce 4.x.
  - Every FFmpeg flag, the audio retry/backoff behaviour, `audio_active` and
    `audio_error` are unchanged.
- **`ApiClient`.**
  - `from_config(api, **overrides)` maps the regrouped `retries` / `outbox` /
    `pool` keys.
  - The outbox limits accept `None` as unbounded.
- **`YOLOWrapper` / `YOLOEWrapper`.**
  - `from_config(model, **overrides)` uses `model.resolved_path`.
  - The model dict takes a top-level `task`.
- **`utils.hide_camera_timestamp_and_add_current_time`** takes a `corner_kicker`
  argument; the caption above the corner label was previously hard-coded.

### Logging
- **Redaction.** A global patcher scrubs every record's message and `extra` before
  any sink sees it. It covers URL passwords, auth headers, Bearer tokens and
  registered secrets together with their URL-encoded forms. Core's sinks also
  scrub tracebacks and run with `diagnose=False`.
- **New helpers.** `add_secret()`, `redact()`, `add_patcher()` and `redacting_sink()`.
- **`configure_logging(config.logging, secrets=config.secrets())`** only replaces
  core's own sinks, so project sinks added afterwards survive later calls.
  `OFF` / null disables a sink.
- **`get_logger(name=None)`.** The level parameters and the deprecated
  `cli_debug` / `module_name` are removed.
- **The JSONL sink** starts a new file if its directory is removed while running.

### CLI
- New `transformsaicore config validate [--lenient] PATH`, which exits 1 on errors.
- New `config example [-o FILE]` and `config schema`.
- `download models` now uses the new loader and takes `--base-url`.

### Removed
- `DataUploader` / `datasend.py` (use `ApiClient`) and `DatasendConfig`.
- `process_config`, `init_kwargs`, `build_rtsp_url` and `resolve_model_paths`.
- `MediaMTXStreamer.get_hls_url()`.
