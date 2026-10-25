# Migrating to transformsai-ai-core 5.0

5.0 reworks the config system. It is a **clean break**: 4.x configs and 4.x
`from_config` calls do not load. This guide covers every key, call and behaviour
that changed.

What changed, in short:

- **One identity.** `meta.name` (which was used as the node serial), each camera's
  `settings.sn` and `livestream.camera_sn_id` are replaced by a single `node.sn`.
- **Named cameras.** `cameras` is a map keyed by camera name. A plain list still
  works, and its entries are named `camera_0`, `camera_1`, … by position.
- **Streams belong to cameras.** The global `livestream` block is split in two:
  `mediamtx` (the shared server) and `cameras.<name>.stream`. A stream takes its
  size and fps from its camera.
- **Typed sources.** `source: {type: rtsp|device|file|url, ...}` replaces
  `local` / `local_source` / `rtsp_source`.
- **Validation that reports instead of dropping.** Core validates only the keys it
  reads, and every core object has a free `settings: {}`. Unknown keys elsewhere are
  reported with a "did you mean" suggestion, never silently dropped. The report is a
  warning at runtime and an error at the gate.
- **No writes into your config.** `rtsp_url`, the stream URLs and model paths are
  properties of the loaded config.
- **Defaults in one place.** They live in the config models only, not in
  `from_config` or in projects.
- **Removed:** `advanced`, `meta`, `extras`, `DatasendConfig` / `DataUploader`,
  the timestamp overlay, the legacy capture keys, `init_kwargs`,
  `get_formatted_fields` and `get_freeform_fields`.

Check a migrated config with the gate:

```bash
uv run transformsaicore config validate config.yaml
```

Any 4.x key that is still present shows up as an unknown key, together with a hint
saying where it went. The full commented reference is
[config.example.yaml](config.example.yaml).

---

## 1. Old key → new key

### Top level

| 4.x | 5.0 |
|---|---|
| `meta.name` | `node.sn` (REQUIRED). There is no `name` key any more. |
| `meta.version` | `node.version` (optional) |
| `meta.token` | removed (nothing read it) |
| `meta.extras` | `node.settings` |
| `advanced.*` | its children move to the top level, as below |
| `advanced.models` | `models` |
| `advanced.api` | `api` |
| `advanced.livestream` | `mediamtx` (server) + `cameras.<name>.stream` (per camera) |
| `advanced.timings`, `advanced.pipeline`, `advanced.extras` | top-level `settings` (project-owned) |
| `cameras: [ ... ]` addressed by index | `cameras: {name: ...}` (or a list, auto-named `camera_<i>`) |
| every `extras: {}` | the object's `settings: {}` |

### Cameras

| 4.x | 5.0 |
|---|---|
| `local: true` + `local_source: /dev/videoX` | `source: {type: device, path: /dev/videoX}` (or `index: 0`) |
| `local: true` + `local_source: clip.mp4` | `source: {type: file, path: clip.mp4}` (`loop`: default true) |
| `rtsp_source.ip` | `source.host` (with `source.type: rtsp`) |
| `rtsp_source.username` | `source.user` |
| `rtsp_source.password` | `source.password` (secret: URL-encoded in the URL, masked in dumps and logs) |
| `rtsp_source.port`, `rtsp_source.path` | `source.port`, `source.path` |
| a full URL | `source: {type: url, url: ...}`, or shorthand `source: "rtsp://..."` |
| `rtsp_url` (written by `process_config`) | derived: `camera.source.url` |
| `capture.rtsp_transport` | `source.transport` (rtsp sources; camera audio uses it too) |
| `capture.max_frame_age_ms` | `capture.max_frame_age`, in seconds or e.g. `"500ms"` |
| `capture.restart.backoff_jitter` | `capture.restart.jitter` |
| `capture.auto_restart_on_fail` (legacy) | `capture.restart.enabled` |
| `capture.restart_delay` (legacy) | `capture.restart.delay` |
| `capture.driver`, `capture.opencv_backend` | removed (they were never used) |
| `capture.timestamp.*` | removed from core; see [§2](#2-removed-features-and-their-replacements) |
| `settings.sn` | removed (it is `node.sn`); use `stream.name` if a stream needs a different name |
| `settings.version` | stays project data: `settings.version` |
| `settings.resolution`, `settings.fps` (when they meant capture) | `capture.width`, `capture.height`, `capture.fps` |

### Livestream → `mediamtx` + `cameras.<name>.stream`

| 4.x `advanced.livestream.*` | 5.0 |
|---|---|
| `mediamtx_ip` | `mediamtx.host` (required when any stream is enabled) |
| `rtsp_port` | `mediamtx.rtsp_port` |
| (hard-coded 8889) | `mediamtx.webrtc_port`, now honoured |
| `demand_poll_interval`, `demand_grace_period`, `demand_timeout` | `mediamtx.demand.{poll_interval, grace_period, timeout}` |
| `demand_url` (formatted with `{camera_sn_id}`) | `mediamtx.demand_url`, a template with `{host} {rtsp_port} {webrtc_port} {name}` |
| (hard-coded RTSP/WebRTC URLs) | `mediamtx.rtsp_url`, `mediamtx.webrtc_url` templates (null = the 4.x URLs) |
| `enabled` | `cameras.<name>.stream.enabled` |
| `camera_sn_id` | derived: `cam_sn_` + `stream.name`, default `{sn}_{camera}` ([§5](#5-behaviour-changes)) |
| `fps`, `frame_width`, `frame_height` | derived from `capture.fps/width/height`; optional `stream.fps/width/height` overrides |
| `bitrate`, `on_demand`, `debug_log_interval` | `stream.bitrate`, `stream.on_demand`, `stream.debug_log_interval` |
| `hw_encode: true` | `stream.encoder.codec: auto` |
| `encoder.codec: copy` | remove it. It was never read, and the default is `libx264`. `codec` is honoured now. |
| `encoder.preset`, `encoder.queue_size` | `stream.encoder.preset`, `stream.encoder.queue_size` |
| `settings.*` | `stream.settings.*` (project-owned) |

### Audio → `cameras.<name>.stream.audio`

| 4.x `livestream.audio.*` | 5.0 |
|---|---|
| `source: mic` + `mic.device/rate/channels` | `input: {type: alsa, device, rate, channels}` |
| `mic.map_channel` | `input.channel` |
| `source: camera` + `camera.url` | `input: {type: camera}`. It always uses this camera's own source URL; there is no second URL. |
| flat `device/rate/...` (legacy shape) | `input: {type: alsa, ...}` |
| `mic.gain_db`, `camera.gain_db` | `filter.gain_db` (unset: 18 for alsa, 0 for camera) |
| (fixed 20 Hz high-pass) | `filter.highpass_hz` (unset: 20 for alsa, off for camera; 0/null = off; above 20 = 4th-order) |
| (fixed 0.97 limiter) | `filter.limiter` (null = no limiter) |
| `bitrate` | `audio.bitrate` |
| (constructor only) | `audio.retry.{backoff, backoff_max, reset_after}` |

### API

| 4.x `advanced.api.*` | 5.0 `api.*` |
|---|---|
| `max_retries`, `retry_backoff`, `retry_backoff_max`, `retry_on_status` | `retries.{max, backoff, backoff_max, on_status}` |
| `cache_enabled`, `cache_dir`, `cache_retry_interval` | `outbox.{enabled, dir, retry_interval}` |
| `max_cache_items`, `max_cache_age_seconds`, `max_cache_retries` | `outbox.{max_items, max_age, max_retries}`. `max_age` accepts `"24h"`. 0/null = unbounded for items and age; for retries, null = unbounded and 0 = never retried. |
| `pool_connections`, `pool_maxsize`, `max_workers` | `pool.{connections, maxsize, workers}` |
| `default_content_type` | `content_type` |
| `enabled`, `base_url`, `timeout`, `headers`, `auth_keys`, `auth_header`, `success_codes`, `endpoints`, `settings` | unchanged (`base_url` is required when enabled; `auth_keys` also accepts a single string) |
| `advanced.datasend` / `DatasendConfig` | removed: use `api` + `ApiClient` |

### Models

| 4.x `advanced.models.<name>.*` | 5.0 `models.<name>.*` |
|---|---|
| `load_options.task` | `task` |
| `load_options.lib_type` | removed (nothing read it; pick `YOLOWrapper` / `YOLOEWrapper` in code) |
| `set_classes` (silently dropped by 4.x validation) | `set_classes` (typed, YOLOE) |
| `type` (was required) | optional; a model needs one of `path`, `download_key` or `type` |
| `path` (overwritten by `process_config`) | `path`, never rewritten; the resolved path is `model.resolved_path` |
| `batch`, `export`, `export_options`, `download_key` | unchanged |

### Logging

| 4.x | 5.0 |
|---|---|
| `advanced.pipeline.logging.cli_sink_level` (project convention) | `logging.console_level` (`OFF`/null = no console sink) |
| `advanced.pipeline.logging.file_sink_level` | `logging.file_level` (`OFF`/null = no file sink) |

---

## 2. Removed features and their replacements

### Timestamp overlay → `frame_hook` + helper

`capture.timestamp`, `TimestampSettings` and the `timestamp_overlay` /
`timestamp_overlay_options` parameters are gone: the overlay is a project feature.
`VideoCaptureAsync` now takes a generic `frame_hook`. The hook runs on the capture
thread on every published frame, so every consumer sees its result. If it raises,
the exception is logged and the unhooked frame is published, so a hook can never
stop capture. The drawing helper is unchanged and config-free:

```yaml
cameras:
  surveillance:
    settings:
      timestamp_overlay: {enabled: true, rect_ratios: [0.015, 0.05, 0.25, 0.035], corner_label: "Gate 2"}
```

```python
from transformsai_ai_core.utils import hide_camera_timestamp_and_add_current_time

ts = camera.settings.get("timestamp_overlay", {})
hook = None
if ts.get("enabled"):
    options = {
        "camera_ts_rect_ratios": ts.get("rect_ratios"), "camera_ts_rect_coords": ts.get("rect_coords"),
        "hide_rect_color": ts.get("hide_color"), "new_ts_font_color": ts.get("font_color"),
        "new_ts_font_scale": ts.get("font_scale"), "time_format": ts.get("time_format"),
        "corner_label": ts.get("corner_label"),
    }
    options = {k: v for k, v in options.items() if v is not None}

    def hook(frame):
        return hide_camera_timestamp_and_add_current_time(frame, inplace=True, **options)

cap = VideoCaptureAsync.from_camera(camera, frame_hook=hook).start()
```

The badge caption above `corner_label` (previously hard-coded to "AI SURVEILLANCE")
is now the `corner_kicker=` argument. It has the same default, and `""` removes it.

### Other removals

| Removed | Use instead |
|---|---|
| `meta.name` | `node.sn` |
| list cameras addressed by index | named map: `config.cameras["surveillance"]` |
| `process_config(path, resolve_models, download_models)` | `load_config(path)`; `config.download_models()` (or `transformsaicore download models`) |
| `load_config(path, validate=True) -> dict` | `load_config(path, strict=False) -> (Config, ValidationReport)` |
| `build_rtsp_url(rtsp_source)` | `camera.source.url` |
| `resolve_model_paths`, model `path` written into the dict | `model.resolved_path` |
| `init_kwargs` | the typed `from_config` / `from_camera` methods |
| `get_formatted_fields`, `get_freeform_fields` | `Config.schema_json()` / `transformsaicore config schema` |
| `AppConfig`, `MetaConfig`, `AdvancedConfig`, `LivestreamConfig`, `RtspSource` (4.x shape), `CaptureSettings`, `RestartSettings`, `LoadOptions`, `StreamEncoderSettings`, `DatasendConfig`, `TimestampSettings` | `transformsai_ai_core.config`: `Config`, `NodeConfig`, `CameraConfig`, `RtspSource`/`DeviceSource`/`FileSource`/`UrlSource`, `CaptureConfig`, `RestartConfig`, `StreamConfig`, `EncoderConfig`, `AudioConfig`, `MediaMTXConfig`, `ApiConfig`, `ModelConfig`, `LoggingConfig` |
| `DataUploader` / `datasend.py` | `ApiClient` (see below) |
| `MediaMTXStreamer.get_hls_url()` | `get_webrtc_url()` (it only ever returned that) |
| `get_logger(cli_sink_level=, file_sink_level=, cli_debug=, module_name=)` | `configure_logging(...)`; `get_logger(name)` |

### DataUploader → ApiClient

| `DataUploader` | `ApiClient` |
|---|---|
| `DataUploader(base_url, secret_keys=[...], ...)` | `ApiClient.from_config(config.api)` with `api.auth_keys` |
| `send_data(data, files)` (fire-and-forget, cached) | `client.post(path, data=..., files=..., async_=True)` (failed writes go to the outbox) |
| `send_data_sync(...)` | `client.post(...)` → `Response` (check `.ok`) |
| `send_heartbeat(sn, ...)` (never cached) | `client.post("heartbeat", json={...}, cache=False)` |
| `endpoints.heartbeat_endpoint` | `api.endpoints.heartbeat`, then `client.send("heartbeat", json=...)` |
| `cache_file_path`, `max_cache_items`, `retry_delay` | `api.outbox.*`, `api.retries.*` |

---

## 3. Loader and component calls

```python
# 4.x
from transformsai_ai_core import process_config, VideoCaptureAsync, MediaMTXStreamer, ApiClient, YOLOWrapper
cfg = process_config("config.yaml", resolve_models=True)
cam = cfg["cameras"][1]
cap = VideoCaptureAsync(src=cam["rtsp_url"], width=1280, height=720, fps=10,
                        timestamp_overlay=True, timestamp_overlay_options={...})
streamer = MediaMTXStreamer.from_config(cfg["advanced"]["livestream"], camera_sn_id=cfg["meta"]["name"],
                                        fps=10, frame_width=1280, frame_height=720,
                                        audio_url=cam["rtsp_url"])
client = ApiClient.from_config(cfg["advanced"]["api"])
model = YOLOWrapper(cfg["advanced"]["models"]["person-det"])

# 5.0
from transformsai_ai_core import VideoCaptureAsync, MediaMTXStreamer, ApiClient, YOLOWrapper
from transformsai_ai_core.config import load_config
config, report = load_config("config.yaml")          # raises ConfigError only for required-key problems
camera = config.cameras["surveillance"]
cap = VideoCaptureAsync.from_camera(camera, frame_hook=hook)
streamer = MediaMTXStreamer.from_camera(camera, config.mediamtx, config.node)
client = ApiClient.from_config(config.api)
model = YOLOWrapper.from_config(config.models["person-det"])
```

- **Overrides.** Every `from_config` / `from_camera` takes `**overrides`, which win
  over the config. Defaults come only from the config models.
- **`stream.enabled` / `api.enabled`.** These remain the project's flags to check
  before building.
- **Pre-deploy gate.** Use `validate_config(path)`, which is strict and has no side
  effects, or `transformsaicore config validate path`, which exits 1 on errors.
- **Project settings.** Register checks with `register_validator("settings.meter", MeterSettings)`
  and secrets with `register_secret("settings.modbus.password")`.

### Constructor renames

For code that builds components by hand:

| Class | 4.x | 5.0 |
|---|---|---|
| `MediaMTXStreamer` | `mediamtx_ip`, `camera_sn_id` (the URL added `cam_sn_`) | `host`, `stream_name` (still gets `cam_sn_` added: `"pb-00004"` → `cam_sn_pb-00004`) |
| `MediaMTXStreamer` | `hw_encode=True`, `encoder_codec="copy"` (ignored) | `encoder_codec="auto"` / `"libx264"` / `"<encoder>"` |
| `MediaMTXStreamer` | `audio_source="alsa"/"rtsp"`, `audio_map_channel` | `audio_input="alsa"/"camera"` (inferred from `audio_device`/`audio_url`), `audio_channel` |
| `MediaMTXStreamer` | — | `webrtc_port`, `rtsp_url`/`webrtc_url`/`demand_url` templates, `audio_transport`, `audio_highpass_hz`, `audio_limiter` |
| `MediaMTXStreamer` | `fps=30, frame_width=1920, frame_height=1080` | all default to `None`: fps 15, size taken from the first frame |
| `VideoCaptureAsync` | `driver`, `opencv_backend`, `timestamp_overlay`, `timestamp_overlay_options` | removed; new `frame_hook`, `source_type`, `loop` |
| `YOLOWrapper` / `YOLOEWrapper` dict | `load_options: {task}` | top-level `task` |

---

## 4. Logging

- **Configure first, then add your sinks.** `configure_logging(config.logging, secrets=config.secrets())`
  replaces only core's own sinks, so call it first and add project sinks afterwards.
  Calling it again later is safe: in 4.x it called `logger.remove()` and dropped
  every project sink.
- **Turning a sink off.** `console_level` / `file_level` accept `OFF` or null.
  This replaces the custom `TAI_OFF` level trick.
- **Redaction is built in.** It replaces projects' own `tai/redact.py` patchers.
  - Core installs the one global loguru patcher. It scrubs the message and the
    string `extra` values of every record, before any sink sees it.
  - What it scrubs: URL passwords (`scheme://user:***@`); `x-auth-key`,
    `x-secret-key`, `authorization`, `x-fleet-key` and `x-tai-token` header
    values; `Bearer` tokens; and every literal registered with `add_secret()` (or
    passed as `secrets=`), together with its URL-encoded form.
  - Core's console and JSONL sinks also scrub rendered tracebacks, and core now
    passes `diagnose=False` so locals are no longer printed in tracebacks.
- **Patchers and sinks.**
  - Don't call `logger.configure(patcher=...)`; it replaces core's patcher. Chain
    yours with `add_patcher(fn)`.
  - Wrap project sinks that render tracebacks with `redacting_sink(sink)`.
- **`get_logger` signature.** It is now `get_logger(name=None)`. Levels are set only
  via `configure_logging`.

---

## 5. Behaviour changes

- **Stream name.** It is always the fixed prefix `cam_sn_` followed by the
  `stream.name` template, whose default `{sn}_{camera}` is unique per camera. The
  prefix keeps every stream inside the bundled `mediamtx-host`'s `live/cam_sn_*`
  on-demand rule. The 4.x URL was `cam_sn_{camera_sn_id}`; a single-stream node
  that must keep its existing URL sets `stream.name: "{sn}"`. Writing the prefix
  yourself (`"cam_sn_{sn}"`) still works, since it is never doubled, but the gate
  warns. Two enabled streams that resolve to the same name are a validation error.
- **WebRTC port.** `webrtc_port` is honoured in the bare-IP WebRTC URL.
- **Model paths.** An explicit `path` always wins; 4.x replaced it with
  `models/<name>/<name>` when that file existed. Relative paths resolve against the
  config file's directory.
- **Masked sources.** `VideoCaptureAsync` log lines, `RuntimeError` messages and
  `get_stats()["src"]` show the source with its password masked. `cap.src` still
  holds the real URL.
- **Typed sources skip auto-detection.** A `device` source opens as a V4L2 device
  even when its path isn't `/dev/video*` (a udev symlink), and a `file` source opens
  as a file whatever its extension.
- **Audio filters.** Camera audio gets no high-pass unless `filter.highpass_hz` asks
  for one, so with the defaults the FFmpeg command is the same as 4.4.8.
- **FFmpeg log file.** It is `.core-streamer-logs/ffmpeg_<stream name>.log`.
- **Lazy streamer start.** A streamer whose size is unknown starts FFmpeg on the
  first `update_frame`; `start_streaming()` returns True while it waits.

---

## 6. Worked example: one real project (petrobangla node)

| 4.x | 5.0 |
|---|---|
| `meta.name` | `node.sn` |
| `meta.version` | `node.version` |
| `cameras[0]` (`local: true`, `local_source`) | `cameras.ocr.source: {type: device, path}` |
| `cameras[0].settings.*` (resolution, flash, capture, night, …) | `cameras.ocr.settings.*` (project) |
| `cameras[1].rtsp_source.{username,password,ip,port,path}` | `cameras.surveillance.source: {type: rtsp, user, password, host, port, path}` |
| `cameras[1].settings.sn` | removed (= `node.sn`); the stream keeps its URL with `stream.name: "{sn}"` |
| `cameras[1].settings.version` | `cameras.surveillance.settings.version` (project) |
| `cameras[1].settings.resolution` / `fps` | `cameras.surveillance.capture.width` / `height` / `fps` |
| `cameras[1].settings.timestamp_overlay.*` | `cameras.surveillance.settings.timestamp_overlay.*` (project, applied through `frame_hook`) |
| `advanced.livestream.{mediamtx_ip, rtsp_port, demand_*}` | `mediamtx.{host, rtsp_port, demand.*}` |
| `advanced.livestream.{enabled, bitrate, on_demand, encoder}` | `cameras.surveillance.stream.*` (drop `encoder.codec: copy`) |
| `advanced.livestream.hw_encode: false` | drop it (the default codec is `libx264`) |
| `advanced.livestream.audio.{source, mic.*, camera.*}` | `cameras.surveillance.stream.audio.{input, filter, bitrate}` |
| `advanced.livestream.settings.meter_inset` | `cameras.surveillance.stream.settings.meter_inset` (project) |
| `advanced.api.*` | `api.*` (`max_retries` → `retries.max`, `cache_*` → `outbox.*`, `pool_*` → `pool.*`, `default_content_type` → `content_type`; `max_cache_retries: 1000000000` → `outbox.max_retries: null`) |
| `advanced.models.<name>.load_options.task` | `models.<name>.task` |
| `advanced.pipeline.logging.{cli_sink_level, file_sink_level}` | `logging.{console_level, file_level}` |
| `advanced.pipeline.*`, `advanced.timings` | `settings.*` (project) |

The surveillance camera, before and after (abridged):

```yaml
# 4.x
meta: {name: "pb-00004", version: "1.0.0"}
cameras:
  - local: true
    local_source: "/dev/video-camera0"
    settings: {resolution: {width: 3840, height: 2160}, flash: {...}}
  - local: false
    rtsp_source: {username: admin, password: "...", ip: 192.168.1.64, port: 554, path: /Streaming/Channels/101}
    settings: {sn: "pb-00004", version: "1.0.0", resolution: {width: 1280, height: 720}, fps: 10,
               timestamp_overlay: {enabled: true}}
advanced:
  livestream:
    enabled: true
    mediamtx_ip: mediamtx.transformsai.com
    bitrate: 800k
    on_demand: true
    encoder: {preset: veryfast, codec: copy, queue_size: 1}
    audio: {source: mic, bitrate: 24k, mic: {device: inmp441, rate: 16000, channels: 2, map_channel: 0, gain_db: 18}}
    settings: {meter_inset: {enabled: true}}
```

```yaml
# 5.0
node: {sn: "pb-00004", version: "1.0.0"}
cameras:
  ocr:
    source: {type: device, path: /dev/video-camera0}
    settings: {resolution: {width: 3840, height: 2160}, flash: {...}}
  surveillance:
    source: {type: rtsp, host: 192.168.1.64, user: admin, password: "...", path: /Streaming/Channels/101}
    capture: {width: 1280, height: 720, fps: 10}
    stream:
      name: "{sn}"                     # -> cam_sn_pb-00004, the 4.x URL
      bitrate: 800k
      on_demand: true
      encoder: {preset: veryfast, queue_size: 1}
      audio:
        input: {type: alsa, device: inmp441, rate: 16000, channels: 2, channel: 0}
        filter: {gain_db: 18}          # the alsa default; set highpass_hz: 300 for the measured hum cut
        bitrate: 24k
      settings: {meter_inset: {enabled: true}}
    settings: {version: "1.0.0", timestamp_overlay: {enabled: true}}
mediamtx: {host: mediamtx.transformsai.com}
```

For this config, the 5.0 FFmpeg command is identical to 4.4.8's except for one
token: `volume=18.0dB` instead of `volume=18dB`, which is the same gain. To switch
to the camera's microphone, set `audio.input: {type: camera}`; no URL is needed.

**Fleet tooling** that reads the serial by regex on `name:` must read `node.sn`
instead (out of scope for core; do it after adopting 5.0).
