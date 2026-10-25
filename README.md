# transformsai-ai-core

Internal, config-driven CV toolkit for edge ML projects: video capture, YOLO inference,
RTSP/WebRTC streaming, HTTP upload, structured logging, YAML config. Thread-safe, lazy-imported
(heavy deps load on first use). This README is the quick reference — read the docstrings/source
for anything not covered here.

> **5.0 is a clean break from 4.x config.** See [MIGRATION-5.0.md](MIGRATION-5.0.md) and
> [CHANGELOG.md](CHANGELOG.md).

```bash
uv add "transformsai-ai-core @ git+https://github.com/Transforms-AI/transformsai-ai-core.git"
# Inference backends (mutually exclusive — pick one): onnx | rknn | trt
uv add "transformsai-ai-core[rknn] @ git+https://github.com/Transforms-AI/transformsai-ai-core.git"
```

`import transformsai_ai_core as core` — everything in the table below is a top-level export.

| Symbol | What it is |
|---|---|
| `load_config`, `validate_config`, `save_config`, `Config` | YAML → typed config + `ValidationReport` |
| `get_logger`, `configure_logging`, `add_secret`, `redact` | structured JSONL logger with secret redaction |
| `VideoCaptureAsync` | background-thread capture |
| `MediaMTXStreamer` | push frames → MediaMTX (RTSP/WebRTC) |
| `ApiClient`, `Response` | generic pooled HTTP client |
| `YOLOWrapper`, `YOLOEWrapper` | Ultralytics wrappers (auto-export ONNX/TRT/RKNN) |
| `mat_to_response`, `time_to_string` | JPEG-encode a frame / server timestamp format |

The full config API (every model, `register_validator`, `register_secret`, …) lives in
`transformsai_ai_core.config`.

**You have freedom here.** Every runtime class takes plain kwargs and works standalone — config is a
convenience, not a requirement. With a config, build components from the typed objects:
`VideoCaptureAsync.from_camera(camera)`, `MediaMTXStreamer.from_camera(camera, config.mediamtx, config.node)`,
`ApiClient.from_config(config.api)`, `YOLOWrapper.from_config(config.models["name"])`. Keyword overrides
always win over config; defaults live only in the config models.

---

## Caveats (read these)

- **Config is validated, never mutated.** Derived values are properties: `camera.source.url`,
  `config.stream_urls(camera)`, `model.resolved_path`. Nothing is written back into your YAML/dict.
- **Unknown keys are reported, never silently dropped.** Outside a `settings` block, an unknown key is a
  warning at runtime (`load_config`) and an error at the gate (`validate_config` /
  `transformsaicore config validate`). Every core object has a free `settings: {}`; the root has one too.
- **`ApiClient` never raises on network failure** — it returns `Response(status_code=0, ok=False)`. Always
  check `resp.ok`. A failed *write* is kept in the outbox and retried; GET/HEAD/OPTIONS never are.
- **`cap.read(copy=False)` returns a live reference** — fast, but don't mutate it (next frame overwrites it).
- **Call `configure_logging` first, then add your own sinks.** It only ever replaces core's sinks, so
  calling it again later is safe. Don't call `logger.configure(patcher=...)` — use `add_patcher()`.
- **Inference backends conflict.** Installing `trt`, `rknn`, and `onnx` together breaks; pick one per device.
  TensorRT is auto-excluded on `aarch64` at the dependency level.

---

## Config (`config.yaml`)

The full, commented reference — every key with its default and meaning — is
[config.example.yaml](config.example.yaml), generated from the schema
(`uv run transformsaicore config example`). A minimal config:

```yaml
node:
  sn: "pb-00004"                    # REQUIRED: the one node identity

cameras:                            # a map keyed by camera name ...
  surveillance:
    source: {type: rtsp, host: 192.168.1.64, user: admin, password: "..."}
    capture: {fps: 10, width: 1280, height: 720}
    stream: {on_demand: true}       # stream name defaults to cam_sn_{sn}_{camera}
  ocr:
    source: {type: device, path: /dev/video0}
    settings: {flash: {pin: 17}}    # project-owned: core never looks inside

# cameras:                          # ... or a list, auto-named camera_0, camera_1, ...
#   - source: {type: rtsp, host: 10.0.0.11}

mediamtx: {host: mediamtx.example.com}
api: {base_url: "https://api.example.com/api/v1/", endpoints: {heartbeat: "telemetry/heartbeats/"}}
models:
  person-det: {path: models/yolo11n.pt}
logging: {console_level: WARNING, file_level: INFO}
settings: {}                        # the project's own section
```

Required keys: `node.sn`, each camera's `source` (and its type's required field), `api.base_url` when
`api` is enabled, `mediamtx.host` when any stream is enabled, and one of `path` / `download_key` / `type`
per model. Source types: `rtsp` (host), `device` (path or index), `file` (path), `url` (url).

Values convert only when unambiguous (`"10"` → 10, `"true"` → true). Durations take seconds or units
(`"500ms"`, `"3s"`, `"2m"`, `"1h"`); bitrates take `"800k"`, `"1.2M"` or bits/s. The only limits are
physical ones (a period > 0, a port 1–65535) — never policy ranges.

```python
from transformsai_ai_core.config import ConfigError, load_config, validate_config

config, report = load_config("config.yaml")     # lenient: unknown keys / bad optional values -> warnings
for issue in report.warnings:                    # Issue(path, message, suggestion)
    print(issue)                                 # cameras.a.stream.bitrat: unknown key (did you mean 'bitrate'?)
# raises ConfigError(report) when a required key is missing/invalid -- with every problem, not the first

report = validate_config("config.yaml")          # strict, no side effects: the pre-deploy gate
if not report.ok: ...

config.cameras["surveillance"].source.url        # rtsp://admin:...@192.168.1.64:554/... (password URL-encoded)
config.stream_urls("surveillance").webrtc        # https://mediamtx.example.com/live/cam_sn_pb-00004_surveillance/
config.models["person-det"].resolved_path        # Path, relative to the config file's directory
config.secrets()                                 # every secret value (for log redaction)
config.masked_dump("yaml")                       # full config with secrets as ***
```

### Project settings and validators

Project keys go in a `settings` block. Register checks so the same gate covers them:

```python
from pydantic import BaseModel, ConfigDict
from transformsai_ai_core.config import register_secret, register_validator

class MeterSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")   # report typos too
    read_interval: float = 60.0

register_validator("settings.meter", MeterSettings)           # pydantic model or a function
register_validator("cameras.*.settings.flash", check_flash, required=True)
register_secret("settings.modbus.password")                   # in secrets(), masked, redacted in logs
```

Failures are errors in strict mode and warnings in lenient mode (unless `required=True`).

---

## Logger

JSONL to `.core-logs/{date}/{run_id}_{time}.jsonl` + console. Auto-rotates at 10 MB.
`.error()`/`.exception()` capture the current traceback automatically.

```python
from loguru import logger as loguru_logger
from transformsai_ai_core import configure_logging, get_logger

configure_logging(config.logging, secrets=config.secrets())   # 1. entry script, first
loguru_logger.add(my_sink, level="WARNING")                   # 2. then your own sinks (never removed by core)

logger = get_logger()           # auto-detects caller module name
logger = get_logger("MyClass")  # or pass a name / object
logger.info("rtsp://admin:pw@cam/s")  # -> rtsp://admin:***@cam/s on every sink
```

- `console_level` / `file_level`: a level name, or `OFF` / `null` for no sink. Keyword args
  (`configure_logging(console_level="DEBUG")`) override the config.
- Redaction runs on every record (message and `extra`) before any sink: URL passwords, auth headers
  (`x-auth-key`, `x-secret-key`, `authorization`, `x-fleet-key`, `x-tai-token`), `Bearer` tokens, and
  every registered secret (plus its URL-encoded form). Core's sinks also scrub tracebacks; wrap your own
  with `redacting_sink(sink)`. Chain extra patchers with `add_patcher(fn)`.

View a file: `cat run_*.jsonl | logdy` (https://logdy.dev).

---

## Video Capture

Reconnect is supervised and **on by default** — a failed open, a run of failed reads, a stalled stream or
an unexpected exception all reconnect on a jittered exponential backoff.

```python
from transformsai_ai_core import VideoCaptureAsync

cap = VideoCaptureAsync.from_camera(config.cameras["surveillance"], frame_hook=stamp_time).start()

# manual:
cap = VideoCaptureAsync(
    src="rtsp://...",             # URL, file path, device path or index
    source_type=None,             # "device" | "file" | "network" (None = auto-detect)
    width=None, height=None, fps=None,
    buffer_size=1, hw_decode=False, max_frame_age_ms=None, auto_resize=True,
    # --- reconnect policy ---
    auto_restart_on_fail=True, restart_delay=30.0,  # restart_delay = backoff CEILING
    restart_backoff_start=1.0, restart_backoff_jitter=0.2, restart_reset_after=30.0,
    max_restart_attempts=None, stall_timeout=None,  # None = auto max(5, 20/fps)
    open_timeout=5.0, rtsp_transport="tcp", ffmpeg_options=None,
    health_log_interval=60.0,
    on_state_change=None,         # fn(state, stats) on every transition
    frame_hook=None,              # fn(frame) -> frame | None, on the capture thread
    loop=None,                    # file sources: restart at EOF (default True)
).start()

grabbed, frame = cap.read(wait_for_new_frame=True, timeout=1.0, copy=True)
cap.stop(); cap.release()
```

### Frame hook

`frame_hook` runs on the capture thread on every published frame, so every consumer (inference,
stream, upload) sees its result. Return a frame, or `None` if you modified it in place. Exceptions are
caught and logged (first failure, then at most once a minute) and the unhooked frame is published — a
hook can never stop capture. `get_stats()["hook_errors"]` counts failures. The classic use is the
camera-clock fix:

```python
from transformsai_ai_core.utils import hide_camera_timestamp_and_add_current_time

def stamp_time(frame):
    return hide_camera_timestamp_and_add_current_time(frame, inplace=True, corner_label="Gate 2")
```

### Health

```python
cap.state        # idle | connecting | streaming | reconnecting | failed | stopped
cap.is_healthy   # streaming AND a frame arrived within the stall window
cap.get_stats()  # restart_count, current_backoff, last_error, uptime_seconds, measured_fps,
                 # total_frames, frames_dropped_stale, orphaned_threads, hook_errors, ...
```

Log lines and `get_stats()["src"]` show the source with its password masked. While the source is down,
`read()` blocks up to `timeout` and returns `(False, None)` — it never busy-spins. `release()` abandons a
thread wedged inside OpenCV instead of releasing a handle it is still using; such an instance is
terminal — build a new `VideoCaptureAsync`.

---

## MediaMTX Streaming

A stream belongs to a camera (`cameras.<name>.stream`); the server (`mediamtx`) is shared. Size and fps
come from `stream`, else the camera's `capture`; an unknown size is taken from the first frame.

```python
from transformsai_ai_core import MediaMTXStreamer

camera = config.cameras["surveillance"]
streamer = MediaMTXStreamer.from_camera(camera, config.mediamtx, config.node)
# manual: MediaMTXStreamer("mediamtx.example.com", 8554, "cam_sn_pb-00004", fps=10,
#   frame_width=1280, frame_height=720, bitrate="800k", encoder_codec="libx264", ...)

streamer.start_streaming()
streamer.update_frame(frame)      # call in loop — non-blocking
streamer.get_rtsp_url(); streamer.get_webrtc_url()
streamer.stop_streaming()
```

- **Stream name**: `stream.name` template with `{sn}`, `{camera}`, `{index}`; default
  `cam_sn_{sn}_{camera}`. Names must be unique across enabled streams (a validation error otherwise).
- **URLs**: `mediamtx.rtsp_url` / `webrtc_url` / `demand_url` templates (`{host}`, `{rtsp_port}`,
  `{webrtc_port}`, `{name}`); `null` keeps the 4.x URLs (WebRTC: `https://{host}/live/{name}/`, or
  `http://{host}:{webrtc_port}/live/{name}/` for a bare IP).
- **Encoder**: `encoder.codec` is `libx264` (default), `auto` (first available hardware encoder) or any
  FFmpeg H.264 encoder name.
- **Audio** (`stream.audio`, best-effort: a failed audio input drops to video-only and is retried on a
  backoff): `input: {type: alsa, device: ...}` (a local mic) or `input: {type: camera}` (this camera's own
  source). `filter` applies channel select (alsa) → `highpass_hz` → `gain_db` → `limiter` → resample;
  unset keys use the input's defaults (alsa: 20 Hz, 18 dB; camera: off, 0 dB). A high-pass above 20 Hz is
  4th-order. `streamer.audio_active` / `audio_error` report the state.

### On-demand publishing (`on_demand: true`)

`start_streaming()` starts a supervisor thread instead of FFmpeg: it polls the demand flag
(`https://{host}/demand/{name}` by default) and runs FFmpeg only while a viewer is watching, stopping it
`mediamtx.demand.grace_period` seconds after demand ends. A failed/unreadable poll never changes state.
Inject `demand_check=callable` for tests or other backends. **The server side needs matching setup** —
see [mediamtx-host/README.md](mediamtx-host/README.md); its `mediamtx.yml` serves demand flags only for
`live/cam_sn_*` paths.

---

## ApiClient

Generic, pooled (`requests.Session` keep-alive), with a directory-per-request disk outbox + background
retry. A heartbeat is just `post(..., cache=False)`. Returns a frozen `Response` — **never raises** on
transport failure.

```python
from transformsai_ai_core import ApiClient, mat_to_response

if config.api and config.api.enabled:
    client = ApiClient.from_config(config.api)      # endpoints auto-registered; overrides win

resp = client.post("data", json={"event": "detection", "count": 5})
if resp.ok:                       # ALWAYS check — failure → status_code=0, ok=False
    print(resp.status_code, resp.json(), resp.attempts)

client.request("REPORT", "events", json={"x": 1})   # any verb
client.post("alerts", data={"alert": "unsafe"}, files={"image": mat_to_response(frame)})
fut = client.post("data", json={"k": "v"}, async_=True)       # Future[Response]
client.post("heartbeat", json={"alive": True}, cache=False)   # never kept in the outbox
client.send("heartbeat", json={...})                          # by endpoint name
client.shutdown(wait=True)
```

Config → constructor: `retries.{max, backoff, backoff_max, on_status}` → `max_retries`, `retry_*`;
`outbox.{enabled, dir, retry_interval, max_items, max_age, max_retries}` → `cache_*`, `max_cache_*`;
`pool.{connections, maxsize, workers}` → `pool_*`, `max_workers`; `content_type` →
`default_content_type`. Outbox limits: `max_items` / `max_age` 0 or null = unbounded; `max_retries`
null = unbounded, 0 = never retried.

### Never-expiring entries (`persist=True`)

`persist=True` (per call, or an endpoint profile's `persist: true`) makes an outbox entry immune to
`max_age` and `max_retries`: it is retried until it succeeds. `max_items` still applies as a disk
backstop (transient entries go first). `list_cached()` / `remove_cached(fid)` inspect and drop entries.

```yaml
api:
  endpoints:
    alert: "alerts/"            # string form → normal expiry
    critical: {path: "critical/", persist: true}
```

---

## YOLO / YOLOE

```python
from transformsai_ai_core import YOLOWrapper, YOLOEWrapper

model = YOLOWrapper.from_config(config.models["person-det"])   # path = model.resolved_path
model = YOLOWrapper({"path": "yolo11n.pt", "task": "detect", "batch": 1})   # manual dict ctor
# keys: path (REQUIRED), task, batch, export, export_options{...passed to ultralytics}, set_classes (YOLOE)

results = model.predict(source=frame, conf=0.5)   # source: str|Path|ndarray|list ; any ultralytics kwarg
model.original_model_path; model.exported_model_path

model = YOLOEWrapper.from_config(config.models["open-vocab"])   # set_classes: [person, knife]
```

`transformsaicore download models` fetches models that have a `download_key` or `type` and no explicit
`path` into `models/<name>/`.

---

## End-to-end skeleton

```python
import time
from transformsai_ai_core import (
    ApiClient, MediaMTXStreamer, VideoCaptureAsync, YOLOWrapper, configure_logging, get_logger,
)
from transformsai_ai_core.config import load_config

config, report = load_config("config.yaml")
configure_logging(config.logging, secrets=config.secrets())
logger = get_logger(__name__)
for issue in report.warnings:
    logger.warning(f"config: {issue}")

models = {name: YOLOWrapper.from_config(m) for name, m in config.models.items()}
client = ApiClient.from_config(config.api) if config.api and config.api.enabled else None

pipelines = []
for name, camera in config.cameras.items():
    cap = VideoCaptureAsync.from_camera(camera).start()
    streamer = None
    if camera.stream and camera.stream.enabled:
        streamer = MediaMTXStreamer.from_camera(camera, config.mediamtx, config.node)
        streamer.start_streaming()
    pipelines.append((name, cap, streamer))

try:
    while True:
        for name, cap, streamer in pipelines:
            ok, frame = cap.read(timeout=0.5)
            if not ok:
                continue
            if streamer:
                streamer.update_frame(frame)
            # TODO: inference with models[...], uploads with client.post(...)
        time.sleep(0.01)
finally:
    for _, cap, streamer in pipelines:
        cap.release()
        if streamer:
            streamer.stop_streaming()
    if client:
        client.shutdown()
```

---

## CLI

```bash
uv run transformsaicore config validate config.yaml          # strict gate: exit 1 on errors
uv run transformsaicore config validate --lenient config.yaml
uv run transformsaicore config example -o config.example.yaml
uv run transformsaicore config schema > config.schema.json   # JSON Schema for editors
uv run transformsaicore download models --config config.yaml [--base-url URL]
```

## Tests

Mixed `unittest` + `pytest` (pytest is not a declared dep):
```bash
uv run --frozen --with pytest pytest tests/ --ignore tests/test_yolo_wrapper.py
uv run --frozen python -m unittest tests.test_config_schema
uv run --frozen python tests/test_yolo_wrapper.py   # downloads ultralytics models
```
