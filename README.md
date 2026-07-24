# transformsai-ai-core

Internal, config-driven CV toolkit for edge ML projects: video capture, YOLO inference,
RTSP/WebRTC streaming, HTTP upload, structured logging, YAML config. Thread-safe, lazy-imported
(heavy deps load on first use). This README is the quick reference — read the docstrings/source
for anything not covered here.

```bash
uv add "transformsai-ai-core @ git+https://github.com/Transforms-AI/transformsai-ai-core.git"
# Inference backends (mutually exclusive — pick one): onnx | rknn | trt
uv add "transformsai-ai-core[rknn] @ git+https://github.com/Transforms-AI/transformsai-ai-core.git"
```

`import transformsai_ai_core as core` — everything in the table below is a top-level export.

| Symbol | What it is |
|---|---|
| `process_config`, `load_config`, `save_config` | YAML → validated dict |
| `get_logger`, `configure_logging` | structured JSONL logger |
| `VideoCaptureAsync` | background-thread capture |
| `MediaMTXStreamer` | push frames → MediaMTX (RTSP/WebRTC/HLS) |
| `ApiClient`, `Response` | generic pooled HTTP client (**use this**) |
| `DataUploader` | legacy transformsai uploader (still works) |
| `YOLOWrapper`, `YOLOEWrapper` | Ultralytics wrappers (auto-export ONNX/TRT/RKNN) |
| `mat_to_response`, `time_to_string` | JPEG-encode a frame / server timestamp format |

**You have freedom here.** Every runtime class takes plain kwargs and works standalone — config is a
convenience, not a requirement. Use `from_config(section, **overrides)` to build from a config section,
or construct manually. Precedence is always **overrides > typed config fields > `extras`**.

---

## Caveats (read these)

- **`process_config` does not resolve models by default.** Pass `resolve_models=True` to download/export.
  Without it, `model["path"]` stays whatever's in the YAML.
- **The API config key is `advanced.api`, not `datasend`.** `DatasendConfig` the class still exists, but
  there is no `advanced.datasend` section anymore. Legacy `DataUploader` is built from `advanced.api` too.
- **`ApiClient` never raises on network failure** — it returns `Response(status_code=0, ok=False)`. Always
  check `resp.ok`. A failed *write* is cached and retried in the background; GET/HEAD/OPTIONS are never cached.
- **`cap.read(copy=False)` returns a live reference** — fast, but don't mutate it (next frame overwrites it).
- **Logger sink levels are set once**, on the first `configure_logging`/`get_logger` call. Call
  `configure_logging(...)` from your entry script *before* anything else logs. Later level changes are ignored.
- **`type` (model) and `meta.name` are the only truly required fields.** Everything else has a default.
- **`extras: {}` exists on every strict section** — a sanctioned dumping ground for keys we haven't modeled.
  Unknown keys there survive validation and reach `from_config`. Don't add Pydantic validation to freeform
  sections (`settings`, `timings`, `pipeline`, `extras`, `export_options`, `endpoints`).
- **Inference backends conflict.** Installing `trt`, `rknn`, and `onnx` together breaks; pick one per device.
  TensorRT is auto-excluded on `aarch64` at the dependency level.

---

## Config (`config.yaml`)

Single file drives everything. Only `meta.name` and each model's `type` are required; all else defaults.

```yaml
meta:
  name: "project-name"          # REQUIRED
  version: "1.0.0"
  token: ""                     # sent as x-token header by legacy uploader
  extras: {}

cameras:
  - local: false                # true = use local_source instead of RTSP
    local_source: ""            # path to local video file
    rtsp_source:                # process_config builds cam["rtsp_url"] from this
      username: "admin"
      password: ""
      ip: ""                    # REQUIRED if not local
      port: 554
      path: "/Streaming/Channels/101"
      extras: {}
    capture:                    # mirrors VideoCaptureAsync.__init__ (typed/validated)
      buffer_size: 1
      opencv_backend: auto      # 'auto'|'ffmpeg'|'gstreamer'|null
      max_frame_age_ms: null    # drop frames older than this (ms)
      width: null
      height: null
      driver: null
      auto_resize: true
      hw_decode: false
      fps: null
      open_timeout: 5.0         # network open/read timeout (applied via FFmpeg)
      rtsp_transport: tcp       # 'tcp'|'udp'|null (rtsp:// only; null = FFmpeg negotiates)
      ffmpeg_options: null      # verbatim OPENCV_FFMPEG_CAPTURE_OPTIONS override
      health_log_interval: 60.0 # periodic health line (0 = off)
      restart:                  # reconnect policy
        enabled: null           # null → legacy auto_restart_on_fail → true
        delay: null             # backoff CEILING; null → legacy restart_delay → 30.0
        backoff_start: 1.0      # first retry delay; doubles up to `delay`
        backoff_jitter: 0.2     # ±fraction, keeps multi-camera reconnects out of lockstep
        reset_after: 30.0       # healthy seconds before the ramp resets
        max_attempts: null      # null = retry forever
        stall_timeout: null     # null = auto max(5, 20/fps); 0 = off
        extras: {}
      timestamp:                # overwrite the camera OSD timestamp with current system time
        enabled: false          # off by default; stamps every published frame when true
        rect_ratios: null       # (x,y,w,h) hide-rect as frame ratios; null = top-left OSD default
        rect_coords: null       # (x,y,w,h) in pixels; overrides rect_ratios
        hide_color: null        # BGR of hide rectangle; null = white
        font_color: null        # BGR of timestamp text; null = black
        font_scale: null        # null = auto from rect height
        time_format: null       # strftime; null = '%Y-%m-%d %H:%M:%S'
        extras: {}
      auto_restart_on_fail: null  # LEGACY → restart.enabled
      restart_delay: null         # LEGACY → restart.delay
      extras: {}
    settings: {}                # freeform per-camera → config["cameras"][n]["settings"]
    extras: {}

advanced:
  models:
    model_name:                 # key = how you look it up in code
      type: "person-det"        # REQUIRED — category label
      download_key: ""          # server download key (used when resolve_models=True)
      batch: 1
      path: ""                  # auto-filled after download/export
      export: false             # auto-export on load
      load_options:
        lib_type: "YOLO"        # YOLO | YOLOE
        task: "detect"          # detect | classify | segment
        extras: {}
      export_options: {}        # freeform → passed straight to ultralytics export
      extras: {}

  timings: {}                   # freeform — your intervals live here

  api:                          # ApiClient config (see §ApiClient for field meanings)
    enabled: true
    base_url: ""                # REQUIRED if enabled
    headers: {}
    timeout: 30
    success_codes: [200, 201, 202, 204]
    default_content_type: auto  # json | form-data | auto
    auth_keys: []               # rotated per request
    auth_header: X-Secret-Key
    max_retries: 3
    retry_backoff: 1.0
    retry_backoff_max: 30
    retry_on_status: [408, 429, 500, 502, 503, 504]
    max_workers: null           # null = min(2, cpu_count)
    cache_enabled: true
    cache_dir: .core-api-cache
    cache_retry_interval: 100
    max_cache_items: 300
    max_cache_age_seconds: 86400
    max_cache_retries: 5
    endpoints: {}               # freeform {name: path | {path, method, headers, content_type, cache, persist}}
                                # auto-registered by from_config
    pool_connections: 10
    pool_maxsize: 10
    settings: {}                # freeform (e.g. jpeg_quality)
    extras: {}

  livestream:                   # mirrors MediaMTXStreamer.__init__
    enabled: true
    mediamtx_ip: "localhost"
    rtsp_port: 8554
    camera_sn_id: ""            # default, override per camera in from_config
    fps: 30
    frame_width: 1920
    frame_height: 1080
    bitrate: "1500k"
    hw_encode: false            # auto-detect GPU encoder
    debug_log_interval: 60.0
    on_demand: false            # true = push only while a viewer is watching (see §MediaMTX Streaming)
    demand_url: ""              # blank = derive https://{mediamtx_ip}/demand/cam_sn_{camera_sn_id}
    demand_poll_interval: 3.0   # seconds between outbound demand polls
    demand_grace_period: 10.0   # demand must stay OFF this long before FFmpeg stops (keep >= 5)
    demand_timeout: 5.0         # HTTP timeout per poll
    encoder:
      preset: ultrafast
      codec: copy               # copy | libx264
      queue_size: 2
      extras: {}
    settings: {}                # freeform (NOT consumed by from_config)
    extras: {}

  pipeline: {}                  # freeform — all project-specific settings
  extras: {}
```

```python
from transformsai_ai_core import process_config, load_config

config = process_config(
    "config.yaml",         # str | Path — REQUIRED
    base_dir=None,         # project root for model paths (default: config's parent dir)
    resolve_models=False,  # True = resolve model paths
    download_models=False, # True = download missing models (needs resolve_models)
)
# load_config(path, validate=True) = parse + validate only, no RTSP/model resolution

meta       = config["meta"]
cameras    = config["cameras"]            # list — iterate per camera
models_cfg = config["advanced"]["models"]
timings    = config["advanced"]["timings"]
api_cfg    = config["advanced"]["api"]
ls_cfg     = config["advanced"]["livestream"]
pipeline   = config["advanced"]["pipeline"]
```

---

## Logger

JSONL to `.core-logs/{date}/{time}_{run_id}.jsonl` (all levels) + console (INFO+ by default).
Auto-rotates at 10 MB. `.error()`/`.exception()` capture the current traceback automatically.

```python
from transformsai_ai_core import configure_logging, get_logger

configure_logging(cli_sink_level="INFO", file_sink_level="DEBUG")  # entry script, once

logger = get_logger()           # auto-detects caller module name
logger = get_logger("MyClass")  # or pass a name / object
logger.debug("file only by default")
logger.info("console + file")
try:
    risky()
except Exception:
    logger.exception("full traceback → file + console")
```
View a file: `cat run_*.jsonl | logdy` (https://logdy.dev).

---

## Video Capture

Auto-detects source type (device index / RTSP URL / file path). Reconnect is supervised
and **on by default** — a failed open, a run of failed reads, a stalled stream or an
unexpected exception all reconnect on a jittered exponential backoff.

```python
from transformsai_ai_core import VideoCaptureAsync

cap = VideoCaptureAsync.from_config(config["cameras"][0]).start()   # recommended

# manual:
cap = VideoCaptureAsync(
    src=cam["rtsp_url"],          # REQUIRED — RTSP URL (built by process_config), file path, or device index
    width=None, height=None,
    buffer_size=1,                # OpenCV buffer; 1 = minimal lag
    hw_decode=False,
    fps=None,
    opencv_backend="auto",        # "auto" | "ffmpeg" | "gstreamer"
    max_frame_age_ms=None,        # drop stale frames
    auto_resize=True,
    # --- reconnect policy ---
    auto_restart_on_fail=True,    # False = raise on a bad source, die on failure (legacy)
    restart_delay=30.0,           # backoff CEILING
    restart_backoff_start=1.0,    # first retry; doubles 1 → 2 → 4 … → restart_delay
    restart_backoff_jitter=0.2,
    restart_reset_after=30.0,     # healthy seconds before the ramp resets
    max_restart_attempts=None,    # None = forever
    stall_timeout=None,           # None = auto max(5, 20/fps); no frame for this long → reconnect
    open_timeout=5.0,             # bounds a blocking open/grab on a dead network
    rtsp_transport="tcp",         # rtsp:// only; None = let FFmpeg negotiate
    health_log_interval=60.0,
    on_state_change=None,         # fn(state, stats) on every transition
    # --- OSD timestamp overwrite (Hikvision clocks are often wrong) ---
    timestamp_overlay=False,      # True = stamp current system time over the camera OSD on every frame
    timestamp_overlay_options=None,  # dict passed to utils.hide_camera_timestamp_and_add_current_time
).start(loop=False)               # loop=True repeats video files

grabbed, frame = cap.read(wait_for_new_frame=True, timeout=1.0, copy=True)  # copy=False = live ref, don't mutate
cap.stop(); cap.release()
```

### Health

```python
cap.state        # idle | connecting | streaming | reconnecting | failed | stopped
cap.is_healthy   # streaming AND a frame arrived within the stall window
cap.get_stats()  # restart_count, current_backoff, last_error, uptime_seconds,
                 # measured_fps, total_frames, frames_dropped_stale, orphaned_threads, …
```

While the source is down, `read()` blocks up to `timeout` and returns `(False, None)` —
it never busy-spins. After `stop()`/`release()` it returns immediately.

`release()` waits `max(2, open_timeout + 1)`s for the capture thread; if OpenCV is wedged
in a blocking call the thread is abandoned (it releases its own handle when the call
returns) rather than releasing a handle another thread is still inside. Such an instance
is terminal — `start()` on it raises; build a new `VideoCaptureAsync`.

---

## MediaMTX Streaming

```python
from transformsai_ai_core import MediaMTXStreamer

streamer = MediaMTXStreamer.from_config(config["advanced"]["livestream"], camera_sn_id="cam_01")
# manual: MediaMTXStreamer(mediamtx_ip="localhost", rtsp_port=8554, camera_sn_id="cam_01",
#   fps=30, frame_width=1920, frame_height=1080, bitrate="1500k",
#   encoder_preset="ultrafast", encoder_codec="copy", stream_queue_size=2,
#   hw_encode=False, debug_log_interval=60.0)

streamer.start_streaming()
streamer.update_frame(frame)      # call in loop — non-blocking
streamer.get_rtsp_url(); streamer.get_webrtc_url(); streamer.get_hls_url()
streamer.stop_streaming()
```

### On-demand publishing (`on_demand: true`)

With `on_demand: true`, `start_streaming()` starts a supervisor thread instead of FFmpeg: it polls a
demand flag on the MediaMTX host (derived as `https://{mediamtx_ip}/demand/cam_sn_{camera_sn_id}`)
and launches FFmpeg only while a viewer is watching, stopping it `demand_grace_period` seconds after
demand ends. Idle cost is one boolean check per `update_frame` call — zero upstream bandwidth. Your
`start → update_frame loop → stop` code is unchanged, and `on_demand: false` (default) keeps
always-on behavior.

Fail-safe: a failed/unreadable poll never changes state (idle stays idle, live stays live). For
tests or alternate backends, inject `demand_check=callable` (returns bool) via `from_config`
overrides. **The server side needs matching setup** — MediaMTX `runOnDemand` hooks plus the flag
server; see [mediamtx-host/README.md](mediamtx-host/README.md) for the full from-scratch VPS
tutorial (docker compose, nginx, TLS, verification).

---

## ApiClient (preferred HTTP client)

Generic, pooled (`requests.Session` keep-alive), with directory-per-request disk cache + background retry.
No transformsai-specific formatting; a heartbeat is just `post(..., cache=False)`. Returns a frozen
`Response` — **never raises** on transport failure.

```python
from transformsai_ai_core import ApiClient, mat_to_response

client = ApiClient.from_config(config["advanced"]["api"])   # endpoints in config auto-registered

# verbs (sync — blocks through retries, returns Response)
resp = client.post("data", json={"event": "detection", "count": 5})
if resp.ok:                       # ALWAYS check — failure → status_code=0, ok=False
    print(resp.status_code, resp.json(), resp.attempts)

client.request("REPORT", "events", json={"x": 1})   # any verb via the generic core
client.get("status", params={"q": 1})

# files: {"field": ("name.jpg", bytes, "image/jpeg")} or list of tuples per field
client.post("alerts", data={"alert": "unsafe"}, files={"image": mat_to_response(frame)})

# async (fire-and-forget) → Future[Response]
fut = client.post("data", json={"k": "v"}, async_=True)

# heartbeat = a write that is never cached/retried
client.post("heartbeat", json={"alive": True}, cache=False)

# endpoint profiles: register once, send by name (per-call arg > profile > client default)
client.register_endpoint("sentiment", "/v1/sentiment")                    # bare path
client.register_endpoint("heartbeat", {"path": "/hb", "cache": False})    # rich dict
client.send("sentiment", json={"score": 0.9})

client.shutdown(wait=True)        # or:  with ApiClient(...) as client: ...
```

Manual ctor mirrors the `api:` YAML keys 1:1 — `base_url`, `headers`, `timeout`, `success_codes`,
`default_content_type` (`json|form-data|auto`), `auth_keys` (rotated), `auth_header`, retry/backoff knobs,
`retry_on_status` (4xx fails fast), `max_workers`, the `cache_*` knobs, `endpoints`, `pool_*`.
Failed writes are persisted atomically under `cache_dir/` and retried until success or limits;
force with `cache=True`/`cache=False`.

### Never-expiring entries (`persist=True`)

The cache normally gives up eventually: a failed write is retried in the background until
`max_cache_age_seconds` or `max_cache_retries` runs out, then it's dropped. `persist=True` changes
**only that** — the entry never runs out. Same trigger (failures only, `cache=True/False` still decides
*whether* to cache), same background retry, still removed the instant it succeeds; it just can't be aged
out or given up on. Use it for requests that must not be silently lost.

```python
client.post("alerts", json={"alert": "unsafe"}, persist=True)   # retried until it lands
```

```yaml
  api:
    endpoints:
      alert: "alerts/"            # string form → normal expiry
      critical:                   # profile dict
        path: "critical/"
        persist: true             # per-call persist=False still overrides this
```

`max_cache_items` continues to apply as a disk backstop (transient entries are evicted first, persistent
ones only if the cap is still exceeded). `list_cached()` shows everything pending with its `persistent`
flag; `remove_cached(fid)` is the only way to discard one by hand.

### DataUploader (legacy)

Transformsai-specific v3 uploader. Still supported; **prefer `ApiClient` for new code.** Built from
`config["advanced"]["api"]` (there is no `datasend` section). Notable bits: `send_data(...)` (async,
fire-and-forget, cached on failure), `send_data_sync(...)` (blocks, returns dict),
`send_heartbeat(sn, timestamp=time_to_string(t), status_log="", live_url="")` (not cached),
`shutdown(wait=True)`. See docstrings for the full kwarg list.

---

## YOLO / YOLOE

```python
from transformsai_ai_core import YOLOWrapper, YOLOEWrapper

config    = process_config("config.yaml", resolve_models=True)   # need this to populate model["path"]
model_cfg = config["advanced"]["models"]["model_name"]

model = YOLOWrapper(model_cfg)                # dict ctor
model = YOLOWrapper.from_config(model_cfg)    # equivalent
# model_cfg: path (REQUIRED), export, batch, load_options{lib_type, task}, export_options{...freeform}

results = model.predict(source=frame, conf=0.5)   # source: str|Path|ndarray|list ; any ultralytics kwarg
for r in results:
    r.boxes.xyxy, r.boxes.conf, r.boxes.cls
model.original_model_path        # Path
model.exported_model_path        # Path | None

# YOLOE — text/visual prompts
model_cfg["load_options"]["lib_type"] = "YOLOE"
model = YOLOEWrapper.from_config(model_cfg)   # extra key: "set_classes": list[str] | None
names = ["person", "knife"]
model.set_classes(names, model.get_text_pe(names))
results = model.predict(frame, conf=0.5)
```

---

## End-to-end skeleton

Minimal multi-camera loop: capture → stream → periodic inference → upload. Project logic goes in the
`TODO`s. This is a starting template, not a framework — restructure freely.

```python
import time, threading
from collections import deque
from transformsai_ai_core import (
    configure_logging, get_logger, process_config,
    VideoCaptureAsync, MediaMTXStreamer, ApiClient, YOLOWrapper,
    mat_to_response, time_to_string,
)

configure_logging()
logger = get_logger(__name__)

def io_worker(s):
    last = time.time()
    while True:
        ok, frame = s["cap"].read(copy=True)
        if not ok:
            time.sleep(0.1); continue
        if s["streamer"]:
            s["streamer"].update_frame(frame)            # TODO: draw overlays first
        if time.time() - last >= s["interval"]:
            with s["lock"]:
                if len(s["buf"]) >= s["buf"].maxlen:
                    s["buf"].popleft()
                s["buf"].append(frame.copy())
            last = time.time()
        time.sleep(0.01)

def main():
    cfg     = process_config("config.yaml", resolve_models=True)
    timings = cfg["advanced"]["timings"]
    api_cfg = cfg["advanced"]["api"]
    ls_cfg  = cfg["advanced"]["livestream"]

    models = {n: YOLOWrapper(m) for n, m in cfg["advanced"]["models"].items() if m.get("path")}
    client = ApiClient.from_config(api_cfg) if api_cfg.get("enabled") else None

    streams = []
    for i, cam in enumerate(cfg["cameras"]):
        cam_id   = f"cam_{i}"
        streamer = None
        if ls_cfg.get("enabled"):
            streamer = MediaMTXStreamer.from_config(ls_cfg, camera_sn_id=cam_id)
            streamer.start_streaming()
        s = {
            "id": cam_id,
            "cap": VideoCaptureAsync.from_config(cam).start(),
            "streamer": streamer,
            "buf": deque(maxlen=timings.get("frame_buffer_size", 1)),
            "lock": threading.Lock(),
            "interval": timings.get("frame_collect_interval", 3.0),
            "last_inf": 0,
        }
        streams.append(s)
        threading.Thread(target=io_worker, args=(s,), daemon=True).start()

    last_hb = time.time()
    try:
        while True:
            now = time.time()
            for s in streams:
                if now - s["last_inf"] >= timings.get("inference_interval", 5.0):
                    with s["lock"]:
                        frames = list(s["buf"])
                    for model in models.values():
                        if frames:
                            results = model.predict(frames, conf=0.5)
                            # TODO: process results, upload via client.post(...)
                    s["last_inf"] = now
            if client and now - last_hb >= timings.get("heartbeat_interval", 60):
                for s in streams:
                    client.post("heartbeat", json={"sn": s["id"], "ts": time_to_string(now)}, cache=False)
                last_hb = now
            time.sleep(0.1)
    except KeyboardInterrupt:
        pass
    finally:
        for s in streams:
            s["cap"].stop(); s["cap"].release()
            if s["streamer"]:
                s["streamer"].stop_streaming()
        if client:
            client.shutdown()

if __name__ == "__main__":
    main()
```

---

## CLI

```bash
uv run transformsaicore download models --config config.yaml   # currently the only subcommand
```

## Tests

Mixed `unittest` + `pytest` (pytest is not a declared dep). Examples:
```bash
uv run python -m unittest tests.test_logger
uv run python tests/test_config_manager.py
uv run --with pytest pytest tests/test_datasend.py
```
