# TransformsAI AI Core

Internal CV library. Config-driven, thread-safe utilities for video capture, streaming, data upload, logging, config management, and AI inference.

```bash
uv add "transformsai-ai-core @ git+https://github.com/Transforms-AI/transformsai-ai-core.git"
# Extras: onnx, rknn, trt, ultralytics
uv add "transformsai-ai-core[rknn] @ git+https://github.com/Transforms-AI/transformsai-ai-core.git"
```

---

## Config (`config.yaml`)

All components are driven from this single file.

```yaml
meta:
  name: "project-name"          # REQUIRED
  version: "1.0.0"              # default: "1.0.0"
  token: ""                     # default: "" — sent as x-token header

cameras:
  - local: false                # default: false — true = use local_source instead of RTSP
    local_source: ""            # default: "" — path to local video file
    rtsp_source:
      username: "admin"         # default: "admin"
      password: ""              # default: ""
      ip: ""                    # REQUIRED if not local
      port: 554                 # default: 554
      path: "/Streaming/Channels/101"  # default: "/Streaming/Channels/101"
    settings:                   # freeform — accessed as config["cameras"][n]["settings"]
      resolution:
        width: 1920
        height: 1080
      fps: 15
      buffer_size: 1
      hw_decode: false

advanced:
  models:
    model_name:                 # key = model identifier, used to access in code
      download_key: ""          # default: "" — server download key
      type: "person-det"        # REQUIRED — model type/category label
      batch: 1                  # default: 1
      path: ""                  # default: "" — auto-populated after download/export
      export: false             # default: false — auto-export on load
      load_options:
        lib_type: "YOLO"        # default: "YOLO" — options: YOLO | YOLOE
        task: "detect"          # default: "detect" — options: detect | classify | segment
      export_options:           # freeform — passed directly to ultralytics export
        format: "onnx"
        half: false
        dynamic: true
        imgsz: 640

  timings:                      # freeform — all accessed via config["advanced"]["timings"]
    inference_interval: 5.0
    datasend_interval: 180
    heartbeat_interval: 60
    frame_collect_interval: 3.0
    frame_buffer_size: 1

  datasend:
    enabled: true               # default: true
    base_url: ""                # REQUIRED if enabled
    endpoints:                  # freeform — e.g. endpoints["data"], endpoints["alerts"]
      data: "/data"
    secret_keys: []             # default: [] — API auth keys

  livestream:
    enabled: true               # default: true
    mediamtx_ip: "localhost"    # default: "localhost"
    rtsp_port: 8554             # default: 8554
    fps: 30                     # default: 30
    settings:                   # freeform
      queue_size: 2

  pipeline:                     # freeform — all project-specific settings go here
    confidence_threshold: 0.5
```

---

## Full Pipeline Example

```python
import time
import threading
from collections import deque
from transformsai_ai_core import (
    get_logger,
    process_config,
    VideoCaptureAsync,
    MediaMTXStreamer,
    DataUploader,
    YOLOWrapper,
    mat_to_response,
    time_to_string,
)

logger = get_logger(__name__)

def camera_io_worker(stream_cfg: dict):
    cap      = stream_cfg["capture"]
    streamer = stream_cfg["streamer"]
    buf      = stream_cfg["frame_buffer"]
    lock     = stream_cfg["lock"]
    buf_size = stream_cfg["buffer_size"]
    interval = stream_cfg["collect_interval"]
    last     = time.time()

    while True:
        grabbed, frame = cap.read(copy=True)
        if not grabbed:
            time.sleep(0.1)
            continue

        if streamer:
            display = frame.copy()
            # TODO: draw overlays on display
            streamer.update_frame(display)

        now = time.time()
        if now - last >= interval:
            with lock:
                if len(buf) >= buf_size:
                    buf.popleft()
                buf.append(frame.copy())
            last = now

        time.sleep(0.01)

def main():
    config         = process_config("config.yaml", resolve_models=True)
    meta           = config["meta"]
    cameras        = config["cameras"]
    models_cfg     = config["advanced"]["models"]
    timings        = config["advanced"]["timings"]
    datasend_cfg   = config["advanced"]["datasend"]
    livestream_cfg = config["advanced"]["livestream"]
    pipeline_cfg   = config["advanced"]["pipeline"]

    # Models
    models = {}
    for name, m in models_cfg.items():
        if m.get("path"):
            models[name] = YOLOWrapper(m)

    # Uploader
    uploader = None
    if datasend_cfg.get("enabled"):
        headers  = {"x-token": meta["token"]} if meta.get("token") else None
        uploader = DataUploader(
            base_url=datasend_cfg["base_url"],
            headers=headers,
            secret_keys=datasend_cfg.get("secret_keys"),
            source=meta["name"],
            project_version=meta.get("version"),
        )

    # Cameras
    stream_configs = []
    for idx, cam_cfg in enumerate(cameras):
        cam_id  = f"cam_{idx}"
        res     = cam_cfg["settings"]["resolution"]
        capture = VideoCaptureAsync(
            src=cam_cfg["rtsp_url"],
            width=res.get("width", 1920),
            height=res.get("height", 1080),
            buffer_size=cam_cfg["settings"].get("buffer_size", 1),
            hw_decode=cam_cfg["settings"].get("hw_decode", False),
            fps=cam_cfg["settings"].get("fps", 30),
        ).start()

        streamer = None
        if livestream_cfg.get("enabled"):
            streamer = MediaMTXStreamer(
                mediamtx_ip=livestream_cfg["mediamtx_ip"],
                rtsp_port=livestream_cfg["rtsp_port"],
                camera_sn_id=cam_id,
                fps=livestream_cfg.get("fps", 30),
                frame_width=res.get("width", 1920),
                frame_height=res.get("height", 1080),
            ).start_streaming()

        buf_size = timings.get("frame_buffer_size", 1)
        interval = timings.get("frame_collect_interval", 3.0)
        stream_cfg = {
            "camera_sn":           cam_id,
            "capture":             capture,
            "streamer":            streamer,
            "frame_buffer":        deque(maxlen=buf_size),
            "lock":                threading.Lock(),
            "buffer_size":         buf_size,
            "collect_interval":    interval,
            "last_inference_time": 0,
        }
        stream_configs.append(stream_cfg)
        threading.Thread(target=camera_io_worker, args=(stream_cfg,), daemon=True).start()

    last_heartbeat = last_datasend = time.time()

    try:
        while True:
            now = time.time()

            for stream in stream_configs:
                # Inference
                if now - stream["last_inference_time"] >= timings.get("inference_interval", 5.0):
                    with stream["lock"]:
                        frames = list(stream["frame_buffer"])
                    if frames:
                        for name, model in models.items():
                            results = model.predict(frames, conf=pipeline_cfg.get("confidence_threshold", 0.5))
                            # TODO: process results
                    stream["last_inference_time"] = now

                # TODO: project-specific per-camera logic here

            if uploader:
                # Heartbeat
                if now - last_heartbeat >= timings.get("heartbeat_interval", 60):
                    for stream in stream_configs:
                        uploader.send_heartbeat(
                            sn=stream["camera_sn"],
                            timestamp=time_to_string(now),
                            status_log="Running normally",
                            live_url=stream["streamer"].get_hls_url() if stream["streamer"] else "",
                        )
                    last_heartbeat = now

                # Data upload
                if now - last_datasend >= timings.get("datasend_interval", 180):
                    for stream in stream_configs:
                        with stream["lock"]:
                            payload = {
                                "sn":        stream["camera_sn"],
                                "timestamp": time_to_string(now),
                                # TODO: add your data fields
                            }
                        files = None
                        if stream["frame_buffer"]:
                            frame = stream["frame_buffer"][-1].copy()
                            # TODO: draw overlays on frame before upload
                            files = {"image": mat_to_response(frame, max_width=1920, jpeg_quality=65)}
                        uploader.send_data(
                            data=payload,
                            files=files,
                            endpoint_path=datasend_cfg["endpoints"].get("data", "/data"),
                        )
                    last_datasend = now

            time.sleep(0.1)

    except KeyboardInterrupt:
        pass
    finally:
        for stream in stream_configs:
            stream["capture"].stop()
            stream["capture"].release()
            if stream["streamer"]:
                stream["streamer"].stop_streaming()
        if uploader:
            uploader.shutdown()


if __name__ == "__main__":
    main()
```

---

## 1. Logger

```python
from transformsai_ai_core import get_logger

# Outputs: .core-logs/*.jsonl (all levels), console (INFO+ by default)
# View logs: cat run_*.jsonl | logdy  (https://logdy.dev)

logger = get_logger(
    name=__name__,   # default: caller module name — pass self for auto class naming
    cli_debug=False, # default: False — True shows DEBUG+ on console for this logger
)

logger.trace("file only")
logger.debug("file only by default")
logger.info("console + file")
logger.warning("console + file")
logger.error("console + file")
logger.critical("console + file")

try:
    risky()
except Exception:
    logger.exception("full traceback to file + console")
```

---

## 2. Config

```python
from transformsai_ai_core import process_config, load_config

# process_config: loads, validates, resolves RTSP URLs, downloads/exports models
config = process_config(
    config_path="config.yaml",  # str|Path — REQUIRED
    resolve_models=True,        # default: True — download + export models if needed
)

# load_config: loads + validates only, no model resolution
config = load_config(
    config_path="config.yaml",  # str|Path — REQUIRED
    validate=True,              # default: True — validate against Pydantic schema
)

# Access pattern
meta           = config["meta"]
cameras        = config["cameras"]           # list — iterate for per-camera setup
models_cfg     = config["advanced"]["models"]
timings        = config["advanced"]["timings"]
datasend_cfg   = config["advanced"]["datasend"]
livestream_cfg = config["advanced"]["livestream"]
pipeline_cfg   = config["advanced"]["pipeline"]
```

---

## 3. Video Capture

```python
from transformsai_ai_core import VideoCaptureAsync, process_config

config  = process_config("config.yaml")
cam_cfg = config["cameras"][0]
res     = cam_cfg["settings"]["resolution"]

cap = VideoCaptureAsync(
    src=cam_cfg["rtsp_url"],                                    # REQUIRED — RTSP URL (built by process_config), file path, or device index
    width=res.get("width", 1920),                               # default: None — resize width
    height=res.get("height", 1080),                             # default: None — resize height
    buffer_size=cam_cfg["settings"].get("buffer_size", 1),      # default: 1 — OpenCV buffer (1 = minimal lag)
    hw_decode=cam_cfg["settings"].get("hw_decode", False),      # default: False — hardware decode
    fps=cam_cfg["settings"].get("fps", 30),                     # default: None
    auto_restart_on_fail=False,                                 # default: False — restart thread on failure
    restart_delay=30.0,                                         # default: 30.0s
    opencv_backend="auto",                                      # default: "auto" — "auto" | "ffmpeg" | "gstreamer"
    max_frame_age_ms=100,                                       # default: 100ms — drop frames older than this
    auto_resize=True,                                           # default: True — resize in read() if w/h set
).start(
    loop=False  # default: None — True loops video files
)

grabbed, frame = cap.read(
    wait_for_frame=False,  # default: False — block until first frame arrives
    timeout=1.0,           # default: 1.0s — timeout when wait_for_frame=True
    copy=True,             # default: True — False = direct ref (faster, do not modify)
)

cap.stop()
cap.release()
```

---

## 4. MediaMTX Streaming

```python
from transformsai_ai_core import MediaMTXStreamer, process_config

config  = process_config("config.yaml")
ls_cfg  = config["advanced"]["livestream"]
cam_cfg = config["cameras"][0]
res     = cam_cfg["settings"]["resolution"]

streamer = MediaMTXStreamer(
    mediamtx_ip=ls_cfg["mediamtx_ip"],                          # REQUIRED
    rtsp_port=ls_cfg["rtsp_port"],                              # REQUIRED
    camera_sn_id="cam_01",                                      # REQUIRED — unique camera identifier
    fps=ls_cfg.get("fps", 30),                                  # default: 30
    frame_width=res.get("width", 1920),                         # REQUIRED
    frame_height=res.get("height", 1080),                       # REQUIRED
    bitrate="1500k",                                            # default: "1500k"
    encoder_preset="ultrafast",                                 # default: "ultrafast"
    encoder_codec="copy",                                       # default: "copy" — "copy" | "libx264"
    stream_queue_size=ls_cfg["settings"].get("queue_size", 2),  # default: 2
    hw_encode=False,                                            # default: False — auto-detect HW encoder
    debug_log_interval=60.0,                                    # default: 60.0s
)

streamer.start_streaming()
streamer.update_frame(frame)   # call in loop — non-blocking
streamer.get_rtsp_url()        # -> str
streamer.get_webrtc_url()      # -> str
streamer.get_hls_url()         # -> str
streamer.stop_streaming()
```

---

## 5. Data Upload

```python
from transformsai_ai_core import DataUploader, mat_to_response, time_to_string, process_config
import time

config = process_config("config.yaml")
meta   = config["meta"]
ds_cfg = config["advanced"]["datasend"]

uploader = DataUploader(
    base_url=ds_cfg["base_url"],                                 # default: None
    heartbeat_url=None,                                          # default: None — separate heartbeat endpoint
    headers={"x-token": meta["token"]} if meta.get("token") else None,  # default: None
    secret_keys=ds_cfg.get("secret_keys"),                       # default: None — str | list[str]
    secret_key_header="X-Secret-Key",                            # default: "X-Secret-Key"
    max_workers=None,                                            # default: min(2, cpu_count)
    max_retries=5,                                               # default: 5
    retry_delay=1,                                               # default: 1s
    timeout=300,                                                 # default: 300s
    disable_caching=False,                                       # default: False
    cache_file_path="uploader_cache.json",                       # default: "uploader_cache.json"
    cache_files_dir="uploader_cached_files",                     # default: "uploader_cached_files"
    max_cache_retries=5,                                         # default: 5
    cache_retry_interval=100,                                    # default: 100s
    max_cache_items=300,                                         # default: 300
    max_cache_age_seconds=86400,                                 # default: 86400 (24h)
    source=meta["name"],                                         # default: "Frame Processor"
    project_version=meta.get("version"),                         # default: None
)

# Async POST JSON
uploader.send_data(
    data={"event": "detection", "count": 5},
    endpoint_path=ds_cfg["endpoints"].get("data", "/data"),
    heartbeat=False,     # default: False — heartbeats are not cached on failure
    files=None,          # default: None — {"field": ("name.jpg", bytes, "image/jpeg")} or list of tuples
    base_url=None,       # default: None — overrides instance base_url
    method="POST",       # default: "POST" — GET | POST | PUT | PATCH | DELETE
    content_type="auto", # default: "auto" — "json" | "form-data" | "auto"
    url_params=None,     # default: None — dict of query parameters
)

# Async POST with image
image_tuple = mat_to_response(
    frame,
    max_width=1920,       # default: 1920
    jpeg_quality=65,      # default: 65
    filename="image.jpg", # default: "image.jpg"
    timestamp=None,       # default: None — float unix timestamp for overlay
    add_timestamp=False,  # default: False
)  # -> ("image.jpg", bytes, "image/jpeg") | None
uploader.send_data(data={"alert": "unsafe"}, files={"image": image_tuple}, endpoint_path="/alerts")

# Multiple files
uploader.send_data(
    files={"images": [("f1.jpg", bytes1, "image/jpeg"), ("f2.jpg", bytes2, "image/jpeg")]},
    endpoint_path="/batch",
)

# Sync — blocks until response, returns response dict
response = uploader.send_data_sync(data={"query": "status"}, method="GET", endpoint_path="/status")

# Heartbeat
uploader.send_heartbeat(
    sn="cam_01",                           # REQUIRED
    timestamp=time_to_string(time.time()), # REQUIRED — use time_to_string for correct server format
    status_log="Running normally",         # default: ""
    live_url="",                           # default: ""
)

uploader.shutdown(wait=True)  # default: True — wait for pending uploads
```

---

## 6. YOLO / YOLOE

```python
# pip install transformsai-ai-core[ultralytics]
from transformsai_ai_core import YOLOWrapper, YOLOEWrapper, process_config

config    = process_config("config.yaml", resolve_models=True)
model_cfg = config["advanced"]["models"]["model_name"]  # pass the whole model dict directly

# --- YOLOWrapper ---
model = YOLOWrapper(model_cfg)
# model_cfg shape:
# {
#   "path": "/path/to/model.pt",    # REQUIRED
#   "export": False,                # default: False
#   "batch": 1,                     # default: 1
#   "load_options": {
#       "lib_type": "YOLO",         # default: "YOLO" — "YOLO" | "YOLOE"
#       "task": "detect",           # default: "detect" — "detect" | "classify" | "segment"
#   },
#   "export_options": {             # freeform, passed to ultralytics export
#       "format": "onnx",
#       "half": False,
#       "dynamic": True,
#       "imgsz": 640,
#   }
# }

results = model.predict(
    source=frame,   # str | Path | ndarray | list — single image, array, or list of either
    conf=0.5,       # any ultralytics predict kwarg
)

for r in results:
    boxes = r.boxes.xyxy   # bounding boxes
    conf  = r.boxes.conf   # confidence scores
    cls   = r.boxes.cls    # class IDs

model.original_model_path   # Path
model.exported_model_path   # Path | None


# --- YOLOEWrapper (text/visual prompt support) ---
model_cfg["load_options"]["lib_type"] = "YOLOE"
model = YOLOEWrapper(model_cfg)
# Additional model_cfg key:
# "set_classes": None   # default: None — list[str] of class names to set on load

names   = ["person", "knife"]
text_pe = model.get_text_pe(names)       # -> ndarray
model.set_classes(names, text_pe)
results = model.predict(frame, conf=0.5)
```
