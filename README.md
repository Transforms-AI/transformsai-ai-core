# TransformsAI AI Core

Internal library for computer vision projects. Provides thread-safe utilities for video capture, streaming, data upload, logging, configuration management, and AI inference.

## Installation

```bash
# Basic installation
pip install git+https://github.com/yourusername/transformsai-ai-core.git

# With YOLO/YOLOE support (ultralytics + torch)
pip install "git+https://github.com/yourusername/transformsai-ai-core.git#egg=transformsai-ai-core[all]"
```

## Quick Start

```python
from transformsai_ai_core import (
    get_logger, VideoCaptureAsync, MediaMTXStreamer, 
    DataUploader, load_config, process_config
)

# Load and process config
config = load_config("config.yaml", validate=True)
runtime_config = process_config("config.yaml", resolve_models=True)

# Setup components
logger = get_logger(__name__)
cap = VideoCaptureAsync("rtsp://camera:554/stream", auto_restart_on_fail=True).start()
streamer = MediaMTXStreamer("127.0.0.1", 8554, "cam_01", 1280, 720).start_streaming()
uploader = DataUploader(base_url="https://api.example.com")

# Main loop
while True:
    grabbed, frame = cap.read()
    if grabbed:
        streamer.update_frame(frame)

cap.release()
streamer.stop_streaming()
uploader.shutdown()
```

---

## 1. Central Logger

**Module:** `transformsai_ai_core.central_logger`

### `get_logger(name: str | object = None, module_name: str = None) -> LoggerWrapper`

Get pre-configured logger instance with colored console output and rotating file logs.

**Parameters:**
- `name`: str|object - Logger name (string or class instance for auto-naming)
- `module_name`: str - Deprecated, kept for compatibility

**Returns:** LoggerWrapper with standard log methods

**Usage:**
```python
from transformsai_ai_core import get_logger

# Basic usage
logger = get_logger(__name__)
logger.info("Message")
logger.warning("Warning")
logger.debug("Debug info")

# Error logging with traceback control
logger.error("Error occurred")  # Includes traceback in file, not console
logger.error("Error occurred", traceback=False)  # No traceback anywhere
logger.exception("Critical error")  # Always includes full traceback

# Named logger for classes
class MyWorker:
    def __init__(self):
        self.logger = get_logger(self)  # Auto-names as "MyWorker"
```

**Log Output:**
- Console: INFO and above, colored, conditional traceback
- `logs/debug_YYYY-MM-DD.log`: DEBUG and above, full traceback
- `logs/error_YYYY-MM-DD.log`: ERROR and above, full traceback

---

## 2. Configuration Management

**Module:** `transformsai_ai_core.config_loader`

### Expected Config Structure

Complete `config.yaml` schema:

```yaml
# Meta Configuration (MetaConfig)
meta:
  name: "project-name"              # str, REQUIRED: Project identifier
  version: "1.0.0"                  # str, default "1.0.0": Config schema version
  token: ""                         # str, default "":  (header x-token:)

# Cameras (list[CameraConfig])
cameras:
  - local: false                    # bool, default false: Use local file vs RTSP
    local_source: ""                # str, default "": Path to local video file
    rtsp_source:                    # RtspSource object
      username: "admin"             # str, default "admin": RTSP username
      password: ""                  # str, default "": RTSP password
      ip: ""                        # str, default "": Camera IP address
      port: 554                     # int, default 554: RTSP port
      path: "/Streaming/Channels/101"  # str, default: RTSP stream path
    capture:                        # CaptureSettings object
      buffer_size: 1                # int, default 1: OpenCV buffer size
      opencv_backend: null          # str|null, default null: 'auto'|'ffmpeg'|'gstreamer'
      max_frame_age_ms: null        # int|null, default null: Drop frames older than (ms)
    settings: {}                    # dict[str, Any]: Freeform per-camera settings

# Advanced Configuration (AdvancedConfig)
advanced:
  # Models (dict[str, ModelConfig])
  models:
    model_name:                     # Key is model identifier (e.g., "yolo11n", "face-det")
      download_key: ""              # str, default "": Direct download key from server
      type: "person-det"            # str, REQUIRED: Model type/category
      batch: 1                      # int, default 1: Batch size for inference
      path: ""                      # str, default "": Local path (auto-populated after download)
      export: false                 # bool, default false: Enable automatic model export
      load_options:                 # LoadOptions object
        lib_type: "YOLO"            # str, default "YOLO": Model library (YOLO|YOLOE)
        task: "detect"              # str, default "detect": Task type (detect|classify|segment)
      export_options: {}            # dict[str, Any]: Freeform export parameters

  # Timings (dict[str, Any]): Freeform
  timings:
    inference_interval: 0.033       # Example: Custom timing values
    datasend_interval: 120
    
  # Data Sending (DatasendConfig)
  datasend:
    enabled: true                   # bool, default true: Master switch
    base_url: ""                    # str, default "": API base URL
    endpoints: {}                   # dict[str, str]: Freeform endpoint paths
    secret_keys: []                 # list[str], default []: API authentication keys
    settings: {}                    # dict[str, Any]: Freeform settings

  # Livestream (LivestreamConfig)
  livestream:
    enabled: true                   # bool, default true: Master switch
    mediamtx_ip: "localhost"        # str, default "localhost": MediaMTX server IP
    rtsp_port: 8554                 # int, default 8554: MediaMTX RTSP port
    encoder:                        # StreamEncoderSettings object
      preset: "ultrafast"           # str, default "ultrafast": Encoder preset
      codec: "copy"                 # str, default "copy": 'copy'|'libx264'
      queue_size: 2                 # int, default 2: Async frame queue depth
    settings: {}                    # dict[str, Any]: Freeform settings

  # Pipeline (dict[str, Any]): Freeform
  pipeline:
    tracking_threshold: 0.5         # All project-specific settings
```

### `load_config(config_path: str | Path, validate: bool = True) -> dict`

Load and optionally validate config from YAML file.

**Parameters:**
- `config_path`: str|Path - Path to config.yaml
- `validate`: bool, default True - Validate against Pydantic schema

**Returns:** dict - Config as dictionary

**Usage:**
```python
from transformsai_ai_core import load_config

config = load_config("config.yaml", validate=True)
project_name = config["meta"]["name"]
cameras = config["cameras"]
```

### `save_config(config_path: str | Path, config: dict) -> None`

Save config dictionary to YAML file.

**Parameters:**
- `config_path`: str|Path - Output path
- `config`: dict - Config dictionary

**Usage:**
```python
from transformsai_ai_core import save_config

config["meta"]["token"] = "hw-12345"
save_config("config.yaml", config)
```

### `process_config(config_path: str | Path, base_dir: str | Path | None = None, resolve_models: bool = False, download_models: bool = False) -> dict`

Load config and process for runtime use (build RTSP URLs, resolve model paths).

**Parameters:**
- `config_path`: str|Path - Path to config.yaml
- `base_dir`: str|Path|null, default None - Project root (defaults to config parent)
- `resolve_models`: bool, default False - Resolve model paths
- `download_models`: bool, default False - Download missing models

**Returns:** dict - Processed config with RTSP URLs built

**Usage:**
```python
from transformsai_ai_core import process_config

# Build RTSP URLs
runtime_config = process_config("config.yaml")
rtsp_url = runtime_config["cameras"][0]["rtsp_url"]

# Resolve model paths
runtime_config = process_config("config.yaml", resolve_models=True)
model_path = runtime_config["advanced"]["models"]["yolo11n"]["path"]

# Download missing models
runtime_config = process_config("config.yaml", resolve_models=True, download_models=True)
```

### `build_rtsp_url(rtsp_source: dict) -> str`

Build RTSP URL from decomposed components.

**Parameters:**
- `rtsp_source`: dict with keys: username, password, ip, port, path

**Returns:** str - Full RTSP URL

**Usage:**
```python
from transformsai_ai_core import build_rtsp_url

url = build_rtsp_url({
    "username": "admin",
    "password": "pass@123",
    "ip": "192.168.1.100",
    "port": 554,
    "path": "/stream"
})
# Result: rtsp://admin:pass%40123@192.168.1.100:554/stream
```

---

## 3. Video Capture

**Module:** `transformsai_ai_core.video_capture`

### `VideoCaptureAsync`

Asynchronous video capture with auto-restart, hardware decode, and edge device optimization.

**Initialization:**
```python
from transformsai_ai_core import VideoCaptureAsync

cap = VideoCaptureAsync(
    src=0,                        # any: Camera index, RTSP URL, or file path
    width=None,                   # int|null: Target width for resize
    height=None,                  # int|null: Target height for resize
    driver=None,                  # any: Deprecated
    heartbeat_config=None,        # dict|null: Heartbeat settings
    auto_restart_on_fail=False,   # bool, default False: Auto-restart on failure
    restart_delay=30.0,           # float, default 30.0s: Delay between restarts
    buffer_size=1,                # int, default 1: OpenCV buffer (1=minimal lag)
    opencv_backend="auto",        # str, default "auto": 'auto'|'ffmpeg'|'gstreamer'
    max_frame_age_ms=100,         # int, default 100ms: Drop old frames
    auto_resize=True,             # bool, default True: Resize in read() if needed
    hw_decode=False               # bool, default False: Hardware decode
)
```

**Methods:**

### `start(loop: bool = None) -> self`

Start background capture thread.

**Parameters:**
- `loop`: bool|null - For file sources only, loop video

**Returns:** self for chaining

**Usage:**
```python
cap = VideoCaptureAsync("rtsp://camera:554/stream").start()
cap = VideoCaptureAsync("video.mp4").start(loop=True)
```

### `read(wait_for_frame: bool = False, timeout: float = 1.0, copy: bool = True) -> tuple[bool, ndarray|null]`

Read latest captured frame.

**Parameters:**
- `wait_for_frame`: bool, default False - Wait for first frame
- `timeout`: float, default 1.0s - Timeout when waiting
- `copy`: bool, default True - Return copy (False = direct reference, faster)

**Returns:** tuple (grabbed: bool, frame: ndarray|null)

**Usage:**
```python
# Non-blocking read
grabbed, frame = cap.read()

# Wait for first frame
grabbed, frame = cap.read(wait_for_frame=True, timeout=10.0)

# Direct reference (faster, don't modify)
grabbed, frame = cap.read(copy=False)
```

### `get(propId: int) -> float|null`

Get OpenCV capture property.

**Parameters:**
- `propId`: int - cv2.CAP_PROP_* constant

**Returns:** float|null

**Usage:**
```python
fps = cap.get(cv2.CAP_PROP_FPS)
```

### `set(propId: int, value: float) -> bool`

Set OpenCV capture property.

**Parameters:**
- `propId`: int - cv2.CAP_PROP_* constant
- `value`: float - Property value

**Returns:** bool - Success

**Usage:**
```python
cap.set(cv2.CAP_PROP_BRIGHTNESS, 50)
```

### `stop() -> None`

Stop background capture thread.

**Usage:**
```python
cap.stop()
```

### `release() -> None`

Release capture device and cleanup resources.

**Usage:**
```python
cap.release()
```

**Context Manager:**
```python
with VideoCaptureAsync("rtsp://camera:554/stream") as cap:
    grabbed, frame = cap.read()
# Automatically calls release()
```

---

## 4. MediaMTX Streaming

**Module:** `transformsai_ai_core.mediamtx_streamer`

### `MediaMTXStreamer`

Push video frames to MediaMTX RTSP server with hardware encoding support.

**Initialization:**
```python
from transformsai_ai_core import MediaMTXStreamer

streamer = MediaMTXStreamer(
    mediamtx_ip="localhost",      # str: MediaMTX server IP/hostname
    rtsp_port=8554,               # int: MediaMTX RTSP port
    camera_sn_id="cam_01",        # str: Camera identifier
    fps=30,                       # int, default 30: Frame rate
    frame_width=1920,             # int: Frame width
    frame_height=1080,            # int: Frame height
    bitrate="1500k",              # str, default "1500k": Video bitrate
    debug_log_interval=60.0,      # float, default 60.0s: Debug log interval
    encoder_preset="ultrafast",   # str, default "ultrafast": Encoder preset
    encoder_codec="copy",         # str, default "copy": 'copy'|'libx264'
    stream_queue_size=2,          # int, default 2: Async queue depth
    hw_encode=False               # bool, default False: Auto-detect hardware encoder
)
```

**Methods:**

### `start_streaming() -> bool`

Start FFmpeg streaming process.

**Returns:** bool - Success

**Usage:**
```python
if streamer.start_streaming():
    print(f"Streaming to: {streamer.rtsp_url}")
```

### `stop_streaming() -> None`

Stop streaming process and cleanup.

**Usage:**
```python
streamer.stop_streaming()
```

### `update_frame(frame: ndarray) -> None`

Update stream with new frame (non-blocking, uses memoryview).

**Parameters:**
- `frame`: ndarray - OpenCV BGR frame

**Usage:**
```python
grabbed, frame = cap.read()
if grabbed:
    streamer.update_frame(frame)
```

### `get_rtsp_url() -> str`

**Returns:** str - RTSP URL

### `get_webrtc_url() -> str`

**Returns:** str - WebRTC URL

### `get_hls_url() -> str`

**Returns:** str - HLS URL

**Usage:**
```python
print(f"RTSP: {streamer.get_rtsp_url()}")
print(f"WebRTC: {streamer.get_webrtc_url()}")
print(f"HLS: {streamer.get_hls_url()}")
```

---

## 5. Data Upload

**Module:** `transformsai_ai_core.datasend`

### `DataUploader`

Asynchronous HTTP client with caching, retries, and auto-tuned thread pool.

**Initialization:**
```python
from transformsai_ai_core import DataUploader

uploader = DataUploader(
    base_url=None,                        # str|null: API base URL
    heartbeat_url=None,                   # str|null: Heartbeat endpoint URL
    headers=None,                         # dict|null: Default headers
    secret_keys=None,                     # str|list[str]|null: API auth keys
    secret_key_header="X-Secret-Key",     # str, default: Header name for auth
    max_workers=None,                     # int|null, default min(2, cpu_count)
    max_retries=5,                        # int, default 5: Max retry attempts
    retry_delay=1,                        # int, default 1s: Base retry delay
    timeout=300,                          # int, default 300s: Total timeout
    disable_caching=False,                # bool, default False: Disable cache
    cache_file_path="uploader_cache.json",# str: Cache file path
    cache_files_dir="uploader_cached_files",# str: Cache files directory
    max_cache_retries=5,                  # int, default 5: Max cache retry count
    cache_retry_interval=100,             # int, default 100s: Retry interval
    max_cache_items=300,                  # int, default 300: Max cached items
    max_cache_age_seconds=86400,          # int, default 24h: Max item age
    source="Frame Processor",             # str: Source identifier
    project_version=None                  # str|null: Project version
)
```

**Methods:**

### `send_data(data: dict|null = None, heartbeat: bool = False, files: dict|null = None, base_url: str|null = None, method: str = "POST", content_type: str = "auto", endpoint_path: str = "", url_params: dict|null = None) -> None`

Send data asynchronously (non-blocking).

**Parameters:**
- `data`: dict|null - Data payload
- `heartbeat`: bool, default False - Mark as heartbeat (not cached on failure)
- `files`: dict|null - Files to upload: `{"field": ("name.jpg", bytes, "image/jpeg")}`
- `base_url`: str|null - Override instance base_url
- `method`: str, default "POST" - HTTP method: GET|POST|PUT|PATCH|DELETE
- `content_type`: str, default "auto" - "json"| "form-data"|"auto"
- `endpoint_path`: str - API endpoint path
- `url_params`: dict|null - Query parameters

**Usage:**
```python
# JSON POST
uploader.send_data(
    data={"event": "detection", "count": 5},
    endpoint_path="/events"
)

# GET with query params
uploader.send_data(
    method="GET",
    endpoint_path="/status",
    url_params={"camera": "cam_01"}
)

# File upload
from transformsai_ai_core import mat_to_response
image_tuple = mat_to_response(frame, max_width=1920, jpeg_quality=65)
uploader.send_data(
    data={"alert": "unsafe"},
    files={"image": image_tuple},
    endpoint_path="/alerts"
)

# Multiple files
uploader.send_data(
    files={
        "images": [
            ("frame1.jpg", bytes1, "image/jpeg"),
            ("frame2.jpg", bytes2, "image/jpeg")
        ]
    },
    endpoint_path="/batch"
)
```

### `send_heartbeat(sn: str, timestamp: str, status_log: str = "", live_url: str = "") -> None`

Send heartbeat (asynchronous, not cached on failure).

**Parameters:**
- `sn`: str - Serial number / device ID
- `timestamp`: str - ISO timestamp
- `status_log`: str, default "" - Status message
- `live_url`: str, default "" - Live stream URL

**Usage:**
```python
import time
uploader.send_heartbeat(
    sn="camera_01",
    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    status_log="Running normally",
    live_url=streamer.get_hls_url()
)
```

### `shutdown(wait: bool = True) -> None`

Shutdown uploader and cleanup resources.

**Parameters:**
- `wait`: bool, default True - Wait for pending uploads

**Usage:**
```python
uploader.shutdown()
```

---

## 6. YOLO/YOLOE Wrappers

**Module:** `transformsai_ai_core.yolo_wrapper`

**Requires:** `pip install transformsai-ai-core[ultralytics]`

### `YOLOWrapper`

Ultralytics YOLO wrapper with auto-export and sequential batching.

**Initialization:**
```python
from transformsai_ai_core import YOLOWrapper

model = YOLOWrapper(
    model_dict={
        "path": "/path/to/model.pt",      # str, required: Model path
        "export": False,                   # bool, default False: Enable auto-export
        "batch": 1,                        # int, default 1: Batch size
        "load_options": {
            "lib_type": "YOLO",            # str, default "YOLO"
            "task": "detect"               # str, default "detect"
        },
        "export_options": {                # dict: Export parameters
            "format": "onnx",              # Export format
            "half": False,                 # FP16
            "dynamic": True,               # Dynamic batch size
            "imgsz": 640                   # Image size
        }
    }
)
```

**Methods:**

### `predict(source: str | Path | list | array, **kwargs) -> list[Results]`

Run prediction with sequential batching.

**Parameters:**
- `source`: str|Path|list|array - Image path, array, or list of paths
- `**kwargs`: Additional ultralytics predict parameters

**Returns:** list[Results] - List of detection results

**Usage:**
```python
# Single image
results = model.predict("image.jpg", conf=0.5)

# Multiple images (auto-batched)
results = model.predict(["img1.jpg", "img2.jpg", "img3.jpg"])

# NumPy array
results = model.predict(frame_bgr, conf=0.5)

# Access results
for r in results:
    boxes = r.boxes.xyxy  # Bounding boxes
    conf = r.boxes.conf    # Confidence
    cls = r.boxes.cls      # Class IDs
```

**Properties:**
- `original_model_path`: Path - Original model path
- `exported_model_path`: Path|null - Exported model path (if exported)

### `YOLOEWrapper`

Ultralytics YOLOE wrapper with text/visual prompt support.

**Initialization:**
```python
from transformsai_ai_core import YOLOEWrapper

model = YOLOEWrapper(
    model_dict={
        "path": "/path/to/yoloe.pt",
        "export": False,
        "batch": 1,
        "load_options": {
            "lib_type": "YOLOE",
            "task": "detect"
        },
        "export_options": {},
        "set_classes": None                # list[str]|null: Class names for prompts
    }
)
```

**Methods:**

### `predict(source: str | Path | list | array, **kwargs) -> list[Results]`

Same as YOLOWrapper.predict().

### `set_classes(names: list[str], text_pe: ndarray) -> None`

Set class names for YOLOE text prompts.

**Parameters:**
- `names`: list[str] - Class names
- `text_pe`: ndarray - Text positional embeddings

**Usage:**
```python
names = ["person", "knife"]
text_pe = model.get_text_pe(names)
model.set_classes(names, text_pe)
results = model.predict("image.jpg")
```

### `get_text_pe(names: list[str]) -> ndarray`

Get text positional embeddings for class names.

**Parameters:**
- `names`: list[str] - Class names

**Returns:** ndarray - Text embeddings

---

## 7. Utility Functions

**Module:** `transformsai_ai_core.utils`

### `time_to_string(timestamp: float) -> str`

Convert Unix timestamp to ISO format string.

**Parameters:**
- `timestamp`: float - Unix timestamp

**Returns:** str - ISO format "YYYY-MM-DDTHH:MM:SSZ"

**Usage:**
```python
from transformsai_ai_core import time_to_string
ts = time_to_string(time.time())
# "2024-01-15T10:30:45Z"
```

### `mat_to_response(frame: ndarray, max_width: int = 1920, jpeg_quality: int = 65, filename: str = "image.jpg", timestamp: float|null = None, add_timestamp: bool = False) -> tuple|null`

Resize and encode OpenCV frame to JPEG bytes for HTTP upload.

**Parameters:**
- `frame`: ndarray - OpenCV BGR frame
- `max_width`: int, default 1920 - Maximum width
- `jpeg_quality`: int, default 65 - JPEG quality (0-100)
- `filename`: str, default "image.jpg" - Output filename
- `timestamp`: float|null, default None - Timestamp to overlay
- `add_timestamp`: bool, default False - Add timestamp overlay

**Returns:** tuple (filename, bytes, content_type) or None

**Usage:**
```python
from transformsai_ai_core import mat_to_response

# Basic encoding
image_tuple = mat_to_response(frame, max_width=1920, jpeg_quality=65)
uploader.send_data(files={"image": image_tuple}, endpoint_path="/upload")

# With timestamp
image_tuple = mat_to_response(
    frame, 
    add_timestamp=True, 
    timestamp=time.time()
)
```

---

## Recommended Example

Production-ready template for multi-camera AI pipeline with streaming, inference, and data upload:

```python
"""
Production AI Pipeline Template

This example demonstrates a complete multi-camera system with:
- Config-driven initialization
- Multi-threaded camera I/O with frame buffering
- Periodic AI inference (YOLO detection/classification)
- Real-time streaming to MediaMTX
- Smart data upload with debouncing
- Graceful shutdown and state persistence
"""

import time
import threading
import copy
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


def camera_io_worker(stream_config: dict):
    """
    Background thread for each camera.
    Handles frame capture, streaming, and frame buffering for AI processing.
    """
    cam_sn = stream_config['camera_sn']
    capture = stream_config['capture']
    streamer = stream_config['streamer']
    frame_buffer = stream_config['frame_buffer']
    lock = stream_config['lock']
    buffer_size = stream_config['buffer_size']
    collect_interval = stream_config['collect_interval']
    
    last_collect = time.time()
    
    while True:
        grabbed, frame = capture.read(copy=True)
        if not grabbed:
            time.sleep(0.1)
            continue
        
        # Update streamer with processed frame (add overlays if needed)
        if streamer:
            with lock:
                # TODO: Add your visualizations here (boxes, text, etc.)
                display_frame = frame.copy()
            streamer.update_frame(display_frame)
        
        # Buffer frames at intervals for AI processing
        current_time = time.time()
        if current_time - last_collect >= collect_interval:
            with lock:
                if len(frame_buffer) >= buffer_size:
                    frame_buffer.popleft()
                frame_buffer.append(frame.copy())
            last_collect = current_time
        
        time.sleep(0.01)  # Prevent CPU hogging


def main():
    # =========================================================================
    # 1. LOAD CONFIGURATION
    # =========================================================================
    logger.info("Loading configuration...")
    config = process_config(
        "config.yaml",
        resolve_models=True,      # Resolve model paths
    )
    
    # Extract config sections
    meta = config["meta"]
    cameras = config["cameras"]
    models_cfg = config["advanced"]["models"]
    timings = config["advanced"]["timings"]
    datasend_cfg = config["advanced"]["datasend"]
    livestream_cfg = config["advanced"]["livestream"]
    pipeline_cfg = config["advanced"]["pipeline"]  # Project-specific settings
    
    # =========================================================================
    # 2. INITIALIZE AI MODELS
    # =========================================================================
    logger.info("Loading AI models...")
    models = {}
    for model_name, model_dict in models_cfg.items():
        if model_dict.get("path"):
            models[model_name] = YOLOWrapper(model_dict)
            logger.info(f"✓ Loaded model: {model_name}")
    
    # =========================================================================
    # 3. INITIALIZE DATA UPLOADER
    # =========================================================================
    uploader = None
    if datasend_cfg.get("enabled"):
        headers = {"x-token": meta.get("token", "")} if meta.get("token") else None
        uploader = DataUploader(
            base_url=datasend_cfg["base_url"],
            headers=headers,
            secret_keys=datasend_cfg.get("secret_keys"),
            source=meta["name"],
            project_version=meta.get("version"),
        )
        logger.info(f"✓ Data uploader initialized: {datasend_cfg['base_url']}")
    
    # =========================================================================
    # 4. INITIALIZE CAMERA STREAMS
    # =========================================================================
    logger.info(f"Initializing {len(cameras)} camera stream(s)...")
    stream_configs = []
    io_threads = []
    
    for idx, cam_cfg in enumerate(cameras):
        cam_sn = f"cam_{idx}"
        logger.info(f"Setting up {cam_sn}...")
        
        # Video capture
        capture = VideoCaptureAsync(
            src=cam_cfg["rtsp_url"],  # Built by process_config()
            width=cam_cfg["settings"].get("width", 1920),
            height=cam_cfg["settings"].get("height", 1080),
            buffer_size=cam_cfg["capture"]["buffer_size"],
            opencv_backend=cam_cfg["capture"]["opencv_backend"],
            hw_decode=True,
            auto_restart_on_fail=True,
        ).start()
        
        # MediaMTX streamer
        streamer = None
        if livestream_cfg.get("enabled"):
            streamer = MediaMTXStreamer(
                mediamtx_ip=livestream_cfg["mediamtx_ip"],
                rtsp_port=livestream_cfg["rtsp_port"],
                camera_sn_id=cam_sn,
                fps=livestream_cfg.get("fps", 30),
                frame_width=cam_cfg["settings"].get("width", 1920),
                frame_height=cam_cfg["settings"].get("height", 1080),
                encoder_preset=livestream_cfg["encoder"]["preset"],
                stream_queue_size=livestream_cfg["encoder"]["queue_size"],
            ).start_streaming()
        
        # Frame buffer for AI processing
        buffer_size = timings.get("frame_buffer_size", 1)
        collect_interval = timings.get("frame_collect_interval", 3.0)
        frame_buffer = deque(maxlen=buffer_size)
        lock = threading.Lock()
        
        # Stream config (shared state between threads)
        stream_cfg = {
            "camera_sn": cam_sn,
            "capture": capture,
            "streamer": streamer,
            "frame_buffer": frame_buffer,
            "lock": lock,
            "buffer_size": buffer_size,
            "collect_interval": collect_interval,
            # Add your project-specific state here
            "tracked_objects": [],  # Example: persistent tracking state
            "last_inference_time": 0,
        }
        stream_configs.append(stream_cfg)
        
        # Start I/O worker thread
        t = threading.Thread(target=camera_io_worker, args=(stream_cfg,), daemon=True)
        t.start()
        io_threads.append(t)
        
        logger.info(f"✓ {cam_sn} initialized")
    
    # =========================================================================
    # 5. MAIN PROCESSING LOOP
    # =========================================================================
    logger.info("Entering main processing loop...")
    
    last_heartbeat = time.time()
    last_datasend = time.time()
    
    try:
        while True:
            current_time = time.time()
            
            # Process each camera
            for stream in stream_configs:
                # --- AI INFERERENCE (periodic) ---
                inference_interval = timings.get("inference_interval", 5.0)
                if current_time - stream["last_inference_time"] >= inference_interval:
                    with stream["lock"]:
                        frames = list(stream["frame_buffer"])
                    
                    if frames:
                        # Batch inference on buffered frames
                        for model_name, model in models.items():
                            results = model.predict(
                                frames,
                                conf=pipeline_cfg.get("confidence_threshold", 0.5),
                                # Add model-specific parameters
                            )
                            # TODO: Process results and update stream state
                            # Example: stream["tracked_objects"] = process_results(results)
                    
                    stream["last_inference_time"] = current_time
                
                # --- PROJECT-SPECIFIC LOGIC ---
                # Add your custom processing here (access stream via stream["lock"])
            
            # --- HEARTBEAT (periodic) ---
            if uploader:
                heartbeat_interval = timings.get("heartbeat_interval", 60)
                if current_time - last_heartbeat >= heartbeat_interval:
                    for stream in stream_configs:
                        uploader.send_heartbeat(
                            sn=stream["camera_sn"],
                            timestamp=time_to_string(current_time),
                            status_log="Running normally",
                            live_url=stream["streamer"].get_hls_url() if stream["streamer"] else "",
                        )
                    last_heartbeat = current_time
                
                # --- DATA UPLOAD (periodic full sync) ---
                datasend_interval = timings.get("datasend_interval", 180)
                if current_time - last_datasend >= datasend_interval:
                    for stream in stream_configs:
                        with stream["lock"]:
                            # TODO: Build payload from stream state
                            payload = {
                                "sn": stream["camera_sn"],
                                "timestamp": time_to_string(current_time),
                                # Add your data here
                            }
                        
                        # Optional: upload frame with annotations
                        if stream["frame_buffer"]:
                            frame = stream["frame_buffer"][-1].copy()
                            # TODO: Draw overlays on frame
                            image_tuple = mat_to_response(frame, max_width=1920, jpeg_quality=65)
                            uploader.send_data(
                                data=payload,
                                files={"image": image_tuple},
                                endpoint_path=datasend_cfg["endpoints"].get("data", "/data"),
                            )
                    
                    last_datasend = current_time
            
            time.sleep(0.1)  # Main loop interval
    
    except KeyboardInterrupt:
        logger.info("\nShutdown requested...")
    
    finally:
        # =========================================================================
        # 6. CLEANUP
        # =========================================================================
        logger.info("Cleaning up resources...")
        
        for stream in stream_configs:
            stream["capture"].stop()
            stream["capture"].release()
            if stream["streamer"]:
                stream["streamer"].stop_streaming()
        
        if uploader:
            uploader.shutdown()
        
        logger.info("✓ Shutdown complete")


if __name__ == "__main__":
    main()
```
---

## Exported Symbols

All of the following are available via `from transformsai_ai_core import *`:

**Core:**
- `get_logger`, `DataUploader`, `MediaMTXStreamer`, `VideoCaptureAsync`
- `time_to_string`, `mat_to_response`

**Config:**
- `load_config`, `save_config`, `process_config`, `build_rtsp_url`
- `AppConfig`, `MetaConfig`, `CameraConfig`, `RtspSource`
- `ModelConfig`, `DatasendConfig`, `LivestreamConfig`, `AdvancedConfig`

**YOLO (optional):**
- `YOLOWrapper`, `YOLOEWrapper`, `YOLO_AVAILABLE`