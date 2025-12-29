# TransformsAI Core Library

A collection of robust, thread-safe utilities for building AI vision applications. This library provides essential components for video capture, streaming, data transmission, and logging.

## Installation

Install the library directly from the Git repository:

```bash
# Basic installation
uv add git+https://github.com/yourusername/transformsai-ai-core.git

# Install with YOLO/YOLOE support (includes ultralytics and torch)
uv add "git+https://github.com/yourusername/transformsai-ai-core.git#egg=transformsai-ai-core[all]"
```

## Core Components

*   **`VideoCaptureAsync`**: Asynchronous video capture with automatic restart on failure, ideal for handling unreliable IP cameras and RTSP streams.
*   **`MediaMTXStreamer`**: Pushes video frames to a MediaMTX RTSP server using FFmpeg, with detailed performance monitoring.
*   **`DataUploader`**: Asynchronous, thread-safe HTTP client with a persistent cache, automatic retries, and heartbeat functionality.
*   **`central_logger`**: A centralized, singleton logger that provides colored console output and rotating file logs out of the box.
*   **`config_loader`**: Centralized configuration management with structured schemas and dynamic freeform fields for project-specific settings.
*   **`yolo_wrapper`**: Ultralytics YOLO and YOLOE wrapper to have auto batching and export options.

---

## Basic Usage Examples

### 1. Central Logger

Set up consistent logging across your entire application with a single function call.

```python
from transformsai_ai_core.central_logger import get_logger

# Get a logger for custom named module
logger = get_logger(module_name="CustomScript")

logger.info("This is an informational message.")
logger.warning("This is a warning.")
logger.error("This is an error with automatic traceback logging.")
logger.error("This is an error with automatic traceback logging turned off.", traceback = False)
logger.exception("This is an standard extension, will always have traceback.")


class MyWorker:
    def __init__(self):
        # It's easy to create named loggers for classes
        self.logger = get_logger(self)
    
    def do_work(self):
        self.logger.debug("Doing some work...")

```

### 2. Configuration Management

Manage project configurations with structured validation and dynamic freeform fields. Perfect for admin panels that need to control camera settings, models, and project-specific parameters.

```python
from transformsai_ai_core.config_loader import (
    load_config, save_config, process_config, build_rtsp_url
)
from transformsai_ai_core.config_schema import AppConfig

# Load and validate config
config = load_config("config.yaml", validate=True)

# Access structured fields
project_name = config["meta"]["name"]
camera_count = len(config["cameras"])
models = config["advanced"]["models"]  # Now a dict keyed by model name

# Edit structured fields (strict schema)
config["meta"]["token"] = "hw-token-12345"
config["cameras"][0]["local"] = False
config["cameras"][0]["rtsp_source"]["ip"] = "192.168.1.100"
config["advanced"]["datasend"]["enabled"] = True

# Edit freeform fields (arbitrary key-value pairs)
config["cameras"][0]["settings"]["resolution"] = {"width": 1920, "height": 1080}
config["advanced"]["timings"]["inference_interval"] = 0.033
config["advanced"]["pipeline"]["custom_threshold"] = 0.85

# Build RTSP URL from components (useful for admin panels)
rtsp_url = build_rtsp_url({
    "username": "admin",
    "password": "pass@123",
    "ip": "192.168.1.100",
    "port": 554,
    "path": "/stream"
})
# Result: rtsp://admin:pass%40123@192.168.1.100:554/stream

# Validate changes
validated = AppConfig(**config)

# Save modified config
save_config("config.yaml", config)

# Process config for runtime (builds RTSP URLs, resolves model paths)
runtime_config = process_config(
    "config.yaml",
    resolve_models=True,
    download_models=False
)

# Access built RTSP URL
if not runtime_config["cameras"][0]["local"]:
    rtsp_url = runtime_config["cameras"][0]["rtsp_url"]
    print(f"Camera stream: {rtsp_url}")
```

**Config Structure Example:**

```yaml
meta:
  name: "people-count"
  version: "1.0.0"
  token: ""

cameras:
  - local: false
    rtsp_source:
      username: "admin"
      password: "pass123"
      ip: "192.168.1.100"
      port: 554
      path: "/stream"
    settings:  # Freeform: add any key-value pairs
      resolution:
        width: 1920
        height: 1080
      custom_param: "value"

advanced:
  models:
    yolov11s:  # Model key/identifier
      download_key: ""  # Optional: direct download key
      type: "person-det"  # Model type/category
      batch: 1
      path: ""  # Auto-populated after download
      load_options:  # Formatted: loading parameters
        lib_type: "YOLO"
        task: "detect"
      export_options:  # Freeform: model-specific options
        backend: "engine"
        img_size: 640
        half: true
  
  timings:  # Freeform: project-specific timings
    inference_interval: 0.033
    datasend_interval: 120
  
  datasend:
    enabled: true
    base_url: "https://api.example.com"
    endpoints:  # Freeform: define your endpoints
      analytics: "cameras/analytics/"
      heartbeat: "cameras/heartbeat/"
  
  pipeline:  # Freeform: project-specific settings
    tracking_threshold: 0.5
    counting_lines:
      x1: 0.0
      y1: 0.5
      x2: 1.0
      y2: 0.5
```

### 3. Asynchronous Video Capture

Reliably capture frames from a camera, RTSP stream, or video file in a background thread. The `auto_restart_on_fail` feature makes it resilient to network drops.

```python
import cv2
import time
from transformsai_ai_core import VideoCaptureAsync, get_logger

logger = get_logger(__name__)
RTSP_URL = "rtsp://your_camera_stream_url"

# Initialize with auto-restart enabled
cap = VideoCaptureAsync(RTSP_URL, auto_restart_on_fail=True, restart_delay=5.0)

# Start the capture thread
cap.start()

logger.info("Capture started. Waiting for the first frame...")

while True:
    # The read() call is non-blocking
    grabbed, frame = cap.read(wait_for_frame=True, timeout=10.0)
    
    if not grabbed:
        logger.warning("Failed to grab frame, the capture thread will try to restart.")
        time.sleep(1.0)
        continue
    
    # Process the frame
    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
```

### 4. Streaming to MediaMTX

Push processed frames to a MediaMTX (or other RTSP) server.

```python
from transformsai_ai_core import MediaMTXStreamer, get_logger

logger = get_logger(__name__)

# Configure the streamer to connect to your MediaMTX server
streamer = MediaMTXStreamer(
    mediamtx_ip="127.0.0.1",
    rtsp_port=8554,
    camera_sn_id="kitchen_cam_01",
    frame_width=1280,
    frame_height=720,
    fps=20
)

# Start the FFmpeg process
if streamer.start_streaming():
    logger.info(f"Streaming to: {streamer.get_rtsp_url()}")
    
    # In your main loop, after processing a frame:
    # grabbed, frame = cap.read()
    # if grabbed:
    #     # annotated_frame = draw_boxes(frame, results)
    #     streamer.update_frame(annotated_frame)

# To stop
# streamer.stop_streaming()
```

### 5. Uploading Data with Caching

Send data (like analytics or alerts) to a server. If the network is down, `DataUploader` automatically caches the data to disk and retries later.

```python
import time
from transformsai_ai_core import DataUploader, get_logger

logger = get_logger(__name__)

uploader = DataUploader(
    base_url="https://api.yourapp.com",
    heartbeat_url="https://api.yourapp.com/heartbeat",
    max_cache_items=1000, # Store up to 1000 failed requests
    cache_retry_interval=60 # Retry cached items every 60 seconds
)

# --- Asynchronously send detection results ---
detection_data = {
    "timestamp": time.time(),
    "objects": ["person", "knife"],
    "confidence": 0.95
}
logger.info("Sending detection data...")
uploader.send_data(
    endpoint_path="/events",
    data=detection_data,
    method="POST"
)

# --- Send a periodic heartbeat ---
logger.info("Sending heartbeat...")
uploader.send_heartbeat(
    sn="kitchen_cam_01",
    timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    status_log="System is running normally."
)

# --- Upload data with an image file ---
# with open("alert_frame.jpg", "rb") as f:
#     image_bytes = f.read()

# uploader.send_data(
#     endpoint_path="/alerts",
#     data={"alert_type": "unsafe_condition"},
#     files={"image": ("alert.jpg", image_bytes, "image/jpeg")}
# )

# In a real app, you don't need to wait, but for this example:
time.sleep(2) 
uploader.shutdown()
```

### 6. YOLO/YOLOE Wrappers with Auto-Export and Batching

*Available when installed with `[all]` extras.*

The library provides wrapper classes for ultralytics YOLO and YOLOE models with automatic model export to optimized formats (ONNX, TensorRT, etc.) and custom sequential batching for controlled memory usage.

#### Config-Based Usage

```yaml
# config.yaml
advanced:
  models:
    yolo11n:
      type: "person-det"
      path: "/models/yolo11n.pt"
      load_options:
        lib_type: "YOLO"  # or "YOLOE"
        task: "detect"
      export: true  # Enable auto-export
      export_options:
        format: "onnx"  # Export to ONNX format
        half: false
        dynamic: true
        imgsz: 640
      batch: 8  # Process 8 images per batch
```

```python
from transformsai_ai_core import YOLOWrapper, YOLOEWrapper, get_logger
from transformsai_ai_core.config_loader import load_config

logger = get_logger(__name__)

# Load config
config = load_config("config.yaml")
model_config = config["advanced"]["models"]["yolo11n"]

# Initialize wrapper from config
model = YOLOWrapper(
    model_path=model_config["path"],
    export=model_config.get("export", False),
    export_options=model_config.get("export_options", {}),
    batch_size=model_config.get("batch", 1)
)

# The wrapper automatically:
# 1. Checks if yolo11n.onnx exists
# 2. Exports to ONNX if not found
# 3. Loads the exported model (falls back to .pt on error)

# Single image prediction
results = model.predict("image.jpg")

# Multiple images with sequential batching
# Chunks into batches of 8, processes sequentially, returns all results
image_paths = [f"image_{i}.jpg" for i in range(25)]
results = model.predict(image_paths)  # Returns list of 25 Results objects

# Process results (same as ultralytics)
for r in results:
    boxes = r.boxes.xyxy  # Bounding boxes
    conf = r.boxes.conf   # Confidence scores
    cls = r.boxes.cls     # Class indices
    logger.info(f"Detected {len(boxes)} objects in {r.path}")
```

#### Direct Usage

```python
from transformsai_ai_core import YOLOWrapper, YOLOEWrapper

# YOLO wrapper with ONNX export
yolo_model = YOLOWrapper(
    model_path="yolo11n.pt",
    export=True,
    export_options={
        "format": "onnx",
        "half": False,
        "dynamic": True,
        "imgsz": 640
    },
    batch_size=4
)

# YOLOE wrapper with TensorRT export (for NVIDIA GPUs)
yoloe_model = YOLOEWrapper(
    model_path="yoloe-11l-seg.pt",
    export=True,
    export_options={
        "format": "engine",  # TensorRT
        "half": True,        # FP16 optimization
        "workspace": 4       # 4GB workspace
    },
    batch_size=8
)

# YOLOE with text prompts (specific to YOLOE)
names = ["person", "car"]
yoloe_model.set_classes(names, yoloe_model.get_text_pe(names))
results = yoloe_model.predict("street.jpg")

# Access both original and exported model paths
print(f"Original: {yolo_model.original_model_path}")
print(f"Exported: {yolo_model.exported_model_path}")
```

#### Supported Export Formats

The wrappers support all ultralytics export formats:
- **ONNX** (`.onnx`) - Cross-platform
- **TensorRT** (`.engine`) - NVIDIA GPU optimization
- **TorchScript** (`.torchscript`) - PyTorch optimized
- **CoreML** (`.mlpackage`) - Apple devices
- **OpenVINO** (`_openvino_model/`) - Intel hardware
- **TFLite** (`.tflite`) - Mobile/edge devices
- And more: EdgeTPU, NCNN, RKNN, MNN, Paddle

The wrapper automatically:
- Checks if exported model exists using format-specific naming
- Exports only if missing (cached for subsequent runs)
- Loads exported model with automatic fallback to original on error

---
## Putting It All Together: A Complete Example

This shows how all components work together in a typical AI vision application.

```python
import time
import cv2
from transformsai_ai_core import (
    get_logger,
    VideoCaptureAsync,
    MediaMTXStreamer,
    DataUploader
)

# 1. Initialization
logger = get_logger(module_name=__name__)
CAMERA_SN = "kitchen_cam_01"
RTSP_URL = "rtsp://your_camera_stream_url"

cap = VideoCaptureAsync(RTSP_URL, auto_restart_on_fail=True)
streamer = MediaMTXStreamer("127.0.0.1", 8554, CAMERA_SN)
uploader = DataUploader(
    base_url="https://api.yourapp.com",
    heartbeat_url="https://api.yourapp.com/heartbeat"
)

# 2. Start background services
cap.start()
streamer.start_streaming()
logger.info(f"View stream at: {streamer.get_rtsp_url()}")

last_heartbeat = 0

try:
    # 3. Main processing loop
    while True:
        grabbed, frame = cap.read(wait_for_frame=True, timeout=10.0)
        if not grabbed:
            time.sleep(0.5)
            continue

        # --- Your AI model processing would go here ---
        # results = your_model.predict(frame)
        # annotated_frame = draw_boxes(frame, results)
        annotated_frame = frame.copy() # Placeholder

        # 4. Use library components with results
        streamer.update_frame(annotated_frame)
        
        # Send data every 10 seconds (example)
        if time.time() - last_heartbeat > 10:
            logger.info("Sending heartbeat and dummy data...")
            uploader.send_heartbeat(
                sn=CAMERA_SN,
                timestamp=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                live_url=streamer.get_hls_url()
            )
            uploader.send_data(
                endpoint_path="/analytics",
                data={"frame_shape": frame.shape}
            )
            last_heartbeat = time.time()

except KeyboardInterrupt:
    logger.info("Shutting down...")

finally:
    # 5. Cleanup
    cap.release()
    streamer.stop_streaming()
    uploader.shutdown()
    logger.info("Shutdown complete.")

```