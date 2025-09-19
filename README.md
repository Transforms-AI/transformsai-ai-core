# TransformsAI Core Library

A collection of robust, thread-safe utilities for building AI vision applications. This library provides essential components for video capture, streaming, data transmission, and logging.

## Installation

Install the library directly from the Git repository:

```bash
# Basic installation
pip install git+https://github.com/yourusername/transformsai-ai-core.git
```

## Core Components

*   **`VideoCaptureAsync`**: Asynchronous video capture with automatic restart on failure, ideal for handling unreliable IP cameras and RTSP streams.
*   **`MediaMTXStreamer`**: Pushes video frames to a MediaMTX RTSP server using FFmpeg, with detailed performance monitoring.
*   **`DataUploader`**: Asynchronous, thread-safe HTTP client with a persistent cache, automatic retries, and heartbeat functionality.
*   **`central_logger`**: A centralized, singleton logger that provides colored console output and rotating file logs out of the box.

---

## Basic Usage Examples

### 1. Central Logger

Set up consistent logging across your entire application with a single function call.

```python
from transformsai_ai_core import get_logger

# Get a logger for the current module
logger = get_logger(module_name=__name__)

logger.info("This is an informational message.")
logger.warning("This is a warning.")
logger.error("This is an error with automatic traceback logging.")

class MyWorker:
    def __init__(self):
        # It's easy to create named loggers for classes
        self.logger = get_logger(name=self.__class__.__name__)
    
    def do_work(self):
        self.logger.debug("Doing some work...")

```

### 2. Asynchronous Video Capture

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

### 3. Streaming to MediaMTX

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

### 4. Uploading Data with Caching

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