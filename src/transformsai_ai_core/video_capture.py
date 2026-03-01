import threading
import cv2
import time
import os
from .datasend import DataUploader 
from .utils import time_to_string 
from .central_logger import get_logger

# --- Information About Script ---
__name__ = "VideoCaptureAsync with Heartbeat"
__author__ = "TransformsAI"

class VideoCaptureAsync:
    """
    Asynchronous video capture class with robust handling for different source types
    (USB, RTSP, Files). Automatically applies optimal backends, buffer sizes, and 
    hardware settings based on the detected source type.
    """

    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

    def __init__(
        self, 
        src=0, 
        width=None, 
        height=None, 
        driver=None,
        heartbeat_config=None, 
        auto_restart_on_fail=False, 
        restart_delay=30.0,
        buffer_size=1,
        opencv_backend="auto",
        max_frame_age_ms=100,
        auto_resize=True,
        hw_decode=False,
        fps=None
    ):
        # Cast string integers to actual ints for proper USB detection
        self.src = int(src) if isinstance(src, str) and src.isdigit() else src
        self.width = width
        self.height = height
        self.target_fps = fps
        self.auto_resize = auto_resize
        self.hw_decode = hw_decode
        self.driver = driver
        
        # FPS control
        self._min_frame_time = 1.0 / self.target_fps if self.target_fps else 0
        self._last_retrieved_time = 0
        
        self.source_type = self._detect_source_type(self.src)
        self._is_file_source = (self.source_type == 'FILE')
        self.loop = True if self._is_file_source else False 

        self.auto_restart_on_fail = auto_restart_on_fail
        self.restart_delay = restart_delay
        self.buffer_size = buffer_size
        self.opencv_backend = opencv_backend
        self.max_frame_age_ms = max_frame_age_ms

        # State variables
        self.cap = None
        self.started = False
        self._grabbed = False
        self._frame = None
        self._frame_timestamp = 0
        self._frame_id = 0  # Tracks unique frames
        self._read_lock = threading.Lock()
        self._thread = None
        self._fps = 30.0
        self._last_frame_time = 0
        self._frame_count = 0
        self._stop_event = threading.Event()

        self._heartbeat_config = heartbeat_config or {}
        self._data_uploader = None
        self.logger = get_logger(self)

        if self._heartbeat_config.get('enabled', False):
            self._initialize_heartbeat()

        try:
            self._initialize_capture()
            self._send_heartbeat(f"Video source {self.src} initialized successfully.")
        except RuntimeError as e:
            if not self.auto_restart_on_fail:
                self._send_heartbeat(f"Video source {self.src} initialization failed.")
                raise
            else:
                self.logger.warning(f"[{self.src}] Initial capture failed: {e}. Will retry in thread.")

    @property
    def frame_id(self):
        """Returns the ID of the latest grabbed frame to prevent duplicate processing."""
        with self._read_lock:
            return self._frame_id

    def _detect_source_type(self, src) -> str:
        """Classifies the video source to apply specific optimizations."""
        if isinstance(src, int) or (isinstance(src, str) and src.startswith('/dev/video')):
            return 'USB'
        if isinstance(src, str):
            if src.startswith(('rtsp://', 'http://', 'https://', 'udp://', 'tcp://')):
                return 'NETWORK'
            _, ext = os.path.splitext(src)
            if ext.lower() in self.VIDEO_EXTENSIONS:
                return 'FILE'
        return 'UNKNOWN'

    def _initialize_heartbeat(self):
        try:
            self._data_uploader = DataUploader(**self._heartbeat_config.get('uploader_config', {}))
            self.logger.info(f"[{self.src}] Heartbeat initialized.")
        except Exception as e:
            self.logger.error(f"[{self.src}] Heartbeat init failed: {e}")
            self._data_uploader = None

    def _send_heartbeat(self, custom_message=None):
        if self._data_uploader and self._heartbeat_config.get('enabled', False):
            sn = self._heartbeat_config.get('sn', f"capture_{self.src}")
            self._data_uploader.send_heartbeat(sn, time_to_string(time.time()), status_log=custom_message)

    def _initialize_capture(self):
        """Opens the capture device and applies source-specific hardware optimizations."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None
            
        # 1. Select optimal backend
        backend = cv2.CAP_ANY
        if self.opencv_backend != "auto":
            backend_map = {'ffmpeg': cv2.CAP_FFMPEG, 'gstreamer': cv2.CAP_GSTREAMER, 'v4l2': cv2.CAP_V4L2}
            backend = backend_map.get(self.opencv_backend.lower(), cv2.CAP_ANY)
        elif self.source_type == 'USB':
            backend = cv2.CAP_V4L2
        elif self.source_type == 'NETWORK':
            backend = cv2.CAP_FFMPEG

        self.cap = cv2.VideoCapture(self.src, backend)
        if not self.cap.isOpened():
            raise RuntimeError(f"Could not open video source: {self.src}")

        # 2. Apply Source-Specific Optimizations
        if self.source_type == 'USB':
            # Force MJPG compression to save USB bandwidth (crucial for high-res/FPS)
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
            # Set hardware resolution BEFORE reading to prevent CPU downscaling
            if self.width and self.height:
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
        if self.source_type in ('USB', 'NETWORK') and self.buffer_size is not None:
            # Minimize internal buffer to prevent latency/stale frames
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

        if self.hw_decode:
            try:
                self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
            except Exception:
                pass

        # 3. Flush stale hardware buffers for live cameras
        if self.source_type == 'USB':
            for _ in range(4):
                self.cap.grab()

        # 4. Extract metadata
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self._fps = fps if fps and fps > 0 else 30.0
        
        if self._is_file_source:
            self._frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            
        self.logger.info(f"[{self.src}] Initialized {self.source_type} source (FPS: {self._fps:.2f})")

    def get_height_width(self):
        if self.cap and self.cap.isOpened():
            return int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return None, None

    def get(self, propId):
        return self.cap.get(propId) if self.cap else None

    def set(self, propId, value):
        return self.cap.set(propId, value) if self.cap else False

    def _attempt_restart_capture(self):
        self.logger.info(f"[{self.src}] Attempting to restart capture...")
        try:
            self._initialize_capture()
            if self.cap and self.cap.isOpened():
                self._send_heartbeat(f"Video capture for {self.src} recovered.")
                self._last_frame_time = 0
                self._grabbed = False
                return True
        except Exception as e:
            self.logger.error(f"[{self.src}] Restart failed: {e}")
        return False

    def start(self, loop=None):
        if self.started:
            return self
        if loop is not None and self._is_file_source:
            self.loop = loop

        self.started = True
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._update, name=f"VideoCaptureAsync_{self.src}", daemon=True)
        self._thread.start()
        return self

    def _update(self):
        target_frame_duration = 1.0 / self._fps if self._fps > 0 else 0
        consecutive_fails = 0

        while not self._stop_event.is_set():
            # Phase 1: Ensure capture is open
            if not self.cap or not self.cap.isOpened():
                if self.auto_restart_on_fail:
                    if not self._attempt_restart_capture():
                        self._stop_event.wait(self.restart_delay)
                        continue
                else:
                    break

            # Phase 2: Read frame based on source type
            grabbed, frame = False, None
            try:
                if self._is_file_source:
                    # Paced reading for files
                    current_time = time.monotonic()
                    if (current_time - self._last_frame_time) >= target_frame_duration or self._last_frame_time == 0:
                        grabbed, frame = self.cap.read()
                        if not grabbed and self.loop:
                            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                            grabbed, frame = self.cap.read()
                        if grabbed:
                            self._last_frame_time = current_time
                    else:
                        self._stop_event.wait(0.005)
                        continue
                else:
                    # Live sources (USB/Network): Separate grab/retrieve to clear hardware buffers fast
                    grabbed = self.cap.grab()
                    if grabbed:
                        # Only decode if enough time passed
                        current_time = time.monotonic()
                        if self.target_fps is None or (current_time - self._last_retrieved_time) >= self._min_frame_time:
                            ret, frame = self.cap.retrieve()
                            
                            if not ret: 
                                grabbed = False
                            else: 
                                self._last_retrieved_time = current_time
                        else:
                            # Grabbed buffer to clear it, but skip decoding to save CPU if we're ahead of schedule
                            continue

                # Phase 3: Handle outcome
                if not grabbed:
                    consecutive_fails += 1
                    # Tolerate minor drops (up to 30 frames) before full restart
                    if consecutive_fails > 30:
                        self.logger.warning(f"[{self.src}] 30 consecutive read failures. Restarting.")
                        if self.auto_restart_on_fail:
                            self.cap.release()
                            self.cap = None
                            consecutive_fails = 0
                            continue
                        else:
                            break
                    else:
                        self._stop_event.wait(0.01)
                        continue
                else:
                    consecutive_fails = 0
                    with self._read_lock:
                        self._grabbed = True
                        self._frame = frame
                        self._frame_timestamp = time.time()
                        self._frame_id += 1

            except Exception as e:
                self.logger.error(f"[{self.src}] Capture error: {e}")
                if self.auto_restart_on_fail:
                    if self.cap: self.cap.release(); self.cap = None
                    continue
                break

        self.started = False

    def read(self, wait_for_frame=False, timeout=1.0, copy=True):
        """
        Returns (grabbed, frame). 
        Maintains exact interface compatibility while utilizing internal optimizations.
        """
        if wait_for_frame and not self._grabbed and self.started:
            start_time = time.monotonic()
            while not self._grabbed and self.started:
                if time.monotonic() - start_time > timeout or self._stop_event.is_set():
                    return False, None
                time.sleep(0.005)

        with self._read_lock:
            frame = self._frame.copy() if (self._grabbed and self._frame is not None and copy) else self._frame
            grabbed = self._grabbed
        
        # Software resize fallback (only triggers if hardware resize wasn't possible)
        if grabbed and frame is not None and self.auto_resize and self.width and self.height:
            h, w = frame.shape[:2]
            if w != self.width or h != self.height:
                interp = cv2.INTER_AREA if self._is_file_source else cv2.INTER_LINEAR
                frame = cv2.resize(frame, (self.width, self.height), interpolation=interp)
        
        return grabbed, frame

    def stop(self):
        if self.started:
            self._stop_event.set()

    def release(self):
        self.stop()
        if self._thread and self._thread.is_alive():
             self._thread.join(timeout=max(2.0, self.restart_delay + 1.0))

        if self.cap:
            self.cap.release()
            self.cap = None
            
        if self._data_uploader:
            try:
                self._data_uploader.shutdown(wait=False)
            except Exception:
                pass
        self.started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()