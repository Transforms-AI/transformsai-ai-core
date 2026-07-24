import threading
import cv2
import time
import os
import random
from .central_logger import get_logger

# --- Information About Script ---
__name__ = "VideoCaptureAsync"
__author__ = "TransformsAI"


class _CaptureFailure(Exception):
    """Internal signal: the current handle is unusable and must be reconnected."""


class _Backoff:
    """
    Exponential backoff with jitter, capped at a ceiling.

    Mirrors ApiClient._backoff: delay = min(start * 2**n, ceiling), plus a
    +/- jitter fraction so multiple cameras on one NVR don't reconnect in lockstep.
    """

    def __init__(self, start=1.0, ceiling=30.0, jitter=0.2):
        self.start = max(0.0, float(start or 0.0))
        self.ceiling = max(self.start, float(ceiling or 0.0))
        self.jitter = min(max(0.0, float(jitter or 0.0)), 1.0)
        self.attempts = 0

    @property
    def current(self):
        """Delay the next call to next() would use, before jitter."""
        if self.start <= 0:
            return 0.0
        return min(self.start * (2 ** min(self.attempts, 30)), self.ceiling)

    def next(self):
        delay = self.current
        self.attempts += 1
        if delay and self.jitter:
            delay *= 1.0 + random.uniform(-self.jitter, self.jitter)
        return max(0.0, delay)

    def reset(self):
        self.attempts = 0


class VideoCaptureAsync:
    """
    Asynchronous video capture class with robust handling for different source types
    (USB, RTSP, Files). Automatically applies optimal backends, buffer sizes, and
    hardware settings based on the detected source type.

    Reconnect is supervised: a failed open, a run of failed reads, a stalled stream
    (open handle that stops delivering frames) or an unexpected exception all funnel
    through the same backoff policy. Health is exposed via `state`, `is_healthy`,
    `get_stats()` and the optional `on_state_change` callback.
    """

    VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm'}

    # Lifecycle states reported by `.state`
    STATE_IDLE = "idle"
    STATE_CONNECTING = "connecting"
    STATE_STREAMING = "streaming"
    STATE_RECONNECTING = "reconnecting"
    STATE_FAILED = "failed"
    STATE_STOPPED = "stopped"

    # Fast secondary trigger; the stall watchdog is the authoritative one.
    _MAX_CONSECUTIVE_FAILS = 30

    _FFMPEG_ENV_KEY = "OPENCV_FFMPEG_CAPTURE_OPTIONS"
    _env_lock = threading.Lock()  # OPENCV_FFMPEG_CAPTURE_OPTIONS is process-global

    # capture.restart.<key> -> __init__ kwarg
    _RESTART_KEY_MAP = {
        "enabled": "auto_restart_on_fail",
        "delay": "restart_delay",
        "backoff_start": "restart_backoff_start",
        "backoff_jitter": "restart_backoff_jitter",
        "reset_after": "restart_reset_after",
        "max_attempts": "max_restart_attempts",
        "stall_timeout": "stall_timeout",
    }

    # capture.timestamp.<key> -> hide_camera_timestamp_and_add_current_time kwarg
    _TIMESTAMP_KEY_MAP = {
        "rect_ratios": "camera_ts_rect_ratios",
        "rect_coords": "camera_ts_rect_coords",
        "hide_color": "hide_rect_color",
        "font_color": "new_ts_font_color",
        "font_scale": "new_ts_font_scale",
        "time_format": "time_format",
    }

    def __init__(
        self,
        src=0,
        width=None,
        height=None,
        driver=None,
        auto_restart_on_fail=True,
        restart_delay=30.0,
        buffer_size=1,
        opencv_backend="auto",
        max_frame_age_ms=None,
        auto_resize=True,
        hw_decode=False,
        fps=None,
        restart_backoff_start=1.0,
        restart_backoff_jitter=0.2,
        restart_reset_after=30.0,
        max_restart_attempts=None,
        stall_timeout=None,
        open_timeout=5.0,
        rtsp_transport="tcp",
        ffmpeg_options=None,
        health_log_interval=60.0,
        on_state_change=None,
        timestamp_overlay=False,
        timestamp_overlay_options=None,
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

        # Reconnect policy
        self.restart_backoff_start = restart_backoff_start
        self.restart_backoff_jitter = restart_backoff_jitter
        self.restart_reset_after = restart_reset_after
        self.max_restart_attempts = max_restart_attempts
        self.stall_timeout = stall_timeout
        self.open_timeout = open_timeout
        self.rtsp_transport = rtsp_transport
        self.ffmpeg_options = ffmpeg_options
        self.health_log_interval = health_log_interval
        self.on_state_change = on_state_change
        self._backoff = _Backoff(restart_backoff_start, restart_delay, restart_backoff_jitter)

        # Optional OSD timestamp-overwrite overlay (applied once, in the capture thread).
        # Drop None-valued options so the util's own defaults apply for anything unset.
        self._timestamp_overlay = bool(timestamp_overlay)
        self._ts_overlay_options = {
            k: v for k, v in (timestamp_overlay_options or {}).items() if v is not None
        }
        self._ts_overlay_failures = 0

        # State variables
        self.cap = None
        self.started = False
        self._grabbed = False
        self._frame = None
        self._frame_timestamp = 0
        self._frame_id = 0  # Tracks unique frames
        self._new_frame_condition = threading.Condition()
        self._last_read_frame_id = -1 # Tracks the last frame sent to the consumer
        self._thread = None
        self._fps = None
        self._last_frame_time = 0
        self._frame_count = 0
        self._stop_event = threading.Event()

        # Supervision / health state
        self._state = self.STATE_IDLE
        self._state_lock = threading.RLock()
        self._cap_lock = threading.RLock()
        self._ever_connected = False
        self._orphaned = False
        self._orphaned_threads = 0
        self.restart_count = 0
        self._consecutive_failures = 0
        self._last_error = None
        self._last_error_time = None
        self._connected_at = None
        self._streaming_since = None
        self._last_new_frame_mono = 0.0
        self._total_frames = 0
        self._frames_dropped_stale = 0
        self._measured_fps = 0.0
        self._fps_window_start = 0.0
        self._fps_window_frames = 0
        self._last_health_log = 0.0
        self._restarts_since_log = 0

        self.logger = get_logger(self)

        try:
            self._initialize_capture()
        except RuntimeError as e:
            if not self.auto_restart_on_fail:
                raise
            else:
                self._note_error(e)
                self.logger.warning(f"[{self.src}] Initial capture failed: {e}. Will retry in thread.")

    @property
    def frame_id(self):
        """Returns the ID of the latest grabbed frame to prevent duplicate processing."""
        with self._new_frame_condition:
            return self._frame_id

    @property
    def state(self):
        """Lifecycle state: idle | connecting | streaming | reconnecting | failed | stopped."""
        return self._state

    @property
    def is_healthy(self):
        """True when streaming and a frame arrived within the stall window."""
        if self._state != self.STATE_STREAMING:
            return False
        stall = self._effective_stall_timeout()
        if not stall:
            return True
        return (time.monotonic() - self._last_new_frame_mono) <= stall

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

    # ------------------------------------------------------------------
    # Handle ownership
    #
    # While the capture thread is alive it is the sole owner of `self.cap`
    # and always releases it in its own finally block. Callers never release
    # a handle a live thread might still be sitting inside.
    # ------------------------------------------------------------------
    def _drop_handle(self):
        """Release and forget the current handle (safe to call repeatedly)."""
        with self._cap_lock:
            cap, self.cap = self.cap, None
        if cap is not None:
            try:
                cap.release()
            except Exception as e:
                self.logger.debug(f"[{self.src}] Handle release raised: {e}")

    def _has_handle(self):
        with self._cap_lock:
            cap = None if self._orphaned else self.cap
        try:
            return cap is not None and cap.isOpened()
        except Exception:
            return False

    def _build_ffmpeg_options(self):
        """
        Build the OPENCV_FFMPEG_CAPTURE_OPTIONS string for network sources so a dead
        link fails fast instead of blocking inside grab() for the OS default.
        """
        if self.ffmpeg_options is not None:
            return self.ffmpeg_options or None
        if self.source_type != 'NETWORK':
            return None

        parts = []
        # rtsp_transport is an RTSP-demuxer option — meaningless for http/udp/tcp URLs.
        if self.rtsp_transport and isinstance(self.src, str) and self.src.startswith('rtsp://'):
            parts.append(f"rtsp_transport;{self.rtsp_transport}")
        if self.open_timeout and self.open_timeout > 0:
            micros = int(float(self.open_timeout) * 1_000_000)
            # 'timeout' is modern FFmpeg, 'stimeout' the pre-5.0 name; the unused one is ignored.
            parts.append(f"timeout;{micros}")
            parts.append(f"stimeout;{micros}")
        return "|".join(parts) if parts else None

    def _open_capture(self, backend):
        """Construct a cv2.VideoCapture, applying FFmpeg options for network sources."""
        options = self._build_ffmpeg_options()
        if not options:
            return cv2.VideoCapture(self.src, backend)

        # The env var is process-global, so network opens serialize. Bounded by open_timeout.
        with self._env_lock:
            previous = os.environ.get(self._FFMPEG_ENV_KEY)
            os.environ[self._FFMPEG_ENV_KEY] = options
            try:
                return cv2.VideoCapture(self.src, backend)
            finally:
                if previous is None:
                    os.environ.pop(self._FFMPEG_ENV_KEY, None)
                else:
                    os.environ[self._FFMPEG_ENV_KEY] = previous

    def _initialize_capture(self):
            """Opens the capture device and applies optimal settings."""
            self._drop_handle()

            if self.source_type == 'USB':
                # Use our optimized V4L2 initialization
                cap = self._open_capture(cv2.CAP_V4L2)
                if not cap.isOpened():
                    cap.release()
                    raise RuntimeError(f"Could not open USB source: {self.src}")

                # Strict Order: FOURCC -> Width/Height -> FPS -> Buffer
                cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
                if self.width and self.height:
                    cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
                    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
                if self.target_fps:
                    cap.set(cv2.CAP_PROP_FPS, self.target_fps)
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                # Flush stale hardware buffers
                for _ in range(4):
                    cap.grab()

            else:
                # Fallback for Network/Files
                backend = cv2.CAP_FFMPEG if self.source_type == 'NETWORK' else cv2.CAP_ANY
                cap = self._open_capture(backend)
                if not cap.isOpened():
                    cap.release()
                    raise RuntimeError(f"Could not open source: {self.src}")
                if self.buffer_size is not None:
                    cap.set(cv2.CAP_PROP_BUFFERSIZE, self.buffer_size)

            # Enable hardware-accelerated decoding if requested
            if self.hw_decode:
                cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)

            # Extract metadata
            fps = cap.get(cv2.CAP_PROP_FPS)
            self._fps = self.target_fps if self.target_fps else (fps if fps and fps > 0 else 30.0)

            if self._is_file_source:
                self._frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

            with self._cap_lock:
                self.cap = cap
                self._orphaned = False

            now = time.monotonic()
            self._connected_at = now
            self._last_new_frame_mono = now  # start the stall clock at connect
            self._ever_connected = True

            self.logger.info(f"[{self.src}] Initialized {self.source_type} source (FPS: {self._fps:.2f})")

    def get_height_width(self):
        with self._cap_lock:
            cap = None if self._orphaned else self.cap
        if cap and cap.isOpened():
            return int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        return None, None

    def get(self, propId):
        with self._cap_lock:
            cap = None if self._orphaned else self.cap
        return cap.get(propId) if cap else None

    def set(self, propId, value):
        with self._cap_lock:
            cap = None if self._orphaned else self.cap
        return cap.set(propId, value) if cap else False

    def _attempt_restart_capture(self):
        """Single restart attempt. Kept for backwards compatibility — the supervised
        loop uses _connect()/_schedule_retry() so the backoff policy is applied."""
        self.logger.info(f"[{self.src}] Attempting to restart capture...")
        try:
            self._initialize_capture()
            if self._has_handle():
                self._last_frame_time = 0
                with self._new_frame_condition:
                    self._grabbed = False
                return True
        except Exception as e:
            self._note_error(e)
            self.logger.error(f"[{self.src}] Restart failed: {e}")
        return False

    def start(self, loop=None):
        if self.started:
            return self
        if self._orphaned:
            # A previous thread is still wedged inside OpenCV and owns the old handle.
            # Starting a second thread would let it release a handle that thread is using.
            raise RuntimeError(
                f"[{self.src}] previous capture thread was abandoned mid-call — "
                "construct a new VideoCaptureAsync instead of restarting this one"
            )
        if loop is not None and self._is_file_source:
            self.loop = loop

        self.started = True
        self._stop_event.clear()
        self._backoff.reset()
        self._last_health_log = 0.0
        self._thread = threading.Thread(target=self._update, name=f"VideoCaptureAsync_{self.src}", daemon=True)
        self._thread.start()
        return self

    # ------------------------------------------------------------------
    # Supervision
    # ------------------------------------------------------------------
    def _set_state(self, state):
        with self._state_lock:
            if self._state == state:
                return
            previous, self._state = self._state, state

        self.logger.debug(f"[{self.src}] state: {previous} -> {state}")
        callback = self.on_state_change
        if callback is None:
            return
        try:
            callback(state, self.get_stats())
        except Exception as e:
            self.logger.warning(f"[{self.src}] on_state_change callback failed: {e}")

    def _note_error(self, exc):
        self._last_error = f"{type(exc).__name__}: {exc}" if isinstance(exc, BaseException) else str(exc)
        self._last_error_time = time.time()

    def _effective_stall_timeout(self):
        """Seconds without a decoded frame before the stream counts as stalled (0 = off)."""
        if self.stall_timeout is not None:
            return max(0.0, float(self.stall_timeout))
        if self._is_file_source:
            return 0.0  # paced reads / EOF handling, not a liveness signal
        fps = self.target_fps or self._fps or 30.0
        return max(5.0, 20.0 / fps) if fps > 0 else 5.0

    def _schedule_retry(self):
        """Apply the restart policy. Returns False when we should give up or are stopping."""
        if not self.auto_restart_on_fail:
            self.logger.warning(f"[{self.src}] auto_restart_on_fail is disabled — capture thread exiting")
            return False

        if self.max_restart_attempts is not None and self._backoff.attempts >= self.max_restart_attempts:
            self.logger.error(
                f"[{self.src}] Giving up after {self._backoff.attempts} reconnect attempts "
                f"(last error: {self._last_error})"
            )
            self._set_state(self.STATE_FAILED)
            return False

        delay = self._backoff.next()
        self._set_state(self.STATE_RECONNECTING if self._ever_connected else self.STATE_CONNECTING)
        self.logger.info(f"[{self.src}] Reconnecting in {delay:.1f}s (attempt {self._backoff.attempts})")
        self._stop_event.wait(delay)
        return not self._stop_event.is_set()

    def _connect(self):
        """Open the source, retrying per policy. False = gave up or stopping."""
        while not self._stop_event.is_set():
            self._set_state(self.STATE_RECONNECTING if self._ever_connected else self.STATE_CONNECTING)
            try:
                self._initialize_capture()
                return True
            except Exception as e:
                self._note_error(e)
                # A refused/unreachable source is expected and handled — log it without the
                # traceback loguru attaches to .error(), or a camera that is down overnight
                # buries the log in identical stacks.
                log = self.logger.warning if isinstance(e, RuntimeError) else self.logger.error
                log(f"[{self.src}] Connect failed: {e}")
                if not self._schedule_retry():
                    return False
        return False

    def _handle_failure(self, exc, unexpected=False):
        """Drop the handle and apply the restart policy. False = gave up or stopping."""
        if self._stop_event.is_set():
            return False  # failure caused by shutdown — not a restart

        self._note_error(exc)
        self.restart_count += 1
        self._restarts_since_log += 1
        log = self.logger.error if unexpected else self.logger.warning
        log(f"[{self.src}] {exc} — restarting capture")

        self._drop_handle()
        self._streaming_since = None
        self._consecutive_failures = 0
        self._last_frame_time = 0
        with self._new_frame_condition:
            self._grabbed = False
            self._new_frame_condition.notify_all()

        return self._schedule_retry()

    def _apply_timestamp_overlay(self, frame):
        """
        Paint the current system time over the camera's OSD timestamp region.

        Runs in the capture thread on the freshly-decoded buffer (inplace, allocation-free)
        so every consumer inherits the corrected time from one overlay. Fail-open: a bad
        options dict must not raise into the capture loop (that would trigger a spurious
        reconnect), so errors are logged at most once and the original frame is published.
        """
        try:
            from .utils import hide_camera_timestamp_and_add_current_time
            return hide_camera_timestamp_and_add_current_time(
                frame, timestamp=None, inplace=True, **self._ts_overlay_options
            )
        except Exception as e:
            self._ts_overlay_failures += 1
            if self._ts_overlay_failures == 1:
                self.logger.error(
                    f"[{self.src}] timestamp overlay failed ({e}); publishing un-stamped "
                    "frames (this is logged once)"
                )
            return frame

    def _publish_frame(self, frame):
        if self._timestamp_overlay and frame is not None:
            frame = self._apply_timestamp_overlay(frame)

        now = time.monotonic()
        self._last_new_frame_mono = now
        self._total_frames += 1
        self._consecutive_failures = 0

        with self._new_frame_condition:
            self._grabbed = True
            self._frame = frame
            self._frame_timestamp = time.time()
            self._frame_id += 1
            self._new_frame_condition.notify_all()  # Wake up any waiting read() calls

        if self._state != self.STATE_STREAMING:
            self._streaming_since = now
            self._set_state(self.STATE_STREAMING)
        elif (self._backoff.attempts and self._streaming_since
                and (now - self._streaming_since) >= self.restart_reset_after):
            # Stable for long enough — a future blip starts the ramp from the bottom again.
            self._backoff.reset()

        # Rolling measured FPS
        if self._fps_window_start == 0.0:
            self._fps_window_start = now
        self._fps_window_frames += 1
        elapsed = now - self._fps_window_start
        if elapsed >= 5.0:
            self._measured_fps = self._fps_window_frames / elapsed
            self._fps_window_start = now
            self._fps_window_frames = 0

    def _note_soft_fail(self):
        """Tolerate minor drops; escalate to a reconnect once the streak is long enough."""
        self._consecutive_failures += 1
        if self._consecutive_failures > self._MAX_CONSECUTIVE_FAILS:
            raise _CaptureFailure(f"{self._consecutive_failures} consecutive read failures")
        self._stop_event.wait(0.01)

    def _check_stall(self):
        """An open handle that stopped delivering frames is dead — reconnect it."""
        stall = self._effective_stall_timeout()
        if not stall or not self._last_new_frame_mono:
            return
        idle = time.monotonic() - self._last_new_frame_mono
        if idle > stall:
            raise _CaptureFailure(f"stalled — no frame for {idle:.1f}s (limit {stall:.1f}s)")

    def _read_once(self):
        """Read one frame. Raises _CaptureFailure when the handle needs reconnecting."""
        with self._cap_lock:
            cap = self.cap
        if cap is None:
            raise _CaptureFailure("handle disappeared")

        if self._is_file_source:
            # Paced reading for files
            fps = self._fps if self._fps else 30.0
            target_frame_duration = 1.0 / fps if fps > 0 else 0
            current_time = time.monotonic()
            if self._last_frame_time != 0 and (current_time - self._last_frame_time) < target_frame_duration:
                self._stop_event.wait(0.005)
                return

            grabbed, frame = cap.read()
            if not grabbed and self.loop:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                grabbed, frame = cap.read()
            if not grabbed:
                self._note_soft_fail()
                return
            self._last_frame_time = current_time
            self._publish_frame(frame)
            return

        # Live sources (USB/Network): Separate grab/retrieve to clear hardware buffers fast
        if not cap.grab():
            self._note_soft_fail()
            return

        current_time = time.monotonic()
        if self.target_fps is not None and (current_time - self._last_retrieved_time) < self._min_frame_time:
            # Grabbed buffer to clear it, but skip decoding to save CPU if we're ahead of
            # schedule. A successful grab is still proof of life, so clear the streak.
            self._consecutive_failures = 0
            return

        ret, frame = cap.retrieve()
        if not ret:
            self._note_soft_fail()
            return
        self._last_retrieved_time = current_time
        self._publish_frame(frame)

    def _maybe_health_log(self):
        """Periodic health line: DEBUG while healthy (file sink only), INFO when degraded."""
        if not self.health_log_interval or self.health_log_interval <= 0:
            return
        now = time.monotonic()
        if self._last_health_log == 0.0:
            self._last_health_log = now
            return
        if (now - self._last_health_log) < self.health_log_interval:
            return
        self._last_health_log = now

        stats = self.get_stats()
        message = (
            f"[{self.src}] health: state={stats['state']} fps={stats['measured_fps']:.1f} "
            f"frames={stats['total_frames']} restarts={stats['restart_count']} "
            f"uptime={stats['uptime_seconds']:.0f}s"
        )
        if self._restarts_since_log or not stats['is_healthy']:
            self.logger.info(f"{message} last_error={stats['last_error']}")
        else:
            self.logger.debug(message)
        self._restarts_since_log = 0

    def _run_loop(self):
        while not self._stop_event.is_set():
            # Phase 1: Ensure capture is open
            if not self._has_handle():
                if not self._connect():
                    return

            # Phase 2/3: Read a frame and police liveness
            try:
                self._read_once()
                self._check_stall()
            except _CaptureFailure as e:
                if not self._handle_failure(e):
                    return
                continue
            except Exception as e:
                if not self._handle_failure(e, unexpected=True):
                    return
                continue

            self._maybe_health_log()

    def _update(self):
        try:
            self._run_loop()
        finally:
            # The thread owns the handle — it always releases what it opened, even if
            # release() gave up waiting for us and abandoned this thread.
            self._drop_handle()
            with self._new_frame_condition:
                self._grabbed = False
                self._new_frame_condition.notify_all()
            self.started = False
            self._streaming_since = None
            self._set_state(self.STATE_STOPPED if self._stop_event.is_set() else self.STATE_FAILED)

    def read(self, wait_for_new_frame=True, timeout=1.0, copy=True):
            """
            Returns (grabbed, frame).
            If wait_for_new_frame is True, it blocks until a fresh frame is decoded,
            matching standard cv2.VideoCapture behavior and preventing CPU thrashing.

            While the source is reconnecting the call blocks up to `timeout` and returns
            (False, None). After stop()/release() it returns immediately.
            """
            if not self.started:
                # Terminal failure (gave up / auto-restart off): sleep out the timeout so a
                # naive `while True: read()` consumer doesn't spin a core on a dead source.
                if (wait_for_new_frame and timeout and self._state == self.STATE_FAILED
                        and not self._stop_event.is_set()):
                    self._stop_event.wait(timeout)
                return False, None

            with self._new_frame_condition:
                if wait_for_new_frame:
                    # Wait until the background thread produces a frame we haven't read yet
                    self._new_frame_condition.wait_for(
                        lambda: self._frame_id > self._last_read_frame_id or self._stop_event.is_set(),
                        timeout=timeout
                    )

                # Update the tracker so we don't read this exact frame again next time
                self._last_read_frame_id = self._frame_id

                frame = self._frame.copy() if (self._grabbed and self._frame is not None and copy) else self._frame
                grabbed = self._grabbed

                # Drop stale frames if max_frame_age_ms is configured
                if grabbed and self.max_frame_age_ms is not None and self.max_frame_age_ms > 0:
                    frame_age_ms = (time.time() - self._frame_timestamp) * 1000
                    if frame_age_ms > self.max_frame_age_ms:
                        grabbed = False
                        frame = None
                        self._frames_dropped_stale += 1

            # Software resize fallback (only triggers if hardware resize wasn't possible)
            if grabbed and frame is not None and self.auto_resize and self.width and self.height:
                h, w = frame.shape[:2]
                if w != self.width or h != self.height:
                    interp = cv2.INTER_AREA if self._is_file_source else cv2.INTER_LINEAR
                    frame = cv2.resize(frame, (self.width, self.height), interpolation=interp)

            return grabbed, frame

    def get_stats(self):
        """Snapshot of capture health, safe to call from any thread."""
        now = time.monotonic()
        last_frame_age_ms = (time.time() - self._frame_timestamp) * 1000 if self._frame_timestamp else None
        return {
            'src': self.src,
            'source_type': self.source_type,
            'state': self._state,
            'is_healthy': self.is_healthy,
            'started': self.started,
            'restart_count': self.restart_count,
            'consecutive_failures': self._consecutive_failures,
            'restart_attempts': self._backoff.attempts,
            'current_backoff': self._backoff.current,
            'stall_timeout': self._effective_stall_timeout(),
            'last_error': self._last_error,
            'last_error_time': self._last_error_time,
            'uptime_seconds': (now - self._streaming_since) if self._streaming_since else 0.0,
            'total_frames': self._total_frames,
            'frames_dropped_stale': self._frames_dropped_stale,
            'last_frame_age_ms': last_frame_age_ms,
            'target_fps': self.target_fps,
            'source_fps': self._fps,
            'measured_fps': self._measured_fps,
            'orphaned_threads': self._orphaned_threads,
        }

    def stop(self):
        self._stop_event.set()
        with self._new_frame_condition:
            self._new_frame_condition.notify_all()  # wake readers blocked in read()
        if self._thread is None or not self._thread.is_alive():
            self.started = False
            self._set_state(self.STATE_STOPPED)

    def release(self):
        self.stop()

        thread = self._thread
        if thread and thread.is_alive():
            # The backoff sleep is interruptible, so this only has to cover a blocking
            # OpenCV call — which open_timeout already bounds for network sources.
            join_timeout = max(2.0, float(self.open_timeout or 0.0) + 1.0)
            thread.join(timeout=join_timeout)
            if thread.is_alive():
                # Wedged inside cv2. Abandon it rather than releasing a handle it may
                # still be using — the daemon thread releases it when the call returns.
                self._orphaned = True
                self._orphaned_threads += 1
                self.logger.warning(
                    f"[{self.src}] Capture thread still wedged in OpenCV after {join_timeout:.1f}s; "
                    "abandoning the handle (the daemon thread will release it)"
                )
                self.started = False
                return

        self._drop_handle()
        self.started = False

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()

    @classmethod
    def from_config(cls, cfg, **overrides):
        from .config_loader import build_rtsp_url, init_kwargs
        if hasattr(cfg, "model_dump"):
            cfg = cfg.model_dump()
        else:
            cfg = dict(cfg)

        if cfg.get("local", False):
            src = cfg.get("local_source", "")
        else:
            src = cfg.get("rtsp_url")
            if not src:
                src = build_rtsp_url(cfg.get("rtsp_source", {}))

        capture_cfg = dict(cfg.get("capture", {}) or {})

        restart_cfg = capture_cfg.pop("restart", None) or {}
        if hasattr(restart_cfg, "model_dump"):
            restart_cfg = restart_cfg.model_dump()

        # Legacy flat keys first; the nested restart block wins where both are set.
        restart_kwargs = {}
        for legacy in ("auto_restart_on_fail", "restart_delay"):
            value = capture_cfg.pop(legacy, None)
            if value is not None:
                restart_kwargs[legacy] = value
        for key, value in restart_cfg.items():
            if key in cls._RESTART_KEY_MAP and value is not None:
                restart_kwargs[cls._RESTART_KEY_MAP[key]] = value
        restart_kwargs.update(restart_cfg.get("extras", {}) or {})

        # Optional timestamp-overwrite overlay: flatten the nested block into the
        # timestamp_overlay flag + a mapped options dict passed to the util.
        timestamp_cfg = capture_cfg.pop("timestamp", None) or {}
        if hasattr(timestamp_cfg, "model_dump"):
            timestamp_cfg = timestamp_cfg.model_dump()
        timestamp_kwargs = {}
        if timestamp_cfg:
            ts_options = {}
            for key, value in timestamp_cfg.items():
                if key in cls._TIMESTAMP_KEY_MAP and value is not None:
                    ts_options[cls._TIMESTAMP_KEY_MAP[key]] = value
            ts_options.update(timestamp_cfg.get("extras", {}) or {})
            timestamp_kwargs = {
                "timestamp_overlay": bool(timestamp_cfg.get("enabled", False)),
                "timestamp_overlay_options": ts_options,
            }

        kwargs: dict = {"src": src, **capture_cfg, **restart_kwargs, **timestamp_kwargs}
        kwargs = {**cfg.get("extras", {}), **kwargs, **overrides}
        return cls(**init_kwargs(cls, kwargs))
