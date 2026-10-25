import cv2
import subprocess
import time
import threading
import signal
import os
import psutil
import socket
import re
import queue
from threading import Lock
from collections import deque
from .central_logger import get_logger
from .config.schema import AUDIO_INPUT_TYPES, StreamConfig
from .config.types import format_bitrate, parse_bitrate
from .config.urls import build_stream_urls
from .video_capture import _Backoff

# --- Information About Script ---
__name__ = "Streamer for MediaMTX Server"
__author__ = "TransformsAI"

# `audio_highpass_hz` / `audio_gain_db` value meaning "the audio input type's default"
AUTO = "auto"


def _num(value):
    """A number as FFmpeg should see it: 20.0 -> '20', 12.5 -> '12.5'."""
    value = float(value)
    return str(int(value)) if value.is_integer() else str(value)


def _frames(value):
    """A frame count (GOP length) from a possibly fractional fps product."""
    return str(max(1, int(round(float(value)))))


def build_audio_filter(audio_input, *, channel=0, highpass_hz=None, gain_db=0.0, limiter=0.97):
    """The -af chain, in a fixed order: channel select (alsa) -> high-pass -> gain -> limiter -> resample.

    highpass_hz: falsy = off; up to 20 Hz a single 2-pole stage (the MEMS DC-offset
    filter of 4.x); above 20 Hz two cascaded 2-pole stages (4th order: a single
    2nd-order stage was measured too gentle for hum/rumble). gain_db: falsy = no
    volume stage and no limiter. limiter: None = gain without a limiter.
    """
    stages = []
    if audio_input == "alsa":
        # Mic channel select: a stereo I2S MEMS mic carries one live channel
        stages.append(f"pan=mono|c0=c{int(channel)}")
    # Camera audio arrives mono with the camera's own AGC, so by default it gets
    # no channel select/high-pass -- only what the config asks for.
    if highpass_hz:
        cutoff = _num(highpass_hz)
        if float(highpass_hz) > 20:
            stages += [f"highpass=f={cutoff}:poles=2"] * 2
        else:
            stages.append(f"highpass=f={cutoff}")
    if gain_db:
        # Fixed makeup gain through a limiter (raw MEMS audio is near-inaudible)
        stages.append(f"volume={float(gain_db)}dB")
        if limiter:
            stages.append(f"alimiter=limit={limiter}:level=false")
    # The async resampler snaps samples onto the wallclock
    # timestamps by stretching/padding up to 1000 samples/s: raw
    # capture stamps jitter 10-20ms backward ("Non-monotonous
    # DTS") and WebRTC receivers punish that with audible
    # dropouts. Output lands at 48kHz because Opus over RTP is
    # ALWAYS clocked at 48kHz (RFC 7587) -- encoding at the
    # capture rate puts receivers on a stretched timeline.
    stages.append("aresample=48000:async=1000")
    return ",".join(stages)


class MediaMTXStreamer:
    """
    Pushes one camera's frames to a MediaMTX server through FFmpeg.

    Build it from config with ``MediaMTXStreamer.from_camera(camera, config.mediamtx,
    config.node)``. Frame size may be left unset: it is then taken from the first
    frame passed to ``update_frame`` (FFmpeg starts once it is known).
    """

    # How long FFmpeg must stay up after launch to count as started
    _start_check_s = 2.0

    def __init__(self, host, rtsp_port, stream_name, fps=None,
                 frame_width=None, frame_height=None, bitrate="1500k",
                 debug_log_interval=60.0, encoder_preset="ultrafast",
                 encoder_codec="libx264", stream_queue_size=2,
                 on_demand=False, demand_poll_interval=3.0,
                 demand_grace_period=10.0, demand_timeout=5.0, demand_check=None,
                 webrtc_port=8889, rtsp_url=None, webrtc_url=None, demand_url=None,
                 audio_input=None, audio_device=None, audio_url="", audio_transport="tcp",
                 audio_rate=16000, audio_channels=2, audio_channel=0,
                 audio_highpass_hz=AUTO, audio_gain_db=AUTO, audio_limiter=0.97,
                 audio_bitrate="24k", audio_retry_backoff=60.0,
                 audio_retry_backoff_max=900.0, audio_reset_after=60.0):
        self.host = host
        self.rtsp_port = rtsp_port
        self.webrtc_port = webrtc_port
        self.stream_name = stream_name
        self.fps = fps or StreamConfig.DEFAULT_FPS
        # None = adopt the size of the first frame pushed
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.bitrate = format_bitrate(parse_bitrate(bitrate))

        # Encoder settings for resource optimization
        self.encoder_preset = encoder_preset  # ultrafast, fast, medium, etc.
        self.encoder_codec = encoder_codec  # libx264, 'auto' (detect hardware), or an FFmpeg encoder name
        self.stream_queue_size = stream_queue_size  # Async queue depth

        # On-demand publishing: gate the FFmpeg push on viewer demand
        self.on_demand = on_demand
        self.demand_poll_interval = demand_poll_interval
        self.demand_grace_period = demand_grace_period
        self.demand_timeout = demand_timeout
        self.demand_check = demand_check  # optional Callable[[], bool] override

        # Optional live audio track, captured by FFmpeg itself. 'alsa' = a
        # local mic through the sound system: audio_device should be a
        # dsnoop/plug PCM when anything else (e.g. a recorder service)
        # captures the same mic -- a bare hw: device admits only one client.
        # 'camera' = the audio track of the camera's own stream (audio_url).
        # Opus is used on the wire: it is the one codec the WebRTC playback
        # path accepts, and mediamtx passes it through to HLS.
        if audio_input is None:
            audio_input = "camera" if audio_url else ("alsa" if audio_device else None)
        if audio_input is not None and audio_input not in AUDIO_INPUT_TYPES:
            raise ValueError(f"audio_input must be one of {sorted(AUDIO_INPUT_TYPES)} or None, got {audio_input!r}")
        self.audio_input = audio_input
        self.audio_device = audio_device or None
        self.audio_url = audio_url or ""
        self.audio_transport = audio_transport
        self.audio_rate = audio_rate
        self.audio_channels = audio_channels
        self.audio_channel = audio_channel  # 0 = left (INMP441 L/R->GND)
        defaults = AUDIO_INPUT_TYPES[audio_input].filter_defaults if audio_input else {}
        self.audio_highpass_hz = defaults.get("highpass_hz") if audio_highpass_hz == AUTO else audio_highpass_hz
        self.audio_gain_db = defaults.get("gain_db", 0.0) if audio_gain_db == AUTO else (audio_gain_db or 0.0)
        self.audio_limiter = audio_limiter
        self.audio_bitrate = format_bitrate(parse_bitrate(audio_bitrate))

        # Audio is best-effort, video is not: when FFmpeg can't open (or
        # loses) the audio input, the stream goes on video-only and audio is
        # retried at a later fresh start, on this backoff. A run that keeps
        # audio for audio_reset_after seconds proves the input healthy and
        # resets it -- surviving the start check alone does not, since an
        # RTSP audio open can hang for up to its 20s timeout first.
        self.audio_reset_after = audio_reset_after
        self._audio_backoff = _Backoff(audio_retry_backoff, audio_retry_backoff_max, jitter=0)
        self._audio_retry_at = 0.0
        self._audio_since = None  # start of the current run with audio
        self._audio_active = False
        self._audio_error = None

        # Setup logging using central logger
        self.logger = get_logger(self)

        # The one URL builder shared with Config.stream_urls()
        self.urls = build_stream_urls(host, stream_name, rtsp_port=rtsp_port, webrtc_port=webrtc_port,
                                      rtsp_url=rtsp_url, webrtc_url=webrtc_url, demand_url=demand_url)
        self.rtsp_url = self.urls.rtsp
        self._demand_url = self.urls.demand

        # Demand supervisor state
        self._demand_thread = None
        self._demand_stop_event = threading.Event()
        self._demand_off_since = None
        self._demand_client = None

        # Process management
        self.ffmpeg_process = None
        self.ffmpeg_log_file = None  # Track log file for cleanup
        self.is_streaming = False
        self._start_pending = False  # start requested before the frame size was known
        self.frame_lock = Lock()
        self.restart_count = 0
        self.last_restart_time = 0
        
        # Async frame queue to decouple encoding from processing
        self._frame_queue = None
        self._writer_thread = None
        self._writer_stop_event = threading.Event()
        
        # Performance tracking for debug logging
        self.last_debug_log_time = 0
        self.debug_log_interval = debug_log_interval
        self.frame_count = 0
        self.frames_since_last_log = 0
        self.last_frame_time = 0
        
        # Performance metrics
        self.frame_write_times = deque(maxlen=100)  # Last 100 frame write times
        self.frame_resize_times = deque(maxlen=100)  # Last 100 resize times
        self.network_ping_times = deque(maxlen=20)  # Last 20 ping times
        self.dropped_frames = 0
        self.buffer_full_count = 0
        self.last_network_check = 0
        self.network_check_interval = 30.0  # Optimized: check network every 30s (was 10s)
        self.frame_queue_size = 0
        self.max_frame_queue_size = 0
        
        # Timing windows for analysis
        self.processing_bottleneck_threshold = 0.020  # 20ms threshold
        self.network_bottleneck_threshold = 0.100  # 100ms threshold
        
        # Create logs directory once at init (not per start_streaming call)
        self.logs_dir = os.path.join(os.getcwd(), '.core-streamer-logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        safe_name = re.sub(r'[^A-Za-z0-9_.-]', '_', str(stream_name))
        self._ffmpeg_log_path = os.path.join(self.logs_dir, f'ffmpeg_{safe_name}.log')
        
        # Resolve the encoder: 'auto' detects hardware, else the named encoder
        self.detected_encoder = self._resolve_encoder(encoder_codec)
        
        self.logger.info(f"[Stream] Initialized streamer {self.stream_name}: {self.rtsp_url}")
        self.logger.info(f"[Stream] Using encoder: {self.detected_encoder}")

    def _resolve_encoder(self, codec):
        codec = (codec or "libx264").strip()
        if codec.lower() == "auto":
            return self._detect_best_encoder()
        if codec.lower() == "copy":
            # Frames arrive as raw BGR: there is nothing to copy
            self.logger.warning("[Stream] encoder codec 'copy' is not possible for raw frames; using libx264")
            return "libx264"
        return codec
    
    def _detect_best_encoder(self):
        """Detect best available hardware encoder, fallback to software.
        Priority: NVIDIA > Intel QSV > AMD VAAPI > ARM V4L2M2M > Software
        """
        hw_encoders = ['h264_nvenc', 'h264_qsv', 'h264_vaapi', 'h264_v4l2m2m']
        
        for encoder in hw_encoders:
            if self._test_encoder(encoder):
                self.logger.info(f"[Stream] Hardware encoder detected: {encoder}")
                return encoder
        
        self.logger.info(f"[Stream] No hardware encoder detected, using software: libx264")
        return 'libx264'
    
    def _test_encoder(self, encoder):
        """Quick test if FFmpeg encoder is available."""
        try:
            result = subprocess.run(
                ['ffmpeg', '-hide_banner', '-encoders'],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                timeout=2,
                text=True
            )
            return encoder in result.stdout
        except:
            return False
    
    def _map_preset_to_encoder(self, preset, encoder):
        """Map libx264-style presets to hardware encoder equivalents.
        
        Args:
            preset: libx264-style preset (ultrafast, fast, medium, slow, etc.)
            encoder: target encoder (h264_nvenc, h264_qsv, h264_vaapi, libx264, etc.)
        
        Returns:
            str: preset compatible with the target encoder
        """
        # NVIDIA h264_nvenc: supports 'default', 'fast', 'medium', 'slow'
        if 'nvenc' in encoder:
            preset_map = {
                'ultrafast': 'fast',
                'fast': 'fast',
                'medium': 'medium',
                'slow': 'slow',
                'slower': 'slow',
                'veryslow': 'slow',
            }
            return preset_map.get(preset, 'fast')
        
        # Intel QSV: supports 'veryfast', 'fast', 'balanced', 'slow', 'veryslow'
        elif 'qsv' in encoder:
            preset_map = {
                'ultrafast': 'veryfast',
                'fast': 'fast',
                'medium': 'balanced',
                'slow': 'slow',
                'slower': 'veryslow',
                'veryslow': 'veryslow',
            }
            return preset_map.get(preset, 'fast')
        
        # AMD VAAPI: supports 'default', 'fast', 'medium', 'slow' (driver dependent)
        elif 'vaapi' in encoder:
            preset_map = {
                'ultrafast': 'fast',
                'fast': 'fast',
                'medium': 'medium',
                'slow': 'slow',
                'slower': 'slow',
                'veryslow': 'slow',
            }
            return preset_map.get(preset, 'fast')
        
        # libx264/libx265 software encoders: native support for all presets
        else:
            return preset
    
    def start_streaming(self):
        """Start streaming. In on-demand mode, starts the demand supervisor
        (FFmpeg launches only when a viewer is watching); otherwise starts
        FFmpeg immediately -- or, with the frame size still unknown, on the
        first frame passed to update_frame (True is returned)."""
        if self.on_demand:
            return self._start_demand_supervisor()
        return self._ensure_ffmpeg_started() or self._start_pending

    def stop_streaming(self):
        """Stop the streaming process (and the demand supervisor, if any)."""
        self._start_pending = False
        if self.on_demand:
            self._stop_demand_supervisor()
        self._ensure_ffmpeg_stopped()

    @property
    def _audio_configured(self):
        if self.audio_input == 'camera':
            return bool(self.audio_url)
        return self.audio_input == 'alsa' and bool(self.audio_device)

    @property
    def _size_known(self):
        return bool(self.frame_width and self.frame_height)

    @property
    def audio_active(self):
        """True while the running stream carries the audio track."""
        return self._audio_active

    @property
    def audio_error(self):
        """Short reason for the last audio failure; None once audio is back."""
        return self._audio_error

    def _ensure_ffmpeg_started(self):
        """Start FFmpeg streaming process with diagnostics."""
        if self.is_streaming:
            return True
        if not self._size_known:
            if not self._start_pending:
                self.logger.info("[Stream] Frame size not set: FFmpeg starts on the first frame")
            self._start_pending = True
            return False
        
        current_time = time.time()
        
        # Prevent rapid restarts (minimum 5 seconds between restarts)
        if current_time - self.last_restart_time < 5.0:
            self.logger.warning("[Stream] Preventing rapid restart - waiting...")
            return False
        
        try:
            with_audio = self._audio_configured and current_time >= self._audio_retry_at
            if self._audio_configured and not with_audio:
                self.logger.debug(f"[Audio] Starting video-only, audio retry in "
                                  f"{self._audio_retry_at - current_time:.0f}s")

            error_log = self._spawn_ffmpeg(self._build_ffmpeg_cmd(with_audio), prime=with_audio)
            if error_log is not None and with_audio:
                # Audio is best-effort: go again without it, right away. The
                # failure is charged to audio only if video then works --
                # when both die (e.g. MediaMTX down) audio wasn't the cause.
                self.logger.error(f"[FFmpeg] Failed to start with audio, retrying video-only. "
                                  f"Log file: {self._ffmpeg_log_path}. Error output: {error_log[-500:]}")
                with_audio = False
                audio_log, error_log = error_log, self._spawn_ffmpeg(self._build_ffmpeg_cmd(False))
                if error_log is None:
                    self._note_audio_failure(self._audio_failure_line(audio_log)
                                             or self._last_line(audio_log))
            if error_log is not None:
                self.logger.error(f"[FFmpeg] Failed to start. Log file: {self._ffmpeg_log_path}")
                self.logger.error(f"[FFmpeg] Error output: {error_log[-500:]}")
                return False

            self.is_streaming = True
            self.restart_count += 1
            self.last_restart_time = current_time
            if with_audio:
                if self._audio_error:
                    self.logger.info("[Audio] Stream started with audio again")
                self._audio_error = None
                self._audio_since = time.time()
            self._audio_active = with_audio

            # Start async frame writer thread
            if self.stream_queue_size > 0:
                self._frame_queue = queue.Queue(maxsize=self.stream_queue_size)
                self._writer_stop_event.clear()
                self._writer_thread = threading.Thread(
                    target=self._frame_writer_loop,
                    name=f"MediaMTX_Writer_{self.stream_name}",
                    daemon=True
                )
                self._writer_thread.start()
                self.logger.info(f"[Stream] Async writer thread started (queue size: {self.stream_queue_size})")
            
            # Print stream information
            self._print_stream_info()
            
            self.logger.info(f"[Stream] Streaming started successfully (restart #{self.restart_count})")
            return True
                
        except Exception as e:
            self.logger.error(f"[Stream] Error starting stream: {e}")
            return False

    def _spawn_ffmpeg(self, ffmpeg_cmd, prime=False):
        """Launch FFmpeg and wait out the start check. Returns None while it
        runs, else the tail of its log (the dead process is dropped)."""
        self.ffmpeg_log_file = open(self._ffmpeg_log_path, 'w')
        self.ffmpeg_process = subprocess.Popen(
            ffmpeg_cmd,
            stdin=subprocess.PIPE,
            stdout=self.ffmpeg_log_file,
            stderr=subprocess.STDOUT,  # Redirect stderr to log file
            bufsize=0,
            preexec_fn=os.setsid if os.name != 'nt' else None
        )
        if prime:
            # FFmpeg (9 at least) reads the piped video before it opens the
            # audio input: with nothing on stdin a dead mic sails through the
            # check and only kills the stream once frames flow. One blank
            # frame makes it open -- and fail on -- the audio input now.
            threading.Thread(target=self._prime_ffmpeg, args=(self.ffmpeg_process,),
                             daemon=True).start()

        # Wait longer for FFmpeg to initialize
        time.sleep(self._start_check_s)
        if self.ffmpeg_process.poll() is None:
            return None

        self.ffmpeg_log_file.close()
        self.ffmpeg_log_file = None
        self.ffmpeg_process = None
        return self._read_log_tail()

    def _prime_ffmpeg(self, process):
        # Under frame_lock so the writer can't interleave a frame into this one
        with self.frame_lock:
            try:
                process.stdin.write(bytes(self.frame_width * self.frame_height * 3))
                process.stdin.flush()
            except (OSError, ValueError):
                pass  # FFmpeg already gone or stopped: the start check reports it

    def _read_log_tail(self, max_bytes=4096):
        try:
            with open(self._ffmpeg_log_path, 'rb') as f:
                f.seek(0, os.SEEK_END)
                f.seek(max(0, f.tell() - max_bytes))
                return f.read().decode('utf-8', 'replace')
        except OSError:
            return ""

    @staticmethod
    def _last_line(log):
        lines = [line.strip() for line in log.splitlines() if line.strip()]
        return lines[-1][:200] if lines else "FFmpeg exited"

    def _audio_failure_line(self, log):
        """The first FFmpeg log line blaming the audio input, shortened, or
        None. Never probes the device itself (that would race the dsnoop share
        and its other reader); FFmpeg's own failure is the signal. Input #1 is
        always the audio (#0 is the piped video): FFmpeg >= 6.1 tags its lines
        `[in#1`, older builds `[alsa @`, and every version names the input
        (device or URL) when it fails to open. Output errors (`[out#0`) never
        match, so a MediaMTX outage is not blamed on audio."""
        ident = self.audio_url if self.audio_input == 'camera' else self.audio_device
        for line in log.splitlines():
            if '[in#1' in line or '[alsa @' in line or (ident and ident in line):
                return re.sub(r'^\[[^\]]*\]\s*', '', line.strip())[:200]
        return None

    def _note_audio_failure(self, reason):
        """Charge a failure to the audio input: the next fresh start goes
        video-only until the backoff has elapsed."""
        self._end_audio_run()
        delay = self._audio_backoff.next()
        self._audio_retry_at = time.time() + delay
        # Goes to health reports: never carry RTSP credentials along
        reason = re.sub(r'://[^/@\s]+@', '://***@', reason)
        self._audio_error = reason
        self.logger.warning(f"[Audio] Audio input failed ({reason}); streaming video-only, "
                            f"retrying audio in {delay:.0f}s")

    def _end_audio_run(self):
        """Close the bookkeeping of the current run with audio. One that
        lasted audio_reset_after seconds proves the input healthy, so the
        retry backoff starts over from its first step."""
        if (self._audio_since is not None
                and time.time() - self._audio_since >= self.audio_reset_after):
            self._audio_backoff.reset()
        self._audio_since = None
        self._audio_active = False

    def _build_ffmpeg_cmd(self, audio_on):
        """FFmpeg argv for this stream, with or without the audio input.
        Video-only is exactly the command of a node with no audio configured."""
        # Build FFmpeg command with auto-detected encoder
        codec = self.detected_encoder
        
        # Base command
        ffmpeg_cmd = [
            'ffmpeg', '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-pix_fmt', 'bgr24',
            '-s', f'{self.frame_width}x{self.frame_height}',
            '-r', _num(self.fps),
        ]
        if audio_on:
            # Wallclock timestamps on the piped video so it shares a clock
            # with the realtime audio capture; without this the two inputs
            # drift whenever the delivered frame rate differs from -r.
            #
            # The small queue is deliberate: at 512 a not-quite-realtime
            # encoder hoarded 1.4GB of raw frames and the live stream
            # drifted MINUTES behind. 32 frames (~3s) bounds the worst
            # lag; past that the pipe blocks and the app-side queue
            # drops frames instead -- degraded fps beats growing delay.
            ffmpeg_cmd.extend([
                '-use_wallclock_as_timestamps', '1',
                '-thread_queue_size', '32',
            ])
        ffmpeg_cmd.extend(['-i', '-'])
        if audio_on and self.audio_input == 'camera':
            if self.audio_url.startswith('rtsp://'):
                ffmpeg_cmd.extend([
                    # SETUP only the audio track: without this the session
                    # also pulls the source's video just to discard it --
                    # wasted bandwidth and extra load on a fragile camera.
                    '-allowed_media_types', 'audio',
                ])
                if self.audio_transport:
                    ffmpeg_cmd.extend(['-rtsp_transport', self.audio_transport])
            ffmpeg_cmd.extend([
                # Fail fast instead of hanging: an overloaded camera can
                # sit on the DESCRIBE for minutes; better to exit and let
                # the demand supervisor retry than to block the publish.
                '-timeout', '20000000',
                '-thread_queue_size', '8192',
                '-use_wallclock_as_timestamps', '1',
                '-i', self.audio_url,
                '-map', '0:v:0', '-map', '1:a:0',
            ])
        elif audio_on:
            ffmpeg_cmd.extend([
                '-f', 'alsa',
                '-ac', str(self.audio_channels),
                '-ar', str(self.audio_rate),
                # Deep enough to ride out multi-second uplink stalls
                # without an ALSA xrun (1024 blocked and dropped audio
                # whenever the TCP push backpressured).
                '-thread_queue_size', '8192',
                # Same clock as the video input: with the two inputs on
                # different timelines (audio 0-based, video wallclock) the
                # muxed session's timestamp mapping sawtoothed and viewers
                # heard periodic dropouts; audio-only publishes were clean.
                '-use_wallclock_as_timestamps', '1',
                '-i', self.audio_device,
                '-map', '0:v:0', '-map', '1:a:0',
            ])
        ffmpeg_cmd.extend(['-c:v', codec])

        # Hardware-specific settings
        encoder_preset = self._map_preset_to_encoder(self.encoder_preset, codec)

        if 'nvenc' in codec:
            # NVIDIA: Use mapped preset (nvenc doesn't support -tune option)
            ffmpeg_cmd.extend(['-preset', encoder_preset])
        elif 'qsv' in codec or 'vaapi' in codec:
            # Intel/AMD: Use mapped preset
            ffmpeg_cmd.extend(['-preset', encoder_preset])
        elif codec == 'libx264':
            # Software: Full control with tuning
            ffmpeg_cmd.extend(['-preset', encoder_preset, '-tune', 'zerolatency'])

        # Common settings
        ffmpeg_cmd.extend([
            '-profile:v', 'main',
            '-pix_fmt', 'yuv420p',
            '-b:v', self.bitrate,
            '-g', _frames(self.fps),
        ])
        if audio_on:
            # VBV-cap the video so keyframes go out as a stream, not a
            # burst that floods the server's per-viewer WebRTC queue. A
            # full second of buffer: the half-second one visibly "breathed"
            # -- each keyframe starved the frames after it and quality
            # cycled bad-to-good on the GOP cadence.
            #
            # The -g here overrides the short one in the common settings
            # (last option wins): a keyframe every ~1s was the other half
            # of that quality cycle, spending a third of the bitrate on
            # I-frames. ~4s GOP is still a fast WebRTC join.
            ffmpeg_cmd.extend([
                '-maxrate', self.bitrate,
                '-bufsize', self.bitrate,
                '-g', _frames(self.fps * 4),
                # Explicit slice threads: with tune zerolatency x264 was
                # effectively single-threaded here and 720p encode ran at
                # 0.9x realtime -- the input backlog above was the result.
                '-threads', '4',
            ])
        if audio_on:
            # Stamp video frames with their real arrival time. The
            # rawvideo demuxer paces pts at the DECLARED -r regardless of
            # -use_wallclock_as_timestamps, so with fewer delivered than
            # declared fps the video timeline runs slow (9.5 real / 15
            # declared = 0.63x) -- and the server's WebRTC A/V sync then
            # culls the audio running "ahead" of that slow video clock
            # (measured: audio delivery ratio == fps ratio, at any
            # bitrate). RTSP readers do no track sync, hence unaffected.
            ffmpeg_cmd.extend([
                '-vf', 'setpts=(RTCTIME-RTCSTART)/(TB*1000000)',
            ])
            # Filter chain and its rationale: see build_audio_filter()
            afilter = build_audio_filter(
                self.audio_input, channel=self.audio_channel, highpass_hz=self.audio_highpass_hz,
                gain_db=self.audio_gain_db, limiter=self.audio_limiter,
            )
            ffmpeg_cmd.extend([
                '-af', afilter,
                '-c:a', 'libopus',
                '-b:a', self.audio_bitrate,
                '-application', 'voip',
                # Never let the muxer hold audio waiting to interleave
                # with late video frames: held audio leaves in ~1s TCP
                # bursts that the server's realtime WebRTC pacer sheds
                # (measured: 33 of 50 pkt/s delivered, rest concealed).
                # 100ms keeps the wire smooth for live viewers.
                '-max_interleave_delta', '100000',
            ])
        ffmpeg_cmd.extend([
            '-f', 'rtsp',
            '-rtsp_transport', 'tcp',
            '-loglevel', 'warning',
            self.rtsp_url
        ])
        return ffmpeg_cmd

    def _print_stream_info(self):
        """Print stream information."""
        self.logger.info(f"[Stream] RTSP URL: {self.rtsp_url}")
        self.logger.info(f"[Stream] WebRTC URL: {self.urls.webrtc}")
        self.logger.info(f"[Stream] Resolution: {self.frame_width}x{self.frame_height} @ {self.fps}fps, Bitrate: {self.bitrate}")
    
    def _ensure_ffmpeg_stopped(self):
        """Stop the FFmpeg process, reaping it even if is_streaming was
        already flipped off (e.g. after the process died)."""
        if not self.is_streaming and not self.ffmpeg_process and not self._writer_thread:
            return

        self.is_streaming = False
        self._end_audio_run()

        # Stop async writer thread
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_stop_event.set()
            self._writer_thread.join(timeout=2.0)
            self.logger.info("[Stream] Async writer thread stopped")
        self._writer_thread = None

        if self.ffmpeg_process:
            try:
                # Graceful shutdown
                if self.ffmpeg_process.stdin:
                    self.ffmpeg_process.stdin.close()
                
                # Send SIGTERM
                if os.name != 'nt':
                    os.killpg(os.getpgid(self.ffmpeg_process.pid), signal.SIGTERM)
                else:
                    self.ffmpeg_process.terminate()
                
                # Wait briefly for graceful exit
                try:
                    self.ffmpeg_process.wait(timeout=3.0)
                except subprocess.TimeoutExpired:
                    # Force kill if needed
                    if os.name != 'nt':
                        os.killpg(os.getpgid(self.ffmpeg_process.pid), signal.SIGKILL)
                    else:
                        self.ffmpeg_process.kill()
                
            except Exception as e:
                self.logger.warning(f"[Stream] Error during cleanup: {e}")
            
            self.ffmpeg_process = None
        
        # Close log file handle
        if self.ffmpeg_log_file:
            try:
                self.ffmpeg_log_file.close()
                self.ffmpeg_log_file = None
            except Exception as e:
                self.logger.warning(f"[Stream] Error closing log file: {e}")
        
        self.logger.info("[Stream] Streaming stopped")

    def _start_demand_supervisor(self):
        """Start the background thread that polls viewer demand and gates
        FFmpeg on it. Idempotent."""
        if self._demand_thread and self._demand_thread.is_alive():
            return True

        if self.demand_check is None and self._demand_client is None:
            from .api_client import ApiClient
            self._demand_client = ApiClient(
                base_url=None,
                timeout=int(max(1, round(self.demand_timeout))),
                max_retries=0,
                cache_enabled=False,
                max_workers=1,
            )

        self._demand_off_since = None
        self._demand_stop_event.clear()
        self._demand_thread = threading.Thread(
            target=self._demand_supervisor_loop,
            name=f"MediaMTX_Demand_{self.stream_name}",
            daemon=True
        )
        self._demand_thread.start()
        self.logger.info(f"[Stream] Demand supervisor started, polling {self._demand_url} "
                         f"every {self.demand_poll_interval}s (grace {self.demand_grace_period}s)")
        return True

    def _stop_demand_supervisor(self):
        """Stop the demand supervisor thread and its poll client."""
        if self._demand_thread and self._demand_thread.is_alive():
            self._demand_stop_event.set()
            self._demand_thread.join(timeout=self.demand_poll_interval + 2.0)
            self.logger.info("[Stream] Demand supervisor stopped")
        self._demand_thread = None
        if self._demand_client:
            self._demand_client.shutdown()
            self._demand_client = None

    def _demand_supervisor_loop(self):
        """Poll demand and start/stop FFmpeg accordingly. Fail-safe: an
        unknown poll result (None) leaves the current state alone."""
        while not self._demand_stop_event.is_set():
            demand = self._check_demand()

            if demand is True:
                self._demand_off_since = None
                if not self.is_streaming:
                    self.logger.info("[Stream] Viewer demand detected - starting stream")
                    self._ensure_ffmpeg_started()
            elif demand is False:
                if self.is_streaming:
                    now = time.time()
                    if self._demand_off_since is None:
                        self._demand_off_since = now
                    elif now - self._demand_off_since >= self.demand_grace_period:
                        self.logger.info("[Stream] No viewer demand - stopping stream")
                        self._ensure_ffmpeg_stopped()
                        self._demand_off_since = None
                else:
                    self._demand_off_since = None

            self._demand_stop_event.wait(self.demand_poll_interval)

    def _check_demand(self):
        """Return True (viewer demand), False (no demand), or None (unknown -
        poll failed or unrecognized body). Never raises."""
        if self.demand_check is not None:
            try:
                return bool(self.demand_check())
            except Exception as e:
                self.logger.warning(f"[Stream] demand_check raised: {e}")
                return None

        response = self._demand_client.get(self._demand_url)
        if not response.ok:
            return None
        body = response.text.strip().lower()
        if body in ("on", "1", "true", "yes"):
            return True
        if body in ("off", "0", "false", "no", ""):
            return False
        return None

    def update_frame(self, frame):
        """
        Send frame to stream with improved error handling and performance tracking.
        """
        # Size not configured: the first frame decides it, then a pending
        # always-on start goes ahead (on-demand waits for the supervisor)
        if not self._size_known:
            if frame is None:
                return False
            self.frame_height, self.frame_width = frame.shape[:2]
            self.logger.info(f"[Stream] Frame size from first frame: {self.frame_width}x{self.frame_height}")
            if self._start_pending and not self.on_demand:
                self._start_pending = False
                if not self._ensure_ffmpeg_started():
                    return False

        # Quick exit if not streaming
        if not self.is_streaming or not self.ffmpeg_process:
            return False
        
        # Optimize: call time.time() once and reuse
        frame_start_time = time.time()
        
        # Process health check with less frequent restarts
        if self.ffmpeg_process.poll() is not None:
            # Process died, but don't restart immediately
            self.logger.warning(f"[FFmpeg] Process died (exit code: {self.ffmpeg_process.returncode})")
            # An audio-input death sends the restart below (or the demand
            # supervisor's) video-only while the audio backoff runs
            if self._audio_active:
                reason = self._audio_failure_line(self._read_log_tail())
                if reason:
                    self._note_audio_failure(reason)
            self._ensure_ffmpeg_stopped()

            # In on-demand mode the supervisor re-launches on the next poll
            # if demand is still present
            if self.on_demand:
                return False

            # Only attempt restart if not too frequent
            if frame_start_time - self.last_restart_time > 10.0:  # 10 second cooldown
                self.logger.info("[Stream] Attempting to restart stream...")
                if self._ensure_ffmpeg_started():
                    return self.update_frame(frame)  # Retry once

            return False
        
        try:
            # Resize if needed
            if frame.shape[:2] != (self.frame_height, self.frame_width):
                resize_start_time = time.time()
                frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                resize_time = time.time() - resize_start_time
                self.frame_resize_times.append(resize_time)
            
            # Use async queue if enabled, else write synchronously
            if self._frame_queue:
                try:
                    # Non-blocking put: drop frame if queue full (backpressure)
                    self._frame_queue.put_nowait(frame)
                    self.frame_count += 1
                    self.frames_since_last_log += 1
                except:
                    self.dropped_frames += 1
                    return False
            else:
                # Synchronous write (fallback)
                with self.frame_lock:
                    write_start_time = time.time()
                    # Optimized: use memoryview to avoid copy for large frames
                    frame_data = memoryview(frame)
                    self.ffmpeg_process.stdin.write(frame_data)
                    self.ffmpeg_process.stdin.flush()
                    write_time = time.time() - write_start_time
                    self.frame_write_times.append(write_time)
                    
                    if write_time > self.processing_bottleneck_threshold:
                        self.buffer_full_count += 1
                    
                    self.frame_count += 1
                    self.frames_since_last_log += 1
            
            self.last_frame_time = frame_start_time
            
            # Periodic debug logging
            if frame_start_time - self.last_debug_log_time >= self.debug_log_interval:
                self._log_performance_debug()
            
            # Periodic network check
            if frame_start_time - self.last_network_check >= self.network_check_interval:
                self._check_network_health()
            
            return True
                
        except (BrokenPipeError, OSError) as e:
            self.logger.warning(f"[Stream] Pipe error: {e}")
            self.is_streaming = False
            self.dropped_frames += 1
            return False
        except Exception as e:
            self.logger.error(f"[Stream] Frame write error: {e}")
            self.dropped_frames += 1
            return False
    
    def _check_network_health(self):
        """Check network connectivity and latency to MediaMTX server."""
        self.last_network_check = time.time()
        
        try:
            # Simple TCP ping to MediaMTX server
            ping_start = time.time()
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(2.0)
            result = sock.connect_ex((self.host, self.rtsp_port))
            sock.close()
            
            if result == 0:
                ping_time = time.time() - ping_start
                self.network_ping_times.append(ping_time)
            else:
                self.network_ping_times.append(999.0)  # Connection failed
                
        except Exception as e:
            self.logger.warning(f"[Network] Health check failed: {e}")
            self.network_ping_times.append(999.0)
    
    def _get_system_metrics(self):
        """Get current system resource usage."""
        try:
            # Optimized: non-blocking CPU measurement (interval=None uses cached value)
            cpu_percent = psutil.cpu_percent(interval=None)
            
            # Memory usage
            memory = psutil.virtual_memory()
            memory_percent = memory.percent
            
            # FFmpeg process specific metrics
            ffmpeg_cpu = 0
            ffmpeg_memory = 0
            if self.ffmpeg_process:
                try:
                    process = psutil.Process(self.ffmpeg_process.pid)
                    ffmpeg_cpu = process.cpu_percent()
                    ffmpeg_memory = process.memory_percent()
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            return {
                'system_cpu': cpu_percent,
                'system_memory': memory_percent,
                'ffmpeg_cpu': ffmpeg_cpu,
                'ffmpeg_memory': ffmpeg_memory
            }
        except Exception:
            return {
                'system_cpu': 0,
                'system_memory': 0,
                'ffmpeg_cpu': 0,
                'ffmpeg_memory': 0
            }
    
    def _log_performance_debug(self):
        """Log performance debug information."""
        current_time = time.time()
        time_elapsed = current_time - self.last_debug_log_time
        
        # Calculate average FPS since last log
        avg_fps = self.frames_since_last_log / time_elapsed if time_elapsed > 0 else 0
        
        # Get process status
        process_status = "Running" if self.is_active() else "Dead/None"
        if self.ffmpeg_process:
            process_status += f" (PID: {self.ffmpeg_process.pid})"
        
        # Get system metrics
        system_metrics = self._get_system_metrics()
        
        # Calculate timing statistics
        if self.frame_write_times:
            avg_write = sum(self.frame_write_times) / len(self.frame_write_times)
            max_write = max(self.frame_write_times)
            self.logger.debug(f"[Performance] Frame write: avg={avg_write*1000:.1f}ms, max={max_write*1000:.1f}ms")
        
        if self.frame_resize_times:
            avg_resize = sum(self.frame_resize_times) / len(self.frame_resize_times)
            max_resize = max(self.frame_resize_times)
            self.logger.debug(f"[Performance] Frame resize: avg={avg_resize*1000:.1f}ms, max={max_resize*1000:.1f}ms")
        
        if self.network_ping_times:
            recent_pings = [p for p in self.network_ping_times if p < 999.0]
            if recent_pings:
                avg_ping = sum(recent_pings) / len(recent_pings)
                max_ping = max(recent_pings)
                self.logger.debug(f"[Performance] Network ping: avg={avg_ping*1000:.1f}ms, max={max_ping*1000:.1f}ms")
            else:
                self.logger.debug(f"[Performance] Network ping: CONNECTION FAILED")
        
        # Log performance summary
        self.logger.debug(f"[Performance] FPS: {avg_fps:.2f}/{self.fps} | Frames: {self.frame_count} | Dropped: {self.dropped_frames} | Process: {process_status}")
        self.logger.debug(f"[Performance] System: CPU={system_metrics['system_cpu']:.1f}%, RAM={system_metrics['system_memory']:.1f}%")
        self.logger.debug(f"[Performance] FFmpeg: CPU={system_metrics['ffmpeg_cpu']:.1f}%, RAM={system_metrics['ffmpeg_memory']:.1f}%")
        self.logger.debug(f"[Performance] Buffer issues: {self.buffer_full_count}")
        
        # Reset counters
        self.frames_since_last_log = 0
        self.last_debug_log_time = current_time
        self.buffer_full_count = 0  # Reset buffer count
        self.dropped_frames = 0     # Reset dropped frames count

    def _frame_writer_loop(self):
        """Async frame writer thread - decouples encoding from main processing."""
        self.logger.info("[Stream] Frame writer thread started")
        
        while not self._writer_stop_event.is_set():
            try:
                # Get frame from queue with timeout
                frame = self._frame_queue.get(timeout=0.1)
                
                # Write to FFmpeg stdin
                with self.frame_lock:
                    write_start_time = time.time()
                    # Optimized: use memoryview to avoid copy
                    frame_data = memoryview(frame)
                    self.ffmpeg_process.stdin.write(frame_data)
                    self.ffmpeg_process.stdin.flush()
                    write_time = time.time() - write_start_time
                    
                    self.frame_write_times.append(write_time)
                    if write_time > self.processing_bottleneck_threshold:
                        self.buffer_full_count += 1
                        
            except Exception:
                # Queue empty or write error - continue loop
                continue
        
        self.logger.info("[Stream] Frame writer thread stopped")
    
    def is_active(self):
        """Quick health check."""
        return (self.is_streaming and 
                self.ffmpeg_process and 
                self.ffmpeg_process.poll() is None)
    
    def get_rtsp_url(self):
        """Get the RTSP URL."""
        return self.rtsp_url
    
    def get_webrtc_url(self):
        """Get the WebRTC URL."""
        return self.urls.webrtc
    
    def get_stats(self):
        """Get comprehensive streaming statistics."""
        # Calculate current performance metrics
        avg_write_time = 0
        avg_resize_time = 0
        avg_ping_time = 0

        if self.frame_write_times:
            avg_write_time = sum(self.frame_write_times) / len(self.frame_write_times)

        if self.frame_resize_times:
            avg_resize_time = sum(self.frame_resize_times) / len(self.frame_resize_times)

        if self.network_ping_times:
            recent_pings = [p for p in self.network_ping_times if p < 999.0]
            if recent_pings:
                avg_ping_time = sum(recent_pings) / len(recent_pings)

        system_metrics = self._get_system_metrics()

        return {
            'is_streaming': self.is_streaming,
            'restart_count': self.restart_count,
            'rtsp_url': self.rtsp_url,
            'webrtc_url': self.get_webrtc_url(),
            'resolution': f"{self.frame_width}x{self.frame_height}",
            'fps': self.fps,
            'process_active': self.is_active(),
            'total_frames': self.frame_count,
            'dropped_frames': self.dropped_frames,
            'avg_write_time_ms': avg_write_time * 1000,
            'avg_resize_time_ms': avg_resize_time * 1000,
            'avg_ping_time_ms': avg_ping_time * 1000,
            'buffer_issues': self.buffer_full_count,
            'audio_active': self.audio_active,
            'audio_error': self.audio_error,
            'system_metrics': system_metrics
        }

    @classmethod
    def from_camera(cls, camera, server, node, **overrides):
        """
        Build the streamer of one camera from typed config.

        Args:
            camera: a ``CameraConfig`` with a ``stream`` block (``config.cameras["name"]``).
            server: the shared ``MediaMTXConfig`` (``config.mediamtx``).
            node: the ``NodeConfig`` (``config.node``) whose ``sn`` fills ``{sn}``.
            overrides: any __init__ kwarg (e.g. ``demand_check=``); wins over config.

        Size and fps come from ``stream`` or else the camera's ``capture``; an
        unknown size is taken from the first frame. ``stream.enabled`` is the
        caller's to check.
        """
        stream = camera.stream
        if stream is None:
            raise ValueError(f"camera {camera.name!r} has no stream block")
        if server is None or not server.host:
            raise ValueError("mediamtx.host is required to stream")
        width, height = camera.stream_size()
        kwargs = {
            "host": server.host,
            "rtsp_port": server.rtsp_port,
            "webrtc_port": server.webrtc_port,
            "rtsp_url": server.rtsp_url,
            "webrtc_url": server.webrtc_url,
            "demand_url": server.demand_url,
            "demand_poll_interval": server.demand.poll_interval,
            "demand_grace_period": server.demand.grace_period,
            "demand_timeout": server.demand.timeout,
            "stream_name": camera.stream_name(node.sn),
            "fps": camera.stream_fps(),
            "frame_width": width,
            "frame_height": height,
            "bitrate": format_bitrate(stream.bitrate),
            "on_demand": stream.on_demand,
            "debug_log_interval": stream.debug_log_interval,
            "encoder_preset": stream.encoder.preset,
            "encoder_codec": stream.encoder.codec,
            "stream_queue_size": stream.encoder.queue_size,
        }
        audio = stream.audio
        if audio is not None:
            resolved = audio.resolved_filter()
            kwargs.update({
                "audio_input": audio.input.type,
                "audio_highpass_hz": resolved.highpass_hz,
                "audio_gain_db": resolved.gain_db,
                "audio_limiter": resolved.limiter,
                "audio_bitrate": format_bitrate(audio.bitrate),
                "audio_retry_backoff": audio.retry.backoff,
                "audio_retry_backoff_max": audio.retry.backoff_max,
                "audio_reset_after": audio.retry.reset_after,
            })
            if audio.input.type == "alsa":
                kwargs.update({
                    "audio_device": audio.input.device,
                    "audio_rate": audio.input.rate,
                    "audio_channels": audio.input.channels,
                    "audio_channel": audio.input.channel,
                })
            else:
                # The camera's own source, never a second URL
                kwargs.update({
                    "audio_url": camera.source.target,
                    "audio_transport": getattr(camera.source, "transport", "tcp"),
                })
        kwargs.update(overrides)
        return cls(**kwargs)
