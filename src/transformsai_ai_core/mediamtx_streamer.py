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
from typing import List, Optional
from .central_logger import get_logger
from .cpu_affinity import set_thread_affinity, set_process_affinity

# --- Information About Script ---
__name__ = "Streamer for MediaMTX Server"
__version__ = "2.1.2" 
__author__ = "TransformsAI"

class MediaMTXStreamer:
    """
    MediaMTX streamer with improved diagnostics and stream information.
    """
    
    def __init__(self, mediamtx_ip, rtsp_port, camera_sn_id, fps=30, 
                 frame_width=1920, frame_height=1080, bitrate="1500k",
                 debug_log_interval=60.0, encoder_preset="ultrafast",
                 encoder_codec="copy", stream_queue_size=2, hw_encode=False,
                 cpu_affinity: Optional[List[int]] = None):
        self.mediamtx_ip = mediamtx_ip
        self.rtsp_port = rtsp_port
        self.camera_sn_id = camera_sn_id
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.bitrate = bitrate
        self.webrtc_port = 8889  # Default MediaMTX WebRTC port
        
        # Encoder settings for resource optimization
        self.encoder_preset = encoder_preset  # ultrafast, fast, medium, etc.
        self.encoder_codec = encoder_codec  # 'copy' to avoid re-encoding, or 'libx264'
        self.hw_encode = hw_encode  # Auto-detect hardware encoder if True
        self.stream_queue_size = stream_queue_size  # Async queue depth
        self.cpu_affinity = cpu_affinity  # Pin writer thread and FFmpeg to these cores (Linux only)
        
        # Setup logging using central logger
        self.logger = get_logger(self)
        
        # Build URLs using helper methods
        self.rtsp_url = self._build_rtsp_url()
        
        # Process management
        self.ffmpeg_process = None
        self.ffmpeg_log_file = None  # Track log file for cleanup
        self.is_streaming = False
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
        self.logs_dir = os.path.join(os.getcwd(), 'logs')
        os.makedirs(self.logs_dir, exist_ok=True)
        
        # Detect best encoder (hardware if available, software fallback)
        self.detected_encoder = self._detect_best_encoder() if hw_encode else 'libx264'
        
        self.logger.info(f"[Stream] Initialized streamer for camera {self.camera_sn_id}: {self.rtsp_url}")
        self.logger.info(f"[Stream] Using encoder: {self.detected_encoder}")
    
    def _is_ip_address(self, host):
        """Check if host is an IP address (IPv4 or IPv6)."""
        # Match IPv4 pattern
        ipv4_pattern = r'^(\d{1,3}\.){3}\d{1,3}$'
        # Match IPv6 pattern (simplified)
        ipv6_pattern = r'^([0-9a-fA-F]{0,4}:){2,7}[0-9a-fA-F]{0,4}$'
        return bool(re.match(ipv4_pattern, host) or re.match(ipv6_pattern, host))
    
    def _build_rtsp_url(self):
        """Build RTSP URL with port."""
        return f"rtsp://{self.mediamtx_ip}:{self.rtsp_port}/live/cam_sn_{self.camera_sn_id}"
    
    def _build_webrtc_url(self):
        """Build WebRTC URL with port only for IP addresses.
        Currently it assumes, if a hostname is used, HTTPS is preferred with no port.
        """
        if self._is_ip_address(self.mediamtx_ip):
            return f"http://{self.mediamtx_ip}:{self.webrtc_port}/live/cam_sn_{self.camera_sn_id}/"
        else:
            return f"https://{self.mediamtx_ip}/live/cam_sn_{self.camera_sn_id}/"
    
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
        """Start FFmpeg streaming process with diagnostics."""
        if self.is_streaming:
            return True
        
        current_time = time.time()
        
        # Prevent rapid restarts (minimum 5 seconds between restarts)
        if current_time - self.last_restart_time < 5.0:
            self.logger.warning("[Stream] Preventing rapid restart - waiting...")
            return False
        
        try:
            # Build FFmpeg command with auto-detected encoder
            codec = self.detected_encoder
            
            # Base command
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{self.frame_width}x{self.frame_height}',
                '-r', str(self.fps),
                '-i', '-',
                '-c:v', codec,
            ]
            
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
                '-g', str(self.fps),
                '-f', 'rtsp',
                '-rtsp_transport', 'tcp',
                '-loglevel', 'warning',
                self.rtsp_url
            ])
            
            # Create log file for FFmpeg output
            log_file_path = os.path.join(self.logs_dir, f'ffmpeg_cam_{self.camera_sn_id}.log')
            self.ffmpeg_log_file = open(log_file_path, 'w')
            
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=self.ffmpeg_log_file,
                stderr=subprocess.STDOUT,  # Redirect stderr to log file
                bufsize=0,
                preexec_fn=os.setsid if os.name != 'nt' else None
            )
            
            # Wait longer for FFmpeg to initialize
            time.sleep(2.0)
            
            if self.ffmpeg_process.poll() is None:
                self.is_streaming = True
                self.restart_count += 1
                self.last_restart_time = current_time

                # Pin FFmpeg process to requested cores (from parent, avoids preexec_fn conflict)
                if self.cpu_affinity:
                    set_process_affinity(self.ffmpeg_process.pid, self.cpu_affinity,
                                         f"FFmpeg_{self.camera_sn_id}")

                # Start async frame writer thread
                if self.stream_queue_size > 0:
                    self._frame_queue = queue.Queue(maxsize=self.stream_queue_size)
                    self._writer_stop_event.clear()
                    self._writer_thread = threading.Thread(
                        target=self._frame_writer_loop,
                        name=f"MediaMTX_Writer_{self.camera_sn_id}",
                        daemon=True
                    )
                    self._writer_thread.start()
                    self.logger.info(f"[Stream] Async writer thread started (queue size: {self.stream_queue_size})")
                
                # Print stream information
                self._print_stream_info()
                
                self.logger.info(f"[Stream] Streaming started successfully (restart #{self.restart_count})")
                return True
            else:
                # Process died immediately - check log
                if self.ffmpeg_log_file:
                    self.ffmpeg_log_file.close()
                    self.ffmpeg_log_file = None
                with open(log_file_path, 'r') as f:
                    error_log = f.read()
                self.logger.error(f"[FFmpeg] Failed to start. Log file: {log_file_path}")
                self.logger.error(f"[FFmpeg] Error output: {error_log[-500:]}")
                return False
                
        except Exception as e:
            self.logger.error(f"[Stream] Error starting stream: {e}")
            return False
    
    def _print_stream_info(self):
        """Print stream information."""
        self.logger.info(f"[Stream] RTSP URL: {self.rtsp_url}")
        self.logger.info(f"[Stream] WebRTC URL: {self._build_webrtc_url()}")
        self.logger.info(f"[Stream] Resolution: {self.frame_width}x{self.frame_height} @ {self.fps}fps, Bitrate: {self.bitrate}")
    
    def stop_streaming(self):
        """Stop the streaming process."""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        
        # Stop async writer thread
        if self._writer_thread and self._writer_thread.is_alive():
            self._writer_stop_event.set()
            self._writer_thread.join(timeout=2.0)
            self.logger.info("[Stream] Async writer thread stopped")
        
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
    
    def update_frame(self, frame):
        """
        Send frame to stream with improved error handling and performance tracking.
        """
        # Quick exit if not streaming
        if not self.is_streaming or not self.ffmpeg_process:
            return False
        
        # Optimize: call time.time() once and reuse
        frame_start_time = time.time()
        
        # Process health check with less frequent restarts
        if self.ffmpeg_process.poll() is not None:
            # Process died, but don't restart immediately
            self.logger.warning(f"[FFmpeg] Process died (exit code: {self.ffmpeg_process.returncode})")
            self.is_streaming = False
            self.stop_streaming()
            
            # Only attempt restart if not too frequent
            if frame_start_time - self.last_restart_time > 10.0:  # 10 second cooldown
                self.logger.info("[Stream] Attempting to restart stream...")
                if self.start_streaming():
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
            result = sock.connect_ex((self.mediamtx_ip, self.rtsp_port))
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
        # Pin this thread to the requested cores on first entry
        if self.cpu_affinity:
            set_thread_affinity(self.cpu_affinity, f"MediaMTX_Writer_{self.camera_sn_id}")

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
        return self._build_webrtc_url()
    
    def get_hls_url(self, hls_port=8889):
        """This is deprecated, will just return webrtc for legacy support."""
        return self.get_webrtc_url()
    
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
            'system_metrics': system_metrics
        }