import cv2
import subprocess
import time
import threading
import signal
import os
import psutil
import socket
from threading import Lock
from collections import deque
from libraries.central_logger import create_class_logger

# --- Information About Script ---
__name__ = "Streamer for MediaMTX Server"
__version__ = "2.1.2" 
__author__ = "TransformsAI"

class MediaMTXStreamer:
    """
    MediaMTX streamer with improved diagnostics and stream information.
    """
    
    def __init__(self, mediamtx_ip, rtsp_port, camera_sn_id, fps=30, 
                 frame_width=1920, frame_height=1080, bitrate="1500k"):
        self.mediamtx_ip = mediamtx_ip
        self.rtsp_port = rtsp_port
        self.camera_sn_id = camera_sn_id
        self.fps = fps
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.bitrate = bitrate
        
        # Setup logging using central logger
        self.logger = create_class_logger(self)
        
        # Build RTSP URL
        self.rtsp_url = f"rtsp://{self.mediamtx_ip}:{self.rtsp_port}/live/cam_sn_{self.camera_sn_id}"
        
        # Process management
        self.ffmpeg_process = None
        self.is_streaming = False
        self.frame_lock = Lock()
        self.restart_count = 0
        self.last_restart_time = 0
        
        # Performance tracking for debug logging
        self.last_debug_log_time = 0
        self.debug_log_interval = 30.0  # Log performance every 30 seconds
        self.frame_count = 0
        self.frames_since_last_log = 0
        self.last_frame_time = 0
        
        # Enhanced performance metrics
        self.frame_write_times = deque(maxlen=100)  # Last 100 frame write times
        self.frame_resize_times = deque(maxlen=100)  # Last 100 resize times
        self.network_ping_times = deque(maxlen=20)  # Last 20 ping times
        self.dropped_frames = 0
        self.buffer_full_count = 0
        self.last_network_check = 0
        self.network_check_interval = 10.0  # Check network every 10 seconds
        self.frame_queue_size = 0
        self.max_frame_queue_size = 0
        
        # Timing windows for analysis
        self.processing_bottleneck_threshold = 0.020  # 20ms threshold
        self.network_bottleneck_threshold = 0.100  # 100ms threshold
        
        self.logger.info(f"Streamer initialized: {self.rtsp_url}")
    
    def start_streaming(self):
        """Start FFmpeg streaming process with diagnostics."""
        if self.is_streaming:
            return True
        
        current_time = time.time()
        
        # Prevent rapid restarts (minimum 5 seconds between restarts)
        if current_time - self.last_restart_time < 5.0:
            self.logger.warning("Preventing rapid restart - waiting...")
            return False
        
        try:
            # Optimized FFmpeg command with better diagnostics
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-f', 'rawvideo',
                '-vcodec', 'rawvideo',
                '-pix_fmt', 'bgr24',
                '-s', f'{self.frame_width}x{self.frame_height}',
                '-r', str(self.fps),
                '-i', '-',
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-tune', 'zerolatency',
                '-profile:v', 'main',
                '-pix_fmt', 'yuv420p',
                '-b:v', self.bitrate,
                '-g', str(self.fps),
                '-f', 'rtsp',
                '-rtsp_transport', 'tcp',
                '-loglevel', 'warning',
                self.rtsp_url
            ]
            
            # Create logs directory if it doesn't exist and set log file path
            logs_dir = os.path.join(os.getcwd(), 'logs')
            os.makedirs(logs_dir, exist_ok=True)
            log_file_path = os.path.join(logs_dir, f'ffmpeg_cam_{self.camera_sn_id}.log')
            
            # Create log file for FFmpeg output
            log_file = open(log_file_path, 'w')
            
            self.ffmpeg_process = subprocess.Popen(
                ffmpeg_cmd,
                stdin=subprocess.PIPE,
                stdout=log_file,
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
                
                # Print comprehensive stream information
                self._print_stream_info()
                
                self.logger.info(f"✅ Streaming started successfully (restart #{self.restart_count})")
                return True
            else:
                # Process died immediately - check log
                log_file.close()
                with open(log_file_path, 'r') as f:
                    error_log = f.read()
                self.logger.error(f"❌ FFmpeg failed to start. Log: {error_log[-500:]}")
                return False
                
        except Exception as e:
            self.logger.error(f"❌ Error starting stream: {e}")
            return False
    
    def _print_stream_info(self):
        """Print comprehensive information about the stream."""
        # Get log file path
        logs_dir = os.path.join(os.getcwd(), 'logs')
        log_file_path = os.path.join(logs_dir, f'ffmpeg_cam_{self.camera_sn_id}.log')
        
        self.logger.info("📺 STREAM INFORMATION:")
        self.logger.info(f"   🎥 RTSP URL: {self.rtsp_url}")
        self.logger.info(f"   🔗 VLC Command: vlc {self.rtsp_url}")
        self.logger.info(f"   🔗 FFplay Command: ffplay {self.rtsp_url}")
        self.logger.info(f"   🔗 HLS URL: {self.get_hls_url()}")
        self.logger.info(f"   ⚙️ Resolution: {self.frame_width}x{self.frame_height} @ {self.fps}fps")
        self.logger.info(f"   📊 Bitrate: {self.bitrate}")
        
        # Additional MediaMTX web interface info (if available)
        web_port = 8889  # Default MediaMTX web port
        self.logger.info(f"   🌐 MediaMTX Web UI: http://{self.mediamtx_ip}:{web_port}")
        self.logger.info(f"   📋 FFMPEG Serivce Log File: {log_file_path}")
    
    def stop_streaming(self):
        """Stop the streaming process."""
        if not self.is_streaming:
            return
        
        self.is_streaming = False
        
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
                self.logger.warning(f"Error during cleanup: {e}")
            
            self.ffmpeg_process = None
        
        self.logger.info("⏹️  Streaming stopped")
    
    def update_frame(self, frame):
        """
        Send frame to stream with improved error handling and performance tracking.
        """
        # Quick exit if not streaming
        if not self.is_streaming or not self.ffmpeg_process:
            return False
        
        frame_start_time = time.time()
        
        # Process health check with less frequent restarts
        if self.ffmpeg_process.poll() is not None:
            # Process died, but don't restart immediately
            self.logger.warning(f"⚠️  FFmpeg process died (exit code: {self.ffmpeg_process.returncode})")
            self.is_streaming = False
            self.stop_streaming()
            
            # Only attempt restart if not too frequent
            if frame_start_time - self.last_restart_time > 10.0:  # 10 second cooldown
                self.logger.info("🔄 Attempting to restart stream...")
                if self.start_streaming():
                    return self.update_frame(frame)  # Retry once
            
            return False
        
        try:
            with self.frame_lock:
                # Track resize timing
                resize_start_time = time.time()
                
                # Resize if needed
                if frame.shape[:2] != (self.frame_height, self.frame_width):
                    frame = cv2.resize(frame, (self.frame_width, self.frame_height))
                
                resize_time = time.time() - resize_start_time
                self.frame_resize_times.append(resize_time)
                
                # Track frame write timing
                write_start_time = time.time()
                
                # Write frame
                self.ffmpeg_process.stdin.write(frame.tobytes())
                self.ffmpeg_process.stdin.flush()
                
                write_time = time.time() - write_start_time
                self.frame_write_times.append(write_time)
                
                # Check for slow writes (potential bottleneck)
                if write_time > self.processing_bottleneck_threshold:
                    self.buffer_full_count += 1
                
                # Update frame counters
                self.frame_count += 1
                self.frames_since_last_log += 1
                self.last_frame_time = time.time()
                
                # Periodic debug logging
                current_time = time.time()
                if current_time - self.last_debug_log_time >= self.debug_log_interval:
                    self._log_performance_debug()
                
                # Periodic network check
                if current_time - self.last_network_check >= self.network_check_interval:
                    self._check_network_health()
                
                return True
                
        except (BrokenPipeError, OSError) as e:
            self.logger.warning(f"⚠️  Pipe error: {e}")
            self.is_streaming = False
            self.dropped_frames += 1
            return False
        except Exception as e:
            self.logger.error(f"❌ Frame write error: {e}")
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
            self.logger.warning(f"Network check failed: {e}")
            self.network_ping_times.append(999.0)
    
    def _get_system_metrics(self):
        """Get current system resource usage."""
        try:
            # CPU usage
            cpu_percent = psutil.cpu_percent(interval=0.1)
            
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
    
    def _analyze_bottlenecks(self):
        """Analyze performance data to identify bottlenecks."""
        bottlenecks = []
        
        # Analyze frame write times
        if self.frame_write_times:
            avg_write_time = sum(self.frame_write_times) / len(self.frame_write_times)
            max_write_time = max(self.frame_write_times)
            
            if avg_write_time > self.processing_bottleneck_threshold:
                bottlenecks.append(f"SLOW FRAME WRITES (avg: {avg_write_time*1000:.1f}ms)")
            
            if max_write_time > 0.100:  # 100ms
                bottlenecks.append(f"FRAME WRITE SPIKES (max: {max_write_time*1000:.1f}ms)")
        
        # Analyze resize times
        if self.frame_resize_times:
            avg_resize_time = sum(self.frame_resize_times) / len(self.frame_resize_times)
            max_resize_time = max(self.frame_resize_times)
            
            if avg_resize_time > 0.010:  # 10ms
                bottlenecks.append(f"SLOW FRAME RESIZE (avg: {avg_resize_time*1000:.1f}ms)")
        
        # Analyze network latency
        if self.network_ping_times:
            recent_pings = [p for p in self.network_ping_times if p < 999.0]
            if recent_pings:
                avg_ping = sum(recent_pings) / len(recent_pings)
                if avg_ping > self.network_bottleneck_threshold:
                    bottlenecks.append(f"HIGH NETWORK LATENCY (avg: {avg_ping*1000:.1f}ms)")
            
            # Check for connection failures
            failed_pings = len([p for p in self.network_ping_times if p >= 999.0])
            if failed_pings > 0:
                bottlenecks.append(f"NETWORK CONNECTION ISSUES ({failed_pings}/{len(self.network_ping_times)} failed)")
        
        # Check buffer issues
        if self.buffer_full_count > 10:
            bottlenecks.append(f"FFMPEG BUFFER ISSUES ({self.buffer_full_count} slow writes)")
        
        # Check dropped frames
        if self.dropped_frames > 0:
            bottlenecks.append(f"DROPPED FRAMES ({self.dropped_frames} total)")
        
        return bottlenecks
    
    def _log_performance_debug(self):
        """Log comprehensive debug performance information."""
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
        write_stats = ""
        resize_stats = ""
        network_stats = ""
        
        if self.frame_write_times:
            avg_write = sum(self.frame_write_times) / len(self.frame_write_times)
            max_write = max(self.frame_write_times)
            write_stats = f"avg: {avg_write*1000:.1f}ms, max: {max_write*1000:.1f}ms"
        
        if self.frame_resize_times:
            avg_resize = sum(self.frame_resize_times) / len(self.frame_resize_times)
            max_resize = max(self.frame_resize_times)
            resize_stats = f"avg: {avg_resize*1000:.1f}ms, max: {max_resize*1000:.1f}ms"
        
        if self.network_ping_times:
            recent_pings = [p for p in self.network_ping_times if p < 999.0]
            if recent_pings:
                avg_ping = sum(recent_pings) / len(recent_pings)
                max_ping = max(recent_pings)
                network_stats = f"avg: {avg_ping*1000:.1f}ms, max: {max_ping*1000:.1f}ms"
            else:
                network_stats = "CONNECTION FAILED"
        
        # Analyze bottlenecks
        bottlenecks = self._analyze_bottlenecks()
        
        # Log comprehensive debug information
        self.logger.debug("📊 DETAILED PERFORMANCE ANALYSIS:")
        self.logger.debug(f"   📈 Current FPS: {avg_fps:.2f} / {self.fps} (target)")
        self.logger.debug(f"   🎬 Total Frames: {self.frame_count} | Dropped: {self.dropped_frames}")
        self.logger.debug(f"   ⚡ Process Status: {process_status}")
        
        self.logger.debug("🔍 TIMING BREAKDOWN:")
        if write_stats:
            self.logger.debug(f"   ✍️  Frame Write: {write_stats}")
        if resize_stats:
            self.logger.debug(f"   🔄 Frame Resize: {resize_stats}")
        if network_stats:
            self.logger.debug(f"   🌐 Network Ping: {network_stats}")
        
        self.logger.debug("💻 SYSTEM RESOURCES:")
        self.logger.debug(f"   🖥️  System CPU: {system_metrics['system_cpu']:.1f}%")
        self.logger.debug(f"   🧠 System RAM: {system_metrics['system_memory']:.1f}%")
        self.logger.debug(f"   🎞️  FFmpeg CPU: {system_metrics['ffmpeg_cpu']:.1f}%")
        self.logger.debug(f"   📊 FFmpeg RAM: {system_metrics['ffmpeg_memory']:.1f}%")
        
        if bottlenecks:
            self.logger.warning("⚠️  IDENTIFIED BOTTLENECKS:")
            for bottleneck in bottlenecks:
                self.logger.warning(f"   🔴 {bottleneck}")
        else:
            self.logger.debug("   ✅ No major bottlenecks detected")
        
        self.logger.debug(f"   📡 RTSP URL: {self.rtsp_url}")
        self.logger.debug(f"   🔢 Buffer Issues: {self.buffer_full_count}")
        
        # Reset counters
        self.frames_since_last_log = 0
        self.last_debug_log_time = current_time
        self.buffer_full_count = 0  # Reset buffer count
        self.dropped_frames = 0     # Reset dropped frames count

    def is_active(self):
        """Quick health check."""
        return (self.is_streaming and 
                self.ffmpeg_process and 
                self.ffmpeg_process.poll() is None)
    
    def get_rtsp_url(self):
        """Get the RTSP URL."""
        return self.rtsp_url
    
    def get_hls_url(self, hls_port=8889):
        """Get the HLS URL for the stream (served by MediaMTX)."""
        return f"http://{self.mediamtx_ip}:{hls_port}/live/cam_sn_{self.camera_sn_id}/"
    
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
        bottlenecks = self._analyze_bottlenecks()
        
        return {
            'is_streaming': self.is_streaming,
            'restart_count': self.restart_count,
            'rtsp_url': self.rtsp_url,
            'resolution': f"{self.frame_width}x{self.frame_height}",
            'fps': self.fps,
            'process_active': self.is_active(),
            'total_frames': self.frame_count,
            'dropped_frames': self.dropped_frames,
            'avg_write_time_ms': avg_write_time * 1000,
            'avg_resize_time_ms': avg_resize_time * 1000,
            'avg_ping_time_ms': avg_ping_time * 1000,
            'buffer_issues': self.buffer_full_count,
            'system_metrics': system_metrics,
            'bottlenecks': bottlenecks
        }