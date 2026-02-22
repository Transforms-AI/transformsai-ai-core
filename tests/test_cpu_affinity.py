"""
Tests for CPU affinity utility and its integration with library components.

Run with:  python tests/test_cpu_affinity.py
"""

import os
import sys
import threading
import platform
import unittest
import time
from unittest.mock import patch, MagicMock, call

# Import the module under test
from transformsai_ai_core.cpu_affinity import (
    set_thread_affinity,
    set_process_affinity,
    get_current_affinity,
)

IS_LINUX = platform.system() == "Linux"


# =============================================================================
# cpu_affinity module tests
# =============================================================================

class TestGetCurrentAffinity(unittest.TestCase):
    def test_returns_set_or_none(self):
        result = get_current_affinity()
        if IS_LINUX:
            self.assertIsInstance(result, set)
            self.assertTrue(len(result) > 0)
        else:
            self.assertIsNone(result)


class TestSetThreadAffinity(unittest.TestCase):
    
    @unittest.skipUnless(IS_LINUX, "sched_setaffinity is Linux-only")
    def test_pin_to_single_core(self):
        """Pin calling thread to core 0, verify via get_current_affinity."""
        original = get_current_affinity()
        try:
            ok = set_thread_affinity([0], "test")
            self.assertTrue(ok)
            self.assertEqual(get_current_affinity(), {0})
        finally:
            # Restore
            if original:
                os.sched_setaffinity(0, original)

    @unittest.skipUnless(IS_LINUX, "sched_setaffinity is Linux-only")
    def test_pin_to_multiple_cores(self):
        cpu_count = os.cpu_count() or 1
        cores = list(range(min(2, cpu_count)))
        original = get_current_affinity()
        try:
            ok = set_thread_affinity(cores, "test_multi")
            self.assertTrue(ok)
            self.assertEqual(get_current_affinity(), set(cores))
        finally:
            if original:
                os.sched_setaffinity(0, original)

    def test_empty_cores_returns_false(self):
        self.assertFalse(set_thread_affinity([], "test"))

    def test_invalid_cores_returns_false(self):
        cpu_count = os.cpu_count() or 1
        self.assertFalse(set_thread_affinity([cpu_count + 100], "test"))

    def test_noop_on_unsupported_platform(self):
        """On any platform: patching away sched_setaffinity must return False gracefully."""
        with patch("transformsai_ai_core.cpu_affinity.os.sched_setaffinity",
                   side_effect=AttributeError):
            result = set_thread_affinity([0], "test")
        self.assertFalse(result)

    @unittest.skipUnless(IS_LINUX, "sched_setaffinity is Linux-only")
    def test_pinning_runs_in_child_thread(self):
        """Verify the child thread is pinned without affecting the current thread."""
        target_core = [0]
        result = {}

        def worker():
            set_thread_affinity(target_core, "child")
            result["affinity"] = get_current_affinity()

        t = threading.Thread(target=worker)
        t.start()
        t.join()
        
        self.assertEqual(result["affinity"], set(target_core))
        
        # Main thread affinity should be unchanged (checking current thread)
        main_affinity = get_current_affinity()
        self.assertTrue(len(main_affinity) > 0)


class TestSetProcessAffinity(unittest.TestCase):
    def test_empty_cores_returns_false(self):
        self.assertFalse(set_process_affinity(os.getpid(), [], "test"))

    def test_invalid_cores_returns_false(self):
        cpu_count = os.cpu_count() or 1
        self.assertFalse(set_process_affinity(os.getpid(), [cpu_count + 100], "test"))

    @unittest.skipUnless(IS_LINUX, "sched_setaffinity is Linux-only")
    def test_pin_current_process(self):
        original = get_current_affinity()
        try:
            ok = set_process_affinity(os.getpid(), [0], "test_proc")
            self.assertTrue(ok)
        finally:
            if original:
                os.sched_setaffinity(0, original)

    def test_noop_on_unsupported_platform(self):
        with patch("transformsai_ai_core.cpu_affinity.os.sched_setaffinity",
                   side_effect=AttributeError):
            result = set_process_affinity(os.getpid(), [0], "test")
        self.assertFalse(result)


# =============================================================================
# VideoCaptureAsync integration
# =============================================================================

class TestVideoCaptureAsyncAffinity(unittest.TestCase):
    """Verify cpu_affinity is stored and applied when _update() runs."""

    def test_constructor_stores_affinity(self):
        """cpu_affinity kwarg must be stored without actually opening a device."""
        with patch("transformsai_ai_core.video_capture.cv2.VideoCapture") as mock_cap:
            mock_cap.return_value.isOpened.return_value = True
            mock_cap.return_value.get.return_value = 30.0
            from transformsai_ai_core.video_capture import VideoCaptureAsync
            cap = VideoCaptureAsync(src=0, cpu_affinity=[0, 1])
            self.assertEqual(cap.cpu_affinity, [0, 1])
            cap.release()

    def test_none_affinity_is_default(self):
        with patch("transformsai_ai_core.video_capture.cv2.VideoCapture") as mock_cap:
            mock_cap.return_value.isOpened.return_value = True
            mock_cap.return_value.get.return_value = 30.0
            from transformsai_ai_core.video_capture import VideoCaptureAsync
            cap = VideoCaptureAsync(src=0)
            self.assertIsNone(cap.cpu_affinity)
            cap.release()

    def test_set_thread_affinity_called_when_thread_starts(self):
        """set_thread_affinity must be called once inside _update when affinity is set."""
        with patch("transformsai_ai_core.video_capture.cv2.VideoCapture") as mock_cap, \
             patch("transformsai_ai_core.video_capture.set_thread_affinity") as mock_pin:
            
            mock_cap.return_value.isOpened.return_value = True
            mock_cap.return_value.get.return_value = 30.0
            mock_cap.return_value.read.return_value = (False, None)
            
            from transformsai_ai_core.video_capture import VideoCaptureAsync
            cap = VideoCaptureAsync(src=0, cpu_affinity=[0, 1], auto_restart_on_fail=False)
            cap.start()
            
            # Give the thread a moment to enter _update
            time.sleep(0.1)
            cap.release()
            
            mock_pin.assert_called_once_with([0, 1], "VideoCaptureAsync_0")


# =============================================================================
# MediaMTXStreamer integration
# =============================================================================

class TestMediaMTXStreamerAffinity(unittest.TestCase):
    """Verify cpu_affinity is stored and applied to writer thread and FFmpeg process."""

    def _make_streamer(self, cpu_affinity=None):
        from transformsai_ai_core.mediamtx_streamer import MediaMTXStreamer
        with patch("transformsai_ai_core.mediamtx_streamer.MediaMTXStreamer._detect_best_encoder",
                   return_value="libx264"):
            return MediaMTXStreamer(
                mediamtx_ip="127.0.0.1",
                rtsp_port=8554,
                camera_sn_id="test01",
                cpu_affinity=cpu_affinity,
            )

    def test_constructor_stores_affinity(self):
        streamer = self._make_streamer(cpu_affinity=[0, 1])
        self.assertEqual(streamer.cpu_affinity, [0, 1])

    def test_none_affinity_is_default(self):
        streamer = self._make_streamer()
        self.assertIsNone(streamer.cpu_affinity)

    def test_writer_thread_pins_on_start(self):
        """set_thread_affinity called once inside _frame_writer_loop."""
        streamer = self._make_streamer(cpu_affinity=[0, 1])
        
        with patch("transformsai_ai_core.mediamtx_streamer.set_thread_affinity") as mock_pin, \
             patch("transformsai_ai_core.mediamtx_streamer.set_process_affinity"), \
             patch("transformsai_ai_core.mediamtx_streamer.subprocess.Popen") as mock_popen, \
             patch("builtins.open", MagicMock()), \
             patch("transformsai_ai_core.mediamtx_streamer.time.sleep"):
            
            proc = MagicMock()
            proc.poll.return_value = None
            proc.pid = 12345
            mock_popen.return_value = proc
            
            streamer.start_streaming()
            time.sleep(0.1)
            streamer.stop_streaming()
            
            mock_pin.assert_called_with([0, 1], "MediaMTX_Writer_test01")

    def test_ffmpeg_process_pinned_after_popen(self):
        """set_process_affinity called with FFmpeg PID after successful Popen."""
        streamer = self._make_streamer(cpu_affinity=[0, 1])
        
        with patch("transformsai_ai_core.mediamtx_streamer.set_thread_affinity"), \
             patch("transformsai_ai_core.mediamtx_streamer.set_process_affinity") as mock_proc_pin, \
             patch("transformsai_ai_core.mediamtx_streamer.subprocess.Popen") as mock_popen, \
             patch("builtins.open", MagicMock()), \
             patch("transformsai_ai_core.mediamtx_streamer.time.sleep"):
            
            proc = MagicMock()
            proc.poll.return_value = None
            proc.pid = 99999
            mock_popen.return_value = proc
            
            streamer.start_streaming()
            streamer.stop_streaming()
            
            mock_proc_pin.assert_called_once_with(99999, [0, 1], "FFmpeg_test01")


# =============================================================================
# DataUploader integration
# =============================================================================

class TestDataUploaderAffinity(unittest.TestCase):
    """Verify cpu_affinity wires into ThreadPoolExecutor initializer."""

    def test_executor_created_with_initializer_when_affinity_set(self):
        with patch("transformsai_ai_core.datasend.ThreadPoolExecutor") as mock_tpe:
            from transformsai_ai_core.datasend import DataUploader
            DataUploader(cpu_affinity=[2, 3])
            
            _, kwargs = mock_tpe.call_args
            self.assertIs(kwargs["initializer"], set_thread_affinity)
            self.assertEqual(kwargs["initargs"], ([2, 3], "DataUploader"))

    def test_executor_created_without_initializer_when_no_affinity(self):
        with patch("transformsai_ai_core.datasend.ThreadPoolExecutor") as mock_tpe:
            from transformsai_ai_core.datasend import DataUploader
            DataUploader()
            
            _, kwargs = mock_tpe.call_args
            self.assertIsNone(kwargs.get("initializer"))


# =============================================================================
# Config schema integration
# =============================================================================

class TestCpuAffinityConfig(unittest.TestCase):
    def test_default_values(self):
        from transformsai_ai_core.config_schema import CpuAffinityConfig
        cfg = CpuAffinityConfig()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.streaming_cores, [])
        self.assertEqual(cfg.inference_cores, [])

    def test_present_in_advanced_config(self):
        from transformsai_ai_core.config_schema import AdvancedConfig
        adv = AdvancedConfig()
        self.assertTrue(hasattr(adv, "cpu_affinity"))

    def test_parses_from_dict(self):
        from transformsai_ai_core.config_schema import CpuAffinityConfig
        cfg = CpuAffinityConfig(enabled=True, streaming_cores=[0, 1], inference_cores=[2, 3, 4, 5])
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.streaming_cores, [0, 1])
        self.assertEqual(cfg.inference_cores, [2, 3, 4, 5])


if __name__ == "__main__":
    unittest.main(verbosity=2)