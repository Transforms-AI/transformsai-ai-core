"""
Tests for VideoCaptureAsync's frame_hook and from_camera.

A hook runs on the capture thread on every published frame. It may transform
the frame; if it raises, the frame is published unhooked and capture goes on.

Run: uv run python -m unittest tests.test_capture_frame_hook
"""

import os
import shutil
import tempfile
import time
import unittest
from unittest import mock

import cv2
import numpy as np

from transformsai_ai_core.config import load_config
from transformsai_ai_core.video_capture import VideoCaptureAsync


def _wait_until(pred, timeout=5.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return pred()


def tearDownModule():
    from loguru import logger
    logger.complete()


class FrameHookTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.mkdtemp(prefix="frame-hook-")
        cls.video = os.path.join(cls.tmp, "clip.avi")
        writer = cv2.VideoWriter(cls.video, cv2.VideoWriter_fourcc(*"MJPG"), 30, (64, 48))
        for _ in range(10):
            writer.write(np.full((48, 64, 3), 100, dtype=np.uint8))
        writer.release()

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def setUp(self):
        self.cap = None

    def tearDown(self):
        if self.cap is not None:
            self.cap.release()

    def start(self, **kwargs):
        self.cap = VideoCaptureAsync(self.video, health_log_interval=0, **kwargs).start()
        return self.cap

    def read_frame(self):
        # The very first read() can return before any frame was decoded; poll for a real one
        deadline = time.time() + 5.0
        while time.time() < deadline:
            grabbed, frame = self.cap.read(timeout=1.0)
            if grabbed:
                return frame
        self.fail("no frame within 5s")

    def test_hook_transforms_frames(self):
        def stamp(frame):
            frame[:, :] = 7
            return frame

        self.start(frame_hook=stamp)
        self.assertTrue((self.read_frame() == 7).all())

    def test_hook_returning_none_means_in_place(self):
        def in_place(frame):
            frame[0, 0] = (1, 2, 3)

        self.start(frame_hook=in_place)
        self.assertEqual(tuple(self.read_frame()[0, 0]), (1, 2, 3))

    def test_raising_hook_never_kills_capture(self):
        calls = []

        def broken(frame):
            calls.append(1)
            raise RuntimeError("overlay font missing")

        cap = self.start(frame_hook=broken)
        first = self.read_frame()
        self.assertAlmostEqual(first.mean(), 100, delta=5)  # unhooked frame published
        self.assertTrue(_wait_until(lambda: len(calls) >= 5))
        self.read_frame()
        self.assertTrue(cap.started)
        self.assertEqual(cap.restart_count, 0)  # no spurious reconnect
        self.assertGreaterEqual(cap.get_stats()["hook_errors"], 5)

    def test_no_hook(self):
        self.start()
        self.assertAlmostEqual(self.read_frame().mean(), 100, delta=5)
        self.assertEqual(self.cap.get_stats()["hook_errors"], 0)


class TestFromCamera(unittest.TestCase):
    def build(self, camera, **overrides):
        config, _ = load_config({"node": {"sn": "n"}, "cameras": {"c": camera}})
        with mock.patch("cv2.VideoCapture") as vc:
            vc.return_value.isOpened.return_value = False
            return VideoCaptureAsync.from_camera(config.cameras["c"], **overrides)

    def test_rtsp_camera(self):
        cap = self.build({
            "source": {"type": "rtsp", "host": "10.0.0.5", "user": "admin", "password": "s3cret!", "transport": "udp"},
            "capture": {"fps": 10, "width": 1280, "height": 720, "max_frame_age": "500ms",
                        "restart": {"delay": "2m", "jitter": 0, "max_attempts": 3}},
        })
        self.assertEqual(cap.src, "rtsp://admin:s3cret%21@10.0.0.5:554/Streaming/Channels/101")
        self.assertNotIn("s3cret", cap._label)
        self.assertNotIn("s3cret", str(cap.get_stats()["src"]))
        self.assertEqual(cap.source_type, "NETWORK")
        self.assertEqual(cap.rtsp_transport, "udp")
        self.assertEqual((cap.width, cap.height, cap.target_fps), (1280, 720, 10))
        self.assertEqual(cap.max_frame_age_ms, 500)
        self.assertEqual(cap._backoff.ceiling, 120.0)
        self.assertEqual(cap.max_restart_attempts, 3)
        cap.release()

    def test_device_symlink_is_a_device(self):
        cap = self.build({"source": {"type": "device", "path": "/dev/cam-ocr"}})
        self.assertEqual(cap.source_type, "USB")  # auto-detect alone would say UNKNOWN
        cap.release()

    def test_file_loop_and_overrides(self):
        hook = lambda frame: frame
        cap = self.build({"source": {"type": "file", "path": "clip.weird", "loop": False}}, frame_hook=hook)
        self.assertEqual(cap.source_type, "FILE")
        self.assertFalse(cap.loop)
        self.assertIs(cap.frame_hook, hook)
        cap.release()

if __name__ == "__main__":
    unittest.main()
