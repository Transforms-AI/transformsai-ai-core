"""
Reconnect / health tests for VideoCaptureAsync.

No camera, no network: cv2.VideoCapture is replaced by a scriptable FakeCapture
(same monkeypatch approach as test_fps_crash.py). Timings are compressed to
milliseconds so the whole suite runs in a few seconds.

Run:
    uv run python -m unittest tests.test_capture_restart -v
"""

import os
import shutil
import threading
import time
import unittest
from unittest import mock

import cv2
import numpy as np

from transformsai_ai_core.video_capture import VideoCaptureAsync, _Backoff

FRAME = np.zeros((4, 4, 3), dtype=np.uint8)

_LOG_DIR_PREEXISTED = os.path.isdir(".core-logs")


def tearDownModule():
    if not _LOG_DIR_PREEXISTED:
        shutil.rmtree(".core-logs", ignore_errors=True)


class FakeCapture:
    """
    Scriptable stand-in for cv2.VideoCapture.

    behavior:
        "ok"      — grab/retrieve always deliver a frame
        "fail"    — grab() always returns False (read-failure streak)
        "silent"  — grab() succeeds but retrieve()/read() never deliver (stall)
        "raise"   — grab() raises
        "closed"  — isOpened() is False (open failure)
        "block"   — grab() blocks until `gate` is set (wedged inside OpenCV)
    """

    def __init__(self, behavior="ok", fps=30.0, gate=None):
        self.behavior = behavior
        self.fps = fps
        self.gate = gate
        self.released = False
        self.grabs = 0
        self.props = {}

    # --- cv2 API -----------------------------------------------------
    def isOpened(self):
        return self.behavior != "closed" and not self.released

    def get(self, prop):
        if prop == cv2.CAP_PROP_FPS:
            return self.fps
        return self.props.get(prop, 0.0)

    def set(self, prop, value):
        self.props[prop] = value
        return True

    def grab(self):
        self.grabs += 1
        if self.behavior == "raise":
            raise RuntimeError("simulated grab failure")
        if self.behavior == "block":
            self.gate.wait(30)
            return False
        if self.behavior == "fail":
            return False
        time.sleep(0.001)
        return True

    def retrieve(self):
        if self.behavior == "silent":
            return False, None
        return True, FRAME.copy()

    def read(self):
        if self.behavior in ("fail", "silent"):
            return False, None
        return True, FRAME.copy()

    def release(self):
        self.released = True


class Opener:
    """Records every cv2.VideoCapture construction and hands back scripted fakes."""

    def __init__(self, *behaviors, fps=30.0, gate=None):
        self.behaviors = list(behaviors) or ["ok"]
        self.fps = fps
        self.gate = gate
        self.caps = []
        self.env_seen = []
        self._lock = threading.Lock()

    def __call__(self, src, backend=None):
        with self._lock:
            index = len(self.caps)
            behavior = self.behaviors[min(index, len(self.behaviors) - 1)]
            cap = FakeCapture(behavior, fps=self.fps, gate=self.gate)
            self.caps.append(cap)
            self.env_seen.append(os.environ.get(VideoCaptureAsync._FFMPEG_ENV_KEY))
        return cap

    @property
    def open_count(self):
        with self._lock:
            return len(self.caps)


def wait_for(predicate, timeout=5.0, interval=0.01):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(interval)
    return predicate()


FAST = dict(
    restart_backoff_start=0.01,
    restart_delay=0.05,
    restart_backoff_jitter=0.0,
    restart_reset_after=0.05,
    health_log_interval=0,
)


class TestBackoff(unittest.TestCase):
    def test_ramp_caps_at_ceiling(self):
        backoff = _Backoff(start=1.0, ceiling=5.0, jitter=0.0)
        self.assertEqual([backoff.next() for _ in range(5)], [1.0, 2.0, 4.0, 5.0, 5.0])

    def test_reset_returns_to_start(self):
        backoff = _Backoff(start=1.0, ceiling=30.0, jitter=0.0)
        backoff.next(); backoff.next()
        backoff.reset()
        self.assertEqual(backoff.attempts, 0)
        self.assertEqual(backoff.next(), 1.0)

    def test_jitter_stays_within_band(self):
        backoff = _Backoff(start=1.0, ceiling=1.0, jitter=0.2)
        for _ in range(20):
            self.assertTrue(0.8 <= backoff.next() <= 1.2)


class CaptureTestCase(unittest.TestCase):
    """Patches cv2.VideoCapture and guarantees the capture is released."""

    def setUp(self):
        self.cap = None
        patcher = mock.patch("cv2.VideoCapture")
        self.addCleanup(patcher.stop)
        self.mock_vc = patcher.start()
        self.addCleanup(self._cleanup_capture)

    def _cleanup_capture(self):
        if self.cap is not None:
            self.cap.release()

    def build(self, opener, src="rtsp://fake/stream", **kwargs):
        self.mock_vc.side_effect = opener
        options = dict(FAST)
        options.update(kwargs)
        self.cap = VideoCaptureAsync(src=src, **options)
        return self.cap


class TestReconnect(CaptureTestCase):
    def test_recovers_from_failed_opens(self):
        opener = Opener("closed", "closed", "ok")
        cap = self.build(opener).start()

        self.assertTrue(wait_for(lambda: cap.state == cap.STATE_STREAMING),
                        f"never reached streaming (state={cap.state})")
        self.assertEqual(opener.open_count, 3)  # one in __init__, two in the thread
        self.assertTrue(cap.is_healthy)

        grabbed, frame = cap.read(timeout=2.0)
        self.assertTrue(grabbed)
        self.assertIsNotNone(frame)

    def test_read_failure_streak_reconnects(self):
        opener = Opener("fail", "ok")
        cap = self.build(opener).start()

        self.assertTrue(wait_for(lambda: cap.state == cap.STATE_STREAMING))
        self.assertGreaterEqual(cap.restart_count, 1)
        self.assertTrue(opener.caps[0].released, "dead handle was not released")
        self.assertIn("consecutive read failures", cap.get_stats()["last_error"])

    def test_exception_during_grab_reconnects(self):
        opener = Opener("raise", "ok")
        cap = self.build(opener).start()

        self.assertTrue(wait_for(lambda: cap.state == cap.STATE_STREAMING))
        self.assertGreaterEqual(cap.restart_count, 1)
        self.assertIn("simulated grab failure", cap.get_stats()["last_error"])

    def test_stall_watchdog_reconnects_a_silent_stream(self):
        # fps=1 throttles decoding, so grabs succeed and the consecutive-failure
        # counter never fires — only the stall watchdog can catch this.
        opener = Opener("silent", "ok")
        cap = self.build(opener, fps=1, stall_timeout=0.2).start()

        self.assertTrue(wait_for(lambda: cap.state == cap.STATE_STREAMING, timeout=5.0),
                        f"stall was not detected (state={cap.state})")
        self.assertGreaterEqual(cap.restart_count, 1)
        self.assertIn("stalled", cap.get_stats()["last_error"])

    def test_stall_timeout_auto_derives_from_fps(self):
        opener = Opener("ok")
        cap = self.build(opener, fps=2)
        self.assertEqual(cap.get_stats()["stall_timeout"], 10.0)  # max(5, 20/2)

        cap.stall_timeout = 0
        self.assertEqual(cap.get_stats()["stall_timeout"], 0.0)  # explicitly disabled

    def test_backoff_resets_after_a_healthy_run(self):
        opener = Opener("fail", "ok")
        cap = self.build(opener, restart_reset_after=0.05).start()

        self.assertTrue(wait_for(lambda: cap.state == cap.STATE_STREAMING))
        self.assertTrue(wait_for(lambda: cap.get_stats()["restart_attempts"] == 0),
                        "backoff never reset after the stream went healthy")


class TestGivingUp(CaptureTestCase):
    def test_max_attempts_ends_in_failed_state(self):
        opener = Opener("closed")
        cap = self.build(opener, max_restart_attempts=2).start()

        self.assertTrue(wait_for(lambda: cap.state == cap.STATE_FAILED),
                        f"never gave up (state={cap.state})")
        self.assertTrue(wait_for(lambda: not cap.started))
        # __init__ open + the thread's first connect + 2 backoff-delayed retries
        self.assertEqual(opener.open_count, 4)

    def test_read_does_not_busy_spin_when_failed(self):
        opener = Opener("closed")
        cap = self.build(opener, max_restart_attempts=1).start()
        self.assertTrue(wait_for(lambda: cap.state == cap.STATE_FAILED))

        started = time.monotonic()
        grabbed, frame = cap.read(timeout=0.3)
        elapsed = time.monotonic() - started

        self.assertFalse(grabbed)
        self.assertIsNone(frame)
        self.assertGreaterEqual(elapsed, 0.25, "read() returned instantly — consumers will spin")

    def test_read_returns_immediately_after_stop(self):
        opener = Opener("ok")
        cap = self.build(opener).start()
        self.assertTrue(wait_for(lambda: cap.state == cap.STATE_STREAMING))
        cap.stop()
        self.assertTrue(wait_for(lambda: not cap.started))

        started = time.monotonic()
        grabbed, _ = cap.read(timeout=2.0)
        self.assertFalse(grabbed)
        self.assertLess(time.monotonic() - started, 0.5)

    def test_auto_restart_off_still_raises_on_bad_source(self):
        self.mock_vc.side_effect = Opener("closed")
        with self.assertRaises(RuntimeError):
            VideoCaptureAsync(src="rtsp://fake/stream", auto_restart_on_fail=False)


class TestShutdown(CaptureTestCase):
    def test_release_stops_thread_and_frees_handle(self):
        opener = Opener("ok")
        cap = self.build(opener).start()
        self.assertTrue(wait_for(lambda: cap.state == cap.STATE_STREAMING))

        cap.release()

        self.assertFalse(cap.started)
        self.assertEqual(cap.state, cap.STATE_STOPPED)
        self.assertTrue(opener.caps[0].released)
        self.assertFalse(cap._thread.is_alive())

    def test_release_abandons_a_wedged_thread_without_double_release(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        opener = Opener("block", gate=gate)
        cap = self.build(opener, open_timeout=0.5).start()
        wedged = opener.caps[0]
        self.assertTrue(wait_for(lambda: wedged.grabs > 0))

        started = time.monotonic()
        cap.release()  # join window is max(2.0, open_timeout + 1.0)
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 4.0, "release() hung on a wedged thread")
        self.assertEqual(cap.get_stats()["orphaned_threads"], 1)
        self.assertFalse(wedged.released,
                         "handle was released while the thread was still inside grab()")

        # The abandoned daemon thread cleans up after itself once OpenCV returns.
        gate.set()
        cap._thread.join(timeout=5.0)
        self.assertTrue(wait_for(lambda: wedged.released))

    def test_restarting_an_abandoned_capture_is_refused(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        cap = self.build(Opener("block", gate=gate), open_timeout=0.5).start()
        cap.release()

        # A second thread would race the wedged one over the same handle.
        with self.assertRaises(RuntimeError):
            cap.start()

    def test_accessors_are_inert_after_abandoning(self):
        gate = threading.Event()
        self.addCleanup(gate.set)
        cap = self.build(Opener("block", gate=gate), open_timeout=0.5).start()
        cap.release()

        self.assertIsNone(cap.get(cv2.CAP_PROP_FPS))
        self.assertFalse(cap.set(cv2.CAP_PROP_FPS, 10))
        self.assertEqual(cap.get_height_width(), (None, None))


class TestObservability(CaptureTestCase):
    def test_state_change_callback_sees_the_full_lifecycle(self):
        seen = []
        opener = Opener("fail", "ok")
        cap = self.build(opener, on_state_change=lambda state, info: seen.append(state)).start()

        self.assertTrue(wait_for(lambda: cap.state == cap.STATE_STREAMING))
        cap.release()

        self.assertIn(cap.STATE_RECONNECTING, seen)
        self.assertIn(cap.STATE_STREAMING, seen)
        self.assertEqual(seen[-1], cap.STATE_STOPPED)

    def test_callback_exceptions_do_not_kill_the_capture(self):
        def boom(state, info):
            raise ValueError("callback blew up")

        cap = self.build(Opener("ok"), on_state_change=boom).start()
        self.assertTrue(wait_for(lambda: cap.state == cap.STATE_STREAMING))
        self.assertTrue(cap.read(timeout=2.0)[0])

    def test_stats_shape_and_counters(self):
        opener = Opener("fail", "ok")
        cap = self.build(opener).start()
        self.assertTrue(wait_for(lambda: cap.state == cap.STATE_STREAMING))
        self.assertTrue(wait_for(lambda: cap.get_stats()["total_frames"] > 0))

        stats = cap.get_stats()
        for key in ("src", "source_type", "state", "is_healthy", "restart_count",
                    "consecutive_failures", "restart_attempts", "current_backoff",
                    "stall_timeout", "last_error", "last_error_time", "uptime_seconds",
                    "total_frames", "frames_dropped_stale", "last_frame_age_ms",
                    "target_fps", "source_fps", "measured_fps", "orphaned_threads"):
            self.assertIn(key, stats)

        self.assertEqual(stats["source_type"], "NETWORK")
        self.assertGreaterEqual(stats["restart_count"], 1)
        self.assertTrue(stats["is_healthy"])


class TestFfmpegOptions(CaptureTestCase):
    def test_rtsp_source_gets_transport_and_timeouts(self):
        opener = Opener("ok")
        self.build(opener, src="rtsp://fake/stream", open_timeout=3.0)

        options = opener.env_seen[0]
        self.assertIn("rtsp_transport;tcp", options)
        self.assertIn("timeout;3000000", options)
        self.assertIn("stimeout;3000000", options)

    def test_env_is_restored_after_opening(self):
        key = VideoCaptureAsync._FFMPEG_ENV_KEY
        sentinel = "user;value"
        os.environ[key] = sentinel
        self.addCleanup(os.environ.pop, key, None)

        self.build(Opener("ok"))
        self.assertEqual(os.environ.get(key), sentinel)

    def test_non_rtsp_network_source_gets_timeouts_only(self):
        opener = Opener("ok")
        self.build(opener, src="http://fake/stream.mjpg")
        self.assertNotIn("rtsp_transport", opener.env_seen[0])
        self.assertIn("timeout;", opener.env_seen[0])

    def test_file_source_is_untouched(self):
        opener = Opener("ok")
        self.build(opener, src="clip.mp4")
        self.assertIsNone(opener.env_seen[0])

    def test_explicit_options_override_everything(self):
        opener = Opener("ok")
        self.build(opener, src="rtsp://fake/stream", ffmpeg_options="custom;1")
        self.assertEqual(opener.env_seen[0], "custom;1")

    def test_transport_can_be_disabled(self):
        opener = Opener("ok")
        self.build(opener, src="rtsp://fake/stream", rtsp_transport=None)
        self.assertNotIn("rtsp_transport", opener.env_seen[0])


if __name__ == "__main__":
    unittest.main()
