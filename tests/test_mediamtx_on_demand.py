"""
Tests for MediaMTXStreamer on-demand publishing (demand supervisor).

No FFmpeg process is launched: _ensure_ffmpeg_started/_ensure_ffmpeg_stopped
are monkeypatched to record calls and toggle is_streaming, and demand is
injected via the demand_check callable.

Run: uv run python -m unittest tests.test_mediamtx_on_demand
"""

import os
import shutil
import time
import unittest

import numpy as np

from transformsai_ai_core.mediamtx_streamer import MediaMTXStreamer


def _wait_until(pred, timeout=3.0, interval=0.01):
    """Poll pred() until it returns True or timeout elapses."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return pred()


class FakeResponse:
    def __init__(self, ok, text=""):
        self.ok = ok
        self.text = text


class FakeDemandClient:
    """Stand-in for the ApiClient used by the default HTTP backend."""

    def __init__(self, response):
        self.response = response
        self.requests = []
        self.shutdown_called = False

    def get(self, url, **kw):
        self.requests.append(url)
        return self.response

    def shutdown(self):
        self.shutdown_called = True


class OnDemandTestCase(unittest.TestCase):
    """Base: builds an on_demand streamer with fake FFmpeg lifecycle."""

    def make_streamer(self, demand_check, poll_interval=0.05, grace_period=0.2,
                      **kwargs):
        s = MediaMTXStreamer(
            mediamtx_ip="vps.example.com",
            rtsp_port=8554,
            camera_sn_id="testcam",
            on_demand=True,
            demand_poll_interval=poll_interval,
            demand_grace_period=grace_period,
            demand_check=demand_check,
            **kwargs,
        )
        s.start_calls = 0
        s.stop_calls = 0

        def fake_start():
            s.start_calls += 1
            s.is_streaming = True
            return True

        def fake_stop():
            s.stop_calls += 1
            s.is_streaming = False

        s._ensure_ffmpeg_started = fake_start
        s._ensure_ffmpeg_stopped = fake_stop
        self.streamer = s
        return s

    def tearDown(self):
        if getattr(self, "streamer", None):
            self.streamer.stop_streaming()
            self.streamer = None


def tearDownModule():
    from loguru import logger
    logger.remove()  # detach sinks before deleting the log directory
    for d in (".core-streamer-logs", ".core-logs"):
        shutil.rmtree(os.path.join(os.getcwd(), d), ignore_errors=True)


class TestDemandSupervisor(OnDemandTestCase):

    def test_starts_on_demand_true_no_double_start(self):
        s = self.make_streamer(demand_check=lambda: True)
        self.assertTrue(s.start_streaming())
        self.assertTrue(_wait_until(lambda: s.is_streaming))
        # Let several polls pass; must not restart while already streaming
        time.sleep(0.3)
        self.assertEqual(s.start_calls, 1)
        # start_streaming is idempotent (supervisor already alive)
        self.assertTrue(s.start_streaming())
        time.sleep(0.15)
        self.assertEqual(s.start_calls, 1)

    def test_stops_only_after_grace_period(self):
        demand = {"value": True}
        s = self.make_streamer(demand_check=lambda: demand["value"])
        s.start_streaming()
        self.assertTrue(_wait_until(lambda: s.is_streaming))

        demand["value"] = False
        # Well within the 0.2s grace period the stream must still be up
        time.sleep(0.1)
        self.assertTrue(s.is_streaming)
        self.assertEqual(s.stop_calls, 0)
        # After grace elapses it stops
        self.assertTrue(_wait_until(lambda: not s.is_streaming, timeout=2.0))
        self.assertEqual(s.stop_calls, 1)

    def test_no_flap_when_demand_returns_within_grace(self):
        demand = {"value": True}
        s = self.make_streamer(demand_check=lambda: demand["value"],
                               grace_period=0.5)
        s.start_streaming()
        self.assertTrue(_wait_until(lambda: s.is_streaming))

        demand["value"] = False
        time.sleep(0.15)  # < grace
        demand["value"] = True
        time.sleep(0.6)  # past what would have been the grace deadline
        self.assertTrue(s.is_streaming)
        self.assertEqual(s.stop_calls, 0)
        self.assertEqual(s.start_calls, 1)

    def test_poll_failure_while_live_keeps_stream_up(self):
        state = {"value": True}

        def check():
            if state["value"] is None:
                raise RuntimeError("poll failed")
            return state["value"]

        s = self.make_streamer(demand_check=check)
        s.start_streaming()
        self.assertTrue(_wait_until(lambda: s.is_streaming))

        state["value"] = None  # every poll now fails -> None -> fail-safe
        time.sleep(0.5)  # far beyond grace
        self.assertTrue(s.is_streaming)
        self.assertEqual(s.stop_calls, 0)

    def test_poll_failure_while_idle_stays_idle(self):
        s = self.make_streamer(demand_check=lambda: (_ for _ in ()).throw(RuntimeError()))
        s.start_streaming()
        time.sleep(0.3)
        self.assertFalse(s.is_streaming)
        self.assertEqual(s.start_calls, 0)

    def test_update_frame_idle_returns_false_and_never_starts(self):
        s = self.make_streamer(demand_check=lambda: False)
        s.start_streaming()
        frame = np.zeros((4, 4, 3), dtype=np.uint8)
        for _ in range(5):
            self.assertFalse(s.update_frame(frame))
        time.sleep(0.2)
        self.assertEqual(s.start_calls, 0)
        self.assertIsNone(s.ffmpeg_process)

    def test_stop_streaming_joins_supervisor_thread(self):
        s = self.make_streamer(demand_check=lambda: True)
        s.start_streaming()
        self.assertTrue(_wait_until(lambda: s.is_streaming))
        thread = s._demand_thread
        self.assertTrue(thread.is_alive())
        s.stop_streaming()
        self.assertFalse(thread.is_alive())
        self.assertIsNone(s._demand_thread)
        self.assertFalse(s.is_streaming)


class TestCheckDemandParsing(OnDemandTestCase):

    def parse(self, ok, text):
        s = self.make_streamer(demand_check=None)
        s._demand_client = FakeDemandClient(FakeResponse(ok, text))
        result = s._check_demand()
        self.streamer = None  # supervisor never started; nothing to stop
        return result

    def test_body_parsing(self):
        for text in ("on", "1", "true", "yes", "ON\n", " True "):
            self.assertIs(self.parse(True, text), True, text)
        for text in ("off", "0", "false", "no", ""):
            self.assertIs(self.parse(True, text), False, text)
        # Failed poll or garbage body -> None (fail-safe)
        self.assertIsNone(self.parse(False, "on"))
        self.assertIsNone(self.parse(True, "garbage"))

    def test_demand_check_exception_returns_none(self):
        def boom():
            raise RuntimeError("backend down")

        s = self.make_streamer(demand_check=boom)
        self.assertIsNone(s._check_demand())
        self.streamer = None

    def test_demand_url_derived_and_override(self):
        s = self.make_streamer(demand_check=lambda: False)
        self.assertEqual(s._demand_url,
                         "https://vps.example.com/demand/cam_sn_testcam")
        self.streamer = None
        s2 = self.make_streamer(
            demand_check=lambda: False,
            demand_url="http://10.0.0.1:8080/flags/{camera_sn_id}",
        )
        self.assertEqual(s2._demand_url, "http://10.0.0.1:8080/flags/testcam")
        self.streamer = None


if __name__ == "__main__":
    unittest.main()
