"""
Tests for the optional OSD timestamp-overwrite overlay in VideoCaptureAsync.

No camera, no network: a bogus local-file source makes _initialize_capture fail
fast (swallowed because auto_restart_on_fail defaults to True), so instances are
constructed without starting the capture thread. The overlay methods are then
exercised directly on synthetic frames.

Run:
    uv run python -m unittest tests.test_capture_timestamp -v
"""

import os
import shutil
import unittest

import numpy as np

from transformsai_ai_core.video_capture import VideoCaptureAsync
from transformsai_ai_core.utils import hide_camera_timestamp_and_add_current_time

# A source that classifies as FILE and fails to open immediately (no device/network).
BOGUS_SRC = "/nonexistent_timestamp_overlay_test.mp4"

_LOG_DIR_PREEXISTED = os.path.isdir(".core-logs")


def tearDownModule():
    if not _LOG_DIR_PREEXISTED:
        shutil.rmtree(".core-logs", ignore_errors=True)


def _blank_frame():
    return np.zeros((1080, 1920, 3), dtype=np.uint8)


def _make_capture(**kwargs):
    # auto_restart_on_fail=True (default) → the failed open is swallowed in __init__,
    # leaving a constructed instance with no running thread.
    return VideoCaptureAsync(src=BOGUS_SRC, **kwargs)


class TestTimestampOverlay(unittest.TestCase):
    def test_apply_overlay_paints_osd_region(self):
        """The overlay reuses the util and modifies the frame (default top-left OSD)."""
        cap = _make_capture(timestamp_overlay=True)
        out = cap._apply_timestamp_overlay(_blank_frame())
        self.assertTrue(np.any(out != 0), "overlay should paint over the OSD region")

    def test_publish_frame_stamps_when_enabled(self):
        cap = _make_capture(timestamp_overlay=True)
        cap._publish_frame(_blank_frame())
        self.assertIsNotNone(cap._frame)
        self.assertTrue(np.any(cap._frame != 0), "published frame should be stamped")
        self.assertEqual(cap._frame_id, 1)

    def test_publish_frame_untouched_when_disabled(self):
        cap = _make_capture(timestamp_overlay=False)
        cap._publish_frame(_blank_frame())
        self.assertTrue(np.all(cap._frame == 0), "disabled overlay must not modify frames")

    def test_none_options_are_dropped(self):
        cap = _make_capture(
            timestamp_overlay=True,
            timestamp_overlay_options={"new_ts_font_scale": None, "time_format": "%H:%M:%S"},
        )
        self.assertNotIn("new_ts_font_scale", cap._ts_overlay_options)
        self.assertEqual(cap._ts_overlay_options.get("time_format"), "%H:%M:%S")

    def test_fail_open_on_bad_options(self):
        """A broken option must not raise into the capture loop; frame passes through."""
        cap = _make_capture(
            timestamp_overlay=True,
            # Only two values -> the util's `xr, yr, wr, hr = ...` unpack raises ValueError.
            timestamp_overlay_options={"camera_ts_rect_ratios": [0.1, 0.1]},
        )
        frame = _blank_frame()
        out = cap._apply_timestamp_overlay(frame)
        self.assertTrue(np.all(out == 0), "failed overlay should return the un-stamped frame")
        self.assertEqual(cap._ts_overlay_failures, 1)


class TestFromConfig(unittest.TestCase):
    def test_timestamp_block_maps_to_kwargs(self):
        cfg = {
            "local": True,
            "local_source": BOGUS_SRC,
            "capture": {
                "timestamp": {
                    "enabled": True,
                    "rect_ratios": [0.02, 0.05, 0.25, 0.04],
                    "time_format": "%H:%M:%S",
                    "font_scale": None,  # dropped
                },
            },
        }
        cap = VideoCaptureAsync.from_config(cfg)
        self.assertTrue(cap._timestamp_overlay)
        self.assertEqual(cap._ts_overlay_options["camera_ts_rect_ratios"], [0.02, 0.05, 0.25, 0.04])
        self.assertEqual(cap._ts_overlay_options["time_format"], "%H:%M:%S")
        self.assertNotIn("new_ts_font_scale", cap._ts_overlay_options)

    def test_absent_timestamp_block_is_off(self):
        cfg = {"local": True, "local_source": BOGUS_SRC, "capture": {}}
        cap = VideoCaptureAsync.from_config(cfg)
        self.assertFalse(cap._timestamp_overlay)
        self.assertEqual(cap._ts_overlay_options, {})


class TestUtilTimeFormat(unittest.TestCase):
    def test_custom_time_format(self):
        frame = _blank_frame()
        out = hide_camera_timestamp_and_add_current_time(frame, time_format="%H:%M:%S", inplace=False)
        self.assertEqual(out.shape, frame.shape)
        self.assertTrue(np.any(out != 0))


if __name__ == "__main__":
    unittest.main()
