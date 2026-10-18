"""
Tests for MediaMTXStreamer's audio fallback: a failed audio input must never
cost the video.

FFmpeg is replaced by a fake `ffmpeg` script put first on PATH. It prints the
same messages real FFmpeg 9 prints for each failure (captured from the real
binary), opens the audio input only once a first video frame arrives (as real
FFmpeg 9 does), and is driven by control files in a temp dir:

  audio_fail   "always", or a count N: the next N opens of the audio input fail
  audio_die    seconds: audio opens, then the input dies mid-run
  video_fail   present: the push to MediaMTX fails (an output, not audio, error)
  calls.jsonl  one argv per launch, appended by the fake

Run: uv run python -m unittest tests.test_mediamtx_audio_fallback
"""

import json
import os
import shutil
import sys
import tempfile
import time
import unittest

import numpy as np

from transformsai_ai_core.mediamtx_streamer import MediaMTXStreamer


FAKE_FFMPEG = r'''#!{python}
import json, os, sys, time
ctrl = os.environ["FAKE_FFMPEG_CTRL"]
argv = sys.argv[1:]
with open(os.path.join(ctrl, "calls.jsonl"), "a") as f:
    f.write(json.dumps(["ffmpeg"] + argv) + "\n")

def read(name):
    try:
        with open(os.path.join(ctrl, name)) as f:
            return f.read().strip()
    except OSError:
        return None

audio = None
if "alsa" in argv:
    rest = argv[argv.index("alsa"):]
    audio = rest[rest.index("-i") + 1]
elif "-allowed_media_types" in argv:
    rest = argv[argv.index("-allowed_media_types"):]
    audio = rest[rest.index("-i") + 1]

def fail_audio():
    if audio.startswith("rtsp://"):
        sys.stderr.write(
            "[tcp @ 0x5652abb93600] Connection to tcp://127.0.0.1:2?timeout=20000000 failed: Connection refused\n"
            "[in#1 @ 0x5652abb90b40] Error opening input: Connection refused\n"
            "Error opening input file %s.\n"
            "Error opening input files: Connection refused\n" % audio)
    else:
        sys.stderr.write(
            "ALSA lib pcm.c:2722:(snd_pcm_open_noupdate) [error.pcm] Unknown PCM %s\n"
            "[in#1 @ 0x55d3cb7b0e80] cannot open audio device %s (No such file or directory)\n"
            "[in#1 @ 0x55d3cb7b0b80] Error opening input: Input/output error\n"
            "Error opening input file %s.\n"
            "Error opening input files: Input/output error\n" % (audio, audio, audio))
    sys.stderr.flush()
    sys.exit(251)

if audio is not None:
    # Like real FFmpeg 9: the piped video (input #0) is read before the audio
    # input is opened, so nothing fails until the first frame arrives
    if sys.stdin.buffer.read(1) == b"":
        sys.exit(0)
    mode = read("audio_fail")
    if mode == "always":
        fail_audio()
    if mode and mode.isdigit() and int(mode) > 0:
        with open(os.path.join(ctrl, "audio_fail"), "w") as f:
            f.write(str(int(mode) - 1))
        fail_audio()

if read("video_fail") is not None:
    sys.stderr.write(
        "[tcp @ 0x7f0d34651400] Connection to tcp://127.0.0.1:1?timeout=0 failed: Connection refused\n"
        "[out#0/rtsp @ 0x55c8e1250a00] Could not write header (incorrect codec parameters ?): Connection refused\n")
    sys.stderr.flush()
    sys.exit(145)

kind, die = "output", read("die")
if not die and audio is not None and read("audio_die"):
    kind, die = "audio", read("audio_die")
deadline = time.time() + float(die) if die else None
os.set_blocking(0, False)
while deadline is None or time.time() < deadline:
    try:
        if sys.stdin.buffer.read(1 << 20) == b"":
            sys.exit(0)
    except (BlockingIOError, TypeError):
        pass
    time.sleep(0.02)
if kind == "output":
    sys.stderr.write("[out#0/rtsp @ 0x55c8e1250a00] Error writing trailer: Broken pipe\n")
else:
    sys.stderr.write("[in#1/alsa @ 0x55d3cb7b0e80] Error during demuxing: Input/output error\n")
sys.stderr.flush()
sys.exit(1)
'''

# Exact argv of 4.4.7 (recorded from the unmodified code by stubbing Popen),
# for MediaMTXStreamer("vps.example.com", 8554, "testcam", fps=15,
# frame_width=640, frame_height=480, bitrate="800k", **audio_kwargs).
GOLDEN = {
    "none": [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt',
        'bgr24', '-s', '640x480', '-r', '15', '-i', '-', '-c:v', 'libx264',
        '-preset', 'ultrafast', '-tune', 'zerolatency', '-profile:v',
        'main', '-pix_fmt', 'yuv420p', '-b:v', '800k', '-g', '15', '-f',
        'rtsp', '-rtsp_transport', 'tcp', '-loglevel', 'warning',
        'rtsp://vps.example.com:8554/live/cam_sn_testcam',
    ],
    "alsa": [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt',
        'bgr24', '-s', '640x480', '-r', '15',
        '-use_wallclock_as_timestamps', '1', '-thread_queue_size', '32',
        '-i', '-', '-f', 'alsa', '-ac', '2', '-ar', '16000',
        '-thread_queue_size', '8192', '-use_wallclock_as_timestamps', '1',
        '-i', 'inmp441', '-map', '0:v:0', '-map', '1:a:0', '-c:v',
        'libx264', '-preset', 'ultrafast', '-tune', 'zerolatency',
        '-profile:v', 'main', '-pix_fmt', 'yuv420p', '-b:v', '800k', '-g',
        '15', '-maxrate', '800k', '-bufsize', '800k', '-g', '60',
        '-threads', '4', '-vf', 'setpts=(RTCTIME-RTCSTART)/(TB*1000000)',
        '-af',
        'pan=mono|c0=c0,highpass=f=20,volume=18.0dB,alimiter=limit=0.97:level=false,aresample=48000:async=1000',
        '-c:a', 'libopus', '-b:a', '24k', '-application', 'voip',
        '-max_interleave_delta', '100000', '-f', 'rtsp', '-rtsp_transport',
        'tcp', '-loglevel', 'warning',
        'rtsp://vps.example.com:8554/live/cam_sn_testcam',
    ],
    "rtsp": [
        'ffmpeg', '-y', '-f', 'rawvideo', '-vcodec', 'rawvideo', '-pix_fmt',
        'bgr24', '-s', '640x480', '-r', '15',
        '-use_wallclock_as_timestamps', '1', '-thread_queue_size', '32',
        '-i', '-', '-allowed_media_types', 'audio', '-rtsp_transport',
        'tcp', '-timeout', '20000000', '-thread_queue_size', '8192',
        '-use_wallclock_as_timestamps', '1', '-i', 'rtsp://cam.local/audio',
        '-map', '0:v:0', '-map', '1:a:0', '-c:v', 'libx264', '-preset',
        'ultrafast', '-tune', 'zerolatency', '-profile:v', 'main',
        '-pix_fmt', 'yuv420p', '-b:v', '800k', '-g', '15', '-maxrate',
        '800k', '-bufsize', '800k', '-g', '60', '-threads', '4', '-vf',
        'setpts=(RTCTIME-RTCSTART)/(TB*1000000)', '-af',
        'aresample=48000:async=1000', '-c:a', 'libopus', '-b:a', '24k',
        '-application', 'voip', '-max_interleave_delta', '100000', '-f',
        'rtsp', '-rtsp_transport', 'tcp', '-loglevel', 'warning',
        'rtsp://vps.example.com:8554/live/cam_sn_testcam',
    ],
}

GOLDEN_KWARGS = {
    "none": {},
    "alsa": dict(audio_device="inmp441"),
    "rtsp": dict(audio_source="rtsp", audio_url="rtsp://cam.local/audio",
                 audio_gain_db=0),
}

FRAME = np.zeros((480, 640, 3), dtype=np.uint8)


def _wait_until(pred, timeout=5.0, interval=0.02):
    deadline = time.time() + timeout
    while time.time() < deadline:
        if pred():
            return True
        time.sleep(interval)
    return pred()


class RecordingLogger:
    def __init__(self):
        self.records = []

    def _log(self, level):
        return lambda msg, *a, **k: self.records.append((level, str(msg)))

    def __getattr__(self, level):
        return self._log(level)

    def count(self, level, needle=""):
        return sum(1 for lv, m in self.records if lv == level and needle in m)


def tearDownModule():
    from loguru import logger
    logger.remove()
    for d in (".core-streamer-logs", ".core-logs"):
        shutil.rmtree(os.path.join(os.getcwd(), d), ignore_errors=True)


class FakeFFmpegTestCase(unittest.TestCase):

    def setUp(self):
        self.ctrl = tempfile.mkdtemp(prefix="fake-ffmpeg-")
        path = os.path.join(self.ctrl, "ffmpeg")
        with open(path, "w") as f:
            f.write(FAKE_FFMPEG.replace("{python}", sys.executable))
        os.chmod(path, 0o755)
        self._env = {k: os.environ.get(k) for k in ("PATH", "FAKE_FFMPEG_CTRL")}
        os.environ["PATH"] = self.ctrl + os.pathsep + os.environ["PATH"]
        os.environ["FAKE_FFMPEG_CTRL"] = self.ctrl
        self.streamer = None

    def tearDown(self):
        if self.streamer:
            self.streamer.stop_streaming()
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        shutil.rmtree(self.ctrl, ignore_errors=True)

    def control(self, name, value=""):
        with open(os.path.join(self.ctrl, name), "w") as f:
            f.write(str(value))

    def calls(self):
        try:
            with open(os.path.join(self.ctrl, "calls.jsonl")) as f:
                return [json.loads(line) for line in f]
        except OSError:
            return []

    def make(self, **kwargs):
        s = MediaMTXStreamer("vps.example.com", 8554, "testcam", fps=15,
                             frame_width=640, frame_height=480, bitrate="800k",
                             **kwargs)
        s._start_check_s = 0.3
        s.logger = RecordingLogger()
        self.streamer = s
        return s

    def restart(self, s):
        """A fresh start, past the rapid-restart guard."""
        s.stop_streaming()
        s.last_restart_time = 0
        return s.start_streaming()

    @staticmethod
    def has_audio(argv):
        return "-allowed_media_types" in argv or "alsa" in argv


class TestCommandLine(unittest.TestCase):

    def test_golden_argv_unchanged(self):
        for name, kwargs in GOLDEN_KWARGS.items():
            s = MediaMTXStreamer("vps.example.com", 8554, "testcam", fps=15,
                                 frame_width=640, frame_height=480,
                                 bitrate="800k", **kwargs)
            self.assertEqual(s._build_ffmpeg_cmd(s._audio_configured),
                             GOLDEN[name], name)
            # The fallback stream is exactly a configured no-audio stream
            self.assertEqual(s._build_ffmpeg_cmd(False), GOLDEN["none"], name)

    def test_from_config_without_audio_unchanged(self):
        s = MediaMTXStreamer.from_config({
            "mediamtx_ip": "vps.example.com", "fps": 15, "frame_width": 640,
            "frame_height": 480, "bitrate": "800k",
        }, camera_sn_id="testcam")
        self.assertFalse(s._audio_configured)
        self.assertEqual(s._build_ffmpeg_cmd(s._audio_configured), GOLDEN["none"])
        self.assertFalse(s.audio_active)
        self.assertIsNone(s.audio_error)


class TestAudioFailureLine(unittest.TestCase):

    def line(self, log, **kwargs):
        kwargs = kwargs or dict(audio_device="inmp441")
        s = MediaMTXStreamer("vps.example.com", 8554, "testcam", **kwargs)
        return s._audio_failure_line(log)

    def test_ffmpeg9_alsa_open_failure(self):
        log = ("ALSA lib pcm.c:2722:(snd_pcm_open_noupdate) [error.pcm] Unknown PCM inmp441\n"
               "[in#1 @ 0x55d3cb7b0e80] cannot open audio device inmp441 (Device or resource busy)\n"
               "Error opening input files: Input/output error\n")
        self.assertIn("inmp441", self.line(log))

    def test_ffmpeg9_rtsp_open_failure(self):
        log = ("[tcp @ 0x5652abb93600] Connection to tcp://127.0.0.1:2?timeout=20000000 failed: Connection refused\n"
               "[in#1 @ 0x5652abb90b40] Error opening input: Connection refused\n")
        self.assertEqual(self.line(log, audio_source="rtsp", audio_url="rtsp://127.0.0.1:2/cam"),
                         "Error opening input: Connection refused")

    def test_older_ffmpeg_formats(self):
        self.assertEqual(self.line("inmp441: Input/output error\n"),
                         "inmp441: Input/output error")
        self.assertIsNotNone(self.line(
            "[alsa @ 0x1] cannot open audio device hw:1 (Device or resource busy)\n",
            audio_device="dsnoop_mic"))
        self.assertIsNotNone(self.line(
            "rtsp://cam.local/audio: Connection timed out\n",
            audio_source="rtsp", audio_url="rtsp://cam.local/audio"))

    def test_audio_error_never_carries_credentials(self):
        s = MediaMTXStreamer("vps.example.com", 8554, "testcam", audio_source="rtsp",
                             audio_url="rtsp://admin:s3cret@cam.local/audio")
        s.logger = RecordingLogger()
        s._note_audio_failure(s._audio_failure_line(
            "rtsp://admin:s3cret@cam.local/audio: Connection timed out\n"))
        self.assertNotIn("s3cret", s.audio_error)
        self.assertNotIn("s3cret", s.logger.records[-1][1])
        self.assertIn("Connection timed out", s.audio_error)

    def test_output_failure_is_not_audio(self):
        log = ("[tcp @ 0x7f0d34651400] Connection to tcp://127.0.0.1:1?timeout=0 failed: Connection refused\n"
               "[out#0/rtsp @ 0x55c8e1250a00] Could not write header (incorrect codec parameters ?): Connection refused\n"
               "[vf#0:0 @ 0x55c8e1254700] Error sending frames to consumers: Connection refused\n")
        self.assertIsNone(self.line(log))
        self.assertIsNone(self.line(log, audio_source="rtsp", audio_url="rtsp://cam.local/audio"))


class TestStartFallback(FakeFFmpegTestCase):

    def assert_video_only_fallback(self, s, ident):
        self.assertTrue(s.start_streaming())
        self.assertTrue(s.is_active())
        first, last = self.calls()
        self.assertTrue(self.has_audio(first))
        self.assertEqual(last, s._build_ffmpeg_cmd(False))
        self.assertFalse(s.audio_active)
        self.assertIn(ident, s.audio_error)
        self.assertEqual(s.logger.count("error"), 1)
        self.assertEqual(s.logger.count("warning"), 1)
        self.assertEqual(s.logger.count("warning", "rapid restart"), 0)
        stats = s.get_stats()
        self.assertFalse(stats["audio_active"])
        self.assertEqual(stats["audio_error"], s.audio_error)

    def test_missing_alsa_pcm_falls_back_to_video_only(self):
        self.control("audio_fail", "always")
        s = self.make(audio_device="nonexistent_pcm")
        self.assert_video_only_fallback(s, "nonexistent_pcm")

    def test_unreachable_rtsp_audio_falls_back_to_video_only(self):
        self.control("audio_fail", "always")
        s = self.make(audio_source="rtsp", audio_url="rtsp://127.0.0.1:2/cam")
        self.assert_video_only_fallback(s, "Connection refused")

    def test_audio_retried_after_backoff_and_heals(self):
        self.control("audio_fail", 1)
        s = self.make(audio_device="inmp441", audio_retry_backoff=0.5)
        self.assertTrue(s.start_streaming())
        self.assertFalse(s.audio_active)

        # Inside the backoff: a fresh start goes straight to video-only, quietly
        self.assertTrue(self.restart(s))
        self.assertFalse(self.has_audio(self.calls()[-1]))
        self.assertEqual(len(self.calls()), 3)
        self.assertEqual(s.logger.count("warning"), 1)

        # After it: audio is tried again and comes back
        time.sleep(0.6)
        self.assertTrue(self.restart(s))
        self.assertTrue(self.has_audio(self.calls()[-1]))
        self.assertTrue(s.audio_active)
        self.assertIsNone(s.audio_error)
        self.assertEqual(s.logger.count("info", "with audio again"), 1)
        self.assertEqual(s.get_stats()["audio_active"], True)

    def test_backoff_doubles_while_failing_and_is_capped(self):
        self.control("audio_fail", "always")
        s = self.make(audio_device="nonexistent_pcm", audio_retry_backoff=60,
                      audio_retry_backoff_max=900)
        delays = []
        for _ in range(6):
            s._audio_retry_at = 0  # let every start try audio
            self.assertTrue(self.restart(s))
            delays.append(round(s._audio_retry_at - time.time()))
        self.assertEqual(delays, [60, 120, 240, 480, 900, 900])

    def test_long_healthy_audio_run_resets_backoff(self):
        s = self.make(audio_device="inmp441", audio_retry_backoff=60,
                      audio_reset_after=60)
        self.assertTrue(s.start_streaming())
        self.assertTrue(s.audio_active)
        s._audio_backoff.attempts = 3  # earlier failures
        s._audio_since -= 120          # ...but this run has been healthy for 2 min
        s.stop_streaming()
        self.assertEqual(s._audio_backoff.attempts, 0)

    def test_short_audio_run_keeps_backoff(self):
        s = self.make(audio_device="inmp441", audio_reset_after=60)
        self.assertTrue(s.start_streaming())
        s._audio_backoff.attempts = 3
        s.stop_streaming()
        self.assertEqual(s._audio_backoff.attempts, 3)

    def test_both_attempts_fail_leaves_audio_state_alone(self):
        self.control("audio_fail", "always")
        self.control("video_fail")
        s = self.make(audio_device="inmp441")
        self.assertFalse(s.start_streaming())
        self.assertEqual(len(self.calls()), 2)
        self.assertFalse(s.is_streaming)
        self.assertIsNone(s.audio_error)
        self.assertEqual(s._audio_backoff.attempts, 0)
        self.assertEqual(s._audio_retry_at, 0.0)

    def test_no_audio_configured_single_launch(self):
        s = self.make()
        primed = []
        s._prime_ffmpeg = primed.append
        self.assertTrue(s.start_streaming())
        self.assertEqual(self.calls(), [GOLDEN["none"]])
        self.assertEqual(primed, [])  # no blank frame on a no-audio stream
        self.assertFalse(s.audio_active)
        self.assertIsNone(s.audio_error)


class TestMidRunDeath(FakeFFmpegTestCase):

    def die_and_restart(self, s):
        self.assertTrue(_wait_until(lambda: s.ffmpeg_process.poll() is not None))
        s.last_restart_time = 0  # past update_frame's restart cooldown
        s.update_frame(FRAME)
        self.assertTrue(s.is_streaming)

    def test_audio_input_death_restarts_video_only(self):
        self.control("audio_die", 0.6)
        s = self.make(audio_device="inmp441")
        self.assertTrue(s.start_streaming())
        self.assertTrue(s.audio_active)
        self.die_and_restart(s)
        self.assertFalse(self.has_audio(self.calls()[-1]))
        self.assertEqual(len(self.calls()), 2)  # no wasted audio attempt
        self.assertFalse(s.audio_active)
        self.assertIn("Input/output error", s.audio_error)
        self.assertEqual(s.logger.count("warning", "video-only"), 1)

    def test_non_audio_death_restarts_with_audio(self):
        self.control("die", 0.6)
        s = self.make(audio_device="inmp441")
        self.assertTrue(s.start_streaming())
        os.remove(os.path.join(self.ctrl, "die"))
        self.die_and_restart(s)
        self.assertTrue(self.has_audio(self.calls()[-1]))
        self.assertTrue(s.audio_active)
        self.assertIsNone(s.audio_error)


if __name__ == "__main__":
    unittest.main()
