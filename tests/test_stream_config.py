"""
Tests for per-camera streams: stream names, MediaMTX URL templates, the audio
filter chain, camera-audio wiring, size/fps derivation, and that constructor
defaults never drift from the config models (the models own every default).

Run: uv run python -m unittest tests.test_stream_config
"""

import inspect
import os
import shutil
import unittest

import numpy as np

from transformsai_ai_core.api_client import ApiClient
from transformsai_ai_core.config import (
    AlsaAudioInput,
    ApiConfig,
    AudioRetryConfig,
    CaptureConfig,
    ConfigError,
    DemandConfig,
    EncoderConfig,
    MediaMTXConfig,
    OutboxConfig,
    PoolConfig,
    RestartConfig,
    RetriesConfig,
    RtspSource,
    StreamConfig,
    build_stream_urls,
    clear_registry,
    load_config,
    parse_bitrate,
    validate_config,
)
from transformsai_ai_core.config.schema import AudioFilter
from transformsai_ai_core.mediamtx_streamer import MediaMTXStreamer, build_audio_filter
from transformsai_ai_core.video_capture import VideoCaptureAsync


def tearDownModule():
    shutil.rmtree(os.path.join(os.getcwd(), ".core-streamer-logs"), ignore_errors=True)


def config_with(cameras, **extra):
    data = {"node": {"sn": "pb-00004"}, "cameras": cameras, "mediamtx": {"host": "mediamtx.example.com"}}
    data.update(extra)
    return data


RTSP = {"type": "rtsp", "host": "192.168.1.64", "user": "admin", "password": "pw1234"}


class TestStreamNames(unittest.TestCase):
    def setUp(self):
        clear_registry()

    def test_default_name_includes_camera(self):
        config, _ = load_config(config_with({"surveillance": {"source": RTSP, "stream": {}}}))
        self.assertEqual(config.stream_name("surveillance"), "cam_sn_pb-00004_surveillance")

    def test_placeholders(self):
        config, _ = load_config(config_with([
            {"source": RTSP, "stream": {"name": "{sn}-{index}"}},
            {"source": RTSP, "stream": {"name": "{sn}-{camera}"}},
        ]))
        self.assertEqual(config.stream_name("camera_0"), "cam_sn_pb-00004-0")
        self.assertEqual(config.stream_name("camera_1"), "cam_sn_pb-00004-camera_1")

    def test_4x_name_kept_explicitly(self):
        config, report = load_config(config_with({"surveillance": {"source": RTSP, "stream": {"name": "{sn}"}}}))
        self.assertEqual(config.stream_urls("surveillance").rtsp,
                         "rtsp://mediamtx.example.com:8554/live/cam_sn_pb-00004")
        self.assertEqual(report.warnings, [])

    def test_prefix_is_fixed(self):
        config, report = load_config(config_with({"a": {"source": RTSP, "stream": {"name": "front_{sn}", "on_demand": True}}}))
        self.assertEqual(config.stream_name("a"), "cam_sn_front_pb-00004")
        self.assertEqual(config.stream_urls("a").demand, "https://mediamtx.example.com/demand/cam_sn_front_pb-00004")
        self.assertEqual(report.warnings, [])

    def test_redundant_prefix_is_not_doubled(self):
        config, report = load_config(config_with({"a": {"source": RTSP, "stream": {"name": "cam_sn_{sn}"}}}))
        self.assertEqual(config.stream_name("a"), "cam_sn_pb-00004")
        self.assertEqual([(i.path, i.suggestion) for i in report.warnings], [("cameras.a.stream.name", "{sn}")])

    def test_streamer_adds_the_prefix_too(self):
        s = MediaMTXStreamer("mediamtx.example.com", 8554, "testcam")
        self.assertEqual(s.stream_name, "cam_sn_testcam")
        self.assertEqual(s.rtsp_url, "rtsp://mediamtx.example.com:8554/live/cam_sn_testcam")
        self.assertEqual(MediaMTXStreamer("h", 8554, "cam_sn_testcam").stream_name, "cam_sn_testcam")

    def test_names_must_be_unique(self):
        data = config_with({
            "a": {"source": RTSP, "stream": {"name": "{sn}"}},
            "b": {"source": RTSP, "stream": {"name": "{sn}"}},
        })
        with self.assertRaises(ConfigError) as ctx:
            load_config(data)
        issue = ctx.exception.report.errors[0]
        self.assertEqual(issue.path, "cameras.b.stream.name")
        self.assertIn("also used by camera 'a'", issue.message)

    def test_default_names_never_clash(self):
        cameras = {f"cam{i}": {"source": RTSP, "stream": {}} for i in range(6)}
        self.assertTrue(validate_config(config_with(cameras)).ok)

    def test_disabled_streams_dont_clash(self):
        data = config_with({
            "a": {"source": RTSP, "stream": {"name": "x"}},
            "b": {"source": RTSP, "stream": {"name": "x", "enabled": False}},
        })
        self.assertTrue(validate_config(data).ok)

    def test_unknown_placeholder(self):
        report = validate_config(config_with({"a": {"source": RTSP, "stream": {"name": "cam_{serial}"}}}))
        self.assertEqual([i.path for i in report.errors], ["cameras.a.stream.name"])

    def test_invalid_path_characters(self):
        report = validate_config(config_with({"a": {"source": RTSP, "stream": {"name": "cam {sn}"}}}))
        self.assertIn("MediaMTX", report.errors[0].message)

    def test_streams_need_a_server_host(self):
        data = config_with({"a": {"source": RTSP, "stream": {}}})
        del data["mediamtx"]
        report = validate_config(data)
        self.assertEqual([i.path for i in report.errors], ["mediamtx.host"])


class TestUrlTemplates(unittest.TestCase):
    def urls(self, host, **kw):
        return build_stream_urls(host, "cam_sn_x", rtsp_port=8554, webrtc_port=8889, **kw)

    def test_4x_defaults_for_a_hostname(self):
        urls = self.urls("mediamtx.example.com")
        self.assertEqual(urls.rtsp, "rtsp://mediamtx.example.com:8554/live/cam_sn_x")
        self.assertEqual(urls.webrtc, "https://mediamtx.example.com/live/cam_sn_x/")
        self.assertEqual(urls.demand, "https://mediamtx.example.com/demand/cam_sn_x")

    def test_4x_defaults_for_a_bare_ip(self):
        urls = self.urls("10.0.0.2")
        self.assertEqual(urls.rtsp, "rtsp://10.0.0.2:8554/live/cam_sn_x")
        self.assertEqual(urls.webrtc, "http://10.0.0.2:8889/live/cam_sn_x/")
        self.assertEqual(urls.demand, "https://10.0.0.2/demand/cam_sn_x")

    def test_webrtc_port_is_honoured(self):
        urls = build_stream_urls("10.0.0.2", "n", rtsp_port=9554, webrtc_port=9889)
        self.assertEqual(urls.webrtc, "http://10.0.0.2:9889/live/cam_sn_n/")
        self.assertEqual(urls.rtsp, "rtsp://10.0.0.2:9554/live/cam_sn_n")

    def test_custom_templates(self):
        urls = self.urls("h", rtsp_url="rtsp://{host}:{rtsp_port}/{name}",
                         webrtc_url="http://{host}:{webrtc_port}/{name}",
                         demand_url="http://{host}:8080/flags/{name}")
        self.assertEqual(urls.rtsp, "rtsp://h:8554/cam_sn_x")
        self.assertEqual(urls.webrtc, "http://h:8889/cam_sn_x")
        self.assertEqual(urls.demand, "http://h:8080/flags/cam_sn_x")

    def test_template_placeholders_validated(self):
        report = validate_config({"node": {"sn": "x"}, "mediamtx": {"host": "h", "rtsp_url": "rtsp://{ip}/{name}"}})
        self.assertEqual([i.path for i in report.errors], ["mediamtx.rtsp_url"])

    def test_streamer_and_config_agree(self):
        config, _ = load_config(config_with(
            {"a": {"source": RTSP, "stream": {}, "capture": {"width": 640, "height": 480}}},
            mediamtx={"host": "10.0.0.2", "webrtc_port": 9000},
        ))
        streamer = MediaMTXStreamer.from_camera(config.cameras["a"], config.mediamtx, config.node)
        urls = config.stream_urls("a")
        self.assertEqual(streamer.rtsp_url, urls.rtsp)
        self.assertEqual(streamer.get_webrtc_url(), urls.webrtc)
        self.assertEqual(streamer._demand_url, urls.demand)
        self.assertEqual(urls.webrtc, "http://10.0.0.2:9000/live/cam_sn_pb-00004_a/")


class TestAudioFilter(unittest.TestCase):
    TAIL = "aresample=48000:async=1000"

    def chain(self, audio_input, filter_cfg):
        config, _ = load_config(config_with({"a": {
            "source": RTSP,
            "stream": {"audio": {"input": audio_input, "filter": filter_cfg}},
        }}))
        resolved = config.cameras["a"].stream.audio.resolved_filter()
        channel = audio_input.get("channel", 0) if isinstance(audio_input, dict) else 0
        return build_audio_filter(config.cameras["a"].stream.audio.input.type, channel=channel,
                                  highpass_hz=resolved.highpass_hz, gain_db=resolved.gain_db,
                                  limiter=resolved.limiter)

    ALSA = {"type": "alsa", "device": "inmp441"}

    def test_alsa_defaults_reproduce_4x(self):
        self.assertEqual(self.chain(self.ALSA, {}),
                         "pan=mono|c0=c0,highpass=f=20,volume=18.0dB,alimiter=limit=0.97:level=false," + self.TAIL)

    def test_highpass_matrix(self):
        cases = {
            None: "pan=mono|c0=c0,",
            20: "pan=mono|c0=c0,highpass=f=20,",
            300: "pan=mono|c0=c0,highpass=f=300:poles=2,highpass=f=300:poles=2,",
        }
        gains = {0: "", 18: "volume=18.0dB,alimiter=limit=0.97:level=false,"}
        for highpass, hp_text in cases.items():
            for gain, gain_text in gains.items():
                with self.subTest(highpass=highpass, gain=gain):
                    self.assertEqual(self.chain(self.ALSA, {"highpass_hz": highpass, "gain_db": gain}),
                                     hp_text + gain_text + self.TAIL)

    def test_highpass_zero_is_off(self):
        self.assertEqual(self.chain(self.ALSA, {"highpass_hz": 0, "gain_db": 0}), "pan=mono|c0=c0," + self.TAIL)

    def test_channel_select(self):
        chain = self.chain({"type": "alsa", "device": "d", "channels": 2, "channel": 1}, {"gain_db": 0})
        self.assertTrue(chain.startswith("pan=mono|c0=c1,highpass=f=20,"))

    def test_channel_must_exist(self):
        config, report = load_config(config_with({"a": {"source": RTSP, "stream": {"audio": {
            "input": {"type": "alsa", "device": "d", "channels": 1, "channel": 1}}}}}))
        self.assertEqual(config.cameras["a"].stream.audio.input.channel, 0)
        self.assertEqual([i.path for i in report.warnings], ["cameras.a.stream.audio.input.channel"])

    def test_camera_defaults_reproduce_4x(self):
        self.assertEqual(self.chain({"type": "camera"}, {}), self.TAIL)
        self.assertEqual(self.chain({"type": "camera"}, {"gain_db": 18}),
                         "volume=18.0dB,alimiter=limit=0.97:level=false," + self.TAIL)
        self.assertEqual(self.chain({"type": "camera"}, {"highpass_hz": 300, "gain_db": 6, "limiter": None}),
                         "highpass=f=300:poles=2,highpass=f=300:poles=2,volume=6.0dB," + self.TAIL)

    def test_filter_defaults_documented_in_model(self):
        self.assertEqual(AlsaAudioInput.filter_defaults, {"highpass_hz": 20.0, "gain_db": 18.0})
        self.assertEqual(AudioFilter.model_fields["limiter"].default, 0.97)


class TestCameraAudio(unittest.TestCase):
    def streamer(self, source, **stream):
        config, _ = load_config(config_with({"a": {
            "source": source, "capture": {"width": 640, "height": 480, "fps": 15},
            "stream": {"audio": {"input": {"type": "camera"}}, **stream},
        }}))
        return MediaMTXStreamer.from_camera(config.cameras["a"], config.mediamtx, config.node)

    def test_uses_the_camera_rtsp_url(self):
        s = self.streamer(RTSP)
        argv = s._build_ffmpeg_cmd(True)
        url = "rtsp://admin:pw1234@192.168.1.64:554/Streaming/Channels/101"
        self.assertEqual(argv[argv.index("-allowed_media_types") + 1], "audio")
        self.assertEqual(argv[argv.index("-i", argv.index("-allowed_media_types")) + 1], url)
        self.assertEqual(s.audio_gain_db, 0.0)

    def test_source_transport_is_reused(self):
        s = self.streamer({**RTSP, "transport": "udp"})
        argv = s._build_ffmpeg_cmd(True)
        self.assertEqual(argv[argv.index("-allowed_media_types") + 2: argv.index("-allowed_media_types") + 4],
                         ["-rtsp_transport", "udp"])

    def test_non_rtsp_url_skips_rtsp_flags(self):
        s = self.streamer({"type": "url", "url": "http://10.0.0.9/audio.mjpg"})
        argv = s._build_ffmpeg_cmd(True)
        self.assertNotIn("-allowed_media_types", argv)
        self.assertIn("http://10.0.0.9/audio.mjpg", argv)
        # the output push keeps its own -rtsp_transport tcp
        self.assertEqual(argv.count("-rtsp_transport"), 1)

    def test_needs_a_network_source(self):
        report = validate_config(config_with({"a": {
            "source": {"type": "device", "index": 0},
            "stream": {"audio": {"input": {"type": "camera"}}}}}))
        self.assertEqual([i.path for i in report.errors], ["cameras.a.stream.audio.input.type"])


class TestSizeAndFps(unittest.TestCase):
    def build(self, capture=None, stream=None):
        config, _ = load_config(config_with({"a": {"source": RTSP, "capture": capture or {}, "stream": stream or {}}}))
        return MediaMTXStreamer.from_camera(config.cameras["a"], config.mediamtx, config.node)

    def test_from_capture(self):
        s = self.build({"width": 1280, "height": 720, "fps": 10})
        self.assertEqual((s.frame_width, s.frame_height, s.fps), (1280, 720, 10))

    def test_stream_overrides(self):
        s = self.build({"width": 1920, "height": 1080, "fps": 25}, {"width": 640, "height": 360, "fps": 5})
        self.assertEqual((s.frame_width, s.frame_height, s.fps), (640, 360, 5))

    def test_unknown_size_is_lazy_and_fps_defaults(self):
        s = self.build()
        self.assertEqual((s.frame_width, s.frame_height, s.fps), (None, None, StreamConfig.DEFAULT_FPS))
        started = []
        self.assertFalse(s._ensure_ffmpeg_started())  # waits for the first frame
        self.assertTrue(s._start_pending)
        s._ensure_ffmpeg_started = lambda: started.append((s.frame_width, s.frame_height)) or False
        s.update_frame(np.zeros((360, 640, 3), dtype=np.uint8))
        self.assertEqual(started, [(640, 360)])

    def test_fractional_fps_in_argv(self):
        s = self.build({"width": 640, "height": 480, "fps": 12.5})
        argv = s._build_ffmpeg_cmd(False)
        self.assertEqual(argv[argv.index("-r") + 1], "12.5")
        self.assertEqual(argv[argv.index("-g") + 1], "12")

    def test_encoder_codec_is_honoured(self):
        s = self.build({"width": 640, "height": 480}, {"encoder": {"codec": "h264_v4l2m2m"}})
        argv = s._build_ffmpeg_cmd(False)
        self.assertEqual(argv[argv.index("-c:v") + 1], "h264_v4l2m2m")

    def test_no_stream_block(self):
        config, _ = load_config(config_with({"a": {"source": RTSP}}))
        with self.assertRaises(ValueError):
            MediaMTXStreamer.from_camera(config.cameras["a"], config.mediamtx, config.node)
        self.assertIsNone(config.stream_urls("a"))


class TestConstructorDefaultsMatchModels(unittest.TestCase):
    """The models own the defaults: a manual constructor call must behave like an empty config."""

    @staticmethod
    def defaults(klass):
        return {n: p.default for n, p in inspect.signature(klass.__init__).parameters.items()
                if p.default is not inspect.Parameter.empty}

    @staticmethod
    def model_default(model, field):
        return model.model_fields[field].get_default(call_default_factory=True)

    def assert_same(self, ctor, mapping):
        for param, (model, field) in mapping.items():
            with self.subTest(param=param):
                expected = self.model_default(model, field)
                actual = ctor[param]
                if isinstance(actual, tuple):
                    actual = list(actual)
                if field == "bitrate":
                    expected, actual = parse_bitrate(expected), parse_bitrate(actual)
                self.assertEqual(actual, expected)

    def test_video_capture(self):
        self.assert_same(self.defaults(VideoCaptureAsync), {
            "width": (CaptureConfig, "width"), "height": (CaptureConfig, "height"), "fps": (CaptureConfig, "fps"),
            "buffer_size": (CaptureConfig, "buffer_size"), "auto_resize": (CaptureConfig, "auto_resize"),
            "hw_decode": (CaptureConfig, "hw_decode"), "open_timeout": (CaptureConfig, "open_timeout"),
            "ffmpeg_options": (CaptureConfig, "ffmpeg_options"),
            "health_log_interval": (CaptureConfig, "health_log_interval"),
            "auto_restart_on_fail": (RestartConfig, "enabled"), "restart_delay": (RestartConfig, "delay"),
            "restart_backoff_start": (RestartConfig, "backoff_start"),
            "restart_backoff_jitter": (RestartConfig, "jitter"),
            "restart_reset_after": (RestartConfig, "reset_after"),
            "max_restart_attempts": (RestartConfig, "max_attempts"),
            "stall_timeout": (RestartConfig, "stall_timeout"),
            "rtsp_transport": (RtspSource, "transport"),
        })

    def test_streamer(self):
        self.assert_same(self.defaults(MediaMTXStreamer), {
            "bitrate": (StreamConfig, "bitrate"), "debug_log_interval": (StreamConfig, "debug_log_interval"),
            "on_demand": (StreamConfig, "on_demand"),
            "frame_width": (StreamConfig, "width"), "frame_height": (StreamConfig, "height"),
            "encoder_preset": (EncoderConfig, "preset"), "encoder_codec": (EncoderConfig, "codec"),
            "stream_queue_size": (EncoderConfig, "queue_size"),
            "webrtc_port": (MediaMTXConfig, "webrtc_port"),
            "rtsp_url": (MediaMTXConfig, "rtsp_url"), "webrtc_url": (MediaMTXConfig, "webrtc_url"),
            "demand_url": (MediaMTXConfig, "demand_url"),
            "demand_poll_interval": (DemandConfig, "poll_interval"),
            "demand_grace_period": (DemandConfig, "grace_period"), "demand_timeout": (DemandConfig, "timeout"),
            "audio_rate": (AlsaAudioInput, "rate"), "audio_channels": (AlsaAudioInput, "channels"),
            "audio_channel": (AlsaAudioInput, "channel"), "audio_limiter": (AudioFilter, "limiter"),
            "audio_retry_backoff": (AudioRetryConfig, "backoff"),
            "audio_retry_backoff_max": (AudioRetryConfig, "backoff_max"),
            "audio_reset_after": (AudioRetryConfig, "reset_after"),
        })
        self.assertEqual(self.defaults(MediaMTXStreamer)["audio_bitrate"], "24k")
        self.assertIsNone(self.defaults(MediaMTXStreamer)["fps"])  # -> StreamConfig.DEFAULT_FPS

    def test_api_client(self):
        self.assert_same(self.defaults(ApiClient), {
            "base_url": (ApiConfig, "base_url"), "timeout": (ApiConfig, "timeout"),
            "success_codes": (ApiConfig, "success_codes"), "default_content_type": (ApiConfig, "content_type"),
            "auth_header": (ApiConfig, "auth_header"),
            "max_retries": (RetriesConfig, "max"), "retry_backoff": (RetriesConfig, "backoff"),
            "retry_backoff_max": (RetriesConfig, "backoff_max"), "retry_on_status": (RetriesConfig, "on_status"),
            "max_workers": (PoolConfig, "workers"), "pool_connections": (PoolConfig, "connections"),
            "pool_maxsize": (PoolConfig, "maxsize"),
            "cache_enabled": (OutboxConfig, "enabled"), "cache_dir": (OutboxConfig, "dir"),
            "cache_retry_interval": (OutboxConfig, "retry_interval"),
            "max_cache_items": (OutboxConfig, "max_items"),
            "max_cache_age_seconds": (OutboxConfig, "max_age"),
            "max_cache_retries": (OutboxConfig, "max_retries"),
        })


if __name__ == "__main__":
    unittest.main()
