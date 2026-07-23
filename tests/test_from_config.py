"""
Tests for from_config classmethods on config-driven constructors.

Verifies that ApiClient, VideoCaptureAsync, MediaMTXStreamer, and
YOLOWrapper/YOLOEWrapper can be constructed directly from config sections.
"""

import tempfile
import shutil
from pathlib import Path

from transformsai_ai_core.config_schema import (
    ApiConfig, AdvancedConfig, AppConfig, CameraConfig, CaptureSettings,
    LivestreamConfig, MetaConfig,
)
from transformsai_ai_core.config_loader import process_config, save_config, load_config, init_kwargs
from transformsai_ai_core.api_client import ApiClient
from transformsai_ai_core.video_capture import VideoCaptureAsync
from transformsai_ai_core.mediamtx_streamer import MediaMTXStreamer


def print_test_header(test_num: int, description: str):
    print(f"\n{'='*70}")
    print(f"Test {test_num}: {description}")
    print('='*70)


# =============================================================================
# SECTION 1: ApiClient.from_config
# =============================================================================

print_test_header(1, "ApiClient.from_config from ApiConfig model")
api_cfg = ApiConfig(
    base_url="https://api.example.com",
    timeout=15,
    endpoints={"heartbeat": "cameras/heartbeat/"},
    auth_keys=["key1", "key2"],
)
client = ApiClient.from_config(api_cfg)
assert client.base_url == "https://api.example.com"
assert client.timeout == 15
assert len(client.auth_keys) == 2
assert "heartbeat" in client._endpoints
print("✓ ApiClient built from ApiConfig model")


print_test_header(2, "ApiClient.from_config with overrides")
api_cfg_dict = {
    "base_url": "https://old.example.com",
    "timeout": 10,
    "max_retries": 1,
    "endpoints": {},
    "enabled": True,
}
client = ApiClient.from_config(api_cfg_dict, base_url="https://new.example.com", timeout=60)
assert client.base_url == "https://new.example.com"
assert client.timeout == 60
assert client.max_retries == 1
print("✓ Override beats config value")

client.shutdown()


print_test_header(3, "ApiClient.from_config from process_config output")
temp_dir = tempfile.mkdtemp()
try:
    temp_config = Path(temp_dir) / "api_test.yaml"
    cfg = load_config("tests/test-config-1.yaml", validate=False)
    cfg["advanced"]["api"] = {
        "base_url": "https://process.example.com",
        "timeout": 25,
        "endpoints": {"hb": "heartbeat/"},
    }
    save_config(temp_config, cfg)
    processed = process_config(temp_config)
    api_section = processed["advanced"]["api"]
    client = ApiClient.from_config(api_section)
    assert client.base_url == "https://process.example.com"
    assert client.timeout == 25
    assert "hb" in client._endpoints
    print("✓ ApiClient built from process_config output")
    client.shutdown()
finally:
    shutil.rmtree(temp_dir, ignore_errors=True)


print_test_header("3a", "ApiClient.from_config auto-registers endpoints (string + dict profiles)")
api_cfg_eps = ApiConfig(
    base_url="https://ep.example.com",
    endpoints={
        "heartbeat": "cameras/heartbeat/",
        "data": {"path": "cameras/data/", "method": "POST", "cache": False},
    },
)
client = ApiClient.from_config(api_cfg_eps)
assert "heartbeat" in client._endpoints
assert "data" in client._endpoints
assert client._endpoints["heartbeat"].path == "cameras/heartbeat/"
assert client._endpoints["data"].path == "cameras/data/"
assert client._endpoints["data"].cache is False
print("✓ Endpoints auto-registered (string + dict profiles)")
client.shutdown()


# =============================================================================
# SECTION 2: VideoCaptureAsync.from_config
# =============================================================================

print_test_header(4, "VideoCaptureAsync.from_config with local source")
local_cfg = {
    "local": True,
    "local_source": "tests/demo/sample.mp4",
    "capture": {
        "buffer_size": 2,
        "opencv_backend": "auto",
        "fps": 15,
        "width": 640,
        "height": 480,
        "auto_restart_on_fail": True,
    },
}
cap = VideoCaptureAsync.from_config(local_cfg)
assert cap.src == "tests/demo/sample.mp4"
assert cap.buffer_size == 2
assert cap.target_fps == 15
print("✓ VideoCaptureAsync with local source — src is local_source")


print_test_header(5, "VideoCaptureAsync.from_config with RTSP source")
rtsp_cfg = {
    "local": False,
    "rtsp_source": {
        "username": "admin",
        "password": "pass",
        "ip": "10.0.0.5",
        "port": 554,
        "path": "/stream",
    },
    "capture": {
        "buffer_size": 1,
        "max_frame_age_ms": 200,
        "auto_restart_on_fail": True,
        "restart_delay": 15.0,
    },
}
cap_rtsp = VideoCaptureAsync.from_config(rtsp_cfg)
assert cap_rtsp.src == "rtsp://admin:pass@10.0.0.5:554/stream"
assert cap_rtsp.max_frame_age_ms == 200
assert cap_rtsp.auto_restart_on_fail is True
assert cap_rtsp.restart_delay == 15.0
print("✓ VideoCaptureAsync with RTSP — src is built RTSP URL")


print_test_header(6, "VideoCaptureAsync.from_config with pre-built rtsp_url")
processed_cfg = {
    "local": False,
    "rtsp_url": "rtsp://prebuilt:9999/live",
    "rtsp_source": {},
    "capture": {"buffer_size": 5, "auto_restart_on_fail": True},
}
cap_url = VideoCaptureAsync.from_config(processed_cfg)
assert cap_url.src == "rtsp://prebuilt:9999/live"
print("✓ VideoCaptureAsync honors pre-built rtsp_url")


print_test_header(7, "VideoCaptureAsync.from_config with overrides")
cap_ov = VideoCaptureAsync.from_config(local_cfg, src="tests/demo/override.mp4", buffer_size=99,
                                         auto_restart_on_fail=True)
assert cap_ov.src == "tests/demo/override.mp4"
assert cap_ov.buffer_size == 99
print("✓ VideoCaptureAsync overrides beat config")


print_test_header("7b", "VideoCaptureAsync.from_config flattens the nested restart block")
restart_cfg = {
    "local": True,
    "local_source": "tests/demo/sample.mp4",
    "capture": {
        "buffer_size": 1,
        "restart": {
            "enabled": True,
            "delay": 12.5,
            "backoff_start": 0.5,
            "backoff_jitter": 0.0,
            "reset_after": 10.0,
            "max_attempts": 4,
            "stall_timeout": 7.5,
        },
    },
}
cap_restart = VideoCaptureAsync.from_config(restart_cfg)
assert cap_restart.auto_restart_on_fail is True
assert cap_restart.restart_delay == 12.5
assert cap_restart.restart_backoff_start == 0.5
assert cap_restart.restart_backoff_jitter == 0.0
assert cap_restart.restart_reset_after == 10.0
assert cap_restart.max_restart_attempts == 4
assert cap_restart.stall_timeout == 7.5
assert cap_restart.get_stats()["stall_timeout"] == 7.5
print("✓ capture.restart.* flattened onto the constructor")


print_test_header("7c", "capture.restart wins over legacy flat keys; legacy still honored alone")
# Nested block present -> it wins
mixed_cfg = {
    "local": True,
    "local_source": "tests/demo/sample.mp4",
    "capture": {
        "auto_restart_on_fail": False,
        "restart_delay": 99.0,
        "restart": {"enabled": True, "delay": 5.0},
    },
}
cap_mixed = VideoCaptureAsync.from_config(mixed_cfg)
assert cap_mixed.auto_restart_on_fail is True
assert cap_mixed.restart_delay == 5.0

# Nested block absent/defaulted (all None) -> legacy flat keys still apply
legacy_cfg = {
    "local": True,
    "local_source": "tests/demo/sample.mp4",
    "capture": {
        "auto_restart_on_fail": True,
        "restart_delay": 42.0,
        "restart": {"enabled": None, "delay": None, "backoff_start": 1.0},
    },
}
cap_legacy = VideoCaptureAsync.from_config(legacy_cfg)
assert cap_legacy.auto_restart_on_fail is True
assert cap_legacy.restart_delay == 42.0

# A full CaptureSettings model (Pydantic materializes restart with None defaults,
# which must not clobber the legacy flat keys the user actually wrote)
cap_model_legacy = VideoCaptureAsync.from_config(
    CameraConfig(
        local=True,
        local_source="tests/demo/sample.mp4",
        capture=CaptureSettings(auto_restart_on_fail=True, restart_delay=17.0),
    )
)
assert cap_model_legacy.auto_restart_on_fail is True
assert cap_model_legacy.restart_delay == 17.0

# Legacy opt-out reaches the constructor: restart off + unopenable source still raises
try:
    VideoCaptureAsync.from_config(
        CameraConfig(
            local=True,
            local_source="tests/demo/sample.mp4",
            capture=CaptureSettings(auto_restart_on_fail=False),
        )
    )
    raise AssertionError("auto_restart_on_fail=False should raise on an unopenable source")
except RuntimeError:
    pass
print("✓ restart precedence: overrides > restart.* > legacy flat > defaults")


print_test_header("7d", "VideoCaptureAsync defaults are robust out of the box")
cap_default = VideoCaptureAsync.from_config({"local": True, "local_source": "tests/demo/sample.mp4"})
assert cap_default.auto_restart_on_fail is True
assert cap_default.restart_delay == 30.0
assert cap_default.restart_backoff_start == 1.0
assert cap_default.max_restart_attempts is None
assert cap_default.state == "idle"
assert cap_default.is_healthy is False
print("✓ restart-on by default, backoff ramp 1.0 → 30.0, retries forever")


# =============================================================================
# SECTION 3: MediaMTXStreamer.from_config
# =============================================================================

print_test_header(8, "MediaMTXStreamer.from_config flattens encoder + top-level fields")
livestream_cfg = {
    "enabled": True,
    "mediamtx_ip": "192.168.1.1",
    "rtsp_port": 9999,
    "camera_sn_id": "",
    "fps": 15,
    "frame_width": 640,
    "frame_height": 480,
    "bitrate": "500k",
    "hw_encode": False,
    "debug_log_interval": 30.0,
    "encoder": {
        "preset": "fast",
        "codec": "libx264",
        "queue_size": 4,
    },
    "settings": {"draw_annotations": True},
}
streamer = MediaMTXStreamer.from_config(livestream_cfg, camera_sn_id="cam1")
assert streamer.mediamtx_ip == "192.168.1.1"
assert streamer.rtsp_port == 9999
assert streamer.camera_sn_id == "cam1"
assert streamer.fps == 15
assert streamer.frame_width == 640
assert streamer.frame_height == 480
assert streamer.bitrate == "500k"
assert streamer.hw_encode is False
assert streamer.encoder_preset == "fast"
assert streamer.encoder_codec == "libx264"
assert streamer.stream_queue_size == 4
print("✓ MediaMTXStreamer built from livestream section")


print_test_header(9, "MediaMTXStreamer.from_config defaults")
streamer_def = MediaMTXStreamer.from_config({})
assert streamer_def.mediamtx_ip == "localhost"
assert streamer_def.rtsp_port == 8554
assert streamer_def.camera_sn_id == ""
assert streamer_def.fps == 30
assert streamer_def.encoder_preset == "ultrafast"
assert streamer_def.encoder_codec == "copy"
assert streamer_def.stream_queue_size == 2
assert streamer_def.on_demand is False  # backward-compat: always-on by default
print("✓ MediaMTXStreamer defaults from empty config")


print_test_header("9a", "MediaMTXStreamer.from_config on-demand knobs + derived demand URL")
od_cfg = {
    "mediamtx_ip": "vps.example.com",
    "on_demand": True,
    "demand_poll_interval": 1.5,
    "demand_grace_period": 20.0,
    "demand_timeout": 4.0,
}
streamer_od = MediaMTXStreamer.from_config(od_cfg, camera_sn_id="cam1")
assert streamer_od.on_demand is True
assert streamer_od.demand_poll_interval == 1.5
assert streamer_od.demand_grace_period == 20.0
assert streamer_od.demand_timeout == 4.0
# Blank demand_url -> derived from mediamtx_ip + per-camera camera_sn_id override
assert streamer_od._demand_url == "https://vps.example.com/demand/cam_sn_cam1"
print("✓ On-demand knobs map through; demand URL derived from mediamtx_ip")

# Explicit demand_url template wins over derivation
streamer_ov = MediaMTXStreamer.from_config(
    {**od_cfg, "demand_url": "http://10.1.2.3:8080/demand/{camera_sn_id}"},
    camera_sn_id="cam2",
)
assert streamer_ov._demand_url == "http://10.1.2.3:8080/demand/cam2"
print("✓ Explicit demand_url override wins")

# LivestreamConfig model carries the new fields with safe defaults
ls_model = LivestreamConfig()
assert ls_model.on_demand is False
assert ls_model.demand_url == ""
assert ls_model.demand_poll_interval == 3.0
assert ls_model.demand_grace_period == 10.0
assert ls_model.demand_timeout == 5.0
print("✓ LivestreamConfig schema defaults")


# =============================================================================
# SECTION 4: init_kwargs helper
# =============================================================================

print_test_header(10, "init_kwargs filters to __init__ signature")
data = {"base_url": "x", "timeout": 1, "enabled": False, "settings": {"a": 1}, "extras": {"b": 2}}
filtered = init_kwargs(ApiClient, data)
assert "base_url" in filtered
assert "timeout" in filtered
assert "enabled" not in filtered
assert "settings" not in filtered
assert "extras" not in filtered
print("✓ init_kwargs drops control/freeform keys")


# =============================================================================
# SECTION 5: extras behavior
# =============================================================================

print_test_header(11, "extras survives process_config round-trip")
temp_dir = tempfile.mkdtemp()
try:
    temp_config = Path(temp_dir) / "extras_test.yaml"
    cfg = load_config("tests/test-config-1.yaml", validate=False)
    cfg["advanced"]["api"] = {
        "base_url": "https://extras.example.com",
        "extras": {"custom_flag": 42, "debug": True},
    }
    save_config(temp_config, cfg)
    processed = process_config(temp_config)
    api_cfg = processed["advanced"]["api"]
    assert api_cfg["base_url"] == "https://extras.example.com"
    assert api_cfg.get("extras", {}).get("custom_flag") == 42
    assert api_cfg.get("extras", {}).get("debug") is True
    print("✓ extras preserved through process_config")
finally:
    shutil.rmtree(temp_dir, ignore_errors=True)


print_test_header(12, "extras forwarded by from_config as lowest-priority kwargs")
client = ApiClient.from_config(
    {"base_url": "https://test.com", "timeout": 5, "extras": {"max_retries": 7}},
)
assert client.base_url == "https://test.com"
assert client.max_retries == 7
print("✓ extras forwarded by from_config")

# Typed field overrides extras
client2 = ApiClient.from_config(
    {"base_url": "https://test.com", "timeout": 5, "extras": {"timeout": 99}},
)
assert client2.timeout == 5
print("✓ Typed field overrides extras of same name")

client.shutdown()
client2.shutdown()


# =============================================================================
# SECTION 6: Round-trip end-to-end (process_config -> from_config for each class)
# =============================================================================

print_test_header(13, "Round-trip: process_config -> ApiClient")
temp_dir = tempfile.mkdtemp()
try:
    temp_config = Path(temp_dir) / "roundtrip.yaml"
    cfg = load_config("tests/test-config-1.yaml", validate=False)
    cfg["advanced"]["api"] = {
        "base_url": "https://roundtrip.example.com",
        "timeout": 12,
        "endpoints": {"test": "api/test/"},
        "auth_keys": ["rt-key"],
    }
    save_config(temp_config, cfg)
    processed = process_config(temp_config)
    client = ApiClient.from_config(processed["advanced"]["api"])
    assert client.base_url == "https://roundtrip.example.com"
    assert client.timeout == 12
    assert "test" in client._endpoints
    print("✓ Round-trip ApiClient OK")
    client.shutdown()
finally:
    shutil.rmtree(temp_dir, ignore_errors=True)


print_test_header(14, "Round-trip: process_config -> VideoCaptureAsync")
temp_dir = tempfile.mkdtemp()
try:
    temp_config = Path(temp_dir) / "roundtrip_cam.yaml"
    cfg = load_config("tests/test-config-1.yaml", validate=False)
    cfg["cameras"][0]["local"] = False
    cfg["cameras"][0]["rtsp_source"]["ip"] = "10.99.99.99"
    cfg["cameras"][0]["capture"] = {"buffer_size": 3, "fps": 10, "auto_restart_on_fail": True}
    save_config(temp_config, cfg)
    processed = process_config(temp_config)
    cam = VideoCaptureAsync.from_config(processed["cameras"][0])
    assert "10.99.99.99" in str(cam.src)
    assert cam.buffer_size == 3
    assert cam.target_fps == 10
    print("✓ Round-trip VideoCaptureAsync OK")
finally:
    shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# SECTION 7: CaptureSettings from_config
# =============================================================================

print_test_header(15, "VideoCaptureAsync.from_config from CameraConfig model")
cam_model = CameraConfig(
    local=True,
    local_source="tests/demo/model-test.mp4",
    capture=CaptureSettings(
        buffer_size=8,
        fps=25,
        width=1280,
        height=720,
        auto_restart_on_fail=True,
    ),
)
cap_model = VideoCaptureAsync.from_config(cam_model)
assert cap_model.src == "tests/demo/model-test.mp4"
assert cap_model.buffer_size == 8
assert cap_model.target_fps == 25
print("✓ VideoCaptureAsync built from CameraConfig model")


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("ALL from_config TESTS PASSED ✓")
print("="*70)
print("\nfrom_config Validated:")
print("  ✓ ApiClient.from_config (model, dict, process_config output, overrides, extras)")
print("  ✓ VideoCaptureAsync.from_config (local, RTSP, pre-built URL, CameraConfig model)")
print("  ✓ VideoCaptureAsync restart policy (nested block, legacy flat keys, precedence)")
print("  ✓ MediaMTXStreamer.from_config (full config, defaults, encoder flattening)")
print("  ✓ init_kwargs filtering")
print("  ✓ extras round-trip survival + forwarding")
print("  ✓ Round-trip: process_config → from_config (ApiClient, VideoCaptureAsync)")
print("="*70)
