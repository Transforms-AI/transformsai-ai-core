#!/usr/bin/env python3
"""Functional tests for config management system."""

import tempfile
import shutil
from pathlib import Path
from pydantic import ValidationError

from transformsai_ai_core.config_schema import AppConfig
from transformsai_ai_core.config_loader import (
    load_config,
    save_config,
    process_config,
    build_rtsp_url,
)


def print_test_header(test_num: int, description: str):
    """Print formatted test header."""
    print(f"\n{'='*70}")
    print(f"Test {test_num}: {description}")
    print('='*70)


# =============================================================================
# SECTION 1: LOADING CONFIGS
# =============================================================================

print_test_header(1, "Load and Validate test-config-1.yaml (Sentiment)")
config1 = load_config("tests/test-config-1.yaml", validate=True)
assert config1["meta"]["name"] == "sentiment"
assert len(config1["cameras"]) == 3
assert len(config1["advanced"]["models"]) == 1
print(f"✓ Loaded {config1['meta']['name']} with {len(config1['cameras'])} cameras")

print_test_header(2, "Load and Validate test-config-2.yaml (People Count)")
config2 = load_config("tests/test-config-2.yaml", validate=True)
assert config2["meta"]["name"] == "people-count"
assert len(config2["cameras"]) == 1
assert len(config2["advanced"]["models"]) == 2
print(f"✓ Loaded {config2['meta']['name']} with {len(config2['advanced']['models'])} models")


# =============================================================================
# SECTION 2: EDITING STRUCTURED FIELDS
# =============================================================================

print_test_header(3, "Edit Meta Fields")
config1["meta"]["token"] = "test-token-12345"
config1["meta"]["version"] = "2.0.0"
validated = AppConfig(**config1)
assert validated.meta.token == "test-token-12345"
assert validated.meta.version == "2.0.0"
print("✓ Meta fields edited successfully")

print_test_header(4, "Edit Camera Structured Fields")
config1["cameras"][0]["local"] = False
config1["cameras"][0]["rtsp_source"]["ip"] = "192.168.1.100"
config1["cameras"][0]["rtsp_source"]["port"] = 8554
config1["cameras"][0]["rtsp_source"]["password"] = "newpass@123"
validated = AppConfig(**config1)
assert validated.cameras[0].local == False
assert validated.cameras[0].rtsp_source.ip == "192.168.1.100"
assert validated.cameras[0].rtsp_source.port == 8554
print("✓ Camera structured fields edited successfully")

print_test_header(5, "Edit Model Structured Fields")
config2["advanced"]["models"][0]["batch"] = 4
config2["advanced"]["models"][0]["backend"] = "onnx"
config2["advanced"]["models"][1]["task"] = "classify"
validated = AppConfig(**config2)
assert validated.advanced.models[0].batch == 4
assert validated.advanced.models[0].backend == "onnx"
assert validated.advanced.models[1].task == "classify"
print("✓ Model structured fields edited successfully")

print_test_header(6, "Edit Datasend Structured Fields")
config1["advanced"]["datasend"]["enabled"] = False
config1["advanced"]["datasend"]["base_url"] = "https://new-api.example.com"
config1["advanced"]["datasend"]["secret_keys"].append("new-key-6789")
validated = AppConfig(**config1)
assert validated.advanced.datasend.enabled == False
assert validated.advanced.datasend.base_url == "https://new-api.example.com"
assert len(validated.advanced.datasend.secret_keys) == 6
print("✓ Datasend structured fields edited successfully")

print_test_header(7, "Edit Livestream Structured Fields")
config2["advanced"]["livestream"]["enabled"] = False
config2["advanced"]["livestream"]["mediamtx_ip"] = "10.0.0.5"
config2["advanced"]["livestream"]["rtsp_port"] = 9554
validated = AppConfig(**config2)
assert validated.advanced.livestream.enabled == False
assert validated.advanced.livestream.mediamtx_ip == "10.0.0.5"
assert validated.advanced.livestream.rtsp_port == 9554
print("✓ Livestream structured fields edited successfully")


# =============================================================================
# SECTION 3: EDITING FREEFORM FIELDS
# =============================================================================

print_test_header(8, "Edit Camera Settings (Freeform)")
config1["cameras"][0]["settings"]["new_param"] = "test_value"
config1["cameras"][0]["settings"]["resolution"]["width"] = 1920
config1["cameras"][0]["settings"]["custom"] = {"nested": {"deep": "value"}}
validated = AppConfig(**config1)
assert validated.cameras[0].settings["new_param"] == "test_value"
assert validated.cameras[0].settings["resolution"]["width"] == 1920
assert validated.cameras[0].settings["custom"]["nested"]["deep"] == "value"
print("✓ Camera freeform settings edited successfully")

print_test_header(9, "Edit Model Export Options (Freeform)")
config2["advanced"]["models"][0]["export_options"]["new_flag"] = True
config2["advanced"]["models"][0]["export_options"]["img_size"] = 1280
config2["advanced"]["models"][0]["export_options"]["custom_dict"] = {"a": 1, "b": 2}
validated = AppConfig(**config2)
assert validated.advanced.models[0].export_options["new_flag"] == True
assert validated.advanced.models[0].export_options["img_size"] == 1280
assert validated.advanced.models[0].export_options["custom_dict"]["b"] == 2
print("✓ Model export_options edited successfully")

print_test_header(10, "Edit Timings (Freeform)")
config1["advanced"]["timings"]["new_interval"] = 60
config1["advanced"]["timings"]["inference_interval"] = 0.05
config1["advanced"]["timings"]["custom_timing"] = {"min": 10, "max": 100}
validated = AppConfig(**config1)
assert validated.advanced.timings["new_interval"] == 60
assert validated.advanced.timings["inference_interval"] == 0.05
assert validated.advanced.timings["custom_timing"]["max"] == 100
print("✓ Timings freeform fields edited successfully")

print_test_header(11, "Edit Datasend Endpoints and Settings (Freeform)")
config1["advanced"]["datasend"]["endpoints"]["new_endpoint"] = "api/new/"
config1["advanced"]["datasend"]["settings"]["timeout"] = 30
config1["advanced"]["datasend"]["settings"]["retry"] = {"max": 3, "delay": 5}
validated = AppConfig(**config1)
assert validated.advanced.datasend.endpoints["new_endpoint"] == "api/new/"
assert validated.advanced.datasend.settings["timeout"] == 30
assert validated.advanced.datasend.settings["retry"]["max"] == 3
print("✓ Datasend freeform fields edited successfully")

print_test_header(12, "Edit Livestream Settings (Freeform)")
config2["advanced"]["livestream"]["settings"]["new_option"] = "enabled"
config2["advanced"]["livestream"]["settings"]["fps"] = 60
config2["advanced"]["livestream"]["settings"]["codec"] = {"name": "h264", "bitrate": 5000}
validated = AppConfig(**config2)
assert validated.advanced.livestream.settings["new_option"] == "enabled"
assert validated.advanced.livestream.settings["fps"] == 60
assert validated.advanced.livestream.settings["codec"]["bitrate"] == 5000
print("✓ Livestream freeform settings edited successfully")

print_test_header(13, "Edit Pipeline (Freeform)")
config2["advanced"]["pipeline"]["new_feature"] = True
config2["advanced"]["pipeline"]["use_shared_memory"] = True
config2["advanced"]["pipeline"]["tracking_zone"]["rx1"] = 0.1
config2["advanced"]["pipeline"]["new_zone"] = {"x": 100, "y": 200, "w": 300, "h": 400}
validated = AppConfig(**config2)
assert validated.advanced.pipeline["new_feature"] == True
assert validated.advanced.pipeline["use_shared_memory"] == True
assert validated.advanced.pipeline["tracking_zone"]["rx1"] == 0.1
assert validated.advanced.pipeline["new_zone"]["w"] == 300
print("✓ Pipeline freeform fields edited successfully")


# =============================================================================
# SECTION 4: VALIDATION ERROR HANDLING
# =============================================================================

print_test_header(14, "Invalid Type for Structured Field")
try:
    invalid_config = config1.copy()
    invalid_config["meta"]["name"] = 123  # Should be string
    AppConfig(**invalid_config)
    assert False, "Should have raised ValidationError"
except ValidationError as e:
    print(f"✓ Validation error caught: {e.error_count()} error(s)")

print_test_header(15, "Invalid Model Task")
try:
    invalid_config = config2.copy()
    invalid_config["advanced"]["models"][0]["task"] = "invalid_task"
    AppConfig(**invalid_config)
    assert False, "Should have raised ValidationError"
except ValidationError as e:
    print(f"✓ Validation error caught: {e.error_count()} error(s)")

print_test_header(16, "Missing Required Field")
try:
    invalid_config = {"cameras": [], "advanced": {}}  # Missing meta
    AppConfig(**invalid_config)
    assert False, "Should have raised ValidationError"
except ValidationError as e:
    print(f"✓ Validation error caught: {e.error_count()} error(s)")


# =============================================================================
# SECTION 5: RTSP URL BUILDING
# =============================================================================

print_test_header(17, "Build RTSP URL with Auth")
rtsp_source = {
    "username": "admin",
    "password": "pass123",
    "ip": "192.168.1.50",
    "port": 554,
    "path": "/stream1"
}
url = build_rtsp_url(rtsp_source)
assert url == "rtsp://admin:pass123@192.168.1.50:554/stream1"
print(f"✓ RTSP URL: {url}")

print_test_header(18, "Build RTSP URL with Special Characters in Password")
rtsp_source = {
    "username": "admin",
    "password": "p@ss#123",
    "ip": "10.0.0.1",
    "port": 8554,
    "path": "/live"
}
url = build_rtsp_url(rtsp_source)
assert "p%40ss%23123" in url
print(f"✓ RTSP URL with encoded password: {url}")

print_test_header(19, "Build RTSP URL without Auth")
rtsp_source = {
    "username": "",
    "password": "",
    "ip": "192.168.1.100",
    "port": 554,
    "path": "/stream"
}
url = build_rtsp_url(rtsp_source)
assert url == "rtsp://192.168.1.100:554/stream"
print(f"✓ RTSP URL without auth: {url}")


# =============================================================================
# SECTION 6: SAVE/LOAD CYCLE
# =============================================================================

print_test_header(20, "Save and Reload Config")
temp_dir = tempfile.mkdtemp()
try:
    temp_config_path = Path(temp_dir) / "test_config.yaml"
    
    # Prepare modified config (reload from file to get clean state)
    test_config = load_config("tests/test-config-1.yaml", validate=True)
    test_config["meta"]["token"] = "saved-token"
    test_config["cameras"][0]["settings"]["saved_field"] = "saved_value"
    test_config["advanced"]["timings"]["saved_timing"] = 999
    
    # Save
    save_config(temp_config_path, test_config)
    assert temp_config_path.exists()
    print(f"✓ Config saved to {temp_config_path}")
    
    # Reload
    reloaded = load_config(temp_config_path, validate=True)
    assert reloaded["meta"]["token"] == "saved-token"
    assert reloaded["cameras"][0]["settings"]["saved_field"] == "saved_value"
    assert reloaded["advanced"]["timings"]["saved_timing"] == 999
    print("✓ Config reloaded and verified")
    
finally:
    shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# SECTION 7: END-TO-END WORKFLOW
# =============================================================================

print_test_header(21, "Process Config with RTSP URL Building")
temp_dir = tempfile.mkdtemp()
try:
    temp_config_path = Path(temp_dir) / "workflow_config.yaml"
    
    # Create config with non-local camera
    workflow_config = load_config("tests/test-config-2.yaml", validate=True)
    workflow_config["cameras"][0]["local"] = False
    workflow_config["cameras"][0]["rtsp_source"]["ip"] = "192.168.5.10"
    
    save_config(temp_config_path, workflow_config)
    
    # Process config (should build RTSP URLs)
    processed = process_config(temp_config_path, base_dir=temp_dir)
    
    assert "rtsp_url" in processed["cameras"][0]
    assert processed["cameras"][0]["rtsp_url"].startswith("rtsp://")
    assert "192.168.5.10" in processed["cameras"][0]["rtsp_url"]
    print(f"✓ RTSP URL built: {processed['cameras'][0]['rtsp_url']}")
    
finally:
    shutil.rmtree(temp_dir, ignore_errors=True)


# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "="*70)
print("ALL TESTS PASSED ✓")
print("="*70)
print("\nConfig Management System Validated:")
print("  ✓ Load and validate configs")
print("  ✓ Edit structured fields (meta, cameras, models, datasend, livestream)")
print("  ✓ Edit freeform fields (settings, timings, pipeline, export_options)")
print("  ✓ Validation error handling")
print("  ✓ RTSP URL building")
print("  ✓ Save/reload cycle")
print("  ✓ End-to-end workflow")
print("="*70)
