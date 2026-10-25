"""
Tests for the typed config entry points: ApiClient.from_config,
YOLOWrapper/YOLOEWrapper model mapping, and the CLI config gate.
(VideoCaptureAsync.from_camera and MediaMTXStreamer.from_camera are covered in
test_capture_frame_hook and test_stream_config.)

Run: uv run python -m unittest tests.test_from_config
"""

import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import yaml

from transformsai_ai_core.api_client import ApiClient
from transformsai_ai_core.config import load_config

REPO = Path(__file__).resolve().parent.parent


class TestApiClientFromConfig(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="api-from-config-")
        self.addCleanup(shutil.rmtree, self.tmp, True)

    def client(self, api, **overrides):
        config, _ = load_config({"node": {"sn": "n"}, "api": api})
        client = ApiClient.from_config(config.api, **overrides)
        self.addCleanup(client.shutdown)
        return client

    def test_regrouped_keys_map_onto_the_constructor(self):
        client = self.client({
            "base_url": "https://api.example.com/v1/",
            "timeout": "15s",
            "auth_keys": "only-key",
            "auth_header": "X-AUTH-KEY",
            "content_type": "JSON",
            "retries": {"max": 1, "backoff": 0.5, "backoff_max": 4, "on_status": [503]},
            "outbox": {"dir": os.path.join(self.tmp, "outbox"), "retry_interval": 0,
                       "max_items": 0, "max_age": None, "max_retries": None},
            "pool": {"connections": 2, "maxsize": 3, "workers": 1},
        })
        self.assertEqual(client.base_url, "https://api.example.com/v1/")
        self.assertEqual(client.timeout, 15.0)
        self.assertEqual(client.auth_keys, ["only-key"])
        self.assertEqual(client.auth_header, "X-AUTH-KEY")
        self.assertEqual(client.default_content_type, "json")
        self.assertEqual(client.max_retries, 1)
        self.assertEqual(client.retry_on_status, (503,))
        self.assertEqual(client.max_workers, 1)
        self.assertTrue(client.cache.cache_dir.endswith("outbox"))
        self.assertEqual(client.cache.max_cache_items, 0)
        self.assertIsNone(client.cache.max_cache_age_seconds)
        self.assertIsNone(client.cache.max_cache_retries)

    def test_endpoints_registered(self):
        client = self.client({
            "base_url": "https://x",
            "outbox": {"enabled": False},
            "endpoints": {
                "heartbeat": "telemetry/heartbeats/",
                "upload": {"path": "data/", "method": "put", "cache": False, "persist": True},
            },
        })
        self.assertEqual(client._endpoints["heartbeat"].path, "telemetry/heartbeats/")
        self.assertEqual(client._endpoints["heartbeat"].method, "POST")
        self.assertEqual(client._endpoints["upload"].method, "PUT")
        self.assertFalse(client._endpoints["upload"].cache)
        self.assertTrue(client._endpoints["upload"].persist)

    def test_overrides_win(self):
        client = self.client({"base_url": "https://old", "outbox": {"enabled": False}},
                             base_url="https://new", headers={"X-Device-SN": "n"})
        self.assertEqual(client.base_url, "https://new")
        self.assertEqual(client.headers, {"X-Device-SN": "n"})

    def test_unbounded_outbox_keeps_retrying(self):
        client = self.client({"base_url": "https://x",
                              "outbox": {"dir": os.path.join(self.tmp, "o"), "retry_interval": 0,
                                         "max_retries": None}})
        cache = client.cache
        cache._read_meta = lambda fid: {"retry_count": 10_000}
        cache._ids = ["a"]
        cache.is_persistent = lambda fid: False
        self.assertFalse(cache.retries_exhausted("a"))
        cache.max_cache_retries = 3
        self.assertTrue(cache.retries_exhausted("a"))


class TestYoloModelDict(unittest.TestCase):
    def test_model_dict_from_typed_model(self):
        from transformsai_ai_core.yolo_wrapper import _model_dict

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(yaml.safe_dump({"node": {"sn": "n"}, "models": {"seg": {
                "path": "models/yoloe.pt", "task": "segment", "batch": 4, "export": True,
                "export_options": {"format": "onnx"}, "set_classes": ["person"]}}}))
            config, _ = load_config(path)
            self.assertEqual(_model_dict(config.models["seg"]), {
                "path": str(Path(tmp) / "models" / "yoloe.pt"),
                "task": "segment",
                "batch": 4,
                "export": True,
                "export_options": {"format": "onnx"},
                "set_classes": ["person"],
            })


class TestCli(unittest.TestCase):
    def run_cli(self, *args):
        return subprocess.run([sys.executable, "-m", "transformsai_ai_core.main", *args],
                              capture_output=True, text=True, cwd=REPO, timeout=60)

    def test_validate_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            good, bad = Path(tmp) / "good.yaml", Path(tmp) / "bad.yaml"
            good.write_text(yaml.safe_dump({"node": {"sn": "n"}}))
            bad.write_text(yaml.safe_dump({"node": {"sn": "n"}, "logging": {"consle_level": "INFO"}}))
            ok = self.run_cli("config", "validate", str(good))
            self.assertEqual(ok.returncode, 0, ok.stdout + ok.stderr)
            failed = self.run_cli("config", "validate", str(bad))
            self.assertEqual(failed.returncode, 1)
            self.assertIn("did you mean 'console_level'", failed.stdout)
            lenient = self.run_cli("config", "validate", "--lenient", str(bad))
            self.assertEqual(lenient.returncode, 0)
            self.assertIn("warning:", lenient.stdout)

    def test_example_and_schema(self):
        example = self.run_cli("config", "example")
        self.assertEqual(example.returncode, 0)
        self.assertEqual(example.stdout, (REPO / "config.example.yaml").read_text(encoding="utf-8"))
        schema = self.run_cli("config", "schema")
        self.assertEqual(schema.returncode, 0)
        self.assertIn('"node"', schema.stdout)


if __name__ == "__main__":
    unittest.main()
