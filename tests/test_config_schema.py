"""
Tests for the 5.0 config system: schema, lenient/strict validation, reports,
project validators, secrets, JSON schema and the generated example.

Run: uv run python -m unittest tests.test_config_schema
"""

import json
import tempfile
import unittest
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict

from transformsai_ai_core.config import (
    Config,
    ConfigError,
    DeviceSource,
    FileSource,
    Issue,
    RtspSource,
    UrlSource,
    clear_registry,
    generate_example,
    load_config,
    parse_bitrate,
    parse_duration,
    register_secret,
    register_validator,
    save_config,
    validate_config,
)

REPO = Path(__file__).resolve().parent.parent


def minimal(**sections):
    return {"node": {"sn": "node-1"}, **sections}


def paths(issues):
    return [issue.path for issue in issues]


class RegistryCleanup(unittest.TestCase):
    def setUp(self):
        clear_registry()
        self.addCleanup(clear_registry)


class TestSources(RegistryCleanup):
    def camera(self, source):
        config, report = load_config(minimal(cameras={"cam": {"source": source}}))
        return config.cameras["cam"].source, report

    def test_rtsp_url_built_with_encoded_credentials(self):
        source, report = self.camera({"type": "rtsp", "host": "10.0.0.5", "user": "ad min",
                                      "password": "p@ss/w:rd", "path": "Streaming/Channels/102"})
        self.assertIsInstance(source, RtspSource)
        self.assertEqual(source.url, "rtsp://ad%20min:p%40ss%2Fw%3Ard@10.0.0.5:554/Streaming/Channels/102")
        self.assertEqual(source.display, "rtsp://ad%20min:***@10.0.0.5:554/Streaming/Channels/102")
        self.assertEqual(source.transport, "tcp")
        self.assertEqual(source.kind, "network")
        self.assertTrue(report.ok)

    def test_rtsp_without_credentials(self):
        source, _ = self.camera({"type": "rtsp", "host": "cam.local", "port": "8554"})
        self.assertEqual(source.url, "rtsp://cam.local:8554/Streaming/Channels/101")

    def test_rtsp_host_required(self):
        with self.assertRaises(ConfigError) as ctx:
            self.camera({"type": "rtsp"})
        self.assertIn("cameras.cam.source.host", paths(ctx.exception.report.errors))

    def test_device_by_path_or_index(self):
        source, _ = self.camera({"type": "device", "path": "/dev/video-camera0"})
        self.assertIsInstance(source, DeviceSource)
        self.assertEqual(source.target, "/dev/video-camera0")
        self.assertEqual(source.kind, "device")
        source, _ = self.camera({"type": "device", "index": 2})
        self.assertEqual(source.target, 2)
        source, _ = self.camera({"type": "device", "path": "1"})
        self.assertEqual(source.target, 1)

    def test_device_needs_exactly_one(self):
        for bad in ({"type": "device"}, {"type": "device", "path": "/dev/video0", "index": 0}):
            with self.assertRaises(ConfigError) as ctx:
                self.camera(bad)
            self.assertIn("exactly one", str(ctx.exception))

    def test_file_source(self):
        source, _ = self.camera({"type": "file", "path": "clip.mkv"})
        self.assertIsInstance(source, FileSource)
        self.assertTrue(source.loop)
        self.assertEqual(source.kind, "file")
        source, _ = self.camera({"type": "file", "path": "clip.mkv", "loop": "false"})
        self.assertFalse(source.loop)

    def test_url_source_and_shorthand(self):
        source, _ = self.camera({"type": "url", "url": "rtsp://u:secretpw@10.0.0.9/live"})
        self.assertIsInstance(source, UrlSource)
        self.assertEqual(source.display, "rtsp://u:***@10.0.0.9/live")
        self.assertEqual(source.password, "secretpw")
        source, _ = self.camera("http://10.0.0.9/mjpg")
        self.assertIsInstance(source, UrlSource)
        source, _ = self.camera(0)
        self.assertIsInstance(source, DeviceSource)

    def test_type_is_case_insensitive(self):
        source, _ = self.camera({"type": "RTSP", "host": "h"})
        self.assertIsInstance(source, RtspSource)

    def test_unknown_type_suggests(self):
        with self.assertRaises(ConfigError) as ctx:
            self.camera({"type": "rtps", "host": "h"})
        issue = ctx.exception.report.errors[0]
        self.assertEqual(issue.path, "cameras.cam.source.type")
        self.assertEqual(issue.suggestion, "rtsp")


class TestCameraForms(RegistryCleanup):
    def test_list_is_named_by_position(self):
        config, _ = load_config(minimal(cameras=[
            {"source": {"type": "rtsp", "host": "10.0.0.11"}},
            {"source": {"type": "rtsp", "host": "10.0.0.12"}},
        ]))
        self.assertEqual(list(config.cameras), ["camera_0", "camera_1"])
        self.assertEqual(config.cameras["camera_1"].name, "camera_1")
        self.assertEqual(config.cameras["camera_1"].index, 1)
        self.assertEqual(config.cameras["camera_1"].source.host, "10.0.0.12")

    def test_map_keeps_order_and_names(self):
        config, _ = load_config(minimal(cameras={
            "ocr": {"source": {"type": "device", "path": "/dev/video0"}},
            "surveillance": {"source": {"type": "rtsp", "host": "h"}},
        }))
        self.assertEqual([(c.name, c.index) for c in config.cameras.values()],
                         [("ocr", 0), ("surveillance", 1)])

    def test_list_errors_use_generated_names(self):
        with self.assertRaises(ConfigError) as ctx:
            load_config(minimal(cameras=[{"source": {"type": "rtsp", "host": "h"}}, {"source": {"type": "rtsp"}}]))
        self.assertIn("cameras.camera_1.source.host", paths(ctx.exception.report.errors))

    def test_no_cameras(self):
        config, report = load_config(minimal())
        self.assertEqual(config.cameras, {})
        self.assertTrue(report.ok)


class TestUnknownKeys(RegistryCleanup):
    CONFIG = minimal(
        cameras={"cam": {
            "source": {"type": "rtsp", "host": "h", "pasword": "x"},
            "capture": {"fsp": 10},
            "stream": {"bitrat": "800k"},
            "settings": {"anything": {"goes": True}},
        }},
        mediamtx={"host": "m"},
        settings={"free": 1},
    )

    def test_lenient_warns_with_suggestions(self):
        config, report = load_config(self.CONFIG)
        self.assertTrue(report.ok)
        found = {issue.path: issue.suggestion for issue in report.warnings}
        self.assertEqual(found["cameras.cam.source.pasword"], "password")
        self.assertEqual(found["cameras.cam.capture.fsp"], "fps")
        self.assertEqual(found["cameras.cam.stream.bitrat"], "bitrate")
        # nothing under settings is reported
        self.assertFalse([p for p in found if "settings" in p])
        # the unknown value is kept, not silently dropped
        self.assertEqual(config.cameras["cam"].capture.model_extra, {"fsp": 10})

    def test_strict_errors(self):
        report = validate_config(self.CONFIG)
        self.assertFalse(report.ok)
        self.assertEqual(len(report.errors), 3)
        with self.assertRaises(ConfigError):
            load_config(self.CONFIG, strict=True)

    def test_4x_keys_get_a_migration_hint(self):
        report = validate_config({"meta": {"name": "x"}, "node": {"sn": "x"}, "advanced": {}})
        messages = {issue.path: issue.message for issue in report.errors}
        self.assertIn("node", messages["meta"])
        self.assertIn("top level", messages["advanced"])

    def test_issue_string(self):
        self.assertEqual(str(Issue("a.b", "unknown key", "c")), "a.b: unknown key (did you mean 'c'?)")


class TestRequiredAndLenientFallback(RegistryCleanup):
    def test_missing_node_sn(self):
        with self.assertRaises(ConfigError) as ctx:
            load_config({"cameras": {}})
        self.assertIn("node", paths(ctx.exception.report.errors))

    def test_all_problems_reported_at_once(self):
        with self.assertRaises(ConfigError) as ctx:
            load_config({
                "node": {"sn": ""},
                "cameras": [{"source": {"type": "rtsp"}}],
                "api": {"enabled": True},
                "models": {"m": {}},
            })
        errors = paths(ctx.exception.report.errors)
        for expected in ("node.sn", "cameras.camera_0.source.host", "api.base_url", "models.m"):
            self.assertIn(expected, errors)

    def test_invalid_optional_falls_back_with_warning(self):
        config, report = load_config(minimal(cameras={"cam": {
            "source": {"type": "rtsp", "host": "h"},
            "capture": {"fps": -5, "buffer_size": "lots", "open_timeout": "soon"},
        }}))
        capture = config.cameras["cam"].capture
        self.assertIsNone(capture.fps)
        self.assertEqual(capture.buffer_size, 1)
        self.assertEqual(capture.open_timeout, 5.0)
        self.assertEqual(sorted(paths(report.warnings)), [
            "cameras.cam.capture.buffer_size", "cameras.cam.capture.fps", "cameras.cam.capture.open_timeout"])
        self.assertIn("using", report.warnings[0].message)

    def test_strict_rejects_invalid_optional(self):
        report = validate_config(minimal(cameras={"cam": {"source": {"type": "rtsp", "host": "h"},
                                                          "capture": {"fps": -5}}}))
        self.assertEqual(paths(report.errors), ["cameras.cam.capture.fps"])

    def test_api_base_url_required_only_when_enabled(self):
        load_config(minimal(api={"enabled": False}))
        with self.assertRaises(ConfigError):
            load_config(minimal(api={}))

    def test_model_needs_a_source(self):
        for ok in ({"path": "m.pt"}, {"download_key": "k"}, {"type": "person-det"}):
            load_config(minimal(models={"m": ok}))
        with self.assertRaises(ConfigError) as ctx:
            load_config(minimal(models={"m": {"path": "", "download_key": ""}}))
        self.assertIn("models.m", paths(ctx.exception.report.errors))

    def test_yaml_errors_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text("node: [unclosed\n")
            self.assertFalse(validate_config(path).ok)
            with self.assertRaises(ConfigError):
                load_config(path)


class TestTypes(RegistryCleanup):
    def test_no_policy_ranges(self):
        config, report = load_config(minimal(
            cameras={"cam": {"source": {"type": "rtsp", "host": "h"},
                             "capture": {"fps": 0.001, "restart": {"delay": 0.001, "max_attempts": 1}}}},
            mediamtx={"host": "m", "demand": {"poll_interval": 0.001}},
        ))
        self.assertTrue(report.ok)
        self.assertEqual(config.cameras["cam"].capture.fps, 0.001)
        self.assertEqual(config.mediamtx.demand.poll_interval, 0.001)

    def test_physical_limits(self):
        for bad in ({"poll_interval": 0}, {"poll_interval": -1}):
            report = validate_config(minimal(mediamtx={"host": "m", "demand": bad}))
            self.assertEqual(paths(report.errors), ["mediamtx.demand.poll_interval"])
        report = validate_config(minimal(mediamtx={"host": "m", "rtsp_port": 70000}))
        self.assertEqual(paths(report.errors), ["mediamtx.rtsp_port"])

    def test_lax_conversion(self):
        config, _ = load_config(minimal(
            cameras={"cam": {"source": {"type": "rtsp", "host": "h", "port": "554"},
                             "capture": {"fps": "10", "hw_decode": "true"}}},
            logging={"console_level": "warn"},
        ))
        camera = config.cameras["cam"]
        self.assertEqual(camera.source.port, 554)
        self.assertEqual(camera.capture.fps, 10)
        self.assertTrue(camera.capture.hw_decode)
        self.assertEqual(config.logging.console_level, "WARNING")

    def test_numbers_become_strings_where_strings_are_expected(self):
        config, _ = load_config({"node": {"sn": 4, "version": 1.0}})
        self.assertEqual(config.node.sn, "4")
        self.assertEqual(config.node.version, "1.0")

    def test_durations(self):
        self.assertEqual(parse_duration(3), 3.0)
        self.assertEqual(parse_duration("3"), 3.0)
        self.assertEqual(parse_duration("500ms"), 0.5)
        self.assertEqual(parse_duration("3s"), 3.0)
        self.assertEqual(parse_duration("2m"), 120.0)
        self.assertEqual(parse_duration("1h"), 3600.0)
        with self.assertRaises(ValueError):
            parse_duration("3 parsecs")
        config, _ = load_config(minimal(api={"base_url": "https://x", "outbox": {"max_age": "24h"}}))
        self.assertEqual(config.api.outbox.max_age, 86400.0)

    def test_bitrates(self):
        self.assertEqual(parse_bitrate("800k"), 800_000)
        self.assertEqual(parse_bitrate("1.2M"), 1_200_000)
        self.assertEqual(parse_bitrate(800000), 800_000)
        self.assertEqual(parse_bitrate("24K"), 24_000)
        with self.assertRaises(ValueError):
            parse_bitrate("fast")
        config, _ = load_config(minimal(cameras={"cam": {"source": {"type": "rtsp", "host": "h"},
                                                         "stream": {"bitrate": "1.2M"}}},
                                        mediamtx={"host": "m"}))
        self.assertEqual(config.cameras["cam"].stream.bitrate, 1_200_000)
        self.assertEqual(config.model_dump()["cameras"]["cam"]["stream"]["bitrate"], "1200k")

    def test_empty_sections_mean_defaults(self):
        config, report = load_config(yaml.safe_load(
            "node: {sn: x}\ncameras:\n  c:\n    source: {type: rtsp, host: h}\n    capture:\n    settings:\n"))
        self.assertTrue(report.ok)
        self.assertEqual(config.cameras["c"].capture.buffer_size, 1)
        self.assertEqual(config.cameras["c"].settings, {})


class TestProjectValidators(RegistryCleanup):
    class Meter(BaseModel):
        model_config = ConfigDict(extra="forbid")
        read_interval: float = 60.0

    def test_pydantic_model_validator(self):
        register_validator("settings.meter", self.Meter)
        load_config(minimal(settings={"meter": {"read_interval": 5}}))
        report = validate_config(minimal(settings={"meter": {"read_intervall": 5}}))
        self.assertEqual(report.errors[0].path, "settings.meter.read_intervall")
        self.assertEqual(report.errors[0].suggestion, "read_interval")

    def test_lenient_is_warning_unless_required(self):
        register_validator("settings.meter", self.Meter)
        _, report = load_config(minimal(settings={"meter": {"read_interval": "never"}}))
        self.assertEqual(paths(report.warnings), ["settings.meter.read_interval"])
        clear_registry()
        register_validator("settings.meter", self.Meter, required=True)
        with self.assertRaises(ConfigError):
            load_config(minimal(settings={"meter": {"read_interval": "never"}}))

    def test_function_validator_with_wildcard(self):
        def check_flash(value):
            if value.get("pin", 0) > 40:
                return "pin out of range"

        register_validator("cameras.*.settings.flash", check_flash)
        report = validate_config(minimal(cameras={
            "a": {"source": {"type": "device", "index": 0}, "settings": {"flash": {"pin": 99}}},
            "b": {"source": {"type": "device", "index": 1}, "settings": {"flash": {"pin": 3}}},
        }))
        self.assertEqual([(i.path, i.message) for i in report.errors],
                         [("cameras.a.settings.flash", "pin out of range")])

    def test_raising_and_broken_validators_are_contained(self):
        def raises(value):
            raise ValueError("bad meter")

        def broken(value):
            return value["missing"]

        register_validator("settings.meter", raises)
        register_validator("settings.meter", broken)
        report = validate_config(minimal(settings={"meter": {}}))
        self.assertEqual(len(report.errors), 2)
        self.assertIn("bad meter", report.errors[0].message)
        self.assertIn("KeyError", report.errors[1].message)

    def test_absent_path_is_skipped(self):
        register_validator("settings.meter", self.Meter, required=True)
        load_config(minimal())


class TestSecrets(RegistryCleanup):
    CONFIG = minimal(
        cameras={
            "a": {"source": {"type": "rtsp", "host": "h", "user": "admin", "password": "camPass1"}},
            "b": {"source": {"type": "url", "url": "rtsp://u:urlPass2@10.0.0.1/s"}},
        },
        api={"base_url": "https://x", "auth_keys": ["key-aaaa", "key-bbbb"]},
        settings={"modbus": {"password": "modbusPw"}},
    )

    def test_secrets_listed(self):
        register_secret("settings.modbus.password")
        config, _ = load_config(self.CONFIG)
        self.assertEqual(sorted(config.secrets()),
                         sorted(["camPass1", "urlPass2", "key-aaaa", "key-bbbb", "modbusPw"]))

    def test_masked_dump(self):
        register_secret("settings.modbus.password")
        config, _ = load_config(self.CONFIG)
        dump = config.masked_dump()
        self.assertEqual(dump["cameras"]["a"]["source"]["password"], "***")
        self.assertEqual(dump["cameras"]["b"]["source"]["url"], "rtsp://u:***@10.0.0.1/s")
        self.assertEqual(dump["api"]["auth_keys"], ["***", "***"])
        self.assertEqual(dump["settings"]["modbus"]["password"], "***")
        text = config.masked_dump("yaml")
        for secret in config.secrets():
            self.assertNotIn(secret, text)
        # the config itself is untouched
        self.assertEqual(config.cameras["a"].source.password, "camPass1")


class TestSchemaAndExample(RegistryCleanup):
    def test_schema_json(self):
        schema = json.loads(Config.schema_json())
        self.assertIn("node", schema["properties"])
        self.assertIn("node", schema["required"])
        self.assertFalse(schema["additionalProperties"])
        # settings stay open
        self.assertNotEqual(schema["properties"]["settings"].get("additionalProperties"), False)
        self.assertTrue(json.loads(Config.schema_json(strict=False))["additionalProperties"])

    def test_example_validates_strictly(self):
        report = validate_config(yaml.safe_load(generate_example()))
        self.assertTrue(report.ok, str(report))
        self.assertEqual(report.warnings, [])

    def test_committed_example_is_fresh(self):
        committed = (REPO / "config.example.yaml").read_text(encoding="utf-8")
        self.assertEqual(committed, generate_example(),
                         "regenerate: uv run transformsaicore config example -o config.example.yaml")


class TestSaveAndPaths(RegistryCleanup):
    def test_save_writes_only_set_keys_and_refuses_invalid(self):
        config, _ = load_config(minimal(cameras={"c": {"source": {"type": "rtsp", "host": "h"}}}))
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "out.yaml"
            save_config(path, config)
            written = yaml.safe_load(path.read_text())
            self.assertEqual(written, {"node": {"sn": "node-1"},
                                       "cameras": {"c": {"source": {"type": "rtsp", "host": "h"}}}})
            with self.assertRaises(ConfigError):
                save_config(path, {"node": {}})

    def test_model_paths_resolve_against_config_dir(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(yaml.safe_dump(minimal(models={"det": {"path": "models/det.pt"},
                                                           "dl": {"download_key": "abc"}})))
            config, _ = load_config(path)
            self.assertEqual(config.models["det"].resolved_path, Path(tmp) / "models" / "det.pt")
            self.assertEqual(config.models["dl"].resolved_path, Path(tmp) / "models" / "dl" / "dl")
            # a single downloaded artefact in the model dir is found on the next run
            (Path(tmp) / "models" / "dl").mkdir(parents=True)
            (Path(tmp) / "models" / "dl" / "dl-v2.rknn").write_text("")
            self.assertEqual(config.models["dl"].resolved_path, Path(tmp) / "models" / "dl" / "dl-v2.rknn")

    def test_nothing_written_into_the_input(self):
        data = minimal(cameras={"c": {"source": {"type": "rtsp", "host": "h", "password": "pw12"}}})
        snapshot = json.dumps(data, sort_keys=True)
        load_config(data)
        self.assertEqual(json.dumps(data, sort_keys=True), snapshot)


if __name__ == "__main__":
    unittest.main()
