import os
import json
import unittest
import io
from contextlib import redirect_stderr
from datetime import datetime
from pathlib import Path
import transformsai_ai_core.central_logger as cv_logger
from transformsai_ai_core.config import LoggingConfig
from transformsai_ai_core.redact import clear_secrets
from loguru import logger


class TestCentralLogger(unittest.TestCase):
    
    def setUp(self):
        cv_logger._is_configured = False
        cv_logger._run_id = None
        cv_logger._console_level = "INFO"
        cv_logger._file_level = "DEBUG"
        cv_logger._core_sink_ids.clear()
        cv_logger._patchers.clear()
        clear_secrets()
        logger.remove()

    def tearDown(self):
        logger.remove()
        cv_logger._core_sink_ids.clear()
        cv_logger._patchers.clear()
        clear_secrets()
        # Clean up all log files in .core-logs directory
        if os.path.exists(".core-logs"):
            for root, dirs, files in os.walk(".core-logs", topdown=False):
                for f in files:
                    try:
                        os.remove(os.path.join(root, f))
                    except Exception:
                        pass
                for d in dirs:
                    try:
                        os.rmdir(os.path.join(root, d))
                    except Exception:
                        pass
            try:
                os.rmdir(".core-logs")
            except Exception:
                pass

    def _get_log_directory(self):
        """Get the date-based log directory path."""
        now = datetime.now()
        return Path(".core-logs") / now.strftime("%Y-%m-%d")

    def _read_jsonl_logs(self, file_path=None):
        """Read logs from a specific file or find the latest log file."""
        logger.complete()
        logs = []
        
        if file_path is None:
            # Find the latest log file in the date directory
            log_dir = self._get_log_directory()
            if log_dir.exists():
                # Get all jsonl files and sort by name (timestamp-based)
                log_files = sorted(log_dir.glob("*.jsonl"))
                if log_files:
                    file_path = log_files[-1]  # Get the latest file
        
        if file_path and os.path.exists(file_path):
            with open(file_path, 'r') as f:
                for line in f:
                    if line.strip():
                        logs.append(json.loads(line))
        return logs

    def _find_log_files(self):
        """Find all log files in the current date directory."""
        log_dir = self._get_log_directory()
        if log_dir.exists():
            return sorted(log_dir.glob("*.jsonl"))
        return []

    def test_date_based_folder_creation(self):
        """Test that logs are created in a date-based subdirectory."""
        log = cv_logger.get_logger()
        log.info("Testing folder creation")
        logger.complete()
        
        log_dir = self._get_log_directory()
        self.assertTrue(
            log_dir.exists(),
            f"Date-based log directory not created: {log_dir}"
        )
        
        # Verify at least one log file exists in the directory
        log_files = list(log_dir.glob("*.jsonl"))
        self.assertTrue(
            len(log_files) >= 1,
            "No log files found in date directory"
        )

    def test_run_id_naming(self):
        """Test that log files use run_id and timestamp naming format."""
        log = cv_logger.get_logger()
        log.info("Testing run_id naming")
        logger.complete()
        
        log_files = self._find_log_files()
        self.assertTrue(len(log_files) >= 1, "No log files found")
        
        # Check naming format: {run_id}_{timestamp}.jsonl
        filename = log_files[0].name
        self.assertTrue(
            filename.endswith(".jsonl"),
            f"File should end with .jsonl: {filename}"
        )
        
        # Verify format: 6-char hex run_id + underscore + timestamp
        parts = filename.replace(".jsonl", "").split("_")
        self.assertTrue(len(parts) >= 2, f"Filename should have run_id_timestamp format: {filename}")
        self.assertTrue(len(parts[0]) == 6, f"Run ID should be 6 characters: {parts[0]}")

    def test_blind_naming(self):
        """Test logger name auto-detection from calling module."""
        log = cv_logger.get_logger()
        log.info("Testing blind name")
        logs = self._read_jsonl_logs()
        self.assertIn(logs[0]['logger_name'], ['test_logger', 'tests.test_logger', '__main__'])

    def test_explicit_naming(self):
        """Test explicit logger name assignment."""
        log = cv_logger.get_logger(name="DataAugmentationWorker")
        log.info("Testing explicit name")
        logs = self._read_jsonl_logs()
        self.assertEqual(logs[0]['logger_name'], "DataAugmentationWorker")

    def capture_console(self, fn):
        stderr_capture = io.StringIO()
        with redirect_stderr(stderr_capture):
            fn()
            logger.complete()
        return stderr_capture.getvalue()

    def test_console_level_info(self):
        """TRACE and DEBUG are not shown at console with INFO level; the file gets DEBUG."""
        def run():
            cv_logger.configure_logging(console_level="INFO")
            log = cv_logger.get_logger()
            log.trace("This is a trace")
            log.debug("This is a debug")
            log.info("This is an info")

        console_output = self.capture_console(run)
        self.assertIn("This is an info", console_output)
        self.assertNotIn("This is a debug", console_output)

        logs = self._read_jsonl_logs()
        self.assertEqual(len(logs), 2)
        self.assertEqual(logs[0]['level'], "DEBUG")

    def test_console_level_debug_from_config(self):
        def run():
            cv_logger.configure_logging(LoggingConfig(console_level="DEBUG"))
            cv_logger.get_logger().debug("This is a debug")

        self.assertIn("This is a debug", self.capture_console(run))

    def test_off_disables_a_sink(self):
        def run():
            cv_logger.configure_logging(LoggingConfig(console_level="OFF", file_level="INFO"))
            cv_logger.get_logger().warning("quiet console")

        self.assertEqual(self.capture_console(run), "")
        self.assertEqual(self._read_jsonl_logs()[0]["message"], "quiet console")
        cv_logger.configure_logging(file_level=None)
        self.assertEqual(len(cv_logger._core_sink_ids), 0)

    def test_configure_logging_leaves_project_sinks_alone(self):
        cv_logger.configure_logging(console_level="OFF")
        seen = []
        logger.add(lambda m: seen.append(m.record["message"]), level="INFO")
        cv_logger.configure_logging(console_level="OFF", file_level="WARNING")  # reconfigure later
        cv_logger.configure_logging(LoggingConfig(console_level="OFF"))
        cv_logger.get_logger().info("to the project sink")
        logger.complete()
        self.assertEqual(seen, ["to the project sink"])
        self.assertEqual(len(cv_logger._core_sink_ids), 1)

    def test_patcher_scrubs_message_and_extra(self):
        cv_logger.configure_logging(console_level="OFF", secrets=["literal-secret"])
        seen = []
        logger.add(lambda m: seen.append(m.record), level="INFO")
        cv_logger.get_logger().bind(source="rtsp://admin:pw9876@10.0.0.5/s").info(
            "opening rtsp://admin:pw9876@10.0.0.5/s with literal-secret")
        logger.complete()
        self.assertEqual(seen[0]["message"], "opening rtsp://admin:***@10.0.0.5/s with ***")
        self.assertEqual(seen[0]["extra"]["source"], "rtsp://admin:***@10.0.0.5/s")
        payload = self._read_jsonl_logs()[0]
        self.assertNotIn("pw9876", json.dumps(payload))
        self.assertNotIn("literal-secret", json.dumps(payload))

    def test_tracebacks_scrubbed_in_core_sinks(self):
        def run():
            cv_logger.configure_logging(console_level="INFO", secrets=["tb-secret-1"])
            try:
                raise RuntimeError("cannot open rtsp://admin:tb-secret-1@cam/s")
            except RuntimeError:
                cv_logger.get_logger().error("capture failed")

        console = self.capture_console(run)
        self.assertIn("capture failed", console)
        self.assertNotIn("tb-secret-1", console)
        payload = self._read_jsonl_logs()[0]
        self.assertIn("RuntimeError", payload["exception"])
        self.assertNotIn("tb-secret-1", payload["exception"])

    def test_add_patcher_chains_after_redaction(self):
        cv_logger.configure_logging(console_level="OFF")
        cv_logger.add_patcher(lambda record: record["extra"].update(tagged=True))
        seen = []
        logger.add(lambda m: seen.append(m.record["extra"].get("tagged")), level="INFO")
        cv_logger.get_logger().info("x")
        logger.complete()
        self.assertEqual(seen, [True])

    def test_redacting_sink_wrapper(self):
        cv_logger.configure_logging(console_level="OFF", file_level="OFF")
        out = io.StringIO()
        logger.add(cv_logger.redacting_sink(out), level="INFO", format="{message}")
        try:
            raise ValueError("token rtsp://u:wrapped-pw@h/s")
        except ValueError:
            logger.exception("boom")
        logger.complete()
        self.assertIn("boom", out.getvalue())
        self.assertNotIn("wrapped-pw", out.getvalue())

    def test_json_format(self):
        """Test that log entries have correct JSON structure."""
        log = cv_logger.get_logger()
        log.warning("Testing formatting")
        
        logs = self._read_jsonl_logs()
        # Test flat keys
        self.assertIn("level", logs[0])
        self.assertIn("timestamp", logs[0])
        self.assertIn("message", logs[0])
        self.assertIn("logger_name", logs[0])
        self.assertIn("file", logs[0])
        self.assertIn("line", logs[0])
        self.assertEqual(logs[0]['level'], "WARNING")

    def test_file_rotation(self):
        """Test auto-rotation when log file exceeds 10MB."""
        log = cv_logger.get_logger()
        
        # Send ~11MB of text to trigger rotation
        huge_text = "X" * (11 * 1024 * 1024)
        log.info(huge_text)
        
        # Send another log to trigger rotation (file size > 10MB)
        log.info("This log should trigger rotation")
        
        # Wait for async queue to flush
        logger.complete()

        # Find all log files for this run
        log_files = self._find_log_files()
        
        # Should have at least 2 files after rotation
        self.assertTrue(
            len(log_files) >= 2,
            f"Expected at least 2 files after rotation, found {len(log_files)}"
        )
        
        # The latest file should contain only the rotation trigger message
        latest_logs = self._read_jsonl_logs(log_files[-1])
        self.assertEqual(len(latest_logs), 1)
        self.assertEqual(latest_logs[0]["message"], "This log should trigger rotation")
        
        # The first file should contain the huge text
        first_logs = self._read_jsonl_logs(log_files[0])
        self.assertTrue(len(first_logs[0]["message"]) > 10 * 1024 * 1024)


if __name__ == '__main__':
    unittest.main(verbosity=2)