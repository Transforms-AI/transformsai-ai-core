import os
import sys
import json
import glob
import unittest
import io
from contextlib import redirect_stderr
from datetime import datetime
from pathlib import Path
import transformsai_ai_core.central_logger as cv_logger
from loguru import logger


class TestCentralLogger(unittest.TestCase):
    
    def setUp(self):
        cv_logger._is_configured = False
        cv_logger._run_id = None
        logger.remove()

    def tearDown(self):
        logger.remove()
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
        self.assertIn(logs[0]['logger_name'], ['test_logger', '__main__'])

    def test_explicit_naming(self):
        """Test explicit logger name assignment."""
        log = cv_logger.get_logger(name="DataAugmentationWorker")
        log.info("Testing explicit name")
        logs = self._read_jsonl_logs()
        self.assertEqual(logs[0]['logger_name'], "DataAugmentationWorker")

    def test_cli_sink_level_info(self):
        """Test that TRACE and DEBUG are not shown at console with INFO level."""
        stderr_capture = io.StringIO()
        with redirect_stderr(stderr_capture):
            log = cv_logger.get_logger(cli_sink_level="INFO")
            log.trace("This is a trace")
            log.debug("This is a debug")
            log.info("This is an info")
            logger.complete()

        console_output = stderr_capture.getvalue()
        self.assertIn("This is an info", console_output)
        self.assertNotIn("This is a debug", console_output)
        
        logs = self._read_jsonl_logs()
        self.assertEqual(len(logs), 3)
        self.assertEqual(logs[0]['level'], "TRACE")

    def test_cli_sink_level_debug(self):
        """Test that DEBUG is shown at console with DEBUG level."""
        stderr_capture = io.StringIO()
        with redirect_stderr(stderr_capture):
            log = cv_logger.get_logger(cli_sink_level="DEBUG")
            log.debug("This is a debug")
            logger.complete()

        console_output = stderr_capture.getvalue()
        self.assertIn("This is a debug", console_output)

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